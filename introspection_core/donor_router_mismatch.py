"""Metrics for donor-position by router-position interventions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch

from .injected_trials import InjectedTrial, gate_logit


POSITION_COUNT = 10
NONE_COLUMN = 10


def sharded_complete_position_batches(
    trials: Sequence[InjectedTrial],
    *,
    batch_size: int,
    rank: int,
    world_size: int,
) -> list[tuple[InjectedTrial, ...]]:
    """Shard complete concept-cluster position groups into bounded batches."""

    if batch_size < POSITION_COUNT:
        raise ValueError(
            f"batch_size must be at least {POSITION_COUNT} for router patching"
        )
    if not 0 <= rank < world_size:
        raise ValueError("rank must lie in [0, world_size)")
    grouped: dict[tuple[int, int], dict[int, InjectedTrial]] = {}
    for trial in trials:
        key = (int(trial.concept_index), int(trial.cluster_index))
        position = int(trial.position)
        if not 0 <= position < POSITION_COUNT:
            raise ValueError(f"trial position must be in [0, {POSITION_COUNT - 1}]")
        positions = grouped.setdefault(key, {})
        if position in positions:
            raise ValueError(f"duplicate trial coordinate {(*key, position)}")
        positions[position] = trial
    expected_positions = set(range(POSITION_COUNT))
    incomplete = {
        key: sorted(expected_positions - set(positions))
        for key, positions in grouped.items()
        if set(positions) != expected_positions
    }
    if incomplete:
        raise ValueError(f"incomplete position groups: {incomplete}")

    ordered_groups = [
        tuple(positions[position] for position in range(POSITION_COUNT))
        for positions in grouped.values()
    ]
    local_groups = ordered_groups[rank::world_size]
    groups_per_batch = batch_size // POSITION_COUNT
    return [
        tuple(
            trial
            for group in local_groups[start : start + groups_per_batch]
            for trial in group
        )
        for start in range(0, len(local_groups), groups_per_batch)
    ]


def router_source_indices(
    trials: Sequence[InjectedTrial],
    *,
    router_position: int,
) -> torch.Tensor:
    """Map every recipient row to its paired injected router-donor row."""

    if not 0 <= router_position < POSITION_COUNT:
        raise ValueError(
            f"router_position must be in [0, {POSITION_COUNT - 1}]"
        )
    lookup = {
        (
            int(trial.concept_index),
            int(trial.cluster_index),
            int(trial.position),
        ): index
        for index, trial in enumerate(trials)
    }
    if len(lookup) != len(trials):
        raise ValueError("trials contain duplicate coordinates")
    indices: list[int] = []
    for trial in trials:
        key = (
            int(trial.concept_index),
            int(trial.cluster_index),
            router_position,
        )
        if key not in lookup:
            raise ValueError(f"missing router donor coordinate {key}")
        indices.append(lookup[key])
    return torch.tensor(indices, dtype=torch.long)


@dataclass
class DonorRouterAccumulator:
    """Accumulate one fixed donor-position/router-position cell."""

    donor_position: int
    router_position: int
    n: int = 0
    gate_sum: float = 0.0
    none_probability_sum: float = 0.0
    number_probability_sum: float = 0.0
    donor_probability_sum: float = 0.0
    router_probability_sum: float = 0.0
    prediction_counts: list[int] = field(
        default_factory=lambda: [0] * (POSITION_COUNT + 1)
    )

    def __post_init__(self) -> None:
        for name, value in (
            ("donor_position", self.donor_position),
            ("router_position", self.router_position),
        ):
            if not 0 <= int(value) < POSITION_COUNT:
                raise ValueError(f"{name} must be in [0, {POSITION_COUNT - 1}]")
        if len(self.prediction_counts) != POSITION_COUNT + 1:
            raise ValueError("prediction_counts must contain 11 values")

    def update(self, logits: torch.Tensor, *, gate_temperature: float) -> None:
        """Add candidate logits with columns TOKEN 0..9 followed by ``none``."""

        selected = logits.detach().float()
        if selected.dim() != 2 or selected.shape[1] != POSITION_COUNT + 1:
            raise ValueError("logits must have shape [batch, 11]")
        if selected.shape[0] == 0:
            return
        probabilities = selected.softmax(dim=-1)
        predictions = selected.argmax(dim=-1).cpu()
        self.n += int(selected.shape[0])
        self.gate_sum += float(
            gate_logit(selected, temperature=gate_temperature).sum()
        )
        self.none_probability_sum += float(probabilities[:, NONE_COLUMN].sum())
        self.number_probability_sum += float(
            probabilities[:, :POSITION_COUNT].sum()
        )
        self.donor_probability_sum += float(
            probabilities[:, self.donor_position].sum()
        )
        self.router_probability_sum += float(
            probabilities[:, self.router_position].sum()
        )
        counts = torch.bincount(
            predictions, minlength=POSITION_COUNT + 1
        ).tolist()
        self.prediction_counts = [
            old + int(value)
            for old, value in zip(self.prediction_counts, counts)
        ]

    def tensor(self, device: torch.device | str) -> torch.Tensor:
        """Return an all-reduce-friendly representation."""

        return torch.tensor(
            [
                self.n,
                self.gate_sum,
                self.none_probability_sum,
                self.number_probability_sum,
                self.donor_probability_sum,
                self.router_probability_sum,
                *self.prediction_counts,
            ],
            dtype=torch.float64,
            device=device,
        )

    @classmethod
    def from_tensor(
        cls,
        values: torch.Tensor,
        *,
        donor_position: int,
        router_position: int,
    ) -> "DonorRouterAccumulator":
        """Reconstruct an accumulator after distributed reduction."""

        items = values.detach().cpu().tolist()
        if len(items) != 6 + POSITION_COUNT + 1:
            raise ValueError("unexpected reduced accumulator width")
        return cls(
            donor_position=donor_position,
            router_position=router_position,
            n=int(items[0]),
            gate_sum=float(items[1]),
            none_probability_sum=float(items[2]),
            number_probability_sum=float(items[3]),
            donor_probability_sum=float(items[4]),
            router_probability_sum=float(items[5]),
            prediction_counts=[int(value) for value in items[6:]],
        )

    def row(self, *, intervention: str) -> dict[str, object]:
        """Summarize output identity and candidate probability for one cell."""

        if self.n <= 0:
            raise RuntimeError("cannot summarize zero examples")
        donor_count = self.prediction_counts[self.donor_position]
        router_count = self.prediction_counts[self.router_position]
        none_count = self.prediction_counts[NONE_COLUMN]
        number_count = self.n - none_count
        same_position = self.donor_position == self.router_position
        other_number_count = number_count - donor_count
        if not same_position:
            other_number_count -= router_count
        return {
            "intervention": intervention,
            "donor_position": self.donor_position,
            "router_position": self.router_position,
            "position_relation": "diagonal" if same_position else "mismatch",
            "n_trials": self.n,
            "none_rate": none_count / self.n,
            "number_rate": number_count / self.n,
            "overall_donor_accuracy": donor_count / self.n,
            "overall_router_accuracy": router_count / self.n,
            "other_number_rate": other_number_count / self.n,
            "mean_none_probability": self.none_probability_sum / self.n,
            "mean_number_probability": self.number_probability_sum / self.n,
            "mean_donor_probability": self.donor_probability_sum / self.n,
            "mean_router_probability": self.router_probability_sum / self.n,
            "mean_gate_score": self.gate_sum / self.n,
        }


def aggregate_position_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """Compute trial-weighted diagonal/mismatch summaries by intervention."""

    materialized = list(rows)
    metrics = (
        "none_rate",
        "number_rate",
        "overall_donor_accuracy",
        "overall_router_accuracy",
        "other_number_rate",
        "mean_none_probability",
        "mean_number_probability",
        "mean_donor_probability",
        "mean_router_probability",
        "mean_gate_score",
    )
    summaries: list[dict[str, object]] = []
    interventions = list(
        dict.fromkeys(str(row["intervention"]) for row in materialized)
    )
    for intervention in interventions:
        for relation in ("diagonal", "mismatch"):
            selected = [
                row
                for row in materialized
                if row["intervention"] == intervention
                and row["position_relation"] == relation
            ]
            if not selected:
                continue
            total = sum(int(row["n_trials"]) for row in selected)
            summary: dict[str, object] = {
                "intervention": intervention,
                "position_relation": relation,
                "n_cells": len(selected),
                "n_trials": total,
            }
            for metric in metrics:
                summary[metric] = sum(
                    float(row[metric]) * int(row["n_trials"])
                    for row in selected
                ) / total
            summaries.append(summary)
    return summaries
