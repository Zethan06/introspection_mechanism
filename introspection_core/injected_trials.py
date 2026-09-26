"""Injected-trial helpers shared by the STE, head-patching and Section 4 scripts.

A trial is one (concept, prompt cluster, candidate position) injection. These
helpers build injected batches, classify natural outcomes, persist the prepared
splits, score the gate log-odds d_tau, and start one distributed worker per GPU.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Sequence

import torch

from introspection_core.injection import make_injection_hook
from introspection_core.model import HookedModel


POSITION_LABELS = tuple(str(index) for index in range(10))
NONE_LABEL = "none"
ALL_LABELS = (*POSITION_LABELS, NONE_LABEL)


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int = 0
    world_size: int = 1

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


def initialize_distributed(enabled: bool) -> DistributedContext:
    """Initialize one NCCL worker per launcher-assigned visible GPU."""

    if not enabled:
        return DistributedContext(enabled=False)
    import torch.distributed as dist

    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(f"Distributed environment is missing: {missing}")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(minutes=30),
    )
    return DistributedContext(True, rank=rank, world_size=world_size)


@dataclass(frozen=True)
class InjectedTrial:
    """One injected trial and its natural outcome."""

    concept_index: int
    cluster_index: int
    position: int
    correct: bool
    predicted_none: bool


def parse_layers(spec: str) -> list[int]:
    """Parse comma-separated layers and inclusive ranges."""

    layers: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(value) for value in part.split("-", 1))
            if end < start:
                raise ValueError(f"Invalid descending layer range: {part}")
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))
    if not layers:
        raise ValueError("layer spec must select at least one layer")
    return sorted(layers)


def _csv_bool(value: object) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no", ""}:
        return False
    raise ValueError(f"Cannot parse boolean CSV value: {value!r}")


def load_injected_trials(path: Path) -> list[InjectedTrial]:
    """Load correct and ``none`` injected outcomes, excluding wrong positions."""

    trials: list[InjectedTrial] = []
    seen: set[tuple[int, int, int]] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "concept_index",
            "cluster_index",
            "position",
            "condition",
            "correct",
            "predicted_none",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Outcome CSV is missing columns: {sorted(missing)}")
        for row in reader:
            if row["condition"] != "injected":
                continue
            position = int(row["position"])
            if position not in range(10):
                continue
            correct = _csv_bool(row["correct"])
            predicted_none = _csv_bool(row["predicted_none"])
            if not correct and not predicted_none:
                # Wrong-location outputs mix detection and localization errors.
                continue
            trial = InjectedTrial(
                concept_index=int(row["concept_index"]),
                cluster_index=int(row["cluster_index"]),
                position=position,
                correct=correct,
                predicted_none=predicted_none,
            )
            key = (trial.concept_index, trial.cluster_index, trial.position)
            if key in seen:
                raise ValueError(f"Duplicate injected trial key: {key}")
            seen.add(key)
            trials.append(trial)
    if not trials:
        raise ValueError(f"No usable injected outcomes found in {path}")
    return trials


@torch.inference_mode()
def infer_split_outcomes(
    model: HookedModel,
    *,
    concepts: Sequence[str],
    concept_vectors: torch.Tensor,
    base_tokens: torch.Tensor,
    injection_token_positions: torch.Tensor,
    candidate_token_ids: Sequence[int],
    injection_layer: int,
    strength: float,
    scale_mode: str,
    batch_size: int,
    context: DistributedContext,
    progress_every: int,
    split_name: str = "train",
) -> list[InjectedTrial]:
    """Classify the natural injected outcomes for the current dataset split."""

    local_indices = list(range(context.rank, len(concepts), context.world_size))
    coordinates = [
        (concept_index, cluster_index, position)
        for concept_index in local_indices
        for cluster_index in range(base_tokens.shape[0])
        for position in range(10)
    ]
    local_trials: list[InjectedTrial] = []
    for batch_number, start in enumerate(range(0, len(coordinates), batch_size), 1):
        batch = coordinates[start : start + batch_size]
        placeholders = [
            InjectedTrial(concept_index, cluster_index, position, False, False)
            for concept_index, cluster_index, position in batch
        ]
        tokens, injection_hook = build_injected_batch(
            placeholders,
            base_tokens=base_tokens,
            injection_token_positions=injection_token_positions,
            concept_vectors=concept_vectors,
            model=model,
            injection_layer=injection_layer,
            strength=strength,
            scale_mode=scale_mode,
        )
        logits, _ = model.last_token_candidate_stats(
            tokens,
            candidate_token_ids=candidate_token_ids,
            fwd_hooks=[injection_hook],
        )
        predictions = logits.argmax(dim=-1).tolist()
        for placeholder, prediction in zip(placeholders, predictions, strict=True):
            correct = int(prediction) == placeholder.position
            predicted_none = int(prediction) == 10
            if correct or predicted_none:
                local_trials.append(
                    InjectedTrial(
                        placeholder.concept_index,
                        placeholder.cluster_index,
                        placeholder.position,
                        correct,
                        predicted_none,
                    )
                )
        if progress_every > 0 and batch_number % progress_every == 0:
            print(
                f"rank={context.rank} classified {split_name}-split outcomes "
                f"{min(start + batch_size, len(coordinates))}/{len(coordinates)}",
                flush=True,
            )
    if not context.enabled:
        return local_trials
    import torch.distributed as dist

    gathered: list[list[InjectedTrial] | None] = [None] * context.world_size
    dist.all_gather_object(gathered, local_trials)
    return [trial for shard in gathered for trial in (shard or [])]


def write_prepared_split(
    output_dir: Path,
    *,
    concepts: Sequence[str],
    concept_vectors: torch.Tensor,
    trials: Sequence[InjectedTrial],
    injection_layer: int,
    split_name: str = "train",
) -> None:
    """Persist one split's concept vectors and classified injected outcomes."""

    prepared = output_dir / f"prepared_{split_name}_split"
    prepared.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "concepts": list(concepts),
            "unit_vectors": concept_vectors.float().cpu(),
            "layer": injection_layer,
        },
        prepared / "concept_vectors.pt",
    )
    with (prepared / "outcomes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "concept_index",
                "cluster_index",
                "position",
                "condition",
                "correct",
                "predicted_none",
            ],
        )
        writer.writeheader()
        for trial in sorted(
            trials,
            key=lambda item: (item.concept_index, item.cluster_index, item.position),
        ):
            writer.writerow(
                {
                    "concept_index": trial.concept_index,
                    "cluster_index": trial.cluster_index,
                    "position": trial.position,
                    "condition": "injected",
                    "correct": int(trial.correct),
                    "predicted_none": int(trial.predicted_none),
                }
            )


def gate_logit(
    candidate_logits: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Return normalized d_tau(z) for TOKEN 0..9 versus ``none``."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if candidate_logits.dim() != 2 or candidate_logits.shape[-1] != len(ALL_LABELS):
        raise ValueError(
            "candidate_logits must be [batch, 11] ordered as 0..9, none; "
            f"got {tuple(candidate_logits.shape)}"
        )
    logits = candidate_logits.float()
    number_logit = temperature * torch.logsumexp(
        logits[:, :10] / temperature,
        dim=-1,
    )
    number_logit -= temperature * math.log(len(POSITION_LABELS))
    return number_logit - logits[:, 10]


def validate_token0_9_candidate_layout(examples: Sequence) -> list[int]:
    labels = tuple(examples[0].candidate_token_ids)
    if labels != ALL_LABELS:
        raise ValueError(
            "Prompt candidates must be ordered exactly as 0..9, none; "
            f"got {labels}"
        )
    if tuple(examples[0].positions) != tuple(range(10)):
        raise ValueError(
            f"Prompt positions must be TOKEN 0..9; got {examples[0].positions}"
        )
    return [int(examples[0].candidate_token_ids[label]) for label in ALL_LABELS]


def build_injected_batch(
    trials: Sequence[InjectedTrial],
    *,
    base_tokens: torch.Tensor,
    injection_token_positions: torch.Tensor,
    concept_vectors: torch.Tensor,
    model: HookedModel,
    injection_layer: int,
    strength: float,
    scale_mode: str,
) -> tuple[torch.Tensor, tuple[str, object]]:
    cluster_indices = torch.tensor([trial.cluster_index for trial in trials], dtype=torch.long)
    concept_indices = torch.tensor([trial.concept_index for trial in trials], dtype=torch.long)
    positions = torch.tensor([trial.position for trial in trials], dtype=torch.long)
    tokens = base_tokens.index_select(0, cluster_indices).to(model.bridge.cfg.device)
    token_positions = injection_token_positions[cluster_indices, positions]
    vectors = concept_vectors.index_select(0, concept_indices)
    spans = [(int(position), int(position) + 1) for position in token_positions.tolist()]
    hook = make_injection_hook(
        model,
        positions=spans,
        vector=vectors,
        layer=injection_layer,
        strength=strength,
        scale=scale_mode,
    )
    return tokens, hook
