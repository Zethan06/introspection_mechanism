"""Evaluation loop for the router-head attention redirection (inject i, attend t_j)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from .attention_inputs import AttentionExample, validate_aligned_layout
from .attention_routing import (
    candidate_routing_key_positions,
    one_hot_attention_hook,
)
from .injection import inject


@dataclass
class AttentionRoutingEvaluation:
    """Tables and aggregate statistics produced by a routing sweep."""

    clean_controls: list[dict]
    natural_injected: list[dict]
    sweep_grid: list[dict]
    summary: dict


@dataclass
class _MetricAccumulator:
    """Accumulate localization metrics over batches of (concept, example) pairs.

    ``expected_column`` is the candidate column scored as correct and
    ``routed_column`` the one scored as "followed the forced destination".
    Pass an ``int`` when every example answers a slot with the same label, which
    is the case for every ascending label prompt. Pass ``None`` when the label
    printed at a slot varies by example -- the shuffled-label control draws one
    derangement per cluster -- and hand :meth:`update` that batch's per-row
    columns. A ``None`` column with no per-row columns raises instead of
    silently scoring every cluster against the first one's permutation, and
    ``track_routed`` keeps "report follow metrics" separable from "the followed
    label is shared".
    """

    labels: tuple[str, ...]
    expected_column: int | None
    routed_column: int | None = None
    track_routed: bool = False
    n: int = 0
    correct: int = 0
    correct_prob_sum: float = 0.0
    routed_correct: int = 0
    routed_prob_sum: float = 0.0
    prediction_counts: list[int] | None = None

    def __post_init__(self) -> None:
        self.prediction_counts = [0] * len(self.labels)
        if self.routed_column is not None:
            self.track_routed = True

    def _columns(
        self,
        shared: int | None,
        per_row: torch.Tensor | None,
        predictions: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        if shared is None:
            if per_row is None:
                raise ValueError(
                    f"{name} varies by example; pass {name}s to update()"
                )
            columns = per_row.to(device=predictions.device, dtype=torch.long)
            if columns.shape != predictions.shape:
                raise ValueError(
                    f"{name}s has shape {tuple(columns.shape)} but the batch "
                    f"holds {tuple(predictions.shape)} predictions"
                )
            return columns
        if per_row is not None:
            raise ValueError(
                f"{name} is shared across examples; do not also pass {name}s"
            )
        return torch.full_like(predictions, int(shared))

    def update(
        self,
        candidate_logits: torch.Tensor,
        candidate_log_probs: torch.Tensor,
        *,
        expected_columns: torch.Tensor | None = None,
        routed_columns: torch.Tensor | None = None,
    ) -> None:
        predictions = candidate_logits.argmax(dim=-1)
        count = int(predictions.numel())
        self.n += count
        expected = self._columns(
            self.expected_column, expected_columns, predictions, "expected_column"
        )
        self.correct += int(predictions.eq(expected).sum())
        self.correct_prob_sum += float(
            candidate_log_probs.gather(1, expected.unsqueeze(1)).exp().sum()
        )
        assert self.prediction_counts is not None
        counts = torch.bincount(predictions, minlength=len(self.labels))
        for index, value in enumerate(counts.tolist()):
            self.prediction_counts[index] += int(value)
        if self.track_routed:
            routed = self._columns(
                self.routed_column, routed_columns, predictions, "routed_column"
            )
            self.routed_correct += int(predictions.eq(routed).sum())
            self.routed_prob_sum += float(
                candidate_log_probs.gather(1, routed.unsqueeze(1)).exp().sum()
            )

    def row(self) -> dict:
        if self.n == 0:
            raise RuntimeError("cannot summarize an empty metric accumulator")
        assert self.prediction_counts is not None
        result = {
            "n_trials": self.n,
            "n_correct": self.correct,
            "accuracy": self.correct / self.n,
            "mean_correct_prob": self.correct_prob_sum / self.n,
            "prediction_counts": json.dumps(
                dict(zip(self.labels, self.prediction_counts)),
                sort_keys=True,
            ),
        }
        if self.track_routed:
            result.update(
                {
                    "n_follow_attention": self.routed_correct,
                    "follow_attention_accuracy": self.routed_correct / self.n,
                    "mean_attention_target_prob": self.routed_prob_sum / self.n,
                }
            )
        return result


def _weighted_mean(rows: list[dict], field: str) -> float:
    total = sum(int(row["n_trials"]) for row in rows)
    if total == 0:
        return float("nan")
    return sum(float(row[field]) * int(row["n_trials"]) for row in rows) / total


def _shared_answer_key(examples: list[AttentionExample]) -> bool:
    """Report whether every example answers each slot with the same label.

    True for every ascending label prompt (digits, letters, number words), and
    False for the shuffled-label control, which permutes the label printed at
    each slot per cluster. Callers that score against ``examples[0]`` alone are
    only correct in the shared case.
    """
    if not examples:
        raise ValueError("examples must be non-empty")
    first = examples[0]
    return all(
        example.expected_candidate_by_position == first.expected_candidate_by_position
        for example in examples[1:]
    )


def _answer_key_columns(
    examples: list[AttentionExample],
    positions: Sequence[int],
    label_columns: dict[str, int],
    *,
    shared_answer_key: bool,
) -> tuple[
    dict[int, torch.Tensor],
    Callable[[int], int | None],
    Callable[[int, torch.Tensor], torch.Tensor | None],
]:
    """Build the three ways a slot's scored candidate column gets looked up.

    ``table`` holds one column per (slot, example) in ``examples`` order, so a
    batch's columns are one ``index_select`` on the same example indices used to
    gather its tokens. ``shared`` returns that slot's single column when every
    example agrees on it and ``None`` otherwise, and ``per_batch`` is its
    counterpart: ``None`` in the shared case, that batch's column vector when
    the answer key is per example. Feeding both into
    :class:`_MetricAccumulator` makes the shared case take the scalar path and
    the shuffled case the gathered one, with no branch at the call sites.
    """
    table = {
        int(position): torch.tensor(
            [
                label_columns[
                    example.expected_candidate_by_position[int(position)]
                ]
                for example in examples
            ],
            dtype=torch.long,
        )
        for position in positions
    }

    def shared(position: int) -> int | None:
        if not shared_answer_key:
            return None
        return label_columns[
            examples[0].expected_candidate_by_position[int(position)]
        ]

    def per_batch(
        position: int, example_indices: torch.Tensor
    ) -> torch.Tensor | None:
        if shared_answer_key:
            return None
        return table[int(position)].index_select(0, example_indices)

    return table, shared, per_batch


def _validate_layout(
    examples: list[AttentionExample],
    concept_names: Sequence[str],
    unit_vectors: torch.Tensor,
    injection_position: int,
    attention_positions: Sequence[int] | None,
    *,
    shared_answer_key: bool = True,
) -> tuple[tuple[int, ...], tuple[str, ...], dict[str, int]]:
    validate_aligned_layout(examples, shared_answer_key=shared_answer_key)
    if unit_vectors.dim() != 2 or unit_vectors.shape[0] != len(concept_names):
        raise ValueError(
            f"unit_vectors shape {tuple(unit_vectors.shape)} does not match "
            f"{len(concept_names)} concepts"
        )
    if not concept_names:
        raise ValueError("concept_names must be non-empty")

    positions = tuple(int(position) for position in examples[0].positions)
    labels = tuple(examples[0].candidate_token_ids)
    label_columns = {label: index for index, label in enumerate(labels)}
    # Every example's answer key is checked, not just the first one's: with a
    # per-cluster permutation the first cluster can be fully in range while a
    # later one names a label the scored candidate set never offers.
    for example in examples:
        for position in positions:
            expected = example.expected_candidate_by_position[position]
            if expected not in label_columns:
                raise ValueError(
                    f"position {position} of {example.key!r} expects missing "
                    f"candidate label {expected!r}"
                )
    if int(injection_position) not in positions:
        raise ValueError(
            f"injection position {injection_position} is not in prompt positions "
            f"{positions}"
        )
    if attention_positions is not None:
        unknown = sorted(set(int(x) for x in attention_positions) - set(positions))
        if unknown:
            raise ValueError(
                f"attention positions {unknown} are not in prompt positions {positions}"
            )
    return positions, labels, label_columns


def evaluate_attention_routing(
    model,
    *,
    examples: list[AttentionExample],
    concept_names: Sequence[str],
    unit_vectors: torch.Tensor,
    injection_layer: int,
    strength: float,
    scale_mode: str,
    attention_layer: int,
    heads: Sequence[int],
    batch_size: int,
    injection_position: int,
    attention_positions: Sequence[int] | None = None,
    attention_target: str = "newline",
    query_position: int = -1,
    include_clean_controls: bool = True,
    progress_prefix: str = "",
) -> AttentionRoutingEvaluation:
    """Run natural and one-hot-routed localization conditions.

    Every injected cell contains the full concept-by-example cross product.
    Rows of ``sweep_grid`` hold residual injection at the selected item ``i``
    while final-query attention is routed to each requested item/newline ``j``.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    shared_answer_key = _shared_answer_key(examples)
    positions, labels, label_columns = _validate_layout(
        examples,
        concept_names,
        unit_vectors,
        injection_position,
        attention_positions,
        shared_answer_key=shared_answer_key,
    )
    injection_position = int(injection_position)
    route_positions = (
        positions
        if attention_positions is None
        else tuple(int(position) for position in attention_positions)
    )
    if attention_target not in {"newline", "item"}:
        raise ValueError("attention_target must be 'newline' or 'item'")

    key_by_example: list[dict[int, int]] = []
    for example in examples:
        if attention_target == "newline":
            key_by_example.append(
                candidate_routing_key_positions(model.tokenizer, example)
            )
        else:
            key_by_example.append(
                {
                    int(position): int(example.injection_spans[position].start)
                    for position in positions
                }
            )

    candidate_token_ids = tuple(
        int(examples[0].candidate_token_ids[label]) for label in labels
    )
    column_table, shared_column, batch_columns = _answer_key_columns(
        examples,
        positions,
        label_columns,
        shared_answer_key=shared_answer_key,
    )

    base_tokens = torch.cat([example.input_ids for example in examples], dim=0)
    n_examples = len(examples)
    n_concepts = len(concept_names)
    device = model.bridge.cfg.device
    dtype = model.bridge.cfg.dtype
    vectors = unit_vectors.to(device=device, dtype=dtype)

    clean_controls: list[dict] = []
    if include_clean_controls:
        clean_logits, clean_log_probs = model.last_token_candidate_stats(
            base_tokens,
            candidate_token_ids=candidate_token_ids,
        )
        clean_predictions = clean_logits.argmax(dim=-1)
        clean_controls.append(
            {
                "condition": "natural_clean",
                "attention_position": None,
                "attention_key_token_index": None,
                "n_examples": n_examples,
                "follow_attention_accuracy": None,
                "mean_attention_target_prob": None,
                "prediction_counts": json.dumps(
                    dict(
                        zip(
                            labels,
                            torch.bincount(
                                clean_predictions, minlength=len(labels)
                            ).tolist(),
                        )
                    ),
                    sort_keys=True,
                ),
            }
        )
        for route_position in route_positions:
            route_columns = column_table[int(route_position)]
            routed_logits: list[torch.Tensor] = []
            routed_log_probs: list[torch.Tensor] = []
            for start in range(0, n_examples, batch_size):
                end = min(start + batch_size, n_examples)
                keys = [
                    key_by_example[index][route_position]
                    for index in range(start, end)
                ]
                hook = one_hot_attention_hook(
                    model,
                    layer=attention_layer,
                    heads=heads,
                    key_positions=keys,
                    query_position=query_position,
                )
                logits, log_probs = model.last_token_candidate_stats(
                    base_tokens[start:end],
                    candidate_token_ids=candidate_token_ids,
                    fwd_hooks=[hook],
                )
                routed_logits.append(logits)
                routed_log_probs.append(log_probs)
            logits = torch.cat(routed_logits, dim=0)
            log_probs = torch.cat(routed_log_probs, dim=0)
            predictions = logits.argmax(dim=-1)
            # Rows are every example in order here, not a (concept, example)
            # batch, so the table indexes straight across without a gather.
            route_follow_columns = route_columns.to(predictions.device)
            clean_controls.append(
                {
                    "condition": "onehot_clean",
                    "attention_position": route_position,
                    "attention_key_token_index": key_by_example[0][route_position],
                    "n_examples": n_examples,
                    "follow_attention_accuracy": float(
                        predictions.eq(route_follow_columns).float().mean()
                    ),
                    "mean_attention_target_prob": float(
                        log_probs.gather(1, route_follow_columns.unsqueeze(1))
                        .exp()
                        .mean()
                    ),
                    "prediction_counts": json.dumps(
                        dict(
                            zip(
                                labels,
                                torch.bincount(
                                    predictions, minlength=len(labels)
                                ).tolist(),
                            )
                        ),
                        sort_keys=True,
                    ),
                }
            )

    pair_count = n_concepts * n_examples
    all_pair_indices = torch.arange(pair_count)
    all_concept_indices = torch.div(
        all_pair_indices, n_examples, rounding_mode="floor"
    )
    all_example_indices = all_pair_indices.remainder(n_examples)

    natural_rows: list[dict] = []
    sweep_rows: list[dict] = []
    for injection_offset, injection_position in enumerate(
        (injection_position,), start=1
    ):
        expected_column = shared_column(injection_position)
        natural_metric = _MetricAccumulator(labels, expected_column)

        for start in range(0, pair_count, batch_size):
            end = min(start + batch_size, pair_count)
            concept_indices = all_concept_indices[start:end]
            example_indices = all_example_indices[start:end]
            batch_tokens = base_tokens.index_select(0, example_indices)
            batch_vectors = vectors.index_select(
                0, concept_indices.to(device=device)
            )
            spans = [
                (
                    int(examples[index].injection_spans[injection_position].start),
                    int(examples[index].injection_spans[injection_position].end),
                )
                for index in example_indices.tolist()
            ]
            with inject(
                model,
                layer=injection_layer,
                positions=spans,
                vector=batch_vectors,
                strength=strength,
                scale=scale_mode,
            ):
                logits, log_probs = model.last_token_candidate_stats(
                    batch_tokens,
                    candidate_token_ids=candidate_token_ids,
                )
            natural_metric.update(
                logits,
                log_probs,
                expected_columns=batch_columns(
                    injection_position, example_indices
                ),
            )

        natural_row = {
            "condition": "natural_injected",
            "injection_position": injection_position,
            **natural_metric.row(),
        }
        natural_rows.append(natural_row)

        for route_position in route_positions:
            metric = _MetricAccumulator(
                labels,
                expected_column,
                shared_column(route_position),
                track_routed=True,
            )
            for start in range(0, pair_count, batch_size):
                end = min(start + batch_size, pair_count)
                concept_indices = all_concept_indices[start:end]
                example_indices = all_example_indices[start:end]
                batch_tokens = base_tokens.index_select(0, example_indices)
                batch_vectors = vectors.index_select(
                    0, concept_indices.to(device=device)
                )
                example_index_list = example_indices.tolist()
                spans = [
                    (
                        int(
                            examples[index]
                            .injection_spans[injection_position]
                            .start
                        ),
                        int(
                            examples[index]
                            .injection_spans[injection_position]
                            .end
                        ),
                    )
                    for index in example_index_list
                ]
                keys = [
                    key_by_example[index][route_position]
                    for index in example_index_list
                ]
                hook = one_hot_attention_hook(
                    model,
                    layer=attention_layer,
                    heads=heads,
                    key_positions=keys,
                    query_position=query_position,
                )
                with inject(
                    model,
                    layer=injection_layer,
                    positions=spans,
                    vector=batch_vectors,
                    strength=strength,
                    scale=scale_mode,
                ):
                    logits, log_probs = model.last_token_candidate_stats(
                        batch_tokens,
                        candidate_token_ids=candidate_token_ids,
                        fwd_hooks=[hook],
                    )
                metric.update(
                    logits,
                    log_probs,
                    expected_columns=batch_columns(
                        injection_position, example_indices
                    ),
                    routed_columns=batch_columns(route_position, example_indices),
                )

            row = {
                "condition": "onehot_injected",
                "injection_position": injection_position,
                "attention_position": route_position,
                "attention_key_token_index": key_by_example[0][route_position],
                "is_diagonal": int(injection_position == route_position),
                **metric.row(),
            }
            row["accuracy_delta_vs_natural_injected"] = (
                row["accuracy"] - natural_row["accuracy"]
            )
            sweep_rows.append(row)
            print(
                f"{progress_prefix}[{injection_offset}/1] "
                f"inject={injection_position} attend={route_position} "
                f"acc={row['accuracy']:.4f} "
                f"follow={row['follow_attention_accuracy']:.4f}",
                flush=True,
            )

    diagonal = [row for row in sweep_rows if row["is_diagonal"]]
    off_diagonal = [row for row in sweep_rows if not row["is_diagonal"]]
    diagonal_positions = {
        int(row["injection_position"]) for row in diagonal
    }
    diagonal_natural_rows = [
        row
        for row in natural_rows
        if int(row["injection_position"]) in diagonal_positions
    ]
    summary = {
        "n_concepts": n_concepts,
        "n_examples": n_examples,
        "answer_key": "shared" if shared_answer_key else "per_example",
        "positions": [injection_position],
        "injection_position": injection_position,
        "attention_positions": list(route_positions),
        "trials_per_cell": pair_count,
        "natural_injected_accuracy": _weighted_mean(natural_rows, "accuracy"),
        "onehot_injected_accuracy": _weighted_mean(sweep_rows, "accuracy"),
        "onehot_follow_attention_accuracy": _weighted_mean(
            sweep_rows, "follow_attention_accuracy"
        ),
        "diagonal_accuracy": _weighted_mean(diagonal, "accuracy")
        if diagonal
        else None,
        "off_diagonal_accuracy": _weighted_mean(off_diagonal, "accuracy")
        if off_diagonal
        else None,
        "diagonal_gain_over_natural": (
            _weighted_mean(diagonal, "accuracy")
            - _weighted_mean(diagonal_natural_rows, "accuracy")
        )
        if diagonal
        else None,
    }
    return AttentionRoutingEvaluation(
        clean_controls=clean_controls,
        natural_injected=natural_rows,
        sweep_grid=sweep_rows,
        summary=summary,
    )


