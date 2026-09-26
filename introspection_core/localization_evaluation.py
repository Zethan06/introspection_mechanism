"""Shared cluster-driven token-localization evaluation helpers."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from .injection import inject, normalize_unit_vector
from .prompts import PromptManager


@dataclass(frozen=True)
class ClusterPrompt:
    """One lane CSV row rendered as a token-localization prompt."""

    key: str
    rank: int | None
    choices: tuple[str, ...]
    input_ids: torch.Tensor
    span_starts: tuple[int, ...]
    span_ends: tuple[int, ...]
    candidate_labels: tuple[str, ...]
    candidate_token_ids: tuple[int, ...]


@dataclass
class LocalizationEvaluation:
    """Per-concept aggregate arrays and optional per-trial rows."""

    n_trials: torch.Tensor
    clean_correct: torch.Tensor
    clean_correct_prob_sum: torch.Tensor
    injected_correct: torch.Tensor
    injected_correct_prob_sum: torch.Tensor
    rows: list[dict]


def load_cluster_prompts(
    path: Path,
    prompt_manager: PromptManager,
    *,
    preamble: str,
    template_name: str = "token_localization",
) -> tuple[list[ClusterPrompt], int]:
    """Load a lane CSV and require one uniform, batchable prompt layout."""
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty cluster CSV: {path}")

    examples: list[ClusterPrompt] = []
    expected_n_choices: int | None = None
    layout_signature: tuple | None = None
    for row_index, row in enumerate(rows):
        raw_choices = json.loads(row["choices"])
        if not isinstance(raw_choices, list) or not raw_choices:
            raise ValueError(f"row {row_index} has invalid choices: {row['choices']!r}")
        choices = [str(choice) for choice in raw_choices]
        if expected_n_choices is None:
            expected_n_choices = len(choices)
        elif len(choices) != expected_n_choices:
            raise ValueError(
                f"row {row_index} has {len(choices)} choices; "
                f"expected {expected_n_choices}"
            )

        rendered = prompt_manager.render(
            template_name,
            choices,
            preamble=preamble,
        )
        labels = tuple(rendered.answer_token_by_choice)
        token_ids = tuple(
            int(rendered.answer_token_by_choice[label]) for label in labels
        )
        signature = (
            int(rendered.input_ids.shape[-1]),
            tuple((int(span.start), int(span.end)) for span in rendered.spans),
            labels,
            token_ids,
        )
        if layout_signature is None:
            layout_signature = signature
        elif signature != layout_signature:
            raise ValueError(
                "lane cluster prompts are not batch-aligned; "
                f"row 0 layout={layout_signature}, row {row_index} layout={signature}"
            )

        rank_raw = row.get("rank") or row.get("dataset_rank")
        examples.append(
            ClusterPrompt(
                key=str(row.get("cluster_key") or row_index),
                rank=int(rank_raw) if rank_raw else None,
                choices=tuple(choices),
                input_ids=rendered.input_ids.detach().cpu(),
                span_starts=tuple(int(span.start) for span in rendered.spans),
                span_ends=tuple(int(span.end) for span in rendered.spans),
                candidate_labels=labels,
                candidate_token_ids=token_ids,
            )
        )

    assert expected_n_choices is not None
    return examples, expected_n_choices


def unit_vector_matrix(
    vectors: Sequence[torch.Tensor] | torch.Tensor,
) -> torch.Tensor:
    """Independently normalize vectors and stack them on CPU."""
    if isinstance(vectors, torch.Tensor):
        if vectors.dim() != 2 or vectors.shape[0] == 0:
            raise ValueError(
                f"vectors tensor must be non-empty 2D, got {tuple(vectors.shape)}"
            )
        if not bool(torch.isfinite(vectors).all()):
            raise ValueError("vectors contain non-finite values")
        norms = torch.linalg.vector_norm(vectors.float(), dim=-1, keepdim=True)
        if bool((norms <= 0).any()):
            raise ValueError("vectors contain a zero-norm row")
        return vectors.float() / norms
    if not vectors:
        raise ValueError("vectors must be non-empty")
    return torch.stack([normalize_unit_vector(vector) for vector in vectors], dim=0)


def evaluate_cluster_localization(
    model,
    *,
    examples: list[ClusterPrompt],
    concept_names: list[str],
    unit_vectors: torch.Tensor,
    injection_layer: int,
    strength: float,
    scale_mode: str,
    batch_size: int,
    include_rows: bool = False,
    progress_prefix: str = "",
    progress_every: int = 1,
    trial_mode: str = "cross_product",
) -> LocalizationEvaluation:
    """Evaluate cross-product trials or one fixed balanced trial per concept."""
    if not examples:
        raise ValueError("examples must be non-empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")
    if unit_vectors.dim() != 2 or unit_vectors.shape[0] != len(concept_names):
        raise ValueError(
            f"unit_vectors shape {tuple(unit_vectors.shape)} does not match "
            f"{len(concept_names)} concepts"
        )

    n_concepts = len(concept_names)
    n_examples = len(examples)
    n_choices = len(examples[0].choices)
    example_position_count = n_examples * n_choices
    if trial_mode == "cross_product":
        total_trials = n_concepts * example_position_count
        all_concept_indices = torch.div(
            torch.arange(total_trials),
            example_position_count,
            rounding_mode="floor",
        )
        within_concept = torch.arange(total_trials).remainder(
            example_position_count
        )
    elif trial_mode == "balanced_one_per_concept":
        total_trials = n_concepts
        all_concept_indices = torch.arange(n_concepts)
        within_concept = torch.arange(n_concepts).remainder(
            example_position_count
        )
    else:
        raise ValueError(
            "trial_mode must be 'cross_product' or "
            f"'balanced_one_per_concept', got {trial_mode!r}"
        )
    all_example_indices = torch.div(
        within_concept, n_choices, rounding_mode="floor"
    )
    all_position_indices = within_concept.remainder(n_choices)

    base_tokens = torch.cat([example.input_ids for example in examples], dim=0)
    candidate_labels = examples[0].candidate_labels
    candidate_token_ids_by_example = torch.tensor(
        [example.candidate_token_ids for example in examples],
        dtype=torch.long,
    )
    if any(
        len(example.candidate_labels) != len(candidate_labels)
        for example in examples
    ):
        raise ValueError("all examples must have the same candidate count")
    candidate_counts = {len(example.candidate_labels) for example in examples}
    if candidate_counts == {n_choices + 1}:
        clean_target_index = n_choices
    elif candidate_counts == {n_choices}:
        clean_target_index = None
    else:
        raise ValueError(
            "candidate labels must contain one label per choice and at most "
            "one additional clean-state label"
        )

    clean_logits, clean_log_probs = model.last_token_candidate_stats(
        base_tokens,
        candidate_token_ids=candidate_token_ids_by_example,
    )
    clean_predictions = clean_logits.argmax(dim=-1)
    clean_logits_for_rows = (
        clean_logits.detach().cpu() if include_rows else None
    )
    clean_log_probs_for_rows = (
        clean_log_probs.detach().cpu() if include_rows else None
    )
    clean_predictions_for_rows = (
        clean_predictions.detach().cpu() if include_rows else None
    )

    clean_expected = all_position_indices
    if clean_target_index is not None:
        clean_expected = torch.full_like(
            all_position_indices, clean_target_index
        )
    clean_trial_correct = clean_predictions.index_select(
        0, all_example_indices
    ).eq(clean_expected)
    clean_trial_probs = clean_log_probs[
        all_example_indices, clean_expected
    ].exp()
    n_trials = torch.bincount(
        all_concept_indices, minlength=n_concepts
    )
    clean_correct = torch.bincount(
        all_concept_indices,
        weights=clean_trial_correct.to(torch.float64),
        minlength=n_concepts,
    ).to(torch.long)
    clean_correct_prob_sum = torch.bincount(
        all_concept_indices,
        weights=clean_trial_probs.to(torch.float64),
        minlength=n_concepts,
    )
    injected_correct = torch.zeros(n_concepts, dtype=torch.long)
    injected_correct_prob_sum = torch.zeros(n_concepts, dtype=torch.float64)
    detail_rows: list[dict] = []

    device = model.bridge.cfg.device
    dtype = model.bridge.cfg.dtype
    vectors = unit_vectors.to(device=device, dtype=dtype)

    for batch_start in range(0, total_trials, batch_size):
        batch_end = min(batch_start + batch_size, total_trials)
        concept_indices = all_concept_indices[batch_start:batch_end]
        example_indices = all_example_indices[batch_start:batch_end]
        position_indices = all_position_indices[batch_start:batch_end]

        batch_tokens = base_tokens.index_select(0, example_indices)
        batch_vectors = vectors.index_select(
            0, concept_indices.to(device=device)
        )
        spans = [
            (
                examples[int(example_index)].span_starts[int(position_index)],
                examples[int(example_index)].span_ends[int(position_index)],
            )
            for example_index, position_index in zip(
                example_indices.tolist(), position_indices.tolist()
            )
        ]
        with inject(
            model,
            layer=injection_layer,
            positions=spans,
            vector=batch_vectors,
            strength=strength,
            scale=scale_mode,
        ):
            candidate_logits, candidate_log_probs = (
                model.last_token_candidate_stats(
                    batch_tokens,
                    candidate_token_ids=(
                        candidate_token_ids_by_example.index_select(
                            0, example_indices
                        )
                    ),
                )
            )

        predictions = candidate_logits.argmax(dim=-1)
        expected = position_indices
        correct = predictions.eq(expected)
        row_indices = torch.arange(batch_end - batch_start)
        correct_probs = candidate_log_probs[row_indices, expected].exp()
        injected_correct.add_(
            torch.bincount(
                concept_indices,
                weights=correct.to(torch.float64),
                minlength=n_concepts,
            ).to(torch.long)
        )
        injected_correct_prob_sum.add_(
            torch.bincount(
                concept_indices,
                weights=correct_probs.to(torch.float64),
                minlength=n_concepts,
            )
        )

        if include_rows:
            candidate_logits_for_rows = candidate_logits.detach().cpu()
            predictions_for_rows = predictions.detach().cpu()
            correct_for_rows = correct.detach().cpu()
            correct_probs_for_rows = correct_probs.detach().cpu()
            assert clean_logits_for_rows is not None
            assert clean_log_probs_for_rows is not None
            assert clean_predictions_for_rows is not None
            for offset in range(batch_end - batch_start):
                concept_index = int(concept_indices[offset])
                example_index = int(example_indices[offset])
                position_index = int(position_indices[offset])
                example = examples[example_index]
                clean_prediction_index = int(
                    clean_predictions_for_rows[example_index]
                )
                clean_expected_index = (
                    position_index
                    if clean_target_index is None
                    else clean_target_index
                )
                injected_prediction_index = int(predictions_for_rows[offset])
                detail_rows.append(
                    {
                        "concept": concept_names[concept_index],
                        "cluster_key": example.key,
                        "cluster_rank": example.rank,
                        "target_position": position_index,
                        "target_choice": example.choices[position_index],
                        "clean_argmax_choice": example.candidate_labels[
                            clean_prediction_index
                        ],
                        "clean_correct": int(
                            clean_prediction_index == clean_expected_index
                        ),
                        "clean_correct_prob": float(
                            clean_log_probs_for_rows[
                                example_index, clean_expected_index
                            ].exp()
                        ),
                        "injected_argmax_choice": example.candidate_labels[
                            injected_prediction_index
                        ],
                        "injected_correct": int(correct_for_rows[offset]),
                        "injected_correct_prob": float(
                            correct_probs_for_rows[offset]
                        ),
                        "clean_candidate_logits": json.dumps(
                            [
                                float(value)
                                for value in clean_logits_for_rows[example_index]
                            ]
                        ),
                        "injected_candidate_logits": json.dumps(
                            [
                                float(value)
                                for value in candidate_logits_for_rows[offset]
                            ]
                        ),
                    }
                )

        batch_number = batch_start // batch_size + 1
        if batch_end == total_trials or batch_number % progress_every == 0:
            print(
                f"{progress_prefix}trials {batch_end}/{total_trials}",
                flush=True,
            )

    return LocalizationEvaluation(
        n_trials=n_trials,
        clean_correct=clean_correct,
        clean_correct_prob_sum=clean_correct_prob_sum,
        injected_correct=injected_correct,
        injected_correct_prob_sum=injected_correct_prob_sum,
        rows=detail_rows,
    )


def concept_metric_rows(
    result: LocalizationEvaluation,
    concept_names: list[str],
) -> list[dict]:
    """Convert aggregate tensors into the workflow's metric table."""
    rows = []
    for index, concept in enumerate(concept_names):
        n_trials = int(result.n_trials[index])
        clean_accuracy = float(result.clean_correct[index]) / n_trials
        injected_accuracy = float(result.injected_correct[index]) / n_trials
        rows.append(
            {
                "concept": concept,
                "n_trials": n_trials,
                "clean_argmax_accuracy": clean_accuracy,
                "clean_mean_correct_prob": (
                    float(result.clean_correct_prob_sum[index]) / n_trials
                ),
                "injected_argmax_accuracy": injected_accuracy,
                "injected_mean_correct_prob": (
                    float(result.injected_correct_prob_sum[index]) / n_trials
                ),
                "accuracy_gain_over_clean": injected_accuracy - clean_accuracy,
                "injected_correct_count": int(result.injected_correct[index]),
            }
        )
    return rows
