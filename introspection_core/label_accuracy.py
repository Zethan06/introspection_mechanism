"""Label-agnostic injected-localization evaluation and result persistence."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import torch

from .attention_inputs import AttentionExample
from .extraction import load_concept_vector_payload
from .injected_trials import InjectedTrial, build_injected_batch
from .model import HookedModel


NONE_LABEL = "none"


class _CandidateLayoutExample(Protocol):
    """Prompt fields needed to resolve ordered candidate labels."""

    positions: Sequence[int]
    candidate_token_ids: Mapping[str, int]


@dataclass(frozen=True)
class LabelCandidateLayout:
    """Ordered position labels plus the final clean-state label."""

    labels: tuple[str, ...]
    token_ids: tuple[int, ...]
    position_count: int

    @property
    def none_index(self) -> int:
        return self.position_count


@dataclass
class LabelAccuracyCounts:
    """Mutable injected-trial counts used for summaries and detail tables."""

    correct_by_concept: Counter[int] = field(default_factory=Counter)
    none_by_concept: Counter[int] = field(default_factory=Counter)
    correct_by_position: Counter[int] = field(default_factory=Counter)
    none_by_position: Counter[int] = field(default_factory=Counter)
    trials_by_position: Counter[int] = field(default_factory=Counter)
    correct_by_cluster: Counter[int] = field(default_factory=Counter)
    none_by_cluster: Counter[int] = field(default_factory=Counter)
    trials_by_cluster: Counter[int] = field(default_factory=Counter)
    counting_by_position: Counter[int] = field(default_factory=Counter)
    predicted_labels: Counter[str] = field(default_factory=Counter)
    n_correct: int = 0
    n_none: int = 0
    n_counting: int = 0

    def update(
        self,
        trials: Sequence[InjectedTrial],
        predictions: Sequence[int],
        layout: LabelCandidateLayout,
        expected_index_by_cluster: Sequence[Sequence[int]] | None = None,
    ) -> None:
        """Accumulate one model batch.

        ``expected_index_by_cluster[cluster][position]`` is the candidate index
        that counts as correct for that cluster's slot. It is the identity for
        the ascending-label prompts and a per-cluster permutation for the
        shuffled-label control; ``None`` keeps the identity behaviour.
        """
        if len(trials) != len(predictions):
            raise ValueError("trials and predictions must have equal length")
        for trial, raw_prediction in zip(trials, predictions, strict=True):
            prediction = int(raw_prediction)
            if not 0 <= prediction < len(layout.labels):
                raise ValueError(f"prediction index out of range: {prediction}")
            if expected_index_by_cluster is None:
                expected = trial.position
            else:
                expected = int(
                    expected_index_by_cluster[trial.cluster_index][trial.position]
                )
            self.predicted_labels[layout.labels[prediction]] += 1
            self.trials_by_position[trial.position] += 1
            self.trials_by_cluster[trial.cluster_index] += 1
            # The answer a pure "count the list entries" strategy would give:
            # the canonical label for this slot, whatever letter is printed
            # there. Identical to ``expected`` in the ascending arm, and never
            # equal to it in the shuffled arm, which uses derangements.
            if prediction == trial.position:
                self.n_counting += 1
                self.counting_by_position[trial.position] += 1
            if prediction == expected:
                self.n_correct += 1
                self.correct_by_concept[trial.concept_index] += 1
                self.correct_by_position[trial.position] += 1
                self.correct_by_cluster[trial.cluster_index] += 1
            elif prediction == layout.none_index:
                self.n_none += 1
                self.none_by_concept[trial.concept_index] += 1
                self.none_by_position[trial.position] += 1
                self.none_by_cluster[trial.cluster_index] += 1


@dataclass(frozen=True)
class LabelAccuracyEvaluation:
    """Clean predictions and injected-trial counts for one data split."""

    layout: LabelCandidateLayout
    clean_predicted_labels: Counter[str]
    counts: LabelAccuracyCounts
    n_trials: int
    n_clusters: int
    expected_index_by_cluster: tuple[tuple[int, ...], ...] | None = None
    clean_correct: int = 0

    @property
    def is_permuted(self) -> bool:
        """Whether any cluster's answer key departs from the identity map."""
        if self.expected_index_by_cluster is None:
            return False
        identity = tuple(range(self.layout.position_count))
        return any(row != identity for row in self.expected_index_by_cluster)


@dataclass(frozen=True)
class LabelAccuracyRunMetadata:
    """Inputs and settings that determine one split's persisted results."""

    model: str
    split: str
    prompt_template: str
    cluster_csv: Path
    concepts_json: Path
    concept_vectors: Path
    injection_layer: int
    strength: float
    scale_mode: str
    seed: int
    prompt_preamble: str
    choice_suffix: str
    dtype: str
    max_concepts: int | None
    max_clusters: int | None
    label_permutation: str = "identity"

    def summary_fields(self) -> dict[str, object]:
        """Return the stable metadata stored in and matched against summaries."""
        return {
            "model": self.model,
            "split": self.split,
            "prompt_template": self.prompt_template,
            "cluster_csv": str(self.cluster_csv),
            "concepts_json": str(self.concepts_json),
            "concept_vectors": str(self.concept_vectors),
            "injection_layer": self.injection_layer,
            "strength": self.strength,
            "scale_mode": self.scale_mode,
            "seed": self.seed,
            "prompt_preamble": self.prompt_preamble,
            "choice_suffix": self.choice_suffix,
            "dtype": self.dtype,
            "max_concepts": self.max_concepts,
            "max_clusters": self.max_clusters,
            "label_permutation": self.label_permutation,
        }


def resolve_label_candidate_layout(
    examples: Sequence[_CandidateLayoutExample],
) -> LabelCandidateLayout:
    """Require one label per zero-based position followed by ``none``."""
    if not examples:
        raise ValueError("examples cannot be empty")

    first = examples[0]
    positions = tuple(int(position) for position in first.positions)
    if positions != tuple(range(len(positions))):
        raise ValueError(f"positions must be 0..N-1, got {positions}")
    labels = tuple(str(label) for label in first.candidate_token_ids)
    if len(labels) != len(positions) + 1 or labels[-1] != NONE_LABEL:
        raise ValueError(
            "candidates must be one label per position followed by 'none'; "
            f"got {labels}"
        )
    token_ids = tuple(int(first.candidate_token_ids[label]) for label in labels)
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("candidate labels must map to distinct single tokens")

    for example in examples[1:]:
        actual_positions = tuple(int(position) for position in example.positions)
        actual_labels = tuple(str(label) for label in example.candidate_token_ids)
        actual_token_ids = tuple(
            int(example.candidate_token_ids[label]) for label in actual_labels
        )
        if actual_positions != positions:
            raise ValueError(
                f"clusters disagree on positions: {positions} versus {actual_positions}"
            )
        if actual_labels != labels or actual_token_ids != token_ids:
            raise ValueError("clusters disagree on candidate labels or token ids")

    return LabelCandidateLayout(labels, token_ids, len(positions))


def load_label_accuracy_vectors(
    path: Path,
    *,
    injection_layer: int,
    d_model: int,
    concepts: Sequence[str],
) -> torch.Tensor:
    """Load one split's concepts from the shared frozen vector payload."""
    vectors = load_concept_vector_payload(
        path,
        concepts=[str(name) for name in concepts],
        layer=injection_layer,
    ).cpu()
    expected_shape = (len(concepts), d_model)
    if vectors.shape != expected_shape:
        raise ValueError(
            "Concept-vector shape does not match model: "
            f"{tuple(vectors.shape)} versus {expected_shape}"
        )
    return vectors


@torch.inference_mode()
def evaluate_split_label_accuracy(
    model: HookedModel,
    *,
    examples: Sequence[AttentionExample],
    concept_vectors: torch.Tensor,
    injection_layer: int,
    strength: float,
    scale_mode: str,
    batch_size: int,
    progress_prefix: str,
    progress_every: int,
) -> LabelAccuracyEvaluation:
    """Evaluate every concept × cluster × position injected trial."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")
    layout = resolve_label_candidate_layout(examples)
    if concept_vectors.dim() != 2 or concept_vectors.shape[0] == 0:
        raise ValueError("concept_vectors must be a non-empty 2D tensor")

    base_tokens = torch.cat([example.input_ids for example in examples], dim=0)
    injection_token_positions = torch.tensor(
        [
            [
                int(example.injection_spans[position].start)
                for position in range(layout.position_count)
            ]
            for example in examples
        ],
        dtype=torch.long,
    )
    label_index = {label: index for index, label in enumerate(layout.labels)}
    expected_index_by_cluster = tuple(
        tuple(
            label_index[str(example.expected_candidate_by_position[position])]
            for position in range(layout.position_count)
        )
        for example in examples
    )

    clean_logits, _ = model.last_token_candidate_stats(
        base_tokens,
        candidate_token_ids=list(layout.token_ids),
    )
    clean_predictions = clean_logits.argmax(dim=-1).tolist()
    clean_counts = Counter(layout.labels[index] for index in clean_predictions)
    # One clean forward pass per cluster: the clean prompt does not depend on
    # the concept or the injected position, so the clean denominator is the
    # cluster count, not the injected trial count.
    clean_correct = sum(
        1 for index in clean_predictions if index == layout.none_index
    )

    coordinates = [
        (concept_index, cluster_index, position)
        for concept_index in range(concept_vectors.shape[0])
        for cluster_index in range(base_tokens.shape[0])
        for position in range(layout.position_count)
    ]
    total = len(coordinates)
    counts = LabelAccuracyCounts()
    for batch_number, start in enumerate(range(0, total, batch_size), 1):
        batch = coordinates[start : start + batch_size]
        trials = [
            InjectedTrial(concept_index, cluster_index, position, False, False)
            for concept_index, cluster_index, position in batch
        ]
        tokens, injection_hook = build_injected_batch(
            trials,
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
            candidate_token_ids=list(layout.token_ids),
            fwd_hooks=[injection_hook],
        )
        counts.update(
            trials,
            logits.argmax(dim=-1).tolist(),
            layout,
            expected_index_by_cluster=expected_index_by_cluster,
        )
        if batch_number % progress_every == 0 or start + batch_size >= total:
            done = min(start + batch_size, total)
            print(
                f"{progress_prefix}{done}/{total} trials "
                f"acc={counts.n_correct / done:.4f} "
                f"none={counts.n_none / done:.4f}",
                flush=True,
            )

    return LabelAccuracyEvaluation(
        layout=layout,
        clean_predicted_labels=clean_counts,
        counts=counts,
        n_trials=total,
        n_clusters=int(base_tokens.shape[0]),
        expected_index_by_cluster=expected_index_by_cluster,
        clean_correct=clean_correct,
    )


def build_label_accuracy_summary(
    evaluation: LabelAccuracyEvaluation,
    *,
    metadata: LabelAccuracyRunMetadata,
    concept_count: int,
) -> dict:
    """Build the stable JSON summary for one split."""
    counts = evaluation.counts
    total = evaluation.n_trials
    observed = "shuffled" if evaluation.is_permuted else "identity"
    if observed != metadata.label_permutation:
        raise ValueError(
            f"metadata declares label_permutation={metadata.label_permutation!r} "
            f"but the rendered examples are {observed!r}"
        )
    return {
        **metadata.summary_fields(),
        "candidate_labels": list(evaluation.layout.labels),
        "candidate_token_ids": list(evaluation.layout.token_ids),
        "n_concepts": concept_count,
        "n_clusters": evaluation.n_clusters,
        "n_positions": evaluation.layout.position_count,
        "n_trials": total,
        "expected_label_by_cluster": [
            [evaluation.layout.labels[index] for index in row]
            for row in (evaluation.expected_index_by_cluster or ())
        ],
        # Clean prompts do not vary with concept or injected position, so this
        # arm has one independent trial per cluster. Reported on its own
        # denominator to keep it from borrowing the injected trial count.
        "n_clean_trials": evaluation.n_clusters,
        "n_clean_correct": evaluation.clean_correct,
        "clean_none_rate": (
            evaluation.clean_correct / evaluation.n_clusters
            if evaluation.n_clusters
            else 0.0
        ),
        "n_correct": counts.n_correct,
        "n_predicted_none": counts.n_none,
        "accuracy": counts.n_correct / total,
        "none_rate": counts.n_none / total,
        "other_rate": (total - counts.n_correct - counts.n_none) / total,
        # Rate of answering with the slot's canonical (ascending) label. In the
        # shuffled arm this is the counting strategy's signature: it cannot
        # coincide with a correct answer, because the permutations are
        # derangements.
        "n_counting_consistent": counts.n_counting,
        "counting_consistent_rate": counts.n_counting / total,
        "injected_predicted_label_counts": dict(counts.predicted_labels),
        "clean_predicted_label_counts": dict(evaluation.clean_predicted_labels),
    }


def label_accuracy_position_rows(
    evaluation: LabelAccuracyEvaluation,
) -> list[list[object]]:
    """Return per-position CSV rows without the header."""
    counts = evaluation.counts
    rows: list[list[object]] = []
    permuted = evaluation.is_permuted
    for position in range(evaluation.layout.position_count):
        trials = counts.trials_by_position[position]
        rows.append(
            [
                position,
                # Under a per-cluster permutation no single label belongs to
                # this slot, so naming one here would be a lie.
                "*" if permuted else evaluation.layout.labels[position],
                trials,
                counts.correct_by_position[position],
                counts.none_by_position[position],
                counts.correct_by_position[position] / trials if trials else 0.0,
                counts.counting_by_position[position] / trials if trials else 0.0,
            ]
        )
    return rows


def label_accuracy_concept_rows(
    evaluation: LabelAccuracyEvaluation,
    concepts: Sequence[str],
) -> list[list[object]]:
    """Return per-concept CSV rows without the header."""
    counts = evaluation.counts
    trials = evaluation.n_clusters * evaluation.layout.position_count
    return [
        [
            index,
            concept,
            trials,
            counts.correct_by_concept[index],
            counts.none_by_concept[index],
            counts.correct_by_concept[index] / trials if trials else 0.0,
        ]
        for index, concept in enumerate(concepts)
    ]


def label_accuracy_cluster_rows(
    evaluation: LabelAccuracyEvaluation,
) -> list[list[object]]:
    """Return per-cluster CSV rows without the header.

    Clusters, not trials, are the independent unit of this design: every trial
    inside one cluster reuses the same ten candidate tokens and the same label
    permutation. These rows are what a cluster-level bootstrap resamples.
    """
    counts = evaluation.counts
    rows: list[list[object]] = []
    for cluster in range(evaluation.n_clusters):
        trials = counts.trials_by_cluster[cluster]
        permutation = (
            evaluation.expected_index_by_cluster[cluster]
            if evaluation.expected_index_by_cluster is not None
            else tuple(range(evaluation.layout.position_count))
        )
        rows.append(
            [
                cluster,
                # Separated, not concatenated: number-word labels are
                # multi-character, so a bare join is ambiguous.
                "|".join(evaluation.layout.labels[index] for index in permutation),
                trials,
                counts.correct_by_cluster[cluster],
                counts.none_by_cluster[cluster],
                counts.correct_by_cluster[cluster] / trials if trials else 0.0,
            ]
        )
    return rows


def label_accuracy_outputs_match(
    output_dir: Path,
    *,
    split_name: str,
    expected: Mapping[str, object],
) -> bool:
    """Return whether a complete artifact set matches the current inputs."""
    summary_path = output_dir / f"{split_name}_summary.json"
    companion_paths = [
        output_dir / f"{split_name}_per_position.csv",
        output_dir / f"{split_name}_per_concept.csv",
    ]
    if not summary_path.is_file() or any(
        not path.is_file() or path.stat().st_size == 0 for path in companion_paths
    ):
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    for key, value in expected.items():
        actual = summary.get(key)
        if key == "label_permutation" and key not in summary:
            # Summaries predating this field can only have come from the
            # ascending path, which is what "identity" means.
            actual = "identity"
        if key in {"cluster_csv", "concepts_json", "concept_vectors"}:
            if actual is None or (
                Path(str(actual)).resolve() != Path(str(value)).resolve()
            ):
                return False
        elif actual != value:
            return False
    return True


def write_label_accuracy_outputs(
    output_dir: Path,
    *,
    split_name: str,
    summary: Mapping[str, object],
    position_rows: Sequence[Sequence[object]],
    concept_rows: Sequence[Sequence[object]],
    cluster_rows: Sequence[Sequence[object]] | None = None,
) -> None:
    """Atomically publish detail CSVs, then the completion summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = [
        (
            output_dir / f"{split_name}_per_position.csv",
            [
                "position",
                "label",
                "n_trials",
                "n_correct",
                "n_none",
                "accuracy",
                "counting_consistent_rate",
            ],
            position_rows,
        ),
        (
            output_dir / f"{split_name}_per_concept.csv",
            [
                "concept_index",
                "concept",
                "n_trials",
                "n_correct",
                "n_none",
                "accuracy",
            ],
            concept_rows,
        ),
    ]
    if cluster_rows is not None:
        tables.append(
            (
                output_dir / f"{split_name}_per_cluster.csv",
                [
                    "cluster_index",
                    "display_labels",
                    "n_trials",
                    "n_correct",
                    "n_none",
                    "accuracy",
                ],
                cluster_rows,
            )
        )
    for path, header, rows in tables:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        temporary.replace(path)

    summary_path = output_dir / f"{split_name}_summary.json"
    temporary_summary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary_summary.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_summary.replace(summary_path)
