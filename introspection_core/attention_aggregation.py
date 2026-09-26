"""TransformerLens attention aggregation across examples, concepts, and positions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .attention_inputs import AttentionExample, validate_aligned_layout
from .hooks import write_span
from .injection import inject, normalize_unit_vector
from .layer_cache import ResidualHookPoint, resolve_resid_hook_name


@dataclass
class PositionAttentionSummary:
    position: int
    clean_correct: int
    injected_correct: int
    num_examples: int
    num_concepts: int

    @property
    def denominator(self) -> int:
        return self.num_examples * self.num_concepts

    @property
    def clean_accuracy(self) -> float:
        return self.clean_correct / self.denominator

    @property
    def injected_accuracy(self) -> float:
        return self.injected_correct / self.denominator

    def as_dict(self) -> dict:
        return {
            "target_position": self.position,
            "clean_correct": self.clean_correct,
            "injected_correct": self.injected_correct,
            "num_clusters": self.num_examples,
            "num_concepts": self.num_concepts,
            "num_examples": self.denominator,
            "clean_accuracy": self.clean_accuracy,
            "injected_accuracy": self.injected_accuracy,
        }


@dataclass
class AveragedAttention:
    """Mean attention tensors and position-level task scores."""

    layers: list[int]
    positions: list[int]
    clean_mean: torch.Tensor
    injected_means: dict[int, torch.Tensor]
    summaries: dict[int, PositionAttentionSummary]
    clean_attention_correct: dict[int, torch.Tensor]
    injected_attention_correct: dict[int, torch.Tensor]


@dataclass
class PositionPatchedAttentionSummary:
    """Position-level scores for the injected reference and patched run."""

    position: int
    clean_correct: int
    injected_correct: int
    patched_correct: int
    num_examples: int
    num_concepts: int

    @property
    def denominator(self) -> int:
        return self.num_examples * self.num_concepts

    @property
    def clean_accuracy(self) -> float:
        return self.clean_correct / self.denominator

    @property
    def injected_accuracy(self) -> float:
        return self.injected_correct / self.denominator

    @property
    def patched_accuracy(self) -> float:
        return self.patched_correct / self.denominator

    def as_dict(self) -> dict:
        # ``injected_accuracy`` remains the browser's compatibility field for
        # the displayed non-clean tensor. The full-injection reference is
        # retained separately so the intervention can still be audited.
        return {
            "target_position": self.position,
            "clean_correct": self.clean_correct,
            "full_injected_correct": self.injected_correct,
            "patched_correct": self.patched_correct,
            "num_clusters": self.num_examples,
            "num_concepts": self.num_concepts,
            "num_examples": self.denominator,
            "clean_accuracy": self.clean_accuracy,
            "full_injected_accuracy": self.injected_accuracy,
            "patched_accuracy": self.patched_accuracy,
            "injected_accuracy": self.patched_accuracy,
        }


@dataclass
class AveragedPatchedAttention:
    """Mean clean/patched attention and reference task scores."""

    layers: list[int]
    positions: list[int]
    clean_mean: torch.Tensor
    patched_means: dict[int, torch.Tensor]
    summaries: dict[int, PositionPatchedAttentionSummary]



def unit_concept_matrix(vectors: Sequence[torch.Tensor] | torch.Tensor) -> torch.Tensor:
    """Normalize a concept bank independently; never average vectors first."""
    rows = list(vectors) if not isinstance(vectors, torch.Tensor) else list(vectors)
    if not rows:
        raise ValueError("At least one concept vector is required")
    return torch.stack([normalize_unit_vector(row) for row in rows], dim=0)


def _clean_expected_label(example: AttentionExample, position: int) -> str:
    """Return the prompt-specific clean label, falling back to legacy behavior."""
    if example.clean_target_label is not None:
        return example.clean_target_label
    return example.expected_candidate_by_position[position]


def average_attention(
    model,
    *,
    examples: list[AttentionExample],
    concept_vectors: Sequence[torch.Tensor] | torch.Tensor,
    injection_layer: int,
    positions: Sequence[int] | None = None,
    strength: float = 3.0,
    scale_mode: str = "relative_hidden_norm",
    concept_batch_size: int = 16,
    layers: Sequence[int] | None = None,
) -> AveragedAttention:
    """Average attention after independently injecting every concept.

    The injected mean at each position is over
    ``len(examples) * len(concept_vectors)`` forward examples.  Concept
    vectors are unit-normalized independently and are *not* averaged into
    one direction before intervention.
    """
    validate_aligned_layout(examples)
    if concept_batch_size <= 0:
        raise ValueError("concept_batch_size must be positive")

    available = list(examples[0].positions)
    selected_positions = (
        available if positions is None else sorted(set(int(p) for p in positions))
    )
    unknown = [position for position in selected_positions if position not in available]
    if not selected_positions or unknown:
        raise ValueError(
            f"positions must be a non-empty subset of {available}; invalid={unknown}"
        )
    selected_layers = (
        list(range(int(model.cfg.n_layers)))
        if layers is None
        else [int(layer) for layer in layers]
    )
    invalid_layers = [
        layer
        for layer in selected_layers
        if layer < 0 or layer >= int(model.cfg.n_layers)
    ]
    if not selected_layers or invalid_layers:
        raise ValueError(
            f"layers must be within 0..{int(model.cfg.n_layers) - 1}; "
            f"invalid={invalid_layers}"
        )

    vectors = unit_concept_matrix(concept_vectors).to(
        device=model.bridge.cfg.device,
        dtype=model.bridge.cfg.dtype,
    )
    num_concepts = int(vectors.shape[0])
    candidate_labels = list(examples[0].candidate_token_ids)
    candidate_ids = [
        examples[0].candidate_token_ids[label] for label in candidate_labels
    ]

    clean_sum: torch.Tensor | None = None
    injected_sums: dict[int, torch.Tensor] = {}
    clean_correct = {position: 0 for position in selected_positions}
    injected_correct = {position: 0 for position in selected_positions}
    clean_attention_correct: dict[int, torch.Tensor] = {}
    injected_attention_correct: dict[int, torch.Tensor] = {}

    for example_number, example in enumerate(examples, start=1):
        input_ids = example.input_ids.to(model.bridge.cfg.device)
        clean_result = model.attention_sums_and_candidate_logits(
            input_ids,
            layers=selected_layers,
            candidate_token_ids=candidate_ids,
            localization_token_indices=example.item_token_indices,
        )
        clean_attention, clean_logits, clean_localization = clean_result
        clean_label = candidate_labels[int(torch.argmax(clean_logits[0]))]
        for position in selected_positions:
            expected = _clean_expected_label(example, position)
            clean_correct[position] += int(clean_label == expected) * num_concepts

        if clean_sum is None:
            clean_sum = torch.zeros_like(clean_attention)
            injected_sums = {
                position: torch.zeros_like(clean_attention)
                for position in selected_positions
            }
            clean_attention_correct = {
                position: torch.zeros(
                    clean_localization.shape[1:], dtype=torch.long
                )
                for position in selected_positions
            }
            injected_attention_correct = {
                position: torch.zeros(
                    clean_localization.shape[1:], dtype=torch.long
                )
                for position in selected_positions
            }
        clean_sum.add_(clean_attention)

        for position in selected_positions:
            expected_choice_offset = example.positions.index(position)
            clean_attention_correct[position].add_(
                (clean_localization[0] == expected_choice_offset).long()
                * num_concepts
            )
            span = example.injection_spans[position]
            for batch_start in range(0, num_concepts, concept_batch_size):
                batch_vectors = vectors[
                    batch_start : batch_start + concept_batch_size
                ]
                batch_size = int(batch_vectors.shape[0])
                batch_tokens = input_ids.expand(batch_size, -1)
                spans = [(int(span.start), int(span.end))] * batch_size
                with inject(
                    model,
                    layer=injection_layer,
                    positions=spans,
                    vector=batch_vectors,
                    strength=strength,
                    scale=scale_mode,
                ):
                    attention_result = model.attention_sums_and_candidate_logits(
                        batch_tokens,
                        layers=selected_layers,
                        candidate_token_ids=candidate_ids,
                        localization_token_indices=example.item_token_indices,
                    )
                    attention_sum, candidate_logits, localization = attention_result
                predictions = torch.argmax(candidate_logits, dim=-1).tolist()
                injected_correct[position] += sum(
                    int(
                        candidate_labels[index]
                        == example.expected_candidate_by_position[position]
                    )
                    for index in predictions
                )
                injected_sums[position].add_(attention_sum)
                injected_attention_correct[position].add_(
                    (localization == expected_choice_offset).long().sum(dim=0)
                )

        print(
            f"processed example {example_number}/{len(examples)} "
            f"key={example.key} concepts={num_concepts}",
            flush=True,
        )

    if clean_sum is None:
        raise RuntimeError("No attention tensors were collected")

    clean_mean = clean_sum / float(len(examples))
    denominator = float(len(examples) * num_concepts)
    injected_means = {
        position: tensor / denominator
        for position, tensor in injected_sums.items()
    }
    summaries = {
        position: PositionAttentionSummary(
            position=position,
            clean_correct=clean_correct[position],
            injected_correct=injected_correct[position],
            num_examples=len(examples),
            num_concepts=num_concepts,
        )
        for position in selected_positions
    }
    return AveragedAttention(
        layers=selected_layers,
        positions=selected_positions,
        clean_mean=clean_mean,
        injected_means=injected_means,
        summaries=summaries,
        clean_attention_correct=clean_attention_correct,
        injected_attention_correct=injected_attention_correct,
    )


def _capture_target_hooks(
    model,
    *,
    layers: Sequence[int],
    token_positions: Sequence[int],
    destination: dict[int, torch.Tensor],
    hook_point: ResidualHookPoint = "resid_post",
) -> list[tuple[str, object]]:
    """Build hooks that retain only selected residual-token states."""
    position_index = [int(position) for position in token_positions]
    hooks: list[tuple[str, object]] = []
    for layer in layers:
        hook_name = resolve_resid_hook_name(model, int(layer), hook_point)

        def make_capture(layer_index: int):
            def capture(activation, hook):
                del hook
                destination[layer_index] = (
                    activation[:, position_index, :].detach().float()
                )
                return activation

            return capture

        hooks.append((hook_name, make_capture(int(layer))))
    return hooks


def _injection_hook(
    model,
    *,
    layer: int,
    spans: list[tuple[int, int]],
    vectors: torch.Tensor,
    strength: float,
    scale_mode: str,
) -> tuple[str, object]:
    values = vectors.to(
        device=model.bridge.cfg.device,
        dtype=model.bridge.cfg.dtype,
    )

    def apply_injection(activation, hook):
        del hook
        return write_span(
            activation,
            spans=spans,
            value=values,
            mode="add",
            strength=strength,
            scale=scale_mode,
        )

    return model.resid_hook_name(layer), apply_injection


