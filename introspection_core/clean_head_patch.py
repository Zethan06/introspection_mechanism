"""Metrics and controls for clean-to-injected final-token head patching."""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence

import torch


class CandidateLayoutExample(Protocol):
    """Prompt example fields needed to resolve candidate logits."""

    positions: Sequence[int]
    clean_target_label: str | None
    candidate_token_ids: Mapping[str, int]


@dataclass(frozen=True)
class CleanHeadPatchCondition:
    """One target or layer-matched control intervention."""

    name: str
    kind: str
    components: tuple[tuple[int, int], ...]


def parse_head_components(spec: str) -> list[tuple[int, int]]:
    """Parse ``L17H24,L18H3`` into unique ``(layer, head)`` pairs."""

    components: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for raw_component in spec.split(","):
        component = raw_component.strip().upper()
        if not component:
            continue
        if not component.startswith("L") or "H" not in component:
            raise ValueError(f"Invalid head component: {raw_component!r}")
        raw_layer, raw_head = component[1:].split("H", 1)
        pair = (int(raw_layer), int(raw_head))
        if pair in seen:
            raise ValueError(f"Duplicate head component: L{pair[0]}H{pair[1]}")
        seen.add(pair)
        components.append(pair)
    if not components:
        raise ValueError("At least one target head is required")
    return components


def format_head_components(components: Sequence[tuple[int, int]]) -> str:
    """Return the canonical comma-separated name for a head group."""

    if not components:
        raise ValueError("components cannot be empty")
    return ",".join(f"L{int(layer)}H{int(head)}" for layer, head in components)


def resolve_candidate_layout(
    examples: Sequence[CandidateLayoutExample],
) -> tuple[list[int], list[int], int]:
    """Resolve target labels, candidate token ids, and the clean-label index.

    The returned target indices are offsets into the candidate-logit tensor,
    not user-visible labels.  This keeps the patch experiment valid for both
    TOKEN 0..9 and TOKEN 1..9 prompts.
    """

    if not examples:
        raise ValueError("examples cannot be empty")
    first = examples[0]
    position_labels = [int(position) for position in first.positions]
    expected = [str(position) for position in position_labels]
    clean_label = first.clean_target_label
    if not clean_label:
        raise ValueError("The prompt template must define a clean target label")
    labels = [*expected, str(clean_label)]
    for example in examples:
        if [int(position) for position in example.positions] != position_labels:
            raise ValueError("All examples must use the same candidate positions")
        if str(example.clean_target_label) != str(clean_label):
            raise ValueError("All examples must use the same clean target label")
        missing = [label for label in labels if label not in example.candidate_token_ids]
        if missing:
            raise ValueError(f"Candidate token ids are missing labels: {missing}")
    token_ids = [int(first.candidate_token_ids[label]) for label in labels]
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("Candidate labels must map to distinct single tokens")
    return position_labels, token_ids, len(position_labels)


def layer_matched_control_groups(
    target: Sequence[tuple[int, int]],
    *,
    n_heads: int,
    count: int,
    seed: int,
) -> list[list[tuple[int, int]]]:
    """Sample unique non-target groups with the target's layer histogram."""

    if n_heads <= 0 or count <= 0:
        raise ValueError("n_heads and count must be positive")
    target_pairs = {(int(layer), int(head)) for layer, head in target}
    if len(target_pairs) != len(target) or not target_pairs:
        raise ValueError("target must contain unique heads")
    layer_counts = Counter(layer for layer, _head in target_pairs)
    available: dict[int, list[int]] = {}
    capacity = 1
    for layer, required in layer_counts.items():
        heads = [head for head in range(n_heads) if (layer, head) not in target_pairs]
        if len(heads) < required:
            raise ValueError(
                f"Layer {layer} has only {len(heads)} non-target heads for "
                f"a size-{required} matched control"
            )
        available[layer] = heads
        capacity *= math.comb(len(heads), required)

    generator = random.Random(seed)
    desired = min(count, capacity)
    groups: set[tuple[tuple[int, int], ...]] = set()
    attempts = 0
    max_attempts = max(100, desired * 100)
    while len(groups) < desired and attempts < max_attempts:
        attempts += 1
        group: list[tuple[int, int]] = []
        for layer in sorted(layer_counts):
            selected = generator.sample(available[layer], layer_counts[layer])
            group.extend((layer, head) for head in sorted(selected))
        groups.add(tuple(group))
    if len(groups) != desired:
        raise RuntimeError(
            f"Could only construct {len(groups)}/{desired} unique controls"
        )
    return [list(group) for group in sorted(groups)]


def build_patch_conditions(
    target: Sequence[tuple[int, int]],
    *,
    n_heads: int,
    control_count: int,
    seed: int,
) -> list[CleanHeadPatchCondition]:
    """Build the target, individual-target, and matched-control conditions."""

    normalized = tuple((int(layer), int(head)) for layer, head in target)
    if not normalized:
        raise ValueError("target cannot be empty")
    conditions = [
        CleanHeadPatchCondition(
            name="target_group",
            kind="target_group",
            components=normalized,
        )
    ]
    if len(normalized) > 1:
        conditions.extend(
            CleanHeadPatchCondition(
                name=f"target_{format_head_components([component])}",
                kind="target_individual",
                components=(component,),
            )
            for component in normalized
        )
    conditions.extend(
        CleanHeadPatchCondition(
            name=f"matched_control_{index:03d}",
            kind="matched_control",
            components=tuple(components),
        )
        for index, components in enumerate(
            layer_matched_control_groups(
                normalized,
                n_heads=n_heads,
                count=control_count,
                seed=seed,
            ),
            1,
        )
    )
    return conditions


def format_accuracy_change_sentence(
    *,
    heads: str,
    accuracy_drop_pp: float,
    ci_low: float,
    ci_high: float,
) -> str:
    """Describe a signed injected-minus-patched accuracy effect faithfully."""

    if accuracy_drop_pp > 0.005:
        change = f"reduces exact-number accuracy by **{accuracy_drop_pp:.2f} pp**"
    elif accuracy_drop_pp < -0.005:
        change = (
            "increases exact-number accuracy by "
            f"**{-accuracy_drop_pp:.2f} pp**"
        )
    else:
        change = "does not measurably change exact-number accuracy (**0.00 pp**)"
    return (
        f"Clean-patching **{heads}** at the final prompt token {change}. "
        "The concept-bootstrap 95% CI for the injected-minus-patched effect is "
        f"{100 * ci_low:.2f} to {100 * ci_high:.2f} pp."
    )


@dataclass
class CleanHeadPatchAccumulator:
    """Aggregate exact-number and conditional-probability patch effects."""

    concept_count: int
    n_trials: int = 0
    clean_correct: int = 0
    injected_correct: int = 0
    patched_correct: int = 0
    injected_to_wrong: int = 0
    wrong_to_correct: int = 0
    patched_clean_prediction: int = 0
    clean_correct_probability_sum: float = 0.0
    injected_correct_probability_sum: float = 0.0
    patched_correct_probability_sum: float = 0.0
    concept_trial_count: torch.Tensor = field(init=False)
    concept_accuracy_drop_sum: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        if self.concept_count <= 0:
            raise ValueError("concept_count must be positive")
        self.concept_trial_count = torch.zeros(self.concept_count, dtype=torch.long)
        self.concept_accuracy_drop_sum = torch.zeros(
            self.concept_count, dtype=torch.float64
        )

    def update(
        self,
        *,
        clean_logits: torch.Tensor,
        injected_logits: torch.Tensor,
        patched_logits: torch.Tensor,
        target_indices: torch.Tensor,
        clean_index: int,
        concept_indices: torch.Tensor,
    ) -> None:
        """Add a paired batch; positive accuracy drop means the head mattered."""

        if (
            clean_logits.shape != injected_logits.shape
            or clean_logits.shape != patched_logits.shape
        ):
            raise ValueError("clean, injected, and patched logits must have equal shape")
        if clean_logits.dim() != 2:
            raise ValueError("candidate logits must have shape [batch, candidates]")
        batch = int(clean_logits.shape[0])
        targets = target_indices.detach().cpu().long()
        concepts = concept_indices.detach().cpu().long()
        if targets.shape != (batch,) or concepts.shape != (batch,):
            raise ValueError("target_indices and concept_indices must match the batch")
        if not 0 <= clean_index < clean_logits.shape[1]:
            raise IndexError("clean_index is outside the candidate dimension")
        if bool(targets.lt(0).any()) or bool(targets.ge(clean_index).any()):
            raise IndexError("target indices must refer to number candidates")
        if bool(concepts.lt(0).any()) or bool(concepts.ge(self.concept_count).any()):
            raise IndexError("concept index is outside the configured concept bank")

        clean = clean_logits.detach().cpu().float()
        injected = injected_logits.detach().cpu().float()
        patched = patched_logits.detach().cpu().float()
        clean_prediction = clean.argmax(dim=-1)
        injected_prediction = injected.argmax(dim=-1)
        patched_prediction = patched.argmax(dim=-1)
        clean_hit = clean_prediction.eq(targets)
        injected_hit = injected_prediction.eq(targets)
        patched_hit = patched_prediction.eq(targets)

        gather = targets[:, None]
        clean_probability = clean.softmax(dim=-1).gather(1, gather).squeeze(1)
        injected_probability = injected.softmax(dim=-1).gather(1, gather).squeeze(1)
        patched_probability = patched.softmax(dim=-1).gather(1, gather).squeeze(1)

        self.n_trials += batch
        self.clean_correct += int(clean_hit.sum())
        self.injected_correct += int(injected_hit.sum())
        self.patched_correct += int(patched_hit.sum())
        self.injected_to_wrong += int((injected_hit & ~patched_hit).sum())
        self.wrong_to_correct += int((~injected_hit & patched_hit).sum())
        self.patched_clean_prediction += int(patched_prediction.eq(clean_index).sum())
        self.clean_correct_probability_sum += float(clean_probability.sum())
        self.injected_correct_probability_sum += float(injected_probability.sum())
        self.patched_correct_probability_sum += float(patched_probability.sum())

        ones = torch.ones(batch, dtype=torch.long)
        self.concept_trial_count.scatter_add_(0, concepts, ones)
        drops = injected_hit.double() - patched_hit.double()
        self.concept_accuracy_drop_sum.scatter_add_(0, concepts, drops)

    def row(self) -> dict[str, float | int | None]:
        """Return pooled paired effects using percentage points for accuracy."""

        if self.n_trials <= 0:
            raise RuntimeError("cannot summarize an empty accumulator")
        n = float(self.n_trials)
        clean_accuracy = self.clean_correct / n
        injected_accuracy = self.injected_correct / n
        patched_accuracy = self.patched_correct / n
        injection_gain = injected_accuracy - clean_accuracy
        accuracy_drop = injected_accuracy - patched_accuracy
        return {
            "n_trials": self.n_trials,
            "clean_exact_number_accuracy": clean_accuracy,
            "injected_exact_number_accuracy": injected_accuracy,
            "patched_exact_number_accuracy": patched_accuracy,
            "accuracy_drop": accuracy_drop,
            "accuracy_drop_pp": 100.0 * accuracy_drop,
            "injection_gain_over_clean": injection_gain,
            "fraction_of_injection_gain_removed": (
                accuracy_drop / injection_gain if injection_gain != 0 else None
            ),
            "injected_to_wrong_rate": self.injected_to_wrong / n,
            "wrong_to_correct_rate": self.wrong_to_correct / n,
            "patched_clean_prediction_rate": self.patched_clean_prediction / n,
            "clean_mean_correct_candidate_probability": (
                self.clean_correct_probability_sum / n
            ),
            "injected_mean_correct_candidate_probability": (
                self.injected_correct_probability_sum / n
            ),
            "patched_mean_correct_candidate_probability": (
                self.patched_correct_probability_sum / n
            ),
            "correct_candidate_probability_drop": (
                self.injected_correct_probability_sum
                - self.patched_correct_probability_sum
            ) / n,
        }

    def concept_accuracy_drops(self) -> torch.Tensor:
        """Return one equally weighted mean accuracy drop per observed concept."""

        valid = self.concept_trial_count.gt(0)
        if not bool(valid.any()):
            raise RuntimeError("no concept effects were accumulated")
        return (
            self.concept_accuracy_drop_sum[valid]
            / self.concept_trial_count[valid].double()
        )


def concept_bootstrap_interval(
    concept_effects: torch.Tensor,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    """Percentile 95% CI for a mean, resampling concepts as the unit."""

    values = concept_effects.detach().cpu().double().flatten()
    if values.numel() == 0 or samples <= 0:
        raise ValueError("concept effects and bootstrap samples must be non-empty")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    means: list[torch.Tensor] = []
    remaining = samples
    while remaining:
        chunk = min(remaining, 2048)
        indices = torch.randint(
            values.numel(),
            (chunk, values.numel()),
            generator=generator,
        )
        means.append(values[indices].mean(dim=1))
        remaining -= chunk
    bootstrap = torch.cat(means)
    low, high = torch.quantile(
        bootstrap, torch.tensor([0.025, 0.975], dtype=bootstrap.dtype)
    )
    return float(low), float(high)
