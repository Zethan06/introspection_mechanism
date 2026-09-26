"""Model-agnostic direction scoring for post-injection boundary states.

The functions in this module operate on residual-stream tensors and saved
directional-concentration components.  They deliberately do not load a model
or assume a model family, hidden width, layer count, or number of candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


PRIMITIVE_OUTCOMES = ("none", "exact_number", "wrong_number")


@dataclass(frozen=True)
class BoundaryRepresentationProtocol:
    """Training and held-out semantics for one representation protocol."""

    test_representation: str
    collect_clean_training: bool
    unique_clean_negative: bool
    required_capture_position: str | None
    direction_definition: str
    positive_prototype_definition: str


BOUNDARY_PROTOCOLS = {
    "delta": BoundaryRepresentationProtocol(
        test_representation="delta",
        collect_clean_training=True,
        unique_clean_negative=False,
        required_capture_position=None,
        direction_definition=(
            "normalize(mean_train(unit_delta|positive) - "
            "mean_train(unit_delta|negative))"
        ),
        positive_prototype_definition="normalize(mean_train(unit_delta|positive))",
    ),
    "injected": BoundaryRepresentationProtocol(
        test_representation="injected",
        collect_clean_training=False,
        unique_clean_negative=False,
        required_capture_position=None,
        direction_definition=(
            "normalize(mean_train(unit_injected|positive) - "
            "mean_train(unit_injected|negative))"
        ),
        positive_prototype_definition=(
            "normalize(mean_train(unit_injected|positive))"
        ),
    ),
    "injected_number_vs_unique_clean": BoundaryRepresentationProtocol(
        test_representation="injected",
        collect_clean_training=True,
        unique_clean_negative=True,
        required_capture_position="final_token",
        direction_definition=(
            "normalize(mean_train(unit_injected|number) - "
            "mean_train_clusters(unit_clean))"
        ),
        positive_prototype_definition=(
            "normalize(mean_train(unit_injected|number))"
        ),
    ),
}
BOUNDARY_REPRESENTATIONS = tuple(BOUNDARY_PROTOCOLS)


def boundary_protocol(representation: str) -> BoundaryRepresentationProtocol:
    """Return the centralized semantics for ``representation``."""

    try:
        return BOUNDARY_PROTOCOLS[representation]
    except KeyError as error:
        raise ValueError(
            f"unknown boundary representation {representation!r}; expected one of "
            f"{BOUNDARY_REPRESENTATIONS}"
        ) from error


def boundary_representation(
    injected: torch.Tensor,
    clean: torch.Tensor | None,
    representation: str,
) -> torch.Tensor:
    """Return the residual representation used for direction learning/scoring."""

    protocol = boundary_protocol(representation)
    if protocol.test_representation == "injected":
        return injected
    if clean is None:
        raise ValueError("clean states are required for the delta representation")
    if injected.shape != clean.shape:
        raise ValueError(
            "injected and clean states must have identical shapes for delta: "
            f"{tuple(injected.shape)} vs {tuple(clean.shape)}"
        )
    return injected - clean


def unique_clean_reference_components(
    clean_states: torch.Tensor,
    *,
    source_panel_count: int,
    outcome_count: int,
    none_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Seed component tensors with one unit clean state per cluster."""

    if clean_states.dim() != 4 or clean_states.shape[1] < 1:
        raise ValueError(
            "clean states must have shape [cluster, position, layer, hidden]"
        )
    if source_panel_count < 1 or outcome_count < 1:
        raise ValueError("panel and outcome counts must be positive")
    if not 0 <= none_index < outcome_count:
        raise ValueError("none_index is outside the outcome range")
    first_position = clean_states[:, 0]
    if not torch.allclose(
        clean_states, first_position[:, None].expand_as(clean_states)
    ):
        raise ValueError("final-token clean states unexpectedly vary by position")
    clean_sum = F.normalize(first_position, dim=-1).float().cpu().sum(dim=0)
    cluster_count, layer_count, hidden_width = first_position.shape
    counts = torch.zeros(source_panel_count, outcome_count, dtype=torch.int64)
    unit_sums = torch.zeros(
        source_panel_count,
        outcome_count,
        layer_count,
        hidden_width,
        dtype=torch.float32,
    )
    union_count = torch.zeros(outcome_count, dtype=torch.int64)
    union_unit_sum = torch.zeros(
        outcome_count, layer_count, hidden_width, dtype=torch.float32
    )
    counts[:, none_index] = cluster_count
    unit_sums[:, none_index] = clean_sum
    union_count[none_index] = cluster_count
    union_unit_sum[none_index] = clean_sum
    return counts, unit_sums, union_count, union_unit_sum


def complete_unique_clean_union_statistics(
    counts: torch.Tensor,
    unit_sums: torch.Tensor,
    union_counts: torch.Tensor,
    union_unit_sums: torch.Tensor,
    *,
    none_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Complete union statistics without duplicating the clean reference."""

    completed_counts = union_counts.clone()
    completed_sums = union_unit_sums.clone()
    for outcome_index in range(counts.shape[1]):
        if outcome_index == none_index:
            continue
        completed_counts[outcome_index] = counts[:, outcome_index].sum()
        completed_sums[outcome_index] = unit_sums[:, outcome_index].sum(dim=0)
    return completed_counts, completed_sums


@dataclass(frozen=True)
class DirectionContrast:
    """One binary comparison in the training components and held-out trials."""

    name: str
    positive_component: str
    negative_component: str
    positive_outcomes: tuple[str, ...]
    negative_outcomes: tuple[str, ...]


DEFAULT_CONTRASTS = (
    DirectionContrast(
        "any_number_vs_none",
        "any_number",
        "none",
        ("exact_number", "wrong_number"),
        ("none",),
    ),
    DirectionContrast(
        "exact_number_vs_none",
        "exact_number",
        "none",
        ("exact_number",),
        ("none",),
    ),
    DirectionContrast(
        "wrong_number_vs_none",
        "wrong_number",
        "none",
        ("wrong_number",),
        ("none",),
    ),
    DirectionContrast(
        "exact_number_vs_wrong_number",
        "exact_number",
        "wrong_number",
        ("exact_number",),
        ("wrong_number",),
    ),
)


@dataclass(frozen=True)
class DirectionBank:
    """Directions learned from one or more training-component shards.

    Direction tensors have shape ``[contrast, panel, layer, hidden]``.  Panel
    zero is the count-weighted union of all source panels; subsequent panels
    correspond one-to-one with ``source_panels``.
    """

    layers: tuple[int, ...]
    source_panels: tuple[str, ...]
    panels: tuple[str, ...]
    contrasts: tuple[DirectionContrast, ...]
    contrast_directions: torch.Tensor
    positive_prototypes: torch.Tensor
    training_counts: Mapping[str, Mapping[str, Mapping[str, int]]]

    def to(self, device: torch.device | str) -> "DirectionBank":
        """Return a bank whose direction tensors reside on ``device``."""

        return DirectionBank(
            layers=self.layers,
            source_panels=self.source_panels,
            panels=self.panels,
            contrasts=self.contrasts,
            contrast_directions=self.contrast_directions.to(device),
            positive_prototypes=self.positive_prototypes.to(device),
            training_counts=self.training_counts,
        )


def classify_primitive_outcomes(
    prediction: torch.Tensor,
    target: torch.Tensor,
    none_index: int,
) -> torch.Tensor:
    """Encode predictions as none=0, exact number=1, or wrong number=2."""

    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have identical shapes, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    outcome = torch.full_like(prediction, 2)
    outcome[prediction.eq(int(none_index))] = 0
    outcome[prediction.eq(target)] = 1
    return outcome


def _validate_component_state(
    state: Mapping,
    *,
    path: Path,
    expected_layers: tuple[int, ...],
    expected_panels: tuple[str, ...] | None,
    expected_outcomes: tuple[str, ...] | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    required = {"layers", "panels", "outcomes", "pooled_counts", "pooled_unit_sums"}
    missing = required.difference(state)
    if missing:
        raise ValueError(f"component file {path} is missing keys: {sorted(missing)}")

    layers = tuple(int(layer) for layer in state["layers"])
    panels = tuple(str(panel) for panel in state["panels"])
    outcomes = tuple(str(outcome) for outcome in state["outcomes"])
    if layers != expected_layers:
        raise ValueError(
            f"component layers in {path} are {list(layers)}, expected "
            f"{list(expected_layers)}"
        )
    if expected_panels is not None and panels != expected_panels:
        raise ValueError(
            f"component panels in {path} are {panels}, expected {expected_panels}"
        )
    if expected_outcomes is not None and outcomes != expected_outcomes:
        raise ValueError(
            f"component outcomes in {path} are {outcomes}, expected "
            f"{expected_outcomes}"
        )

    counts = state["pooled_counts"]
    unit_sums = state["pooled_unit_sums"]
    expected_count_shape = (len(panels), len(outcomes))
    if tuple(counts.shape) != expected_count_shape:
        raise ValueError(
            f"pooled_counts in {path} has shape {tuple(counts.shape)}, expected "
            f"{expected_count_shape}"
        )
    if unit_sums.dim() != 4 or tuple(unit_sums.shape[:3]) != (
        len(panels),
        len(outcomes),
        len(layers),
    ):
        raise ValueError(
            f"pooled_unit_sums in {path} has incompatible shape "
            f"{tuple(unit_sums.shape)}"
        )
    return panels, outcomes


def _normalized_or_nan(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    norms = values.norm(dim=-1, keepdim=True)
    normalized = F.normalize(values, dim=-1)
    usable = valid[..., None] & norms.gt(0)
    return torch.where(usable, normalized, torch.full_like(normalized, torch.nan))


def load_direction_bank(
    paths: Sequence[Path],
    expected_layers: Sequence[int],
    *,
    contrasts: Sequence[DirectionContrast] = DEFAULT_CONTRASTS,
) -> DirectionBank:
    """Merge training shards and estimate count-weighted direction prototypes."""

    component_paths = tuple(Path(path).resolve() for path in paths)
    if not component_paths:
        raise ValueError("at least one training-component path is required")
    layers = tuple(int(layer) for layer in expected_layers)
    if not layers or len(set(layers)) != len(layers):
        raise ValueError("expected_layers must be non-empty and unique")
    contrast_specs = tuple(contrasts)
    if not contrast_specs or len({item.name for item in contrast_specs}) != len(
        contrast_specs
    ):
        raise ValueError("contrasts must be non-empty and have unique names")

    states = []
    source_panels: tuple[str, ...] | None = None
    outcomes: tuple[str, ...] | None = None
    for path in component_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        state = torch.load(path, map_location="cpu", weights_only=True)
        source_panels, outcomes = _validate_component_state(
            state,
            path=path,
            expected_layers=layers,
            expected_panels=source_panels,
            expected_outcomes=outcomes,
        )
        states.append(state)
    assert source_panels is not None and outcomes is not None

    required_outcomes = {
        outcome
        for contrast in contrast_specs
        for outcome in (contrast.positive_component, contrast.negative_component)
    }
    missing_outcomes = required_outcomes.difference(outcomes)
    if missing_outcomes:
        raise ValueError(
            "training components do not contain required outcomes: "
            f"{sorted(missing_outcomes)}"
        )

    counts = torch.stack(
        [state["pooled_counts"].to(torch.int64) for state in states]
    ).sum(0)
    unit_sums = torch.stack(
        [state["pooled_unit_sums"].float() for state in states]
    ).sum(0)
    has_union = [
        "pooled_union_counts" in state or "pooled_union_unit_sums" in state
        for state in states
    ]
    if any(has_union) and not all(has_union):
        raise ValueError("component shards must agree on explicit union statistics")
    if all(has_union):
        union_counts = torch.stack(
            [state["pooled_union_counts"].to(torch.int64) for state in states]
        ).sum(0)
        union_unit_sums = torch.stack(
            [state["pooled_union_unit_sums"].float() for state in states]
        ).sum(0)
        if union_counts.shape != counts.shape[1:]:
            raise ValueError("explicit union counts have an incompatible shape")
        if union_unit_sums.shape != unit_sums.shape[1:]:
            raise ValueError("explicit union unit sums have an incompatible shape")
    else:
        union_counts = counts.sum(dim=0)
        union_unit_sums = unit_sums.sum(dim=0)
    panel_counts = torch.cat((union_counts[None], counts), dim=0)
    panel_unit_sums = torch.cat((union_unit_sums[None], unit_sums), dim=0)
    panels = ("all_boundaries", *source_panels)

    contrast_directions = []
    positive_prototypes = []
    training_counts: dict[str, dict[str, dict[str, int]]] = {}
    for contrast in contrast_specs:
        positive_index = outcomes.index(contrast.positive_component)
        negative_index = outcomes.index(contrast.negative_component)
        positive_counts = panel_counts[:, positive_index]
        negative_counts = panel_counts[:, negative_index]
        positive_mean = panel_unit_sums[:, positive_index] / positive_counts.clamp_min(
            1
        )[:, None, None]
        negative_mean = panel_unit_sums[:, negative_index] / negative_counts.clamp_min(
            1
        )[:, None, None]
        positive_valid = positive_counts.gt(0)[:, None].expand(-1, len(layers))
        contrast_valid = (
            positive_counts.gt(0) & negative_counts.gt(0)
        )[:, None].expand(-1, len(layers))
        contrast_directions.append(
            _normalized_or_nan(positive_mean - negative_mean, contrast_valid)
        )
        positive_prototypes.append(
            _normalized_or_nan(positive_mean, positive_valid)
        )
        training_counts[contrast.name] = {
            panel: {
                "positive": int(positive_counts[index]),
                "negative": int(negative_counts[index]),
            }
            for index, panel in enumerate(panels)
        }

    return DirectionBank(
        layers=layers,
        source_panels=source_panels,
        panels=panels,
        contrasts=contrast_specs,
        contrast_directions=torch.stack(contrast_directions),
        positive_prototypes=torch.stack(positive_prototypes),
        training_counts=training_counts,
    )


def boundary_panel_indices(
    boundary_text: Sequence[Sequence[str]],
    source_panels: Sequence[str],
    *,
    newline_panel: str = "literal_newline",
    non_newline_panel: str = "token_marker_mean",
) -> torch.Tensor:
    """Assign each candidate position to a component panel by decoded text.

    The number and location of newline boundaries are inferred from the
    rendered prompts.  A position must have the same newline status in every
    example, which prevents silently mixing incompatible prompt layouts.
    """

    panel_names = tuple(str(panel) for panel in source_panels)
    if newline_panel not in panel_names or non_newline_panel not in panel_names:
        raise ValueError(
            "source panels must contain the configured newline and non-newline "
            f"panels; got {panel_names}"
        )
    if newline_panel == non_newline_panel:
        raise ValueError("newline and non-newline panel names must differ")
    if not boundary_text:
        raise ValueError("boundary_text must contain at least one example")
    position_count = len(boundary_text[0])
    if position_count == 0 or any(len(row) != position_count for row in boundary_text):
        raise ValueError("boundary_text rows must have the same non-zero length")

    indices = []
    for position in range(position_count):
        flags = {"\n" in row[position] for row in boundary_text}
        if len(flags) != 1:
            raise ValueError(
                f"boundary position {position} is a newline in only some examples"
            )
        panel = newline_panel if flags.pop() else non_newline_panel
        indices.append(panel_names.index(panel))
    return torch.tensor(indices, dtype=torch.long)


def score_boundary_deltas(
    deltas: torch.Tensor,
    source_panel_index: torch.Tensor,
    bank: DirectionBank,
) -> dict[str, torch.Tensor]:
    """Score a batch of deltas against every contrast and applicable panel.

    Args:
        deltas: ``[batch, layer, hidden]`` injected-minus-clean states.
        source_panel_index: ``[batch]`` source-panel index for each trial.
        bank: directions learned from disjoint training concepts.

    Returns:
        Three tensors shaped ``[batch, contrast, panel, layer]``.  Scores for
        non-applicable source panels are NaN; the union panel is always filled.
    """

    if deltas.dim() != 3:
        raise ValueError(
            f"deltas must have shape [batch, layer, hidden], got {tuple(deltas.shape)}"
        )
    if deltas.shape[1:] != bank.contrast_directions.shape[2:]:
        raise ValueError(
            "delta layer/hidden dimensions do not match direction bank: "
            f"{tuple(deltas.shape[1:])} vs "
            f"{tuple(bank.contrast_directions.shape[2:])}"
        )
    panel_index = torch.as_tensor(
        source_panel_index, dtype=torch.long, device=deltas.device
    )
    if panel_index.shape != (deltas.shape[0],):
        raise ValueError(
            f"source_panel_index must have shape ({deltas.shape[0]},), got "
            f"{tuple(panel_index.shape)}"
        )
    if bool(((panel_index < 0) | (panel_index >= len(bank.source_panels))).any()):
        raise ValueError("source_panel_index contains an out-of-range panel")

    directions = bank.contrast_directions.to(deltas.device)
    prototypes = bank.positive_prototypes.to(deltas.device)
    unit_deltas = F.normalize(deltas, dim=-1)
    shape = (
        deltas.shape[0],
        len(bank.contrasts),
        len(bank.panels),
        len(bank.layers),
    )
    unit_scores = torch.full(shape, torch.nan, device=deltas.device)
    prototype_scores = torch.full_like(unit_scores, torch.nan)
    raw_scores = torch.full_like(unit_scores, torch.nan)
    for output_panel in range(len(bank.panels)):
        mask = (
            torch.ones_like(panel_index, dtype=torch.bool)
            if output_panel == 0
            else panel_index.eq(output_panel - 1)
        )
        if not bool(mask.any()):
            continue
        selected_direction = directions[:, output_panel]
        selected_prototype = prototypes[:, output_panel]
        unit_scores[mask, :, output_panel] = torch.einsum(
            "bld,cld->bcl", unit_deltas[mask], selected_direction
        )
        prototype_scores[mask, :, output_panel] = torch.einsum(
            "bld,cld->bcl", unit_deltas[mask], selected_prototype
        )
        raw_scores[mask, :, output_panel] = torch.einsum(
            "bld,cld->bcl", deltas[mask], selected_direction
        )
    return {
        "unit_contrast_projection": unit_scores,
        "cosine_to_positive_prototype": prototype_scores,
        "raw_contrast_projection": raw_scores,
    }


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute binary ROC AUC with average ranks for tied scores."""

    labels_array = np.asarray(labels, dtype=bool)
    scores_array = np.asarray(scores, dtype=np.float64)
    if labels_array.shape != scores_array.shape:
        raise ValueError("labels and scores must have identical shapes")
    if not np.isfinite(scores_array).all():
        raise ValueError("scores must be finite")
    n_positive = int(labels_array.sum())
    n_negative = int((~labels_array).sum())
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    order = np.argsort(scores_array, kind="mergesort")
    sorted_scores = scores_array[order]
    ranks = np.empty(len(scores_array), dtype=np.float64)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        # One-indexed average rank for the half-open tied interval [start, end).
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    positive_rank_sum = ranks[labels_array].sum()
    return float(
        (positive_rank_sum - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def contrast_metadata(
    contrasts: Sequence[DirectionContrast],
) -> dict[str, dict[str, list[str]]]:
    """Serialize held-out label definitions without relying on numeric codes."""

    return {
        contrast.name: {
            "positive": list(contrast.positive_outcomes),
            "negative": list(contrast.negative_outcomes),
        }
        for contrast in contrasts
    }
