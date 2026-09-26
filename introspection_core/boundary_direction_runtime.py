"""Shared model runtime for collecting and scoring boundary-state deltas."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import torch
import torch.nn.functional as F

from .attention_routing import candidate_routing_key_positions
from .boundary_direction_auc import boundary_panel_indices
from .localization_evaluation import load_cluster_prompts
from .model import HookedModel
from .prompts import PromptManager


@dataclass(frozen=True)
class BoundaryTask:
    """Tokenized localization prompts and their candidate boundary layout."""

    examples: tuple
    position_count: int
    base_tokens: torch.Tensor
    candidate_positions: torch.Tensor
    boundary_positions: torch.Tensor
    candidate_ids: torch.Tensor
    boundary_text: tuple[tuple[str, ...], ...]
    position_panel_index: torch.Tensor


CAPTURE_POSITIONS = ("routing_boundary", "final_token", "post_injection_token")


def capture_positions(task: BoundaryTask, capture_position: str) -> torch.Tensor:
    """Resolve the residual-stream observation position for every trial site."""

    if capture_position == "post_injection_token":
        positions = task.candidate_positions + 1
        if torch.any(positions >= task.base_tokens.shape[1]):
            raise ValueError("injection position has no following token")
        return positions
    if capture_position == "routing_boundary":
        return task.boundary_positions
    if capture_position == "final_token":
        return torch.full_like(task.boundary_positions, task.base_tokens.shape[1] - 1)
    raise ValueError(
        f"unknown capture position {capture_position!r}; "
        f"expected one of {CAPTURE_POSITIONS}"
    )


def load_concept_vectors(
    concept_csv: Path,
    state_vectors: Path,
) -> tuple[list[str], torch.Tensor]:
    """Load concepts in split order and select their saved unit vectors."""

    with Path(concept_csv).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "concept" not in rows[0]:
        raise ValueError(
            f"concept CSV is empty or lacks a concept column: {concept_csv}"
        )
    concepts = [str(row["concept"]) for row in rows]
    if len(set(concepts)) != len(concepts):
        raise ValueError(f"concept CSV contains duplicates: {concept_csv}")

    payload = torch.load(state_vectors, map_location="cpu", weights_only=True)
    if "concepts" not in payload or "unit_vectors" not in payload:
        raise ValueError(
            f"state-vector artifact lacks concepts or unit_vectors: {state_vectors}"
        )
    state_index = {str(name): index for index, name in enumerate(payload["concepts"])}
    missing = [concept for concept in concepts if concept not in state_index]
    if missing:
        raise ValueError(
            f"state-vector artifact is missing {len(missing)} split concepts; "
            f"first missing concept: {missing[0]!r}"
        )
    vectors = payload["unit_vectors"].float().index_select(
        0, torch.tensor([state_index[concept] for concept in concepts])
    )
    return concepts, F.normalize(vectors, dim=-1)


def routing_positions(tokenizer, example, position_count: int) -> list[int]:
    """Resolve the routing boundary following every candidate span."""

    adapter = SimpleNamespace(
        input_ids=example.input_ids,
        positions=tuple(range(position_count)),
        injection_spans={
            position: SimpleNamespace(
                start=int(example.span_starts[position]),
                end=int(example.span_ends[position]),
            )
            for position in range(position_count)
        },
    )
    resolved = candidate_routing_key_positions(tokenizer, adapter)
    return [int(resolved[position]) for position in range(position_count)]


def residual_capture_hooks(
    model: HookedModel,
    layers: Sequence[int],
    positions: torch.Tensor,
    destination: dict[int, torch.Tensor],
) -> list[tuple]:
    """Capture selected batch-specific positions at residual-post hooks."""

    output_device = torch.device(model.bridge.cfg.device)
    hooks = []
    for layer in layers:
        def make_capture(layer_index: int):
            def capture(activation, hook):
                del hook
                activation_positions = positions.to(activation.device)
                rows = torch.arange(
                    activation_positions.shape[0], device=activation.device
                )
                row_index = rows if activation_positions.dim() == 1 else rows[:, None]
                selected = activation[row_index, activation_positions, :].detach()
                destination[layer_index] = selected.to(output_device)
                return activation

            return capture

        hooks.append((model.resid_hook_name(layer), make_capture(layer)))
    return hooks


def build_boundary_task(
    model: HookedModel,
    cluster_csv: Path,
    *,
    prompt_preamble: str,
    prompt_template: str,
    source_panels: Sequence[str],
    newline_panel: str,
    non_newline_panel: str,
) -> BoundaryTask:
    """Render one split's prompts and infer its model-specific token layout."""

    examples, position_count = load_cluster_prompts(
        cluster_csv,
        PromptManager(model.tokenizer),
        preamble=prompt_preamble,
        template_name=prompt_template,
    )
    if not examples or position_count < 1:
        raise ValueError("cluster CSV produced no examples or candidate positions")
    for example in examples:
        if len(example.candidate_labels) != position_count + 1:
            raise ValueError("each prompt must include an explicit none candidate")
        if any(
            int(end) - int(start) != 1
            for start, end in zip(example.span_starts, example.span_ends)
        ):
            raise ValueError("direction AUC requires single-token candidate spans")

    base_tokens = torch.cat([example.input_ids for example in examples], dim=0)
    candidate_positions = torch.tensor(
        [
            [int(example.span_starts[position]) for position in range(position_count)]
            for example in examples
        ]
    )
    boundaries = torch.tensor(
        [
            routing_positions(model.tokenizer, example, position_count)
            for example in examples
        ]
    )
    candidate_ids = torch.tensor([example.candidate_token_ids for example in examples])
    boundary_ids = base_tokens.gather(1, boundaries)
    boundary_text = tuple(
        tuple(
            model.tokenizer.decode(
                [int(token)], clean_up_tokenization_spaces=False
            )
            for token in row
        )
        for row in boundary_ids.tolist()
    )
    panel_index = boundary_panel_indices(
        boundary_text,
        source_panels,
        newline_panel=newline_panel,
        non_newline_panel=non_newline_panel,
    )
    return BoundaryTask(
        examples=tuple(examples),
        position_count=position_count,
        base_tokens=base_tokens,
        candidate_positions=candidate_positions,
        boundary_positions=boundaries,
        candidate_ids=candidate_ids,
        boundary_text=boundary_text,
        position_panel_index=panel_index,
    )


def collect_clean_boundary_states(
    model: HookedModel,
    task: BoundaryTask,
    layers: Sequence[int],
    *,
    batch_size: int,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Collect clean states as ``[example, trial position, layer, hidden]``.

    By default the state is captured at each routing boundary.  Callers can
    provide another ``[example, trial position]`` tensor, such as the final
    input-token position repeated for every injection site.
    """

    reference_batch_size = max(batch_size, len(task.examples))
    example_index = torch.arange(reference_batch_size).remainder(len(task.examples))
    state_positions = task.boundary_positions if positions is None else positions
    if state_positions.shape != task.boundary_positions.shape:
        raise ValueError(
            "capture positions must have shape "
            f"{tuple(task.boundary_positions.shape)}, got {tuple(state_positions.shape)}"
        )
    destination: dict[int, torch.Tensor] = {}
    model.last_token_candidate_stats(
        task.base_tokens.index_select(0, example_index),
        candidate_token_ids=task.candidate_ids.index_select(0, example_index),
        fwd_hooks=residual_capture_hooks(
            model,
            layers,
            state_positions.index_select(0, example_index),
            destination,
        ),
    )
    reference = torch.stack([destination[layer] for layer in layers], dim=2).float()
    first_row = torch.tensor(
        [
            int((example_index == value).nonzero()[0])
            for value in range(len(task.examples))
        ],
        device=reference.device,
    )
    return reference.index_select(0, first_row)
