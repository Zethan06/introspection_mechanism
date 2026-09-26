#!/usr/bin/env python3
"""Fit the position--none direction of Figure 2a on the fit half of the injected trials.

At every layer, d = norm(mean_{position report} norm(h) - mean_{none} norm(h))
over final-token residuals of injected trials.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from introspection_core.boundary_direction_auc import (  # noqa: E402
    BOUNDARY_REPRESENTATIONS,
    PRIMITIVE_OUTCOMES,
    boundary_protocol,
    boundary_representation,
    classify_primitive_outcomes,
    complete_unique_clean_union_statistics,
    unique_clean_reference_components,
)
from introspection_core.boundary_direction_runtime import (  # noqa: E402
    CAPTURE_POSITIONS,
    build_boundary_task,
    capture_positions,
    collect_clean_boundary_states,
    load_concept_vectors,
    residual_capture_hooks,
)
from introspection_core.injection import make_injection_hook  # noqa: E402
from introspection_core.model import HookedModel, ModelConfig  # noqa: E402


COMPONENT_OUTCOMES = ("all", "none", "any_number", "exact_number", "wrong_number")
SOURCE_PANELS = ("token_marker_mean", "literal_newline")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-concept-csv", type=Path, required=True)
    parser.add_argument("--state-vectors", type=Path, required=True)
    parser.add_argument("--train-cluster-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--injection-layer", type=int, required=True)
    parser.add_argument(
        "--end-layer",
        type=int,
        help="Final layer to inspect (default: the model's final layer).",
    )
    parser.add_argument(
        "--start-layer",
        type=int,
        default=0,
        help="First layer to inspect (default: 0).",
    )
    parser.add_argument("--strength", type=float, required=True)
    parser.add_argument(
        "--representation",
        choices=BOUNDARY_REPRESENTATIONS,
        default="delta",
        help=(
            "Residual representation used to estimate directions: paired "
            "injected-clean delta (default), the injected latent directly, or "
            "injected Number trials versus one clean latent per training cluster."
        ),
    )
    parser.add_argument(
        "--prompt-template",
        default="semantic_highinj_posref_gate_balanced_disrupts",
    )
    parser.add_argument("--prompt-preamble", default="system")
    parser.add_argument(
        "--capture-position",
        choices=CAPTURE_POSITIONS,
        default="routing_boundary",
        help="Residual-stream position used to estimate the outcome direction.",
    )
    parser.add_argument(
        "--scale-mode",
        choices=("unit", "relative_hidden_norm"),
        default="relative_hidden_norm",
    )
    parser.add_argument("--newline-panel", default="literal_newline")
    parser.add_argument("--non-newline-panel", default="token_marker_mean")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.injection_layer < 0 or args.start_layer < 0:
        raise ValueError("injection-layer and start-layer must be non-negative")
    if args.end_layer is not None and args.end_layer < args.start_layer:
        raise ValueError("end-layer must not precede start-layer")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    for field in ("train_concept_csv", "state_vectors", "train_cluster_csv"):
        path = getattr(args, field).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        setattr(args, field, path)
    args.output = args.output.resolve()


def _outcome_masks(primitive: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return (
        torch.ones_like(primitive, dtype=torch.bool),
        primitive.eq(0),
        primitive.ne(0),
        primitive.eq(1),
        primitive.eq(2),
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_args(args)
    concepts, unit_vectors = load_concept_vectors(
        args.train_concept_csv, args.state_vectors
    )
    model = HookedModel(
        ModelConfig(
            name=args.model,
            device=args.device,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
    )
    model_layer_count = int(model.cfg.n_layers)
    model_width = int(model.cfg.d_model)
    if args.injection_layer >= model_layer_count:
        raise ValueError(
            f"injection layer {args.injection_layer} exceeds model range "
            f"0..{model_layer_count - 1}"
        )
    args.end_layer = (
        args.end_layer if args.end_layer is not None else model_layer_count - 1
    )
    if args.end_layer >= model_layer_count:
        raise ValueError(
            f"end layer {args.end_layer} exceeds model range 0..{model_layer_count - 1}"
        )
    if args.start_layer > args.end_layer:
        raise ValueError("start-layer exceeds the model layer range")
    layers = list(range(args.start_layer, args.end_layer + 1))
    if unit_vectors.shape[1] != model_width:
        raise ValueError(
            f"state-vector width {unit_vectors.shape[1]} does not match model width "
            f"{model_width}"
        )
    task = build_boundary_task(
        model,
        args.train_cluster_csv,
        prompt_preamble=args.prompt_preamble,
        prompt_template=args.prompt_template,
        source_panels=SOURCE_PANELS,
        newline_panel=args.newline_panel,
        non_newline_panel=args.non_newline_panel,
    )
    state_positions = capture_positions(task, args.capture_position)
    protocol = boundary_protocol(args.representation)
    if (
        protocol.required_capture_position is not None
        and args.capture_position != protocol.required_capture_position
    ):
        raise ValueError(
            f"{args.representation} requires --capture-position "
            f"{protocol.required_capture_position}"
        )
    clean_boundary = None
    if protocol.collect_clean_training:
        clean_boundary = collect_clean_boundary_states(
            model, task, layers, batch_size=args.batch_size, positions=state_positions
        )
    device = torch.device(model.bridge.cfg.device)

    if protocol.unique_clean_negative:
        assert clean_boundary is not None
        none_index = COMPONENT_OUTCOMES.index("none")
        counts, unit_sums, union_count, union_unit_sum = (
            unique_clean_reference_components(
                clean_boundary,
                source_panel_count=len(SOURCE_PANELS),
                outcome_count=len(COMPONENT_OUTCOMES),
                none_index=none_index,
            )
        )
    else:
        counts = torch.zeros(
            len(SOURCE_PANELS), len(COMPONENT_OUTCOMES), dtype=torch.int64
        )
        unit_sums = torch.zeros(
            len(SOURCE_PANELS),
            len(COMPONENT_OUTCOMES),
            len(layers),
            model_width,
            dtype=torch.float32,
        )
    site_count = len(task.examples) * task.position_count
    total_trials = len(concepts) * site_count
    for batch_number, start in enumerate(
        range(0, total_trials, args.batch_size), start=1
    ):
        end = min(start + args.batch_size, total_trials)
        flat = torch.arange(start, end)
        concept_index = torch.div(flat, site_count, rounding_mode="floor")
        within = flat.remainder(site_count)
        example_index = torch.div(
            within, task.position_count, rounding_mode="floor"
        )
        position_index = within.remainder(task.position_count)
        word_positions = task.candidate_positions[example_index, position_index]
        observation_positions = state_positions[example_index, position_index]
        destination: dict[int, torch.Tensor] = {}
        logits, _ = model.last_token_candidate_stats(
            task.base_tokens.index_select(0, example_index),
            candidate_token_ids=task.candidate_ids.index_select(0, example_index),
            fwd_hooks=[
                make_injection_hook(
                    model,
                    layer=args.injection_layer,
                    positions=[
                        (int(value), int(value) + 1) for value in word_positions
                    ],
                    vector=unit_vectors.index_select(0, concept_index),
                    strength=args.strength,
                    scale=args.scale_mode,
                ),
                *residual_capture_hooks(
                    model, layers, observation_positions, destination
                ),
            ],
        )
        primitive = classify_primitive_outcomes(
            logits.argmax(dim=-1),
            position_index.to(logits.device),
            task.position_count,
        ).cpu()
        injected = torch.stack([destination[layer] for layer in layers], dim=1).float()
        clean = (
            clean_boundary[example_index.to(device), position_index.to(device)]
            if clean_boundary is not None
            else None
        )
        representation = boundary_representation(injected, clean, args.representation)
        unit_representation = F.normalize(representation, dim=-1)
        trial_panels = task.position_panel_index[position_index]
        for panel_index in range(len(SOURCE_PANELS)):
            panel_mask = trial_panels.eq(panel_index)
            for outcome_index, outcome_mask in enumerate(_outcome_masks(primitive)):
                if (
                    protocol.unique_clean_negative
                    and COMPONENT_OUTCOMES[outcome_index] == "none"
                ):
                    continue
                mask = panel_mask & outcome_mask
                count = int(mask.sum())
                counts[panel_index, outcome_index] += count
                if count:
                    unit_sums[panel_index, outcome_index] += unit_representation[
                        mask.to(device)
                    ].sum(dim=0).cpu()
        if batch_number % 25 == 0 or end == total_trials:
            print(f"[train] trials={end}/{total_trials}", flush=True)

    component_payload = {
        "schema_version": 2,
        "layers": layers,
        "panels": SOURCE_PANELS,
        "outcomes": COMPONENT_OUTCOMES,
        "pooled_counts": counts,
        "pooled_unit_sums": unit_sums,
    }
    if protocol.unique_clean_negative:
        union_count, union_unit_sum = complete_unique_clean_union_statistics(
            counts,
            unit_sums,
            union_count,
            union_unit_sum,
            none_index=COMPONENT_OUTCOMES.index("none"),
        )
        component_payload["pooled_union_counts"] = union_count
        component_payload["pooled_union_unit_sums"] = union_unit_sum
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(component_payload, args.output)
    metadata = {
        "schema_version": 3,
        "role": "train_direction_components",
        "model": args.model,
        "model_layer_count": model_layer_count,
        "model_width": model_width,
        "train_concept_csv": str(args.train_concept_csv),
        "train_cluster_csv": str(args.train_cluster_csv),
        "state_vectors": str(args.state_vectors),
        "concepts": concepts,
        "concept_count": len(concepts),
        "cluster_count": len(task.examples),
        "position_count": task.position_count,
        "position_source_panels": [
            SOURCE_PANELS[index] for index in task.position_panel_index.tolist()
        ],
        "layers": layers,
        "start_layer": args.start_layer,
        "injection_layer": args.injection_layer,
        "end_layer": args.end_layer,
        "strength": args.strength,
        "scale_mode": args.scale_mode,
        "prompt_template": args.prompt_template,
        "prompt_preamble": args.prompt_preamble,
        "capture_position": args.capture_position,
        "total_trials": total_trials,
        "representation": args.representation,
        "representation_definition": (
            "unit(injected_resid-clean_resid)"
            if args.representation == "delta"
            else (
                "number=unit(injected_resid); none=one unit(clean_resid) per "
                "training cluster with equal cluster weight"
                if protocol.unique_clean_negative
                else "unit(injected_resid)"
            )
        ),
        "unique_clean_cluster_count": (
            len(task.examples) if protocol.unique_clean_negative else None
        ),
        "component_file": str(args.output),
        "primitive_outcomes": list(PRIMITIVE_OUTCOMES),
    }
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"[done] components={args.output} metadata={metadata_path}", flush=True)


if __name__ == "__main__":
    main()
