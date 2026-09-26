#!/usr/bin/env python3
"""Score the held-out half on the position--none direction (Figure 2a).

Each held-out injected trial is scored by the cosine between its final-token
residual and d at every layer; scores and response groups go to one .npz.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from introspection_core.boundary_direction_auc import (  # noqa: E402
    PRIMITIVE_OUTCOMES,
    BOUNDARY_REPRESENTATIONS,
    boundary_protocol,
    classify_primitive_outcomes,
    boundary_representation,
    contrast_metadata,
    load_direction_bank,
    score_boundary_deltas,
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-concept-csv", type=Path, required=True)
    parser.add_argument("--train-cluster-csv", type=Path, required=True)
    parser.add_argument("--training-components", type=Path, required=True)
    parser.add_argument("--test-concept-csv", type=Path, required=True)
    parser.add_argument("--test-cluster-csv", type=Path, required=True)
    parser.add_argument("--state-vectors", type=Path, required=True)
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
            "Residual representation scored against the training direction: "
            "paired injected-clean delta (default), injected latent directly, "
            "or injected test latents against a Number-vs-unique-clean direction."
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
        help="Residual-stream position scored with the train-estimated direction.",
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
    path_fields = (
        "train_concept_csv",
        "train_cluster_csv",
        "training_components",
        "test_concept_csv",
        "test_cluster_csv",
        "state_vectors",
    )
    for field in path_fields:
        path = getattr(args, field).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        setattr(args, field, path)
    if args.train_concept_csv == args.test_concept_csv:
        raise ValueError("train and test concept CSVs must differ")
    if args.train_cluster_csv == args.test_cluster_csv:
        raise ValueError("train and test cluster CSVs must differ")
    args.output = args.output.resolve()


def _validate_training_provenance(
    args: argparse.Namespace,
    *,
    train_concepts: Sequence[str],
    layers: Sequence[int],
) -> dict:
    metadata_path = args.training_components.with_suffix(".metadata.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"training component provenance is required: {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "role": "train_direction_components",
        "model": args.model,
        "train_concept_csv": str(args.train_concept_csv),
        "train_cluster_csv": str(args.train_cluster_csv),
        "state_vectors": str(args.state_vectors),
        "concepts": list(train_concepts),
        "layers": list(layers),
        "injection_layer": args.injection_layer,
        "end_layer": args.end_layer,
        "strength": args.strength,
        "scale_mode": args.scale_mode,
        "prompt_template": args.prompt_template,
        "prompt_preamble": args.prompt_preamble,
        "capture_position": args.capture_position,
        "representation": args.representation,
    }
    found = dict(metadata)
    mismatches = {
        key: {"found": found.get(key), "expected": value}
        for key, value in expected.items()
        if found.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "training components do not match the declared training run: "
            f"{mismatches}"
        )
    return metadata


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_args(args)
    train_concepts, _ = load_concept_vectors(
        args.train_concept_csv, args.state_vectors
    )
    test_concepts, unit_vectors = load_concept_vectors(
        args.test_concept_csv, args.state_vectors
    )
    overlap = sorted(set(train_concepts).intersection(test_concepts))
    if overlap:
        raise ValueError(
            f"train/test concept leakage: {len(overlap)} overlaps; first={overlap[0]!r}"
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
    training_metadata = _validate_training_provenance(
        args, train_concepts=train_concepts, layers=layers
    )
    protocol = boundary_protocol(args.representation)
    test_representation = protocol.test_representation
    direction_bank = load_direction_bank([args.training_components], layers)
    if unit_vectors.shape[1] != model_width:
        raise ValueError(
            f"state-vector width {unit_vectors.shape[1]} does not match model width "
            f"{model_width}"
        )
    task = build_boundary_task(
        model,
        args.test_cluster_csv,
        prompt_preamble=args.prompt_preamble,
        prompt_template=args.prompt_template,
        source_panels=direction_bank.source_panels,
        newline_panel=args.newline_panel,
        non_newline_panel=args.non_newline_panel,
    )
    state_positions = capture_positions(task, args.capture_position)
    clean_boundary = None
    if test_representation == "delta":
        clean_boundary = collect_clean_boundary_states(
            model, task, layers, batch_size=args.batch_size, positions=state_positions
        )
    device = torch.device(model.bridge.cfg.device)
    direction_bank = direction_bank.to(device)

    site_count = len(task.examples) * task.position_count
    total_trials = len(test_concepts) * site_count
    arrays: dict[str, list[np.ndarray]] = {
        "concept_index": [],
        "example_index": [],
        "position_index": [],
        "primitive_outcome": [],
        "unit_contrast_projection": [],
        "cosine_to_positive_prototype": [],
        "raw_contrast_projection": [],
        "representation_norm": [],
    }
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
        representation = boundary_representation(
            injected, clean, test_representation
        )
        scores = score_boundary_deltas(
            representation,
            task.position_panel_index[position_index],
            direction_bank,
        )
        arrays["concept_index"].append(concept_index.numpy())
        arrays["example_index"].append(example_index.numpy())
        arrays["position_index"].append(position_index.numpy())
        arrays["primitive_outcome"].append(primitive.numpy())
        for name, values in scores.items():
            arrays[name].append(values.float().cpu().numpy())
        arrays["representation_norm"].append(
            representation.norm(dim=-1).float().cpu().numpy()
        )
        if batch_number % 25 == 0 or end == total_trials:
            print(f"[test] trials={end}/{total_trials}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 4,
        "protocol": "train_direction_then_test_once",
        "model": args.model,
        "model_layer_count": model_layer_count,
        "model_width": model_width,
        "train_concept_csv": str(args.train_concept_csv),
        "train_cluster_csv": str(args.train_cluster_csv),
        "training_components": str(args.training_components),
        "training_component_metadata": training_metadata,
        "test_concept_csv": str(args.test_concept_csv),
        "test_cluster_csv": str(args.test_cluster_csv),
        "train_concepts": train_concepts,
        "test_concepts": test_concepts,
        "train_concept_count": len(train_concepts),
        "test_concept_count": len(test_concepts),
        "test_cluster_count": len(task.examples),
        "test_trial_count": total_trials,
        "layers": layers,
        "start_layer": args.start_layer,
        "position_count": task.position_count,
        "source_panels": list(direction_bank.source_panels),
        "position_source_panels": [
            direction_bank.source_panels[index]
            for index in task.position_panel_index.tolist()
        ],
        "panels": list(direction_bank.panels),
        "contrasts": [item.name for item in direction_bank.contrasts],
        "contrast_outcomes": contrast_metadata(direction_bank.contrasts),
        "primitive_outcomes": list(PRIMITIVE_OUTCOMES),
        "training_counts": direction_bank.training_counts,
        "injection_layer": args.injection_layer,
        "end_layer": args.end_layer,
        "strength": args.strength,
        "scale_mode": args.scale_mode,
        "prompt_template": args.prompt_template,
        "prompt_preamble": args.prompt_preamble,
        "capture_position": args.capture_position,
        "representation": args.representation,
        "test_representation": test_representation,
        "direction_definition": protocol.direction_definition,
        "positive_prototype_definition": protocol.positive_prototype_definition,
    }
    np.savez_compressed(
        args.output,
        metadata=np.array(json.dumps(metadata)),
        **{name: np.concatenate(parts) for name, parts in arrays.items()},
    )
    print(f"[done] test_scores={args.output}", flush=True)


if __name__ == "__main__":
    main()
