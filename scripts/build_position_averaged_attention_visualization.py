#!/usr/bin/env python3
"""Build the TransformerLens version of the averaged token-attention browser."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from introspection_core.attention_concepts import load_concept_bank
from introspection_core.cluster_split import file_sha256, require_valid_cluster_split
from introspection_core.attention_inputs import TokenLocalizationCsvTask
from introspection_core.model import HookedModel, ModelConfig
from introspection_core.position_averaged_attention import (
    build_position_averaged_attention_visualization,
)
from introspection_core.prompts import REGISTRY


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF id or local checkpoint")
    parser.add_argument(
        "--task",
        choices=["token_localization"],
        default="token_localization",
        help="Task adapter; add new adapters without changing aggregation/browser code.",
    )
    parser.add_argument("--cluster_file", type=Path, required=True)
    parser.add_argument("--choices_column", default="choices")
    parser.add_argument("--cluster_key_column", default="cluster_key")
    parser.add_argument(
        "--position_index_start",
        type=int,
        choices=[0],
        default=0,
        help="First TOKEN label; the registered prompts number slots from 0.",
    )
    parser.add_argument(
        "--prompt_template",
        choices=sorted(REGISTRY),
        default=None,
        help="Prompt template (default: token_localization).",
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        required=True,
        help="Canonical results/<model> directory",
    )
    parser.add_argument(
        "--calibration_selection",
        type=Path,
        help=(
            "Frozen calibration selection (default: "
            "<results_dir>/calibration/selection.json)"
        ),
    )
    parser.add_argument(
        "--artifact_name",
        choices=["validation_attention"],
        default="validation_attention",
    )
    parser.add_argument("--max_clusters", type=int)
    parser.add_argument("--positions", nargs="+", type=int)
    parser.add_argument("--layers", default="all", help="'all', '0-15', or '0,4,8'")

    parser.add_argument("--concept", default="Genes")
    parser.add_argument("--average_all_concepts", action="store_true")
    parser.add_argument(
        "--vector_source",
        choices=["on_the_fly", "saved", "payload"],
        default="on_the_fly",
    )
    parser.add_argument(
        "--state_vectors",
        type=Path,
        help=(
            "state_vectors.pt covering at least the selected concepts. Implies "
            "--vector_source payload, which slices it by name instead of "
            "re-extracting."
        ),
    )
    parser.add_argument("--concepts_json", type=Path)
    parser.add_argument("--concept_csv", type=Path)
    parser.add_argument("--concept_column", default="concept")
    parser.add_argument("--max_concepts", type=int)
    parser.add_argument(
        "--max_baselines",
        type=int,
        help="Optional extraction smoke-test cap; omit for the full baseline bank.",
    )
    parser.add_argument("--vectors_dir", type=Path)
    parser.add_argument("--vec_type", default="last")
    parser.add_argument("--vector_batch_size", type=int, default=16)
    parser.add_argument("--concept_batch_size", type=int, default=16)

    parser.add_argument("--injection_layer", type=int)
    parser.add_argument("--strength", type=float)
    parser.add_argument(
        "--scale_mode",
        choices=["relative_hidden_norm", "unit"],
        default="relative_hidden_norm",
    )
    parser.add_argument(
        "--prompt_preamble",
        choices=["none", "user", "system"],
        default=None,
        help="Prompt framing (default: system).",
    )
    parser.add_argument(
        "--choice_suffix",
        default="",
        help="Text after every choice; default is empty.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.state_vectors is not None:
        if args.vector_source == "saved":
            parser.error("--state_vectors conflicts with --vector_source saved")
        args.vector_source = "payload"
    elif args.vector_source == "payload":
        parser.error("--vector_source payload requires --state_vectors")
    if args.task == "token_localization":
        missing = [
            name
            for name in (
                "concepts_json",
                "concept_csv",
                "injection_layer",
                "strength",
            )
            if getattr(args, name) is None
        ]
        if missing:
            parser.error(
                "--task token_localization requires: "
                + ", ".join(f"--{name}" for name in missing)
            )
    return args


def resolve(path: Path | None) -> Path | None:
    if path is None or path.is_absolute():
        return path
    return REPO_ROOT / path


def parse_layers(spec: str, n_layers: int) -> list[int] | None:
    if spec == "all":
        return None
    if "-" in spec and "," not in spec:
        start, end = (int(value) for value in spec.split("-", 1))
        layers = list(range(start, end + 1))
    else:
        layers = [int(value) for value in spec.split(",")]
    invalid = [layer for layer in layers if layer < 0 or layer >= n_layers]
    if not layers or invalid:
        raise ValueError(f"Invalid --layers {spec!r}; invalid={invalid}")
    return layers


def validate_calibration_selection(
    path: Path,
    *,
    injection_layer: int,
    strength: float,
) -> dict:
    """Load the frozen calibration choice and reject stale CLI settings."""
    selection = json.loads(path.read_text())
    selected_layer = selection.get("injection_layer")
    selected_strength = selection.get(
        "strength", selection.get("injection_strength")
    )
    mismatches = {}
    if selected_layer is None or int(selected_layer) != int(injection_layer):
        mismatches["injection_layer"] = (selected_layer, injection_layer)
    if selected_strength is None or not math.isclose(
        float(selected_strength),
        float(strength),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        mismatches["strength"] = (selected_strength, strength)
    if mismatches:
        raise ValueError(
            "CLI settings do not match calibration/selection.json: "
            f"{mismatches}"
        )
    return selection


def main(argv=None) -> None:
    args = parse_args(argv)
    results_dir = resolve(args.results_dir)
    calibration_selection_path: Path | None = None
    calibration_selection: dict | None = None
    if args.task == "token_localization":
        calibration_selection_path = (
            resolve(args.calibration_selection)
            if args.calibration_selection is not None
            else results_dir / "calibration" / "selection.json"
        )
        calibration_selection = validate_calibration_selection(
            calibration_selection_path,
            injection_layer=args.injection_layer,
            strength=args.strength,
        )
    torch.manual_seed(args.seed)
    output_dir = results_dir / "visualizations"
    cluster_file = resolve(args.cluster_file)
    require_valid_cluster_split(cluster_file)
    model = HookedModel(
        ModelConfig(
            name=args.model,
            device=args.device,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
    )
    layers = parse_layers(args.layers, int(model.cfg.n_layers))
    position_index_start = args.position_index_start
    prompt_preamble = (
        args.prompt_preamble
        if args.prompt_preamble is not None
        else "system"
    )
    task_kwargs = {
        "path": cluster_file,
        "max_examples": args.max_clusters,
        "choices_column": args.choices_column,
        "key_column": args.cluster_key_column,
        "preamble": prompt_preamble,
        "choice_suffix": args.choice_suffix,
        "position_index_start": position_index_start,
    }
    provenance = {
        "cluster_file": str(resolve(args.cluster_file)),
        "seed": args.seed,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    if calibration_selection_path is not None:
        provenance.update(
            {
                "calibration_selection": str(calibration_selection_path),
                "calibration_selection_sha256": file_sha256(
                    calibration_selection_path
                ),
            }
        )
    task = TokenLocalizationCsvTask(
        **task_kwargs,
        template_name=args.prompt_template,
        name=args.prompt_template or "token_localization",
    )
    concept_bank = load_concept_bank(
        model,
        concepts_json=resolve(args.concepts_json),
        concept_csv=resolve(args.concept_csv),
        concept_column=args.concept_column,
        max_concepts=args.max_concepts,
        max_baselines=args.max_baselines,
        layer=args.injection_layer,
        average_all_concepts=args.average_all_concepts,
        concept=args.concept,
        vector_source=args.vector_source,
        vectors_dir=resolve(args.vectors_dir),
        state_vectors=resolve(args.state_vectors),
        vec_type=args.vec_type,
        extraction_batch_size=args.vector_batch_size,
    )
    print(
        f"task={task.name} examples={args.max_clusters or 'all'} "
        f"concepts={len(concept_bank.names)} positions={args.positions or 'all'}",
        flush=True,
    )
    provenance.update(
        {
            "concepts_json": str(resolve(args.concepts_json)),
            "concept_csv": str(resolve(args.concept_csv)),
            "vector_source": args.vector_source,
            "vector_paths": concept_bank.source_paths,
        }
    )
    browser_path = build_position_averaged_attention_visualization(
        model,
        task=task,
        concept_bank=concept_bank,
        output_dir=output_dir,
        injection_layer=args.injection_layer,
        positions=args.positions,
        layers=layers,
        strength=args.strength,
        scale_mode=args.scale_mode,
        concept_batch_size=args.concept_batch_size,
        provenance=provenance,
        artifact_stem=args.artifact_name,
    )
    print(f"saved {browser_path}", flush=True)


if __name__ == "__main__":
    main()
