#!/usr/bin/env python3
"""Sweep layer and strength on every concept/cluster/position lane trial."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from introspection_core.cluster_split import require_valid_cluster_split
from introspection_core.extraction import (
    extract_concept_vectors,
    extract_concept_vector_matrices,
    load_concepts_from_json,
)
from introspection_core.localization_evaluation import (
    evaluate_cluster_localization,
    load_cluster_prompts,
    unit_vector_matrix,
)
from introspection_core.model import HookedModel, ModelConfig
from introspection_core.prompts import REGISTRY, PromptManager
from introspection_core.results import make_run_dir, write_metadata, write_table
from introspection_core.vocab_token_clean_prior import extract_single_word_tokens


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--concepts_json", type=Path, required=True)
    parser.add_argument("--cluster_csv", type=Path, required=True)
    parser.add_argument("--max_concepts", type=int)
    parser.add_argument(
        "--baseline_mode",
        choices=["manifest", "full_english"],
        default="manifest",
        help=(
            "manifest uses baseline_words from --concepts_json; "
            "full_english uses every eligible English tokenizer word"
        ),
    )
    parser.add_argument("--baseline_min_word_len", type=int, default=1)
    parser.add_argument("--baseline_max_word_len", type=int, default=32)
    parser.add_argument(
        "--baseline_case_filter",
        choices=["all", "lower", "upper_initial"],
        default="all",
    )
    parser.add_argument("--extraction_layer", type=int)
    parser.add_argument(
        "--tie_extraction_layer",
        action="store_true",
        default=True,
        help="extract vectors at each injection layer (default)",
    )
    parser.add_argument(
        "--no_tie_extraction_layer",
        dest="tie_extraction_layer",
        action="store_false",
    )
    parser.add_argument("--layer_start", type=int)
    parser.add_argument(
        "--layer_end",
        type=int,
        help="last layer to scan (default: the last layer of the model)",
    )
    parser.add_argument("--layer_step", type=int, default=1)
    parser.add_argument(
        "--strengths",
        nargs="+",
        type=float,
        default=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    )
    parser.add_argument(
        "--scale_mode",
        choices=["unit", "relative_hidden_norm"],
        default="relative_hidden_norm",
    )
    parser.add_argument(
        "--preamble", choices=["none", "user", "system"], default="system"
    )
    parser.add_argument(
        "--prompt_template",
        choices=sorted(REGISTRY),
        default="token_localization",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--extraction_batch_size", type=int, default=16)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--results_dir", type=Path)
    parser.add_argument("--run_name", default="sweep_injection_localization")
    parser.add_argument("--run_dir", type=Path)
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--date")
    args = parser.parse_args(argv)
    if args.max_concepts is not None and args.max_concepts <= 0:
        parser.error("--max_concepts must be positive")
    if args.baseline_min_word_len <= 0:
        parser.error("--baseline_min_word_len must be positive")
    if args.baseline_max_word_len < args.baseline_min_word_len:
        parser.error(
            "--baseline_max_word_len must be at least "
            "--baseline_min_word_len"
        )
    return args


def _full_english_baseline_words(tokenizer, args: argparse.Namespace) -> list[str]:
    filter_args = SimpleNamespace(
        word_list=None,
        exclude_word_list=None,
        min_word_len=args.baseline_min_word_len,
        max_word_len=args.baseline_max_word_len,
        case_filter=args.baseline_case_filter,
        max_tokens=None,
        seed=args.seed,
    )
    rows = extract_single_word_tokens(tokenizer, filter_args)
    rows.sort(key=lambda row: int(row["token_id"]))
    if not rows:
        raise ValueError("tokenizer has no eligible ASCII-English word tokens")
    return [str(row["word"]) for row in rows]


def _word_list_sha256(words: list[str]) -> str:
    encoded = json.dumps(
        words,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_layer_end(n_layers: int, requested_end: int | None) -> int:
    """Resolve the inclusive sweep end, defaulting to the whole network."""
    if n_layers <= 0:
        raise ValueError("n_layers must be positive")
    if requested_end is not None:
        return requested_end
    return n_layers - 1


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    require_valid_cluster_split(args.cluster_csv)
    if args.run_dir is None and args.results_dir is None:
        raise SystemExit("error: one of --results_dir or --run_dir is required")
    if args.batch_size <= 0 or args.extraction_batch_size <= 0:
        raise ValueError("batch sizes must be positive")

    prefix = f"[worker {args.worker_id}] "
    print(f"{prefix}loading model {args.model}", flush=True)
    model = HookedModel(
        ModelConfig(name=args.model, device=args.device, dtype=args.dtype)
    )
    prompt_manager = PromptManager(model.tokenizer)
    n_layers = int(model.cfg.n_layers)

    examples, n_choices = load_cluster_prompts(
        args.cluster_csv,
        prompt_manager,
        preamble=args.preamble,
        template_name=args.prompt_template,
    )
    concepts, baseline_words = load_concepts_from_json(
        args.concepts_json,
        max_concepts=args.max_concepts,
    )
    if args.baseline_mode == "full_english":
        baseline_words = _full_english_baseline_words(model.tokenizer, args)
    baseline_words_hash = _word_list_sha256(baseline_words)
    print(
        f"{prefix}baseline={args.baseline_mode} "
        f"words={len(baseline_words)} sha256={baseline_words_hash}",
        flush=True,
    )
    layer_start = args.layer_start if args.layer_start is not None else 0
    layer_end = resolve_layer_end(n_layers, args.layer_end)
    if (
        layer_start < 0
        or layer_end >= n_layers
        or layer_start > layer_end
        or args.layer_step <= 0
    ):
        raise ValueError(
            f"invalid layer range {layer_start}..{layer_end} "
            f"step={args.layer_step} for {n_layers} layers"
        )
    injection_layers = list(
        range(layer_start, layer_end + 1, args.layer_step)
    )
    from introspection_core.data_parallel import shard_items

    my_layers = shard_items(
        injection_layers, args.worker_id, args.num_workers
    )
    cells_by_layer = {
        layer: list(args.strengths) for layer in my_layers
    }
    all_cell_count = len(injection_layers) * len(args.strengths)
    my_cell_count = len(my_layers) * len(args.strengths)

    print(
        f"{prefix}{len(concepts)} concepts × {len(examples)} clusters × "
        f"{n_choices} positions; {my_cell_count}/{all_cell_count} cells",
        flush=True,
    )
    if not my_layers:
        print(f"{prefix}no sweep cells assigned; exiting", flush=True)
        return

    grid_rows: list[dict] = []
    fixed_vectors = None
    fixed_extraction_layer = None
    tied_vectors_by_layer = None
    if not args.tie_extraction_layer:
        fixed_extraction_layer = (
            args.extraction_layer
            if args.extraction_layer is not None
            else n_layers // 2
        )
        extracted = extract_concept_vectors(
            model,
            words=concepts,
            baseline_words=baseline_words,
            layer=fixed_extraction_layer,
            batch_size=args.extraction_batch_size,
        )
        fixed_vectors = unit_vector_matrix([item.vector for item in extracted])
    else:
        extraction_layers = list(cells_by_layer)
        print(
            f"{prefix}extracting vectors for {len(extraction_layers)} layers "
            "in shared forward passes",
            flush=True,
        )
        raw_vectors_by_layer = extract_concept_vector_matrices(
            model,
            words=concepts,
            baseline_words=baseline_words,
            layers=extraction_layers,
            batch_size=args.extraction_batch_size,
        )
        tied_vectors_by_layer = {
            layer: unit_vector_matrix(matrix)
            for layer, matrix in raw_vectors_by_layer.items()
        }

    completed_cells = 0
    for injection_layer, strengths in cells_by_layer.items():
        extraction_layer = (
            injection_layer
            if args.tie_extraction_layer
            else fixed_extraction_layer
        )
        assert extraction_layer is not None
        if tied_vectors_by_layer is not None:
            concept_names = concepts
            vectors = tied_vectors_by_layer[extraction_layer]
        else:
            concept_names = concepts
            assert fixed_vectors is not None
            vectors = fixed_vectors

        for strength in strengths:
            completed_cells += 1
            cell_prefix = (
                f"{prefix}[cell {completed_cells}/{my_cell_count} "
                f"L{injection_layer} S{strength:g}] "
            )
            evaluation = evaluate_cluster_localization(
                model,
                examples=examples,
                concept_names=concept_names,
                unit_vectors=vectors,
                injection_layer=injection_layer,
                strength=strength,
                scale_mode=args.scale_mode,
                batch_size=args.batch_size,
                include_rows=False,
                progress_prefix=cell_prefix,
                progress_every=50,
                trial_mode="balanced_one_per_concept",
            )
            n_trials = int(evaluation.n_trials.sum())
            n_correct = int(evaluation.injected_correct.sum())
            mean_correct_prob = (
                float(evaluation.injected_correct_prob_sum.sum()) / n_trials
            )
            clean_correct = int(evaluation.clean_correct.sum())
            clean_mean_correct_prob = (
                float(evaluation.clean_correct_prob_sum.sum()) / n_trials
            )
            row = {
                "injection_layer": injection_layer,
                "extraction_layer": extraction_layer,
                "strength": strength,
                "n_trials": n_trials,
                "n_correct": n_correct,
                "accuracy": n_correct / n_trials,
                "mean_correct_prob": mean_correct_prob,
                "clean_n_correct": clean_correct,
                "clean_accuracy": clean_correct / n_trials,
                "clean_mean_correct_prob": clean_mean_correct_prob,
            }
            grid_rows.append(row)
            print(
                f"{cell_prefix}accuracy={row['accuracy']:.6f} "
                f"mean_correct_prob={mean_correct_prob:.6g}",
                flush=True,
            )

    if args.run_dir is not None:
        run_dir = args.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = make_run_dir(args.results_dir, args.run_name)

    if args.worker_id == 0:
        visible_gpus = [
            value
            for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if value
        ]
        write_metadata(
            run_dir,
            model_name=args.model,
            args=vars(args),
            seed=args.seed,
            date=args.date or datetime.now().strftime("%Y-%m-%d"),
            gpus=visible_gpus or None,
            extra={
                "cluster_csv": str(args.cluster_csv.resolve()),
                "n_choices": n_choices,
                "prompt_template": args.prompt_template,
                "candidate_labels": list(examples[0].candidate_labels),
                "n_clusters": len(examples),
                "n_concepts": len(concepts),
                "baseline_mode": args.baseline_mode,
                "baseline_word_count": len(baseline_words),
                "baseline_words_sha256": baseline_words_hash,
                "n_layers": n_layers,
                "all_layers": injection_layers,
                "trials_per_cell": len(concepts),
                "sweep_trial_assignment": (
                    "concept_index cycles deterministically through the "
                    "lane CSV's cluster/position pairs"
                ),
            },
        )

    if args.run_dir is not None:
        grid_filename = f"workers/worker_{args.worker_id}.csv"
    else:
        grid_filename = "sweep.csv"
    write_table(run_dir, grid_filename, grid_rows)

    if args.run_dir is None:
        best = sorted(
            grid_rows,
            key=lambda row: (
                -row["accuracy"],
                -row["mean_correct_prob"],
                row["strength"],
                row["injection_layer"],
            ),
        )[0]
        (run_dir / "selection.json").write_text(
            json.dumps(best, indent=2) + "\n"
        )

    print(f"{prefix}wrote {run_dir / grid_filename}", flush=True)


if __name__ == "__main__":
    main()
