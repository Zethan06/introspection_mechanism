#!/usr/bin/env python3
"""Redirect the router heads' attention while a concept is injected.

A concept is injected at candidate ``i``; in the selected router heads, the
post-softmax attention row of the final prompt position is set to one-hot on
the successor token ``t_j`` of candidate ``j`` (the token right after it), for
every requested ``j``. Value vectors and all other computation are unchanged.
The injected run without redirection is recorded as the baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from introspection_core import (
    HookedModel,
    ModelConfig,
    PromptManager,
    REGISTRY,
    ShuffledLabelTokenLocalizationCsvTask,
    TokenLocalizationCsvTask,
    cluster_choice_count,
    evaluate_attention_routing,
    load_concept_vector_payload,
    load_concepts_from_json,
    template_slot_labels,
    template_system_prompt,
)
from introspection_core.results import make_run_dir, write_metadata, write_table


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="HF id or local checkpoint")
    parser.add_argument("--cluster_file", type=Path, required=True)
    parser.add_argument("--concepts_json", type=Path, required=True)
    parser.add_argument(
        "--concept_vectors_file",
        type=Path,
        required=True,
        help="Concept-vector payload of the evaluated split (Stage 03).",
    )
    parser.add_argument("--max_clusters", type=int)
    parser.add_argument("--max_concepts", type=int)
    parser.add_argument("--injection_layer", type=int, required=True)
    parser.add_argument(
        "--injection_position", type=int, required=True, help="Injected candidate i."
    )
    parser.add_argument("--strength", type=float, required=True)
    parser.add_argument(
        "--scale_mode",
        choices=["unit", "relative_hidden_norm"],
        default="relative_hidden_norm",
    )
    parser.add_argument(
        "--attention_layer",
        type=int,
        required=True,
        help="Zero-based layer of the router heads.",
    )
    parser.add_argument(
        "--heads",
        type=int,
        nargs="+",
        required=True,
        help="Router heads, redirected jointly.",
    )
    parser.add_argument(
        "--attention_positions",
        type=int,
        nargs="+",
        help="Subset of redirection targets j; defaults to all ten.",
    )
    parser.add_argument(
        "--prompt_preamble", choices=["none", "user", "system"], default="system"
    )
    parser.add_argument(
        "--prompt_template",
        required=True,
        help=(
            "Evaluation prompt or one of its letters_a_j / numwords_one_ten "
            "variants; this selects the label set."
        ),
    )
    parser.add_argument(
        "--label_permutation",
        choices=("identity", "shuffled"),
        default="identity",
        help=(
            "identity keeps the ascending label order; shuffled draws one "
            "derangement of the same label set per cluster."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--run_name", default="attention_routing")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if args.max_clusters is not None and args.max_clusters <= 0:
        parser.error("--max_clusters must be positive")
    if args.max_concepts is not None and args.max_concepts <= 0:
        parser.error("--max_concepts must be positive")
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")
    if args.prompt_template not in REGISTRY:
        parser.error(f"unknown --prompt_template {args.prompt_template!r}")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    cluster_file = _resolve(args.cluster_file)
    concepts_json = _resolve(args.concepts_json)

    print(f"loading model {args.model}", flush=True)
    model = HookedModel(
        ModelConfig(
            name=args.model,
            device=args.device,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
    )
    n_layers = int(model.cfg.n_layers)
    n_heads = int(model.cfg.n_heads)
    if not 0 <= args.injection_layer < n_layers:
        raise ValueError(
            f"injection layer {args.injection_layer} outside 0..{n_layers - 1}"
        )
    if not 0 <= args.attention_layer < n_layers:
        raise ValueError(
            f"attention layer {args.attention_layer} outside 0..{n_layers - 1}"
        )
    bad_heads = [head for head in args.heads if not 0 <= head < n_heads]
    if bad_heads:
        raise ValueError(f"head indices outside 0..{n_heads - 1}: {bad_heads}")
    if len(set(args.heads)) != len(args.heads):
        raise ValueError("router --heads must be unique")

    prompt_manager = PromptManager(model.tokenizer)
    template = REGISTRY[args.prompt_template]
    if args.label_permutation == "shuffled":
        # Only the label permutation moves: the system prompt, candidate list,
        # slot order and scored answer set all come from the same template.
        n_choices = cluster_choice_count(cluster_file)
        task = ShuffledLabelTokenLocalizationCsvTask(
            path=cluster_file,
            canonical_labels=template_slot_labels(template, n_choices),
            system_prompt=template_system_prompt(
                template, args.prompt_preamble, n_choices
            ),
            template_name=args.prompt_template,
            max_examples=args.max_clusters,
            preamble=args.prompt_preamble,
            seed=args.seed,
            name=args.prompt_template,
        )
    else:
        task = TokenLocalizationCsvTask(
            path=cluster_file,
            max_examples=args.max_clusters,
            preamble=args.prompt_preamble,
            template_name=args.prompt_template,
            name=args.prompt_template,
        )
    examples = task.build_examples(prompt_manager)
    slot_labels = tuple(
        examples[0].expected_candidate_by_position[position]
        for position in examples[0].positions
    )
    print(
        f"label set: template={args.prompt_template} "
        f"permutation={args.label_permutation} "
        f"cluster0 slots={list(slot_labels)}",
        flush=True,
    )
    concepts, _baseline_words = load_concepts_from_json(
        concepts_json,
        max_concepts=args.max_concepts,
    )
    vector_path = _resolve(args.concept_vectors_file)
    vectors = load_concept_vector_payload(
        vector_path,
        concepts=concepts,
        layer=args.injection_layer,
    )

    positions = list(examples[0].positions)
    if args.injection_position not in positions:
        raise ValueError(
            f"injection position {args.injection_position} is not in prompt "
            f"positions {positions}"
        )
    route_positions = args.attention_positions or positions
    print(
        f"running {len(concepts)} concepts x {len(examples)} prompts x "
        f"1 injection position ({args.injection_position}) x "
        f"{len(route_positions)} redirection targets; "
        f"L{args.attention_layer}H{args.heads}",
        flush=True,
    )
    evaluation = evaluate_attention_routing(
        model,
        examples=examples,
        concept_names=concepts,
        unit_vectors=vectors,
        injection_layer=args.injection_layer,
        strength=args.strength,
        scale_mode=args.scale_mode,
        attention_layer=args.attention_layer,
        heads=args.heads,
        batch_size=args.batch_size,
        injection_position=args.injection_position,
        attention_positions=args.attention_positions,
    )
    evaluation.summary["label_set"] = {
        "prompt_template": args.prompt_template,
        "label_permutation": args.label_permutation,
        "slot_labels_cluster0": list(slot_labels),
        "scored_candidates": list(examples[0].candidate_token_ids),
    }

    run_dir = make_run_dir(args.results_dir, args.run_name)
    write_table(run_dir, "sweep_grid.csv", evaluation.sweep_grid)
    write_table(run_dir, "natural_injected.csv", evaluation.natural_injected)
    write_table(run_dir, "clean_controls.csv", evaluation.clean_controls)
    (run_dir / "summary.json").write_text(
        json.dumps(evaluation.summary, indent=2), encoding="utf-8"
    )
    write_metadata(
        run_dir,
        model_name=args.model,
        args=vars(args),
        seed=args.seed,
        extra={
            "concept_vector_source": str(args.concept_vectors_file),
            "attention_intervention": (
                "post-softmax one-hot rewrite of the router heads' final-query "
                "row onto the successor token of candidate j"
            ),
            "position_semantics": "i=injection position; j=redirection target",
            "summary": evaluation.summary,
        },
    )

    print(json.dumps(evaluation.summary, indent=2), flush=True)
    print(f"wrote results to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
