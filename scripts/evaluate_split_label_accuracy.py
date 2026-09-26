#!/usr/bin/env python3
"""Measure natural injected localization accuracy for one split and one label set.

This is the label-agnostic counterpart of ``scripts/prepare_injected_outcomes.py``.
That entry point is locked to the digit prompt (``validate_token0_9_candidate_layout``)
and only persists trials that were correct or answered ``none``, because the STE
experiments downstream consume exactly those. Here every trial is counted, so the
denominator is the full ``concepts x clusters x positions`` cross product and the
resulting accuracy is directly comparable across prompt label sets.

Everything except the prompt template is held fixed against the frozen digit run:
the same canonical ``state_vectors.pt`` payload, the same split concept list and
order, the same cluster bank, and the same injection layer and strength. The only
manipulated variable is how the ten candidate positions are labelled.

``--label_permutation shuffled`` adds the second half of that manipulation. The
registered prompts print their labels in ascending order, so the label token is
collinear with the ordinal slot and counting list entries is indistinguishable
from reading the label beside the disrupted candidate. The shuffled mode draws
one derangement of the same label set per cluster, keeping the system prompt,
the candidates, their slot order and the scored answer set identical, so the two
strategies disagree on every trial.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from introspection_core import (  # noqa: E402
    HookedModel,
    LabelAccuracyRunMetadata,
    ModelConfig,
    PromptManager,
    ShuffledLabelTokenLocalizationCsvTask,
    TokenLocalizationCsvTask,
    cluster_choice_count,
    build_label_accuracy_summary,
    evaluate_split_label_accuracy,
    label_accuracy_cluster_rows,
    label_accuracy_concept_rows,
    label_accuracy_outputs_match,
    label_accuracy_position_rows,
    load_label_accuracy_vectors,
    write_label_accuracy_outputs,
)
from introspection_core.extraction import load_concepts_from_json  # noqa: E402
from introspection_core.prompts import (  # noqa: E402
    REGISTRY,
    template_slot_labels,
    template_system_prompt,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--cluster_csv", type=Path, required=True)
    parser.add_argument(
        "--concept_vectors",
        type=Path,
        required=True,
        help="state_vectors.pt from scripts/extract_concept_vector_payload.py.",
    )
    parser.add_argument(
        "--concepts_json",
        type=Path,
        required=True,
        help="This split's concept list; slices the shared payload in its order.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--injection_layer", type=int, required=True)
    parser.add_argument("--strength", type=float, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--prompt_template", required=True)
    parser.add_argument(
        "--prompt_preamble", choices=("none", "user", "system"), default="system"
    )
    parser.add_argument("--choice_suffix", default="")
    parser.add_argument(
        "--scale_mode",
        choices=("unit", "relative_hidden_norm"),
        default="relative_hidden_norm",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress_every", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--label_permutation",
        choices=("identity", "shuffled"),
        default="identity",
        help=(
            "identity keeps the registered ascending label order; shuffled "
            "draws one derangement of the label set per cluster so the label "
            "token stops being collinear with the ordinal slot."
        ),
    )
    parser.add_argument(
        "--max_concepts", type=int, help="Smoke-test cap on the split's concepts."
    )
    parser.add_argument(
        "--max_clusters", type=int, help="Smoke-test cap on the cluster bank."
    )
    return parser.parse_args(argv)


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / f"{args.split_name}_summary.json"
    metadata = LabelAccuracyRunMetadata(
        model=args.model,
        split=args.split_name,
        prompt_template=args.prompt_template,
        cluster_csv=args.cluster_csv,
        concepts_json=args.concepts_json,
        concept_vectors=args.concept_vectors,
        injection_layer=args.injection_layer,
        strength=args.strength,
        scale_mode=args.scale_mode,
        seed=args.seed,
        prompt_preamble=args.prompt_preamble,
        choice_suffix=args.choice_suffix,
        dtype=args.dtype,
        max_concepts=args.max_concepts,
        max_clusters=args.max_clusters,
        label_permutation=args.label_permutation,
    )
    if args.prompt_template not in REGISTRY:
        raise SystemExit(f"unknown prompt template {args.prompt_template!r}")
    if not args.overwrite and label_accuracy_outputs_match(
        args.output_dir,
        split_name=args.split_name,
        expected=metadata.summary_fields(),
    ):
        print(f"reuse {summary_path}", flush=True)
        return

    # The summary is the completion marker. Remove it before regenerating the
    # detail tables so an interrupted run cannot look complete on restart.
    summary_path.unlink(missing_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    model = HookedModel(
        ModelConfig(
            name=args.model,
            device=args.device,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
    )
    if args.injection_layer >= int(model.cfg.n_layers):
        raise ValueError(
            f"Injection layer exceeds model depth {int(model.cfg.n_layers)}"
        )

    concepts, _baseline_words = load_concepts_from_json(
        args.concepts_json, max_concepts=args.max_concepts
    )
    concept_vectors = load_label_accuracy_vectors(
        args.concept_vectors,
        injection_layer=args.injection_layer,
        d_model=int(model.cfg.d_model),
        concepts=concepts,
    )

    prompt_manager = PromptManager(model.tokenizer)
    if args.label_permutation == "shuffled":
        template = REGISTRY[args.prompt_template]
        n_choices = cluster_choice_count(args.cluster_csv)
        task = ShuffledLabelTokenLocalizationCsvTask(
            path=args.cluster_csv,
            canonical_labels=template_slot_labels(template, n_choices),
            system_prompt=template_system_prompt(
                template, args.prompt_preamble, n_choices
            ),
            template_name=args.prompt_template,
            max_examples=args.max_clusters,
            preamble=args.prompt_preamble,
            choice_suffix=args.choice_suffix,
            seed=args.seed,
            name=args.prompt_template,
        )
    else:
        task = TokenLocalizationCsvTask(
            path=args.cluster_csv,
            max_examples=args.max_clusters,
            preamble=args.prompt_preamble,
            choice_suffix=args.choice_suffix,
            position_index_start=0,
            template_name=args.prompt_template,
            name=args.prompt_template,
        )
    examples = task.build_examples(prompt_manager)
    n_positions = len(examples[0].positions)
    total = len(concepts) * len(examples) * n_positions
    print(
        f"[{args.split_name}] {len(concepts)} concepts x {len(examples)} clusters "
        f"x {n_positions} positions = {total} trials",
        flush=True,
    )
    evaluation = evaluate_split_label_accuracy(
        model,
        examples=examples,
        concept_vectors=concept_vectors,
        injection_layer=args.injection_layer,
        strength=args.strength,
        scale_mode=args.scale_mode,
        batch_size=args.batch_size,
        progress_prefix=f"[{args.split_name}] ",
        progress_every=args.progress_every,
    )
    summary = build_label_accuracy_summary(
        evaluation,
        metadata=metadata,
        concept_count=len(concepts),
    )
    write_label_accuracy_outputs(
        args.output_dir,
        split_name=args.split_name,
        summary=summary,
        position_rows=label_accuracy_position_rows(evaluation),
        concept_rows=label_accuracy_concept_rows(evaluation, concepts),
        cluster_rows=label_accuracy_cluster_rows(evaluation),
    )

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
