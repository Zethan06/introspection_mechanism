#!/usr/bin/env python3
"""Classify natural injected outcomes for one split into a prepared bank.

Every downstream STE experiment (head-mask training, head patching, value and
attention transplants) consumes ``prepared_<split>_split/outcomes.csv`` plus its
matching ``concept_vectors.pt``. This entry point produces that bank directly
from a locked concept-vector payload and one frozen cluster bank, without
training any subspace.
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

from introspection_core.attention_inputs import TokenLocalizationCsvTask  # noqa: E402
from introspection_core.extraction import (  # noqa: E402
    load_concept_vector_payload,
    load_concepts_from_json,
)
from introspection_core.injected_trials import (  # noqa: E402
    infer_split_outcomes,
    initialize_distributed,
    validate_token0_9_candidate_layout,
    write_prepared_split,
)
from introspection_core.model import HookedModel, ModelConfig  # noqa: E402
from introspection_core.prompts import PromptManager  # noqa: E402


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
        help=(
            "Restrict the payload to this split's concepts, in its listed order. "
            "The splits are disjoint subsets of the canonical population, so this "
            "slices the shared payload instead of re-extracting vectors."
        ),
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--split_name",
        default="train",
        help="Bank is written to <output_dir>/prepared_<split_name>_split.",
    )
    parser.add_argument("--injection_layer", type=int, required=True)
    parser.add_argument("--strength", type=float, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--prompt_template", default="semantic_highinj_posref_gate_balanced_disrupts"
    )
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
    parser.add_argument("--progress_every", type=int, default=50)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def load_split_vectors(
    path: Path,
    *,
    injection_layer: int,
    d_model: int,
    subset: Sequence[str] | None = None,
) -> tuple[list[str], torch.Tensor]:
    """Read the canonical concept-vector payload and return unit vectors.

    When ``subset`` is given, rows are selected by concept name and returned in
    the subset's order, so the emitted bank and its ``outcomes.csv`` share one
    positional index.
    """

    if subset is None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        names = payload.get("concepts")
        if not isinstance(names, list):
            rows = payload.get("rows")
            if not isinstance(rows, list):
                raise ValueError(f"{path} has no concept names")
            names = [row.get("word", row.get("concept")) for row in rows]
        subset = [str(name) for name in names]
    concepts = [str(name) for name in subset]
    concept_vectors = load_concept_vector_payload(
        path, concepts=concepts, layer=injection_layer
    ).cpu()
    if concept_vectors.shape != (len(concepts), d_model):
        raise ValueError(
            "Concept-vector shape does not match model: "
            f"{tuple(concept_vectors.shape)} versus ({len(concepts)}, {d_model})"
        )
    return concepts, concept_vectors


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    prepared = args.output_dir / f"prepared_{args.split_name}_split"
    if (prepared / "outcomes.csv").exists() and not args.overwrite:
        raise SystemExit(f"{prepared / 'outcomes.csv'} exists; pass --overwrite")

    context = initialize_distributed(args.distributed)
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
    subset = None
    if args.concepts_json is not None:
        subset, _baseline_words = load_concepts_from_json(args.concepts_json)
    concepts, concept_vectors = load_split_vectors(
        args.concept_vectors,
        injection_layer=args.injection_layer,
        d_model=int(model.cfg.d_model),
        subset=subset,
    )
    if args.injection_layer >= int(model.cfg.n_layers):
        raise ValueError(f"Injection layer exceeds model depth {int(model.cfg.n_layers)}")

    examples = TokenLocalizationCsvTask(
        path=args.cluster_csv,
        preamble=args.prompt_preamble,
        choice_suffix=args.choice_suffix,
        position_index_start=0,
        template_name=args.prompt_template,
        name=args.prompt_template,
    ).build_examples(PromptManager(model.tokenizer))
    candidate_token_ids = validate_token0_9_candidate_layout(examples)
    base_tokens = torch.cat([example.input_ids for example in examples], dim=0)
    injection_token_positions = torch.tensor(
        [
            [int(example.injection_spans[position].start) for position in range(10)]
            for example in examples
        ],
        dtype=torch.long,
    )

    trials = infer_split_outcomes(
        model,
        concepts=concepts,
        concept_vectors=concept_vectors,
        base_tokens=base_tokens,
        injection_token_positions=injection_token_positions,
        candidate_token_ids=candidate_token_ids,
        injection_layer=args.injection_layer,
        strength=args.strength,
        scale_mode=args.scale_mode,
        batch_size=args.batch_size,
        context=context,
        progress_every=args.progress_every,
        split_name=args.split_name,
    )
    if not context.is_primary:
        return

    write_prepared_split(
        args.output_dir,
        concepts=concepts,
        concept_vectors=concept_vectors,
        trials=trials,
        injection_layer=args.injection_layer,
        split_name=args.split_name,
    )
    metadata = {
        "model": args.model,
        "cluster_csv": str(args.cluster_csv),
        "concept_vectors": str(args.concept_vectors),
        "concepts_json": None if args.concepts_json is None else str(args.concepts_json),
        "split_name": args.split_name,
        "injection_layer": args.injection_layer,
        "strength": args.strength,
        "scale_mode": args.scale_mode,
        "prompt_template": args.prompt_template,
        "prompt_preamble": args.prompt_preamble,
        "position_index_start": 0,
        "choice_suffix": args.choice_suffix,
        "seed": args.seed,
        "concept_count": len(concepts),
        "cluster_count": len(examples),
        "trial_count": len(trials),
        "correct_count": sum(1 for trial in trials if trial.correct),
        "predicted_none_count": sum(1 for trial in trials if trial.predicted_none),
        "prepared_dir": str(prepared),
    }
    (prepared / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
