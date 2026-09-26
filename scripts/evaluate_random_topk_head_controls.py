#!/usr/bin/env python3
"""Evaluate uniform random-k head controls for the STE Top-k sweep.

The Top-k curve shows what happens when the k heads the STE search selected
are patched. It cannot, on its own, distinguish that selection from the
generic damage of patching any k heads at the same depth. This script patches
random head sets of the same cardinality, drawn uniformly from the candidate
pool the STE search ranged over, and reports the same two transition rates the
Top-k curve plots.

Every draw shares one pass over the evaluation split. The expensive part of a
trial is the paired clean and injected prefix forward, which does not depend on
the mask, so it is computed once per batch and reused by every (k, draw)
cell. Only the final single-token forward is repeated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from introspection_core.cluster_split import file_sha256  # noqa: E402
from introspection_core.injected_trials import (  # noqa: E402
    initialize_distributed,
)
from introspection_core.head_mask_gate import (  # noqa: E402
    TransitionStats,
    load_head_gate_dataset,
    load_head_mask_checkpoint,
    masked_final_token_head_hooks,
    prepare_cached_head_gate_batch,
)
from introspection_core.model import HookedModel, ModelConfig  # noqa: E402
from introspection_core.random_head_controls import (  # noqa: E402
    candidate_components,
    components_to_mask,
    draw_seed,
    sample_random_components,
    summarize_draws,
)


DIRECTIONS = ("on", "off")
TRANSITIONS = {"on": "none_to_number", "off": "number_to_none"}
METRICS = (
    "n_trials",
    "source_prediction_trials",
    "converted_trials",
    "conversion_rate",
    "target_accuracy_before",
    "target_accuracy_after",
    "target_accuracy_delta",
)
# Fields the Top-k curve reads, and therefore the ones the random arm has to
# publish with an interval rather than a single number.
SUMMARY_METRICS = (
    "conversion_rate",
    "target_accuracy_before",
    "target_accuracy_after",
    "target_accuracy_delta",
)
# Settings that must agree between the two directional reference checkpoints,
# because one shared pass over the split serves both.
SHARED_CHECKPOINT_FIELDS = (
    "model",
    "injection_layer",
    "strength",
    "scale_mode",
    "prompt_template",
    "prompt_preamble",
    "gate_temperature",
    "n_heads",
)


@dataclass(frozen=True)
class Draw:
    """One random head set evaluated in both gate directions."""

    top_k: int
    draw: int
    seed: int
    components: tuple[tuple[int, int], ...]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--reference_head_mask_on",
        type=Path,
        required=True,
        help="Any trained gate-on mask; supplies the candidate pool and the "
        "injection and prompt configuration the control must reproduce.",
    )
    parser.add_argument(
        "--reference_head_mask_off", type=Path, required=True
    )
    parser.add_argument("--test_cluster_csv", type=Path, required=True)
    parser.add_argument("--test_concept_vectors", type=Path, required=True)
    parser.add_argument(
        "--dataset_split", choices=("validation", "test"), default="validation"
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--top_k",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16, 32],
        help="Cardinalities to control, matching the Top-k sweep grid.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Independent random head sets per cardinality.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--max_concepts",
        type=int,
        help="Evaluate a seeded subset of held-out concepts, matching "
        "evaluate_ste_topk_validation_sweep.py. The control must use the same "
        "population as the Top-k arm it is compared against.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress_every", type=int, default=5)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.max_concepts is not None and args.max_concepts <= 0:
        parser.error("--max_concepts must be positive")
    if any(value <= 0 for value in args.top_k):
        parser.error("--top_k values must be positive")
    if len(set(args.top_k)) != len(args.top_k):
        parser.error("--top_k values must be unique")
    return args


def build_draws(
    *,
    layers: Sequence[int],
    n_heads: int,
    top_k_values: Sequence[int],
    repeats: int,
    base_seed: int,
) -> list[Draw]:
    """Enumerate every (cardinality, draw) cell with its sampled heads."""

    pool_size = len(candidate_components(layers=layers, n_heads=n_heads))
    draws: list[Draw] = []
    for top_k in top_k_values:
        if int(top_k) > pool_size:
            raise ValueError(
                f"Top{top_k} exceeds the {pool_size}-head candidate pool"
            )
        # Draws are independent, so two of them may coincide by chance; that is
        # part of the sampling distribution and is kept. Only a pool that
        # cannot supply as many distinct head sets as there are draws makes the
        # interval meaningless.
        if math.comb(pool_size, int(top_k)) < int(repeats):
            raise ValueError(
                f"the {pool_size}-head candidate pool has fewer distinct Top"
                f"{top_k} sets than --repeats={repeats}"
            )
        for draw in range(int(repeats)):
            seed = draw_seed(base_seed=base_seed, top_k=int(top_k), draw=draw)
            components = sample_random_components(
                layers=layers, n_heads=n_heads, top_k=int(top_k), seed=seed
            )
            draws.append(
                Draw(
                    top_k=int(top_k),
                    draw=int(draw),
                    seed=seed,
                    components=tuple(components),
                )
            )
    return draws


def aggregate_draw_rows(
    rows: Sequence[dict[str, object]],
    *,
    top_k_values: Sequence[int],
    repeats: int,
) -> list[dict[str, object]]:
    """Reduce per-draw transition rows to one mean-and-interval row per k.

    Column names mirror ``summarize_ste_topk_sweep.py`` with a statistic
    suffix, so the plot reads the two arms through the same field names.
    """

    summary: list[dict[str, object]] = []
    for top_k in top_k_values:
        row: dict[str, object] = {"top_k": int(top_k), "n_draws": int(repeats)}
        for direction in DIRECTIONS:
            transition = TRANSITIONS[direction]
            selected = [
                item
                for item in rows
                if int(item["top_k"]) == int(top_k)
                and item["transition"] == transition
            ]
            if len(selected) != int(repeats):
                raise ValueError(
                    f"Top{top_k} {transition} has {len(selected)} draws, "
                    f"expected {repeats}"
                )
            for metric in SUMMARY_METRICS:
                statistics = summarize_draws(
                    [float(item[metric]) for item in selected]
                )
                row[f"{transition}_{metric}_mean"] = statistics.mean
                row[f"{transition}_{metric}_std"] = statistics.std
                row[f"{transition}_{metric}_ci_low"] = statistics.ci_low
                row[f"{transition}_{metric}_ci_high"] = statistics.ci_high
                row[f"{transition}_{metric}_min"] = statistics.minimum
                row[f"{transition}_{metric}_max"] = statistics.maximum
        summary.append(row)
    return summary


def _load_reference_checkpoints(
    args: argparse.Namespace,
) -> dict[str, dict]:
    checkpoints = {
        "on": load_head_mask_checkpoint(
            args.reference_head_mask_on.resolve(), expected_top_k=None
        ),
        "off": load_head_mask_checkpoint(
            args.reference_head_mask_off.resolve(), expected_top_k=None
        ),
    }
    for direction, checkpoint in checkpoints.items():
        if str(checkpoint["selection_direction"]) != direction:
            raise ValueError(
                f"--reference_head_mask_{direction} is a "
                f"{checkpoint['selection_direction']!r} checkpoint"
            )
        if str(checkpoint.get("model")) != args.model:
            raise ValueError(
                f"--reference_head_mask_{direction} model does not match --model"
            )
    for field in SHARED_CHECKPOINT_FIELDS:
        values = {str(checkpoint.get(field)) for checkpoint in checkpoints.values()}
        if len(values) != 1:
            raise ValueError(
                f"directional reference checkpoints disagree on {field}: {values}"
            )
    if [int(layer) for layer in checkpoints["on"]["layers"]] != [
        int(layer) for layer in checkpoints["off"]["layers"]
    ]:
        raise ValueError("directional reference checkpoints span different layers")
    if str(checkpoints["on"].get("choice_suffix", "")) != str(
        checkpoints["off"].get("choice_suffix", "")
    ):
        raise ValueError("directional reference checkpoints disagree on choice_suffix")
    return checkpoints


def _assert_evaluation_split(
    checkpoints: dict[str, dict],
    *,
    cluster_csv: Path,
    concept_vectors_file: Path,
) -> dict[str, str]:
    """Refuse to score the control on the heads' own training split."""

    hashes = {
        "test_cluster_csv": file_sha256(cluster_csv),
        "test_concept_vectors": file_sha256(concept_vectors_file),
    }
    for direction, checkpoint in checkpoints.items():
        train_hashes = checkpoint.get("input_sha256", {})
        reused = [
            name
            for name, train_key in (
                ("cluster", "train_cluster_csv"),
                ("concept_vectors", "train_concept_vectors"),
            )
            if train_hashes.get(train_key) is not None
            and train_hashes[train_key]
            == hashes[
                "test_cluster_csv"
                if name == "cluster"
                else "test_concept_vectors"
            ]
        ]
        if reused:
            raise ValueError(
                f"evaluation inputs reuse {direction} train artifacts for: "
                + ", ".join(reused)
            )
    return hashes


def sharded_batches(
    items: Sequence,
    *,
    batch_size: int,
    rank: int,
    world_size: int,
) -> list[Sequence]:
    """Shard whole batches so results are stable across worker counts."""

    batches = [
        items[start : start + batch_size]
        for start in range(0, len(items), batch_size)
    ]
    return batches[rank::world_size]


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    context = initialize_distributed(args.distributed)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"non-empty output directory exists: {output_dir}")
    if context.is_primary:
        output_dir.mkdir(parents=True, exist_ok=True)
    if context.enabled:
        torch.distributed.barrier()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    checkpoints = _load_reference_checkpoints(args)
    reference = checkpoints["on"]
    layers = [int(layer) for layer in reference["layers"]]
    n_heads = int(reference["n_heads"])

    cluster_csv = args.test_cluster_csv.resolve()
    concept_vectors = args.test_concept_vectors.resolve()
    input_hashes = _assert_evaluation_split(
        checkpoints, cluster_csv=cluster_csv, concept_vectors_file=concept_vectors
    )
    draws = build_draws(
        layers=layers,
        n_heads=n_heads,
        top_k_values=args.top_k,
        repeats=args.repeats,
        base_seed=args.seed,
    )

    model = HookedModel(
        ModelConfig(
            name=args.model,
            device=args.device,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            n_devices=1 if context.enabled else None,
        )
    )
    if int(model.cfg.n_heads) != n_heads:
        raise ValueError("reference checkpoint head count does not match the model")
    invalid = [layer for layer in layers if not 0 <= layer < int(model.cfg.n_layers)]
    if invalid:
        raise ValueError(f"candidate layers outside the model: {invalid}")
    device = model.bridge.cfg.device
    masks = {
        (item.top_k, item.draw): components_to_mask(
            item.components, layers=layers, n_heads=n_heads
        )
        .float()
        .to(device)
        for item in draws
    }

    dataset = load_head_gate_dataset(
        model,
        cluster_csv=cluster_csv,
        concept_vectors_file=concept_vectors,
        injection_layer=int(reference["injection_layer"]),
        prompt_template=str(reference["prompt_template"]),
        prompt_preamble=str(reference["prompt_preamble"]),
        choice_suffix=str(reference.get("choice_suffix", "")),
    )
    concept_indices = list(range(len(dataset.concepts)))
    if args.max_concepts is not None:
        if args.max_concepts > len(concept_indices):
            raise ValueError(
                f"--max_concepts={args.max_concepts} exceeds {len(concept_indices)}"
            )
        concept_indices = sorted(
            random.Random(args.seed).sample(concept_indices, args.max_concepts)
        )
    selected_concepts = set(concept_indices)
    evaluation_trials = tuple(
        trial
        for trial in dataset.trials
        if trial.concept_index in selected_concepts
    )
    evaluation_population = (
        "full_concept_cluster_position_grid"
        if len(concept_indices) == len(dataset.concepts)
        else "seeded_validation_concept_subset_full_cluster_position_grid"
    )
    local_batches = sharded_batches(
        evaluation_trials,
        batch_size=args.batch_size,
        rank=context.rank,
        world_size=context.world_size,
    )
    local_trial_count = sum(len(batch) for batch in local_batches)

    # One accumulator per (direction, k, draw), plus the unpatched recipient
    # baseline, which is shared by every draw within a direction.
    transitions = {
        (direction, item.top_k, item.draw): TransitionStats()
        for direction in DIRECTIONS
        for item in draws
    }
    if context.is_primary:
        print(
            f"model={args.model} split={args.dataset_split} "
            f"coordinates={len(dataset.trials)} "
            f"selected_coordinates={len(evaluation_trials)} layers={layers} "
            f"pool={len(layers) * n_heads} top_k={list(args.top_k)} "
            f"repeats={args.repeats} cells={len(draws) * len(DIRECTIONS)} "
            f"world_size={context.world_size}",
            flush=True,
        )

    for batch_index, trials in enumerate(local_batches, start=1):
        batch = prepare_cached_head_gate_batch(
            model,
            trials=trials,
            dataset=dataset,
            layers=layers,
            injection_layer=int(reference["injection_layer"]),
            strength=float(reference["strength"]),
            scale_mode=str(reference["scale_mode"]),
        )
        for direction in DIRECTIONS:
            # Gate on restores an injected decision into a clean recipient;
            # gate off restores the clean decision into an injected one.
            if direction == "on":
                baseline_logits = batch.clean_logits
                recipient_cache = batch.clean_cache
                donor_z = batch.injected_z
            else:
                baseline_logits = batch.injected_logits
                recipient_cache = batch.injected_cache
                donor_z = batch.clean_z
            for item in draws:
                patch_hooks = masked_final_token_head_hooks(
                    model,
                    layers=layers,
                    donor_z_by_layer=donor_z,
                    mask=masks[(item.top_k, item.draw)],
                )
                patched = model.incremental_last_token_candidate_stats(
                    batch.last_tokens,
                    prefix_kv_cache=recipient_cache,
                    prefix_length=batch.prefix_length,
                    candidate_token_ids=dataset.candidate_token_ids,
                    fwd_hooks=patch_hooks,
                )[0]
                transitions[(direction, item.top_k, item.draw)].update(
                    baseline_logits, patched, direction=direction
                )
        if (
            context.is_primary
            and args.progress_every > 0
            and batch_index % args.progress_every == 0
        ):
            scored = transitions[(DIRECTIONS[0], draws[0].top_k, 0)].n
            print(
                f"rank0_batch={batch_index}/{len(local_batches)} "
                f"rank0_scored={scored}/{local_trial_count}",
                flush=True,
            )

    ordered_keys = list(transitions)
    packed = torch.stack(
        [transitions[key].tensor(device) for key in ordered_keys]
    )
    if context.enabled:
        torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
    transitions = {
        key: TransitionStats.from_tensor(packed[index])
        for index, key in enumerate(ordered_keys)
    }
    for key, value in transitions.items():
        if value.n != len(evaluation_trials):
            raise RuntimeError(
                f"{args.dataset_split} grid incomplete for {key}: "
                f"{value.n} != {len(evaluation_trials)}"
            )

    draw_rows: list[dict[str, object]] = []
    for item in draws:
        for direction in DIRECTIONS:
            row = transitions[(direction, item.top_k, item.draw)].row(
                direction=direction, router="native"
            )
            draw_rows.append(
                {
                    "top_k": item.top_k,
                    "draw": item.draw,
                    "draw_seed": item.seed,
                    "selection": "random_uniform",
                    "selection_direction": direction,
                    **row,
                    "components": " ".join(
                        f"L{layer}H{head}" for layer, head in item.components
                    ),
                }
            )
    summary_rows = aggregate_draw_rows(
        draw_rows, top_k_values=args.top_k, repeats=args.repeats
    )

    if context.is_primary:
        draws_csv = output_dir / "random_topk_draws.csv"
        with draws_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(draw_rows[0]))
            writer.writeheader()
            writer.writerows(draw_rows)
        summary_csv = output_dir / "random_topk_transition_summary.csv"
        with summary_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)
        summary = {
            "model": args.model,
            "dataset_split": args.dataset_split,
            "evaluation_population": evaluation_population,
            "evaluated_concepts": len(concept_indices),
            "evaluated_coordinates": len(evaluation_trials),
            f"{args.dataset_split}_input_sha256": input_hashes,
            "control": "uniform random heads from the STE candidate pool",
            "candidate_layers": layers,
            "candidate_pool_size": len(layers) * n_heads,
            "sampling": (
                "uniform without replacement over candidate layers x heads; "
                "the layer histogram is not matched to the Top-k solution"
            ),
            "paired_directions": (
                "each draw is evaluated in both gate directions, so the two "
                "random curves share one head set per draw"
            ),
            "reference_head_masks": {
                direction: str(
                    getattr(args, f"reference_head_mask_{direction}").resolve()
                )
                for direction in DIRECTIONS
            },
            "top_k_values": [int(value) for value in args.top_k],
            "repeats": int(args.repeats),
            "base_seed": int(args.seed),
            "patch_site": "attention hook_z at final prompt token only",
            "conversion_denominator": (
                "trials whose baseline prediction is the source class"
            ),
            "target_accuracy_denominator": f"all {args.dataset_split} trials",
            "interval": "Student-t 95% interval of the mean across draws",
            "router": "native",
            "world_size": context.world_size,
            "per_rank_batch_size": args.batch_size,
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "results": summary_rows,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary_rows, indent=2), flush=True)
        print(f"wrote {draws_csv} and {summary_csv}", flush=True)
    if context.enabled:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
