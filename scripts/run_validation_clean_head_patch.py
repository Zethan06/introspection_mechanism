#!/usr/bin/env python3
"""Patch paired clean head outputs into injected validation runs.

Only the final prompt token is recomputed.  The injected prefix KV cache is
built once per batch and reused for the natural injected baseline, the target
head intervention, and every layer-matched random-head control.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from introspection_core.attention_inputs import TokenLocalizationCsvTask  # noqa: E402
from introspection_core.clean_head_patch import (  # noqa: E402
    CleanHeadPatchCondition,
    CleanHeadPatchAccumulator,
    build_patch_conditions,
    concept_bootstrap_interval,
    format_accuracy_change_sentence,
    format_head_components,
    parse_head_components,
    resolve_candidate_layout,
)
from introspection_core.head_output_patch import (  # noqa: E402
    capture_final_token_heads,
    patched_final_token_head_group_logits,
)
from introspection_core.injection import make_injection_hook  # noqa: E402
from introspection_core.model import HookedModel, ModelConfig  # noqa: E402
from introspection_core.prompts import PromptManager  # noqa: E402
from introspection_core.results import write_table  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--cluster_csv", type=Path, required=True)
    parser.add_argument("--concept_vectors", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--target_heads",
        help="Clean-patched causal target, e.g. L17H24 or L24H29,L24H31.",
    )
    selection.add_argument(
        "--sweep_layer",
        type=int,
        help="Patch every attention head in this layer independently.",
    )
    selection.add_argument(
        "--sweep_model",
        action="store_true",
        help="Patch every attention head in every model layer independently.",
    )
    parser.add_argument(
        "--head_shard_index",
        type=int,
        default=0,
        help="Zero-based modulo shard used with --sweep_layer or --sweep_model.",
    )
    parser.add_argument(
        "--head_shard_count",
        type=int,
        default=1,
        help="Number of modulo shards used with --sweep_layer or --sweep_model.",
    )
    parser.add_argument("--injection_layer", type=int, required=True)
    parser.add_argument("--strength", type=float, required=True)
    parser.add_argument("--control_count", type=int, default=32)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--reference_batch_size", type=int, default=32)
    parser.add_argument("--max_concepts", type=int)
    parser.add_argument("--max_clusters", type=int)
    parser.add_argument("--max_trials", type=int)
    parser.add_argument(
        "--candidate_only",
        action="store_true",
        help="Score only answer candidates instead of the full vocabulary.",
    )
    parser.add_argument(
        "--prompt_template",
        default="semantic_highinj_posref_gate_balanced_disrupts",
    )
    parser.add_argument(
        "--prompt_preamble", choices=("none", "user", "system"), default="system"
    )
    parser.add_argument("--position_index_start", type=int, choices=(0,), default=0)
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
    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    for name in ("batch_size", "reference_batch_size", "control_count", "bootstrap_samples"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.head_shard_count <= 0:
        parser.error("--head_shard_count must be positive")
    if not 0 <= args.head_shard_index < args.head_shard_count:
        parser.error("--head_shard_index must be in [0, --head_shard_count)")
    if args.sweep_layer is None and not args.sweep_model and (
        args.head_shard_index != 0 or args.head_shard_count != 1
    ):
        parser.error("head sharding is only valid with --sweep_layer or --sweep_model")
    if args.injection_layer < 0 or not math.isfinite(args.strength) or args.strength <= 0:
        parser.error("--injection_layer must be non-negative and --strength positive")
    for name in ("max_concepts", "max_clusters", "max_trials"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name} must be positive")
    return args


def build_sweep_specs(
    *,
    n_layers: int,
    n_heads: int,
    sweep_layer: int | None,
    sweep_model: bool,
    shard_index: int,
    shard_count: int,
) -> list[CleanHeadPatchCondition]:
    """Build a deterministic modulo shard over layer/head components."""
    layers = range(n_layers) if sweep_model else (sweep_layer,)
    components = [
        (int(layer), head)
        for layer in layers
        for head in range(n_heads)
    ]
    selected = [
        component
        for flat_index, component in enumerate(components)
        if flat_index % shard_count == shard_index
    ]
    if not selected:
        raise ValueError("head shard is empty")
    kind = "model_head" if sweep_model else "layer_head"
    return [
        CleanHeadPatchCondition(
            name=f"head_L{layer}H{head}",
            kind=kind,
            components=((layer, head),),
        )
        for layer, head in selected
    ]


def _load_vectors(
    path: Path, *, injection_layer: int, max_concepts: int | None
) -> tuple[list[str], torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    vectors = payload.get("unit_vectors", payload.get("vectors"))
    if not isinstance(vectors, torch.Tensor) or vectors.dim() != 2:
        raise ValueError(f"{path} must contain a rank-2 unit_vectors or vectors tensor")
    if int(payload.get("layer", injection_layer)) != injection_layer:
        raise ValueError("concept-vector layer does not match --injection_layer")
    vectors = F.normalize(vectors.float(), dim=-1)
    if max_concepts is not None:
        vectors = vectors[:max_concepts]
    names = payload.get("concepts")
    if not isinstance(names, list) or len(names) < vectors.shape[0]:
        names = [str(index) for index in range(vectors.shape[0])]
    else:
        names = [str(name) for name in names[: vectors.shape[0]]]
    return names, vectors.cpu()


def _collect_clean_references(
    model: HookedModel,
    *,
    base_tokens: torch.Tensor,
    candidate_token_ids: list[int],
    layers: list[int],
    batch_size: int,
    candidate_only: bool = False,
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    captured = {layer: [] for layer in layers}
    logits: list[torch.Tensor] = []
    for start in range(0, base_tokens.shape[0], batch_size):
        tokens = base_tokens[start : start + batch_size]
        prefix_length = int(tokens.shape[1]) - 1
        prefix_cache = model.build_prefix_kv_cache(tokens)
        batch_z, batch_logits = capture_final_token_heads(
            model,
            last_tokens=tokens[:, -1:],
            prefix_kv_cache=prefix_cache,
            prefix_length=prefix_length,
            candidate_token_ids=candidate_token_ids,
            layers=layers,
            candidate_only=candidate_only,
        )
        for layer in layers:
            captured[layer].append(batch_z[layer])
        logits.append(batch_logits)
        del prefix_cache
    return (
        {layer: torch.cat(parts, dim=0) for layer, parts in captured.items()},
        torch.cat(logits, dim=0),
    )


def _write_evidence(
    output_dir: Path,
    *,
    target_row: Mapping[str, Any],
    controls: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    control_drop = float(summary["matched_control_accuracy_drop_median"])
    low, high = target_row["accuracy_drop_concept_bootstrap_95_ci"]
    target_change_pp = -float(target_row["accuracy_drop_pp"])
    control_change_pp = -100.0 * control_drop
    if abs(target_change_pp) < 0.005:
        target_change_pp = 0.0
    if abs(control_change_pp) < 0.005:
        control_change_pp = 0.0
    lines = [
        "# Validation clean-head patch evidence",
        "",
        format_accuracy_change_sentence(
            heads=str(target_row["heads"]),
            accuracy_drop_pp=float(target_row["accuracy_drop_pp"]),
            ci_low=float(low),
            ci_high=float(high),
        ),
        "",
        "| condition | exact-number accuracy | change from injected |",
        "|---|---:|---:|",
        f"| Clean (no injection) | {100 * target_row['clean_exact_number_accuracy']:.2f}% | — |",
        f"| Injected | {100 * target_row['injected_exact_number_accuracy']:.2f}% | — |",
        (
            "| Target clean patch | "
            f"{100 * target_row['patched_exact_number_accuracy']:.2f}% | "
            f"{target_change_pp:+.2f} pp |"
        ),
        (
            f"| Matched-control median ({len(controls)} groups) | "
            f"{100 * (target_row['injected_exact_number_accuracy'] - control_drop):.2f}% | "
            f"{control_change_pp:+.2f} pp |"
        ),
        "",
        (
            "The target is compared with random non-target head groups having the "
            "same number of heads in each layer. The empirical one-sided p-value is "
            f"{summary['empirical_control_p_value']:.4g}."
        ),
        "",
        (
            "Primary metric: argmax accuracy on the correct number among all "
            "number labels plus `none`."
        ),
    ]
    (output_dir / "evidence.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"non-empty output directory exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    target = parse_head_components(args.target_heads) if args.target_heads else []
    model = HookedModel(
        ModelConfig(
            name=args.model,
            device=args.device,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
    )
    n_layers, n_heads = int(model.cfg.n_layers), int(model.cfg.n_heads)
    invalid = [pair for pair in target if not 0 <= pair[0] < n_layers or not 0 <= pair[1] < n_heads]
    if invalid:
        raise ValueError(f"target heads outside model dimensions: {invalid}")
    if args.injection_layer >= n_layers:
        raise ValueError(f"injection layer exceeds model depth {n_layers}")

    concept_names, concept_vectors = _load_vectors(
        args.concept_vectors,
        injection_layer=args.injection_layer,
        max_concepts=args.max_concepts,
    )
    if concept_vectors.shape[1] != int(model.cfg.d_model):
        raise ValueError("concept vectors do not match the model hidden dimension")
    examples = TokenLocalizationCsvTask(
        path=args.cluster_csv,
        max_examples=args.max_clusters,
        preamble=args.prompt_preamble,
        choice_suffix=args.choice_suffix,
        position_index_start=args.position_index_start,
        template_name=args.prompt_template,
        name=args.prompt_template,
    ).build_examples(PromptManager(model.tokenizer))
    position_labels, candidate_token_ids, clean_index = resolve_candidate_layout(examples)
    base_tokens = torch.cat([example.input_ids for example in examples], dim=0)
    injection_token_positions = torch.tensor(
        [
            [int(example.injection_spans[label].start) for label in position_labels]
            for example in examples
        ],
        dtype=torch.long,
    )
    prefix_length = int(base_tokens.shape[1]) - 1
    if bool(injection_token_positions.ge(prefix_length).any()):
        raise ValueError("injection positions must all precede the final prompt token")

    if args.sweep_layer is not None or args.sweep_model:
        if args.sweep_layer is not None and not 0 <= args.sweep_layer < n_layers:
            raise ValueError(f"sweep layer exceeds model depth {n_layers}")
        specs = build_sweep_specs(
            n_layers=n_layers,
            n_heads=n_heads,
            sweep_layer=args.sweep_layer,
            sweep_model=args.sweep_model,
            shard_index=args.head_shard_index,
            shard_count=args.head_shard_count,
        )
    else:
        specs = build_patch_conditions(
            target,
            n_heads=n_heads,
            control_count=args.control_count,
            seed=args.seed,
        )
    layers = sorted({layer for spec in specs for layer, _head in spec.components})
    clean_z, clean_logits_by_cluster = _collect_clean_references(
        model,
        base_tokens=base_tokens,
        candidate_token_ids=candidate_token_ids,
        layers=layers,
        batch_size=args.reference_batch_size,
        candidate_only=args.candidate_only,
    )
    accumulators = {
        spec.name: CleanHeadPatchAccumulator(len(concept_names))
        for spec in specs
    }

    coordinates = itertools.product(
        range(len(concept_names)), range(len(examples)), range(len(position_labels))
    )
    if args.max_trials is not None:
        coordinates = itertools.islice(coordinates, args.max_trials)
    trial_count = 0
    batch_number = 0
    while True:
        batch = list(itertools.islice(coordinates, args.batch_size))
        if not batch:
            break
        batch_number += 1
        trial_count += len(batch)
        concept_indices = torch.tensor([row[0] for row in batch], dtype=torch.long)
        cluster_indices = torch.tensor([row[1] for row in batch], dtype=torch.long)
        target_indices = torch.tensor([row[2] for row in batch], dtype=torch.long)
        tokens = base_tokens.index_select(0, cluster_indices).to(model.bridge.cfg.device)
        vector_batch = concept_vectors.index_select(0, concept_indices)
        token_positions = injection_token_positions[cluster_indices, target_indices]
        injection_hook = make_injection_hook(
            model,
            positions=[(int(position), int(position) + 1) for position in token_positions],
            vector=vector_batch,
            layer=args.injection_layer,
            strength=args.strength,
            scale=args.scale_mode,
        )
        injected_cache = model.build_prefix_kv_cache(tokens, fwd_hooks=[injection_hook])
        last_tokens = tokens[:, -1:]
        if args.candidate_only:
            injected_logits = model.incremental_last_token_candidate_logits_only(
                last_tokens,
                prefix_kv_cache=injected_cache,
                prefix_length=prefix_length,
                candidate_token_ids=candidate_token_ids,
            )
        else:
            injected_logits, _ = model.incremental_last_token_candidate_stats(
                last_tokens,
                prefix_kv_cache=injected_cache,
                prefix_length=prefix_length,
                candidate_token_ids=candidate_token_ids,
            )
        clean_logits = clean_logits_by_cluster.index_select(0, cluster_indices)
        source_z = {
            layer: values.index_select(0, cluster_indices) for layer, values in clean_z.items()
        }
        for spec in specs:
            patched_logits = patched_final_token_head_group_logits(
                model,
                last_tokens=last_tokens,
                prefix_kv_cache=injected_cache,
                prefix_length=prefix_length,
                candidate_token_ids=candidate_token_ids,
                components=spec.components,
                source_z_by_layer=source_z,
                candidate_only=args.candidate_only,
            )
            accumulators[spec.name].update(
                clean_logits=clean_logits,
                injected_logits=injected_logits,
                patched_logits=patched_logits,
                target_indices=target_indices,
                clean_index=clean_index,
                concept_indices=concept_indices,
            )
        del injected_cache
        if args.progress_every > 0 and batch_number % args.progress_every == 0:
            print(f"batches={batch_number} trials={trial_count}", flush=True)

    rows: list[dict[str, Any]] = []
    per_concept_rows: list[dict[str, Any]] = []
    for condition_index, spec in enumerate(specs):
        accumulator = accumulators[spec.name]
        row = {
            "condition": spec.name,
            "condition_type": spec.kind,
            "heads": format_head_components(spec.components),
            "n_heads_patched": len(spec.components),
            **accumulator.row(),
        }
        interval = concept_bootstrap_interval(
            accumulator.concept_accuracy_drops(),
            samples=args.bootstrap_samples,
            seed=args.seed + condition_index,
        )
        row["accuracy_drop_ci_low"] = interval[0]
        row["accuracy_drop_ci_high"] = interval[1]
        row["accuracy_drop_concept_bootstrap_95_ci"] = list(interval)
        rows.append(row)
        counts = accumulator.concept_trial_count
        drops = accumulator.concept_accuracy_drop_sum
        for concept_index in counts.gt(0).nonzero(as_tuple=False).flatten().tolist():
            per_concept_rows.append(
                {
                    "condition": spec.name,
                    "condition_type": spec.kind,
                    "heads": row["heads"],
                    "concept_index": concept_index,
                    "concept": concept_names[concept_index],
                    "n_trials": int(counts[concept_index]),
                    "accuracy_drop": float(drops[concept_index] / counts[concept_index]),
                }
            )
    write_table(args.output_dir, "condition_effects.csv", rows)
    write_table(args.output_dir, "per_concept_effects.csv", per_concept_rows)

    if args.sweep_layer is not None or args.sweep_model:
        components_by_name = {
            spec.name: spec.components[0]
            for spec in specs
        }
        ranked = sorted(
            rows,
            key=lambda row: (-float(row["accuracy_drop"]), str(row["heads"])),
        )
        for rank, row in enumerate(ranked, 1):
            layer, head = components_by_name[str(row["condition"])]
            row["layer"] = layer
            row["head"] = head
            row["rank_by_accuracy_drop"] = rank
            row["ci_excludes_zero"] = bool(
                float(row["accuracy_drop_ci_low"]) > 0
                or float(row["accuracy_drop_ci_high"]) < 0
            )
        write_table(args.output_dir, "condition_effects.csv", rows)
        baseline = rows[0]
        top = ranked[0]
        experiment = (
            "validation_clean_head_model_sweep_shard"
            if args.sweep_model
            else "validation_clean_head_layer_sweep_shard"
        )
        summary = {
            "schema_version": 2,
            "experiment": experiment,
            "sweep_layer": args.sweep_layer,
            "head_shard_index": args.head_shard_index,
            "head_shard_count": args.head_shard_count,
            "component_count": len(specs),
            "components": [format_head_components(spec.components) for spec in specs],
            "n_trials": trial_count,
            "concept_count": len(concept_names),
            "cluster_count": len(examples),
            "position_labels": position_labels,
            "clean_exact_number_accuracy": baseline["clean_exact_number_accuracy"],
            "injected_exact_number_accuracy": baseline[
                "injected_exact_number_accuracy"
            ],
            "top_head": top["heads"],
            "top_head_accuracy_drop_pp": top["accuracy_drop_pp"],
            "patch_site": "attention hook_z at the final prompt token only",
            "execution": (
                "clean final-token donors are collected once per cluster; each injected "
                "prefix KV cache is built once per batch and reused by every head in this shard"
            ),
        }
        metadata = {
            **summary,
            "model": args.model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "candidate_labels": [
                *[str(label) for label in position_labels],
                examples[0].clean_target_label,
            ],
            "conditional_probability_space": "number labels plus clean target label",
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        (args.output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2), flush=True)
        return

    target_row = next(row for row in rows if row["condition"] == "target_group")
    controls = [row for row in rows if row["condition_type"] == "matched_control"]
    control_drops = torch.tensor([row["accuracy_drop"] for row in controls], dtype=torch.float64)
    control_quantiles = torch.quantile(
        control_drops,
        torch.tensor([0.05, 0.5, 0.95], dtype=control_drops.dtype),
    )
    exceedances = int(control_drops.ge(float(target_row["accuracy_drop"])).sum())
    summary = {
        "schema_version": 1,
        "target_heads": target_row["heads"],
        "n_trials": trial_count,
        "concept_count": len(concept_names),
        "cluster_count": len(examples),
        "position_labels": position_labels,
        "clean_exact_number_accuracy": target_row["clean_exact_number_accuracy"],
        "injected_exact_number_accuracy": target_row["injected_exact_number_accuracy"],
        "target_patched_exact_number_accuracy": target_row["patched_exact_number_accuracy"],
        "target_accuracy_drop": target_row["accuracy_drop"],
        "target_accuracy_drop_pp": target_row["accuracy_drop_pp"],
        "target_accuracy_drop_concept_bootstrap_95_ci": target_row[
            "accuracy_drop_concept_bootstrap_95_ci"
        ],
        "fraction_of_injection_gain_removed": target_row[
            "fraction_of_injection_gain_removed"
        ],
        "matched_control_count": len(controls),
        "matched_control_accuracy_drop_p05": float(control_quantiles[0]),
        "matched_control_accuracy_drop_median": float(control_quantiles[1]),
        "matched_control_accuracy_drop_p95": float(control_quantiles[2]),
        "target_minus_control_median_accuracy_drop": (
            float(target_row["accuracy_drop"]) - float(control_quantiles[1])
        ),
        "empirical_control_p_value": (1 + exceedances) / (1 + len(controls)),
        "patch_site": "attention hook_z at the final prompt token only",
        "execution": (
            "clean final-token donors are collected once per cluster; each injected "
            "prefix KV cache is built once per batch and reused by all conditions"
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        **summary,
        "model": args.model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "candidate_labels": [
            *[str(label) for label in position_labels],
            examples[0].clean_target_label,
        ],
        "conditional_probability_space": "number labels plus clean target label",
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    _write_evidence(args.output_dir, target_row=target_row, controls=controls, summary=summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
