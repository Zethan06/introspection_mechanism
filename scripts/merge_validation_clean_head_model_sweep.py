#!/usr/bin/env python3
"""Merge disjoint whole-model head shards after validating full coverage."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", nargs="+", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args(argv)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _experiment_args(meta: dict) -> dict:
    """Ignore only settings that identify a shard or its execution location."""
    ignored = {"output_dir", "head_shard_index", "device", "progress_every", "overwrite"}
    return {key: value for key, value in meta["args"].items() if key not in ignored}


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"non-empty output directory: {args.output_dir}")

    summaries = []
    metadata = []
    effects: list[dict[str, str]] = []
    per_concept: list[dict[str, str]] = []
    for shard in args.shards:
        summary = json.loads((shard / "summary.json").read_text(encoding="utf-8"))
        meta = json.loads((shard / "metadata.json").read_text(encoding="utf-8"))
        if summary.get("experiment") != "validation_clean_head_model_sweep_shard":
            raise ValueError(f"not a whole-model head shard: {shard}")
        shard_effects = _read_csv(shard / "condition_effects.csv")
        shard_concepts = _read_csv(shard / "per_concept_effects.csv")
        if len(shard_effects) != int(summary["component_count"]):
            raise ValueError(f"component-count mismatch: {shard}")
        if len(shard_concepts) != len(shard_effects) * int(summary["concept_count"]):
            raise ValueError(f"per-concept coverage mismatch: {shard}")
        head_trials = {row["heads"]: int(row["n_trials"]) for row in shard_effects}
        expected_pairs = {
            (head, concept_index)
            for head in head_trials
            for concept_index in range(int(summary["concept_count"]))
        }
        observed_pairs = [
            (row["heads"], int(row["concept_index"])) for row in shard_concepts
        ]
        if len(observed_pairs) != len(expected_pairs) or set(observed_pairs) != expected_pairs:
            raise ValueError(f"per-concept coverage mismatch: {shard}")
        for row in shard_concepts:
            if int(row["n_trials"]) <= 0 or int(row["n_trials"]) > head_trials[row["heads"]]:
                raise ValueError(f"invalid per-concept trial count: {shard}")
        summaries.append(summary)
        metadata.append(meta)
        effects.extend(shard_effects)
        per_concept.extend(shard_concepts)

    first, first_meta = summaries[0], metadata[0]
    shard_count = int(first["head_shard_count"])
    indices = [int(summary["head_shard_index"]) for summary in summaries]
    if len(summaries) != shard_count or sorted(indices) != list(range(shard_count)):
        raise ValueError(f"shard coverage mismatch: {indices} of {shard_count}")
    for summary, meta in zip(summaries[1:], metadata[1:]):
        for field in (
            "head_shard_count", "n_trials", "concept_count", "cluster_count",
            "position_labels", "clean_exact_number_accuracy",
            "injected_exact_number_accuracy", "patch_site",
        ):
            if summary[field] != first[field]:
                raise ValueError(f"inconsistent shard {field}")
        for field in ("model", "n_layers", "n_heads", "candidate_labels"):
            if meta[field] != first_meta[field]:
                raise ValueError(f"inconsistent shard {field}")
        if _experiment_args(meta) != _experiment_args(first_meta):
            raise ValueError("inconsistent shard experiment arguments")

    n_layers, n_heads = int(first_meta["n_layers"]), int(first_meta["n_heads"])
    expected = {(layer, head) for layer in range(n_layers) for head in range(n_heads)}
    observed = [(int(row["layer"]), int(row["head"])) for row in effects]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("whole-model head coverage is incomplete or duplicated")
    for row in effects:
        layer, head = int(row["layer"]), int(row["head"])
        if row["heads"] != f"L{layer}H{head}" or int(row["n_trials"]) != int(first["n_trials"]):
            raise ValueError(f"invalid head or trial count: {row['heads']}")

    ranked = sorted(
        effects,
        key=lambda row: (-float(row["accuracy_drop"]), int(row["layer"]), int(row["head"])),
    )
    for rank, row in enumerate(ranked, 1):
        row["rank_by_accuracy_drop"] = str(rank)
    effects.sort(key=lambda row: (int(row["layer"]), int(row["head"])))
    per_concept.sort(
        key=lambda row: (
            int(row["heads"].split("H", 1)[0][1:]),
            int(row["heads"].split("H", 1)[1]),
            int(row["concept_index"]),
        )
    )
    top = ranked[0]
    summary = {
        "schema_version": 2,
        "experiment": "validation_clean_head_model_sweep",
        "model": first_meta["model"],
        "n_layers": n_layers,
        "n_heads": n_heads,
        "component_count": len(effects),
        "shard_count": shard_count,
        "n_trials_per_head": int(first["n_trials"]),
        "concept_count": int(first["concept_count"]),
        "cluster_count": int(first["cluster_count"]),
        "position_labels": first["position_labels"],
        "clean_exact_number_accuracy": float(first["clean_exact_number_accuracy"]),
        "injected_exact_number_accuracy": float(first["injected_exact_number_accuracy"]),
        "top_head": top["heads"],
        "top_head_accuracy_drop_pp": float(top["accuracy_drop_pp"]),
        "patch_site": first["patch_site"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "condition_effects.csv", effects)
    _write_csv(args.output_dir / "per_concept_effects.csv", per_concept)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    provenance = {
        **summary,
        "experiment_args": _experiment_args(first_meta),
        "shards": [
            {"index": int(shard_summary["head_shard_index"]), "path": str(shard.resolve())}
            for shard, shard_summary in zip(args.shards, summaries)
        ],
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
