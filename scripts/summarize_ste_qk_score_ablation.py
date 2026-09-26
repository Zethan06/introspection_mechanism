#!/usr/bin/env python3
"""Aggregate complete STE QK score ablations over frozen prompt clusters."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


GROUPS = ("validation100", "bottom100")
CONDITIONS = ("native", "without_query", "without_key")
COUNTS = ("n_trials", "number_count", "exact_count", "none_count",
          "native_number_to_none", "native_none_to_number")


def read_cluster(path: Path, cluster: int, reference: dict | None = None) -> list[dict]:
    """Require a complete cluster and its six balanced outcome rows."""
    marker = path / "manifest.json"
    if not marker.is_file() or not (path / "predictions.pt").is_file():
        raise FileNotFoundError(f"incomplete cluster: {path}")
    manifest = json.loads(marker.read_text())
    if manifest["cluster"] != cluster or manifest["trial_end"] != 2000:
        raise ValueError(f"unexpected manifest at {path}")
    if reference is not None:
        for key in ('source', 'selection', 'model', 'batch_size', 'selected_heads', 'conditions'):
            if manifest[key] != reference[key]:
                raise ValueError(f"incompatible {key} at {path}")
    with (path / "summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if {(r["group"], r["condition"]) for r in rows} != {
        (group, condition) for group in GROUPS for condition in CONDITIONS
    } or len(rows) != 6:
        raise ValueError(f"missing or duplicate outcome rows at {path}")
    if any(int(r["n_trials"]) != 1000 for r in rows):
        raise ValueError(f"wrong trial count at {path}")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--expected-clusters", type=int, default=30)
    parser.add_argument("--models", nargs="+",
                        help="Model folders under --results_dir (default: every folder with cluster_00)")
    parser.add_argument('--baseline-csv', nargs=2, action='append', default=[], metavar=('MODEL', 'CSV'),
                        help='Optional historical native counts; repeat for each model to audit')
    args = parser.parse_args()
    if args.expected_clusters < 1:
        parser.error("expected-clusters must be positive")
    models = args.models or sorted(
        path.name for path in args.results_dir.iterdir()
        if (path / "cluster_00/manifest.json").is_file()
    )
    if not models:
        parser.error(f"no model folders with cluster_00/manifest.json under {args.results_dir}")
    baselines = {}
    for model, path in args.baseline_csv:
        if model not in models or model in baselines:
            parser.error('baseline model must be known and supplied only once')
        with Path(path).open(newline='') as handle:
            native_rows = [row for row in csv.DictReader(handle) if row['condition'] == 'native']
        saved = {(int(row['cluster_index']), row['group']): row for row in native_rows}
        expected = {(c, g) for c in range(args.expected_clusters) for g in GROUPS}
        if len(saved) != len(native_rows) or not expected.issubset(saved):
            raise ValueError(f'missing or duplicate native baseline rows: {path}')
        for key in expected:
            for field in ('number_count', 'exact_count', 'n_trials'):
                int(saved[key][field])
        baselines[model] = saved
    summary, cluster_effects, baseline_audit = [], [], []
    for model in models:
        totals = defaultdict(lambda: defaultdict(int))
        reference = json.loads((args.results_dir / model / 'cluster_00/manifest.json').read_text())
        for cluster in range(args.expected_clusters):
            rows = read_cluster(args.results_dir / model / f"cluster_{cluster:02d}", cluster, reference)
            native = {r["group"]: r for r in rows if r["condition"] == "native"}
            for row in rows:
                group, condition = row["group"], row["condition"]
                counts = {key: int(row[key]) for key in COUNTS}
                item = {"model": model, "cluster": cluster, "group": group,
                        "condition": condition, **counts}
                item["number_rate"] = counts["number_count"] / counts["n_trials"]
                item["exact_rate"] = counts["exact_count"] / counts["n_trials"]
                item["delta_number_pp"] = 100 * (
                    counts["number_count"] - int(native[group]["number_count"])) / counts["n_trials"]
                item["delta_exact_pp"] = 100 * (
                    counts["exact_count"] - int(native[group]["exact_count"])) / counts["n_trials"]
                cluster_effects.append(item)
                for key, value in counts.items():
                    totals[group, condition][key] += value
        for group in GROUPS:
            baseline = totals[group, "native"]
            for condition in CONDITIONS:
                counts = totals[group, condition]
                item = {"model": model, "group": group, "condition": condition, **counts}
                item["number_rate"] = counts["number_count"] / counts["n_trials"]
                item["exact_rate"] = counts["exact_count"] / counts["n_trials"]
                item["delta_number_pp"] = 100 * (
                    counts["number_count"] - baseline["number_count"]) / counts["n_trials"]
                item["delta_exact_pp"] = 100 * (
                    counts["exact_count"] - baseline["exact_count"]) / counts["n_trials"]
                summary.append(item)
    out = args.results_dir / "summary"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "cluster_effects.csv", cluster_effects)
    for model, saved in baselines.items():
        for row in cluster_effects:
            if row["model"] != model or row["condition"] != "native":
                continue
            old = saved[row["cluster"], row["group"]]
            audit = {"model": model, "cluster": row["cluster"], "group": row["group"]}
            for key in ("number_count", "exact_count", "n_trials"):
                audit[f"current_{key}"] = row[key]
                audit[f"prior_{key}"] = int(old[key])
                audit[f"delta_{key}"] = row[key] - int(old[key])
            baseline_audit.append(audit)
    if baseline_audit:
        write_csv(out / "baseline_audit.csv", baseline_audit)
    mismatches = sum(any(row[f"delta_{key}"] != 0 for key in ("number_count", "exact_count", "n_trials"))
                     for row in baseline_audit)
    lines = [f"# STE QK score ablation: {args.expected_clusters} frozen clusters", "",
             "Each model uses gate-on STE heads, 100 valid and 100 bottom concepts, and ten injection positions per cluster.",
             "Number is the argmax among `0`–`9` and `none` landing on `0`–`9`.", "",
             "| Model | Group | Condition | Number | Change vs native | Exact |",
             "|---|---|---|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['model']} | {row['group']} | {row['condition']} | "
                     f"{100*row['number_rate']:.2f}% | {row['delta_number_pp']:+.2f} pp | "
                     f"{100*row['exact_rate']:.2f}% |")
    audit_note = (f"Historical native count mismatches: {mismatches}/{len(baseline_audit)} rows."
                  if baseline_audit else "Historical baseline audit not requested.")
    lines += ["", audit_note, "Per-cluster effects are in the adjacent CSV file.", ""]
    (out / "README.md").write_text("\n".join(lines))
    print(f"COMPLETE models={len(models)} clusters_per_model={args.expected_clusters} "
          f"baseline_mismatches={mismatches}/{len(baseline_audit)} out={out}")


if __name__ == "__main__":
    main()
