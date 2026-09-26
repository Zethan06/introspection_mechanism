#!/usr/bin/env python3
"""Tabulate Fig 3c/d and Table 2 for the frozen digit gate masks on every label arm.

Every cell is read from run artifacts; nothing is transcribed by hand.

* ``test_{on,off}_clean_router_pin/results.csv`` -- router heads pinned to the
  clean run, with the gate native or patched.
* ``test_{on,off}_env_output_patch/aggregate_results.csv`` -- the complete
  gate-donor i x router-donor j grid. The diagonal gives the router-only cell;
  pooling all 100 (i, j) cells gives the gate + router cell; the 90 mismatched
  cells of the clean-recipient run give Table 2.

The digits identity arm reads ``ste_topk_sweep/top32``; the other arms read
``ste_topk_sweep/top32/label_transfer/<arm>`` as written by
``run_sh/04e_gate_label_transfer.sh``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
MODELS = (
    ("qwen3-4b-instruct-2507", "Qwen3-4B-IT"),
    ("llama3.1-8b-instruct", "LLaMA-3.1-8B-IT"),
    ("gemma3-12b-it", "Gemma-3-12B-IT"),
)
ARMS = (
    "digits_identity",
    "letters_identity",
    "words_identity",
    "digits_shuffled",
    "letters_shuffled",
    "words_shuffled",
)


def _read_csv(path: Path) -> list[dict[str, str]] | None:
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pin_rates(rows: list[dict[str, str]], *, column: str) -> dict[str, float]:
    """Return the four matched-grid cells of one router-pin run."""

    by_key = {(row["condition"], row["router"]): float(row[column]) for row in rows}
    baseline = next(key for key in by_key if key[0].endswith("_baseline"))[0]
    patch = next(key for key in by_key if not key[0].endswith("_baseline"))[0]
    return {
        "unmodified": by_key[(baseline, "native")],
        "gate_patch_router_native": by_key[(patch, "native")],
        "router_clean": by_key[(baseline, "clean_patch")],
        "gate_patch_router_clean": by_key[(patch, "clean_patch")],
    }


def grid_rates(rows: list[dict[str, str]], *, column: str) -> dict[str, float]:
    """Return router-only (diagonal) and pooled gate + router cells of one grid."""

    cells = {(row["intervention"], row["position_relation"]): row for row in rows}
    baseline = next(key[0] for key in cells if key[0].endswith("_baseline"))
    patch = next(key[0] for key in cells if not key[0].endswith("_baseline"))

    def pooled(intervention: str) -> float:
        parts = [cells[(intervention, relation)] for relation in ("diagonal", "mismatch")]
        total = sum(float(part["n_trials"]) for part in parts)
        return sum(float(part["n_trials"]) * float(part[column]) for part in parts) / total

    return {
        "router_patch_diagonal": float(cells[(baseline, "diagonal")][column]),
        "gate_patch_router_patch_pooled": pooled(patch),
    }


def cross_position(rows: list[dict[str, str]]) -> dict[str, float]:
    """Table 2: clean recipient, gate from i, router from j != i."""

    cells = {(row["intervention"], row["position_relation"]): row for row in rows}
    patch = next(key[0] for key in cells if not key[0].endswith("_baseline"))
    row = cells[(patch, "mismatch")]
    return {
        "output_i": float(row["overall_donor_accuracy"]),
        "output_j": float(row["overall_router_accuracy"]),
        "other": float(row["other_number_rate"]),
        "none": float(row["none_rate"]),
        "n_trials": float(row["n_trials"]),
    }


def arm_row(base: Path, *, model: str, arm: str) -> dict[str, object] | None:
    pin_off = _read_csv(base / "test_off_clean_router_pin" / "results.csv")
    pin_on = _read_csv(base / "test_on_clean_router_pin" / "results.csv")
    grid_off = _read_csv(base / "test_off_env_output_patch" / "aggregate_results.csv")
    grid_on = _read_csv(base / "test_on_env_output_patch" / "aggregate_results.csv")
    if not any((pin_off, pin_on, grid_off, grid_on)):
        return None
    row: dict[str, object] = {"model": model, "arm": arm}
    # (d) injected recipient, none rate.
    if pin_off:
        pin = pin_rates(pin_off, column="none_rate")
        row["d_unmodified"] = pin["unmodified"]
        row["d_gate_injected_router_clean"] = pin["router_clean"]
        row["d_gate_clean_router_clean"] = pin["gate_patch_router_clean"]
    if grid_off:
        grid = grid_rates(grid_off, column="none_rate")
        row["d_gate_injected_router_clean_diag_check"] = grid["router_patch_diagonal"]
        row["d_gate_clean_router_injected"] = grid["gate_patch_router_patch_pooled"]
    # (c) clean recipient, position-report rate.
    if pin_on:
        pin = pin_rates(pin_on, column="number_rate")
        row["c_unmodified"] = pin["unmodified"]
        row["c_gate_injected_router_clean"] = pin["gate_patch_router_clean"]
    if grid_on:
        grid = grid_rates(grid_on, column="number_rate")
        row["c_gate_clean_router_injected"] = grid["router_patch_diagonal"]
        row["c_gate_injected_router_injected"] = grid["gate_patch_router_patch_pooled"]
        for key, value in cross_position(grid_on).items():
            row[f"table2_{key}"] = value
    return row


def _pct(row: dict[str, object], key: str) -> str:
    value = row.get(key)
    return "--" if value is None else f"{100 * float(value):.1f}"


def markdown(rows: Sequence[dict[str, object]]) -> str:
    lines = [
        "### Fig 3c: clean run, position-report rate (%)",
        "| Model | Arm | Unmodified | Gate clean x router inj | Gate inj x router clean | Gate inj x router inj |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['model']} | {r['arm']} | {_pct(r, 'c_unmodified')} | {_pct(r, 'c_gate_clean_router_injected')} "
        f"| {_pct(r, 'c_gate_injected_router_clean')} | {_pct(r, 'c_gate_injected_router_injected')} |"
        for r in rows
    ]
    lines += [
        "",
        "### Fig 3d: injected run, none rate (%)",
        "| Model | Arm | Unmodified | Gate inj x router clean | Gate clean x router inj | Gate clean x router clean |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['model']} | {r['arm']} | {_pct(r, 'd_unmodified')} | {_pct(r, 'd_gate_injected_router_clean')} "
        f"| {_pct(r, 'd_gate_clean_router_injected')} | {_pct(r, 'd_gate_clean_router_clean')} |"
        for r in rows
    ]
    lines += [
        "",
        "### Table 2: clean run, gate from i, router from j != i (%)",
        "| Model | Arm | Output i | Output j | Other | none |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['model']} | {r['arm']} | {_pct(r, 'table2_output_i')} | {_pct(r, 'table2_output_j')} "
        f"| {_pct(r, 'table2_other')} | {_pct(r, 'table2_none')} |"
        for r in rows
    ]
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT / "results" / "router_label_transfer_summary",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows = []
    for slug, label in MODELS:
        top32 = args.results_root / slug / "ste_topk_sweep" / "top32"
        for arm in ARMS:
            base = top32 if arm == "digits_identity" else top32 / "label_transfer" / arm
            row = arm_row(base, model=label, arm=arm)
            if row is not None:
                rows.append(row)
    if not rows:
        raise SystemExit("no finished router runs found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = sorted(
        {key for row in rows for key in row},
        key=lambda key: (key not in ("model", "arm"), key),
    )
    with (args.output_dir / "router_label_transfer.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    table = markdown(rows)
    (args.output_dir / "router_label_transfer.md").write_text(table, encoding="utf-8")
    (args.output_dir / "router_label_transfer.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    print(table)
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
