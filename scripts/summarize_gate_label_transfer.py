#!/usr/bin/env python3
"""Tabulate Fig 3a/b for the ordered-digit gate masks applied to every label arm.

Reads the frozen digits run (``ste_topk_sweep/top32/test_{on,off}``) and the
transfer runs written by ``run_sh/04e_gate_label_transfer.sh``
(``ste_topk_sweep/top32/label_transfer/<arm>/test_{on,off}``). For each model
and arm it reports the unmodified rate, the gate-patch rate and the target
rate of both directions, and the fraction of the gap the patch closes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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


def read_direction(run_dir: Path) -> dict[str, dict[str, float]] | None:
    """Return the native-router baseline and Top-k rows of one test run."""

    path = run_dir / "results.csv"
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["router"] == "native"]
    by_kind = {}
    for row in rows:
        kind = "baseline" if row["condition"].endswith("_baseline") else "patch"
        by_kind[kind] = {
            key: float(row[key])
            for key in ("n_trials", "none_rate", "number_rate", "exact_target_accuracy")
        }
    if set(by_kind) != {"baseline", "patch"}:
        raise ValueError(f"{path} lacks a native baseline/patch pair")
    return by_kind


def gap_closed(before: float, after: float, target: float) -> float:
    """Fraction of the unmodified-to-target gap the patch closes (NaN if no gap)."""

    gap = target - before
    return (after - before) / gap if abs(gap) > 1e-12 else math.nan


def wilson(rate: float, n: float, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return math.nan, math.nan
    denominator = 1 + z * z / n
    center = (rate + z * z / (2 * n)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denominator
    return center - half, center + half


def transfer_row(
    *,
    model: str,
    arm: str,
    off: dict[str, dict[str, float]] | None,
    on: dict[str, dict[str, float]] | None,
    min_gap: float,
) -> dict[str, object]:
    """Combine one arm's two directions into a Fig 3a/b row.

    Gate off moves an injected run toward the clean ``none`` rate, which the
    gate-on run's clean baseline supplies; gate on moves a clean run toward the
    injected position rate, which the gate-off run's baseline supplies.
    """

    row: dict[str, object] = {"model": model, "arm": arm}
    clean_none = None if on is None else on["baseline"]["none_rate"]
    injected_position = None if off is None else off["baseline"]["number_rate"]
    if off is not None:
        row["injected_localization"] = off["baseline"]["exact_target_accuracy"]
        row["off_unmodified_none"] = off["baseline"]["none_rate"]
        row["off_patch_none"] = off["patch"]["none_rate"]
        low, high = wilson(off["patch"]["none_rate"], off["patch"]["n_trials"])
        row["off_patch_none_ci_low"], row["off_patch_none_ci_high"] = low, high
    if on is not None:
        row["clean_none"] = clean_none
        row["on_unmodified_position"] = on["baseline"]["number_rate"]
        row["on_patch_position"] = on["patch"]["number_rate"]
        low, high = wilson(on["patch"]["number_rate"], on["patch"]["n_trials"])
        row["on_patch_position_ci_low"], row["on_patch_position_ci_high"] = low, high
    if off is not None and clean_none is not None:
        before = off["baseline"]["none_rate"]
        row["off_target_none"] = clean_none
        row["off_gap"] = clean_none - before
        row["off_gap_closed"] = (
            gap_closed(before, off["patch"]["none_rate"], clean_none)
            if abs(clean_none - before) >= min_gap
            else math.nan
        )
    if on is not None and injected_position is not None:
        before = on["baseline"]["number_rate"]
        row["on_target_position"] = injected_position
        row["on_gap"] = injected_position - before
        row["on_gap_closed"] = (
            gap_closed(before, on["patch"]["number_rate"], injected_position)
            if abs(injected_position - before) >= min_gap
            else math.nan
        )
    return row


def _pct(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    return f"{100 * float(value):.1f}"


def markdown(rows: Sequence[dict[str, object]]) -> str:
    header = (
        "| Model | Arm | Loc. | Off: inj none | Off: patch | Off: target | Off gap closed "
        "| On: clean pos | On: patch | On: target | On gap closed |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    body = "".join(
        "| {model} | {arm} | {loc} | {a} | {b} | {c} | {d} | {e} | {f} | {g} | {h} |\n".format(
            model=row["model"],
            arm=row["arm"],
            loc=_pct(row.get("injected_localization")),
            a=_pct(row.get("off_unmodified_none")),
            b=_pct(row.get("off_patch_none")),
            c=_pct(row.get("off_target_none")),
            d=_pct(row.get("off_gap_closed")),
            e=_pct(row.get("on_unmodified_position")),
            f=_pct(row.get("on_patch_position")),
            g=_pct(row.get("on_target_position")),
            h=_pct(row.get("on_gap_closed")),
        )
        for row in rows
    )
    return header + body


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT / "results" / "gate_label_transfer_summary",
    )
    parser.add_argument(
        "--min_gap",
        type=float,
        default=0.10,
        help="Report gap closed only when the target differs from the unmodified rate by this much.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows = []
    for slug, label in MODELS:
        top32 = args.results_root / slug / "ste_topk_sweep" / "top32"
        for arm in ARMS:
            base = top32 if arm == "digits_identity" else top32 / "label_transfer" / arm
            off = read_direction(base / "test_off")
            on = read_direction(base / "test_on")
            if off is None and on is None:
                continue
            rows.append(
                transfer_row(model=label, arm=arm, off=off, on=on, min_gap=args.min_gap)
            )
    if not rows:
        raise SystemExit("no finished runs found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}, key=lambda key: (key not in ("model", "arm"), key))
    with (args.output_dir / "gate_label_transfer.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    table = markdown(rows)
    (args.output_dir / "gate_label_transfer.md").write_text(table, encoding="utf-8")
    (args.output_dir / "gate_label_transfer.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    print(table)
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
