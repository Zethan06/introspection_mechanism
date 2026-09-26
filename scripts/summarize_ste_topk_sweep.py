#!/usr/bin/env python3
"""Aggregate bidirectional STE Top-k evaluation results into one comparison table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence


DEFAULT_TOP_K_VALUES = (1, 2, 4, 8, 16, 32)
TRANSITIONS = {
    "on": "none_to_number",
    "off": "number_to_none",
}
METRICS = (
    "n_trials",
    "source_prediction_trials",
    "converted_trials",
    "conversion_rate",
    "target_accuracy_before",
    "target_accuracy_after",
    "target_accuracy_delta",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--sweep_dir", type=Path, required=True)
    parser.add_argument(
        "--top_k",
        type=int,
        nargs="+",
        default=list(DEFAULT_TOP_K_VALUES),
    )
    parser.add_argument("--router", default="native")
    parser.add_argument(
        "--result_split",
        choices=("validation", "test"),
        default="test",
        help="Evaluation split encoded by the result directory names.",
    )
    parser.add_argument("--output_csv", type=Path)
    parser.add_argument("--output_json", type=Path)
    args = parser.parse_args(argv)
    if any(top_k <= 0 for top_k in args.top_k):
        parser.error("--top_k values must be positive")
    if len(set(args.top_k)) != len(args.top_k):
        parser.error("--top_k values must be unique")
    return args


def _load_transition(
    path: Path,
    *,
    top_k: int,
    direction: str,
    router: str,
) -> tuple[str, dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"missing Top{top_k} {direction} result: {path}"
        ) from None
    if int(payload.get("top_k", -1)) != top_k:
        raise ValueError(f"{path} does not describe Top{top_k}")
    if payload.get("selection_direction") != direction:
        raise ValueError(f"{path} does not describe direction={direction}")
    expected_transition = TRANSITIONS[direction]
    matches = [
        row
        for row in payload.get("transitions", [])
        if row.get("transition") == expected_transition
        and row.get("router") == router
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{path} must contain one {expected_transition} row for router={router}"
        )
    missing = [metric for metric in METRICS if metric not in matches[0]]
    if missing:
        raise ValueError(f"{path} transition row is missing fields: {missing}")
    return str(payload.get("model", "")), matches[0]


def summarize_sweep(
    sweep_dir: Path,
    *,
    top_k_values: Sequence[int],
    router: str = "native",
    result_split: str = "test",
) -> tuple[str, list[dict[str, object]]]:
    """Load all requested cardinalities and return one row per Top-k value."""

    rows: list[dict[str, object]] = []
    models: set[str] = set()
    for top_k in top_k_values:
        row: dict[str, object] = {"top_k": int(top_k)}
        for direction, transition in TRANSITIONS.items():
            model, metrics = _load_transition(
                sweep_dir
                / f"top{top_k}"
                / f"{result_split}_{direction}"
                / "summary.json",
                top_k=int(top_k),
                direction=direction,
                router=router,
            )
            models.add(model)
            for metric in METRICS:
                row[f"{transition}_{metric}"] = metrics[metric]
        rows.append(row)
    if len(models) != 1 or not next(iter(models), ""):
        raise ValueError(f"sweep results must use one non-empty model, found {models}")
    return next(iter(models)), rows


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    sweep_dir = args.sweep_dir.resolve()
    model, rows = summarize_sweep(
        sweep_dir,
        top_k_values=args.top_k,
        router=args.router,
        result_split=args.result_split,
    )
    output_csv = (
        args.output_csv.resolve()
        if args.output_csv is not None
        else sweep_dir
        / (
            "transition_summary.csv"
            if args.result_split == "test"
            else f"{args.result_split}_transition_summary.csv"
        )
    )
    output_json = (
        args.output_json.resolve()
        if args.output_json is not None
        else sweep_dir
        / (
            "transition_summary.json"
            if args.result_split == "test"
            else f"{args.result_split}_transition_summary.json"
        )
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "model": model,
        "router": args.router,
        "evaluation_split": args.result_split,
        "top_k_values": [int(top_k) for top_k in args.top_k],
        "conversion_denominator": "trials whose baseline prediction is the source class",
        "target_accuracy_denominator": f"all {args.result_split} trials",
        "results": rows,
    }
    output_json.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(rows, indent=2), flush=True)
    print(f"wrote {output_csv} and {output_json}", flush=True)


if __name__ == "__main__":
    main()
