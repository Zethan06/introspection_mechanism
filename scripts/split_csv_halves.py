#!/usr/bin/env python3
"""Split one split CSV into two disjoint halves for fit/eval reuse.

The boundary-direction protocol needs a split it fits on and a disjoint split it
scores once. When both must come from a single pool (for example validation),
this produces a deterministic partition -- of concepts or of clusters -- that
keeps every column of the source CSV intact.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Sequence

import numpy as np


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-fit", type=Path, required=True)
    parser.add_argument("--output-eval", type=Path, required=True)
    parser.add_argument(
        "--key-column",
        default="concept",
        help="Column that uniquely identifies a row (default: concept; "
        "use cluster_key for cluster CSVs).",
    )
    parser.add_argument(
        "--fit-fraction",
        type=float,
        default=0.5,
        help="Share of rows assigned to the fit half (default: 0.5).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0.0 < args.fit_fraction < 1.0:
        raise ValueError("fit-fraction must lie strictly between 0 and 1")
    if args.output_fit == args.output_eval:
        raise ValueError("fit and eval outputs must differ")

    with args.input.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames or args.key_column not in fieldnames:
        raise ValueError(f"CSV lacks a {args.key_column} column: {args.input}")
    keys = [str(row[args.key_column]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError(
            f"CSV contains duplicate {args.key_column} values: {args.input}"
        )

    fit_count = int(round(len(rows) * args.fit_fraction))
    if fit_count < 1 or fit_count >= len(rows):
        raise ValueError(
            f"fit-fraction {args.fit_fraction} leaves an empty half for "
            f"{len(rows)} rows"
        )

    # Shuffle before cutting: both concept and cluster CSVs arrive ranked by
    # quality, so a positional cut would hand the fit half every strong row.
    order = np.random.default_rng(args.seed).permutation(len(rows))
    fit_rows = [rows[index] for index in sorted(order[:fit_count])]
    eval_rows = [rows[index] for index in sorted(order[fit_count:])]

    for path, subset in ((args.output_fit, fit_rows), (args.output_eval, eval_rows)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(subset)

    print(
        f"split {len(rows)} rows from {args.input} into "
        f"{len(fit_rows)} fit + {len(eval_rows)} eval "
        f"(key {args.key_column}, seed {args.seed})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
