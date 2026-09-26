#!/usr/bin/env python3
"""Plot whole-model final-token clean-patch effects by layer and head."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effects", type=Path, required=True)
    parser.add_argument("--output_stem", type=Path, required=True)
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args(argv)
    if args.top_k <= 0:
        parser.error("--top_k must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    with args.effects.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty effects table: {args.effects}")

    n_layers = 1 + max(int(row["layer"]) for row in rows)
    n_heads = 1 + max(int(row["head"]) for row in rows)
    matrix = np.full((n_layers, n_heads), np.nan)
    for row in rows:
        matrix[int(row["layer"]), int(row["head"])] = float(row["accuracy_drop_pp"])
    ranked = sorted(rows, key=lambda row: float(row["accuracy_drop"]), reverse=True)
    top = ranked[: min(args.top_k, len(ranked))]

    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42}
    )
    figure, (heat_axis, rank_axis) = plt.subplots(
        1,
        2,
        figsize=(10.0, max(4.2, 0.13 * n_layers)),
        gridspec_kw={"width_ratios": (1.35, 1)},
    )
    limit = max(abs(float(np.nanmin(matrix))), abs(float(np.nanmax(matrix))), 0.01)
    image = heat_axis.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    heat_axis.set_xlabel("Attention head")
    heat_axis.set_ylabel("Layer")
    heat_axis.set_title("All-head causal effect (accuracy drop, pp)")
    figure.colorbar(
        image,
        ax=heat_axis,
        label="Injected − clean-patched accuracy (pp)",
    )

    labels = [row["heads"] for row in reversed(top)]
    values = [float(row["accuracy_drop_pp"]) for row in reversed(top)]
    y = np.arange(len(top))
    rank_axis.barh(y, values, color="#0072B2")
    rank_axis.set_yticks(y, labels)
    rank_axis.axvline(0, color="#222222", linewidth=0.8)
    rank_axis.set_xlabel("Accuracy drop (pp)")
    rank_axis.set_title(f"Top {len(top)} heads")
    rank_axis.grid(axis="x", color="#E1E1E1", linewidth=0.6)
    rank_axis.spines[["top", "right"]].set_visible(False)

    figure.tight_layout()
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(args.output_stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(args.output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
