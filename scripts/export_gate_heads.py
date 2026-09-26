#!/usr/bin/env python3
"""Export frozen STE gate masks as the heads.csv used by the QK and OV analyses.

Each row is one candidate head of the STE search window; ``is_ste`` marks the
Top-k heads the mask selected. Pass one ``--head_mask SLUG=PATH`` per model to
collect several models into one table.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch


def head_rows(slug: str, path: Path) -> list[dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    layers = [int(layer) for layer in payload["layers"]]
    hard = payload["hard_mask"].bool()
    if tuple(hard.shape) != (len(layers), int(payload["n_heads"])):
        raise ValueError(f"{path}: hard_mask does not match layers x heads")
    if int(hard.sum()) != int(payload["top_k"]):
        raise ValueError(f"{path}: hard_mask does not select exactly top_k heads")
    return [
        {
            "model": slug,
            "layer": layer,
            "head": head,
            "is_ste": bool(hard[row, head]),
        }
        for row, layer in enumerate(layers)
        for head in range(hard.shape[1])
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--head_mask",
        action="append",
        required=True,
        metavar="SLUG=PATH",
        help="model slug and its train_<direction>/head_mask.pt",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for entry in args.head_mask:
        slug, separator, path = entry.partition("=")
        if not separator or not slug or not path:
            parser.error(f"--head_mask expects SLUG=PATH, got {entry!r}")
        rows.extend(head_rows(slug, Path(path)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "layer", "head", "is_ste"])
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
