#!/usr/bin/env python3
"""Summarize stability and layer concentration of STE head selections."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence


Head = tuple[int, int]


def _label_for_path(path: Path) -> str:
    return path.parent.parent.name if path.parent.name == "train" else path.stem


def _selected_heads(payload: dict) -> tuple[Head, ...]:
    rows = payload.get("heads")
    if not isinstance(rows, list):
        raise ValueError("selection manifest is missing a list-valued `heads`")
    selected = [
        (int(row["layer"]), int(row["head"]))
        for row in rows
        if isinstance(row, dict) and row.get("selected")
    ]
    if len(set(selected)) != len(selected):
        raise ValueError("selection manifest contains duplicate selected heads")
    return tuple(selected)


def analyze_manifests(selections: Iterable[tuple[str, Path]]) -> dict:
    rounds: list[dict] = []
    sets: dict[str, set[Head]] = {}
    occurrences: list[dict] = []
    for label, path in selections:
        if label in sets:
            raise ValueError(f"duplicate selection label: {label}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        heads = set(_selected_heads(payload))
        if not heads:
            raise ValueError(f"selection manifest has no selected heads: {path}")
        sets[label] = heads
        layer_counts = Counter(layer for layer, _head in heads)
        rounds.append(
            {
                "label": label,
                "path": str(path.resolve()),
                "top_k": int(payload.get("top_k", len(heads))),
                "layers_searched": payload.get("layers_searched"),
                "n_selected": len(heads),
                "layer_counts": {
                    str(layer): count for layer, count in sorted(layer_counts.items())
                },
            }
        )
        for row in payload["heads"]:
            if not isinstance(row, dict) or not row.get("selected"):
                continue
            occurrence = {
                "round": label,
                "layer": int(row["layer"]),
                "head": int(row["head"]),
            }
            for field in ("selection_rank", "score"):
                if row.get(field) is not None:
                    occurrence[field] = float(row[field])
            occurrences.append(occurrence)

    labels = [row["label"] for row in rounds]
    adjacent_overlap = []
    for left, right in zip(labels, labels[1:]):
        intersection = len(sets[left] & sets[right])
        union = len(sets[left] | sets[right])
        adjacent_overlap.append(
            {
                "left": left,
                "right": right,
                "intersection": intersection,
                "recall_from_left": intersection / len(sets[left]),
                "recall_from_right": intersection / len(sets[right]),
                "jaccard": intersection / union,
            }
        )

    frequency: Counter[Head] = Counter()
    ranks: defaultdict[Head, list[float]] = defaultdict(list)
    scores: defaultdict[Head, list[float]] = defaultdict(list)
    for row in occurrences:
        head = (int(row["layer"]), int(row["head"]))
        frequency[head] += 1
        if "selection_rank" in row:
            ranks[head].append(float(row["selection_rank"]))
        if "score" in row:
            scores[head].append(float(row["score"]))
    frequency_rows = []
    for (layer, head), count in sorted(
        frequency.items(), key=lambda item: (-item[1], item[0])
    ):
        row = {"layer": layer, "head": head, "count": count}
        if ranks[(layer, head)]:
            row["mean_selection_rank"] = sum(ranks[(layer, head)]) / len(
                ranks[(layer, head)]
            )
        if scores[(layer, head)]:
            row["mean_score"] = sum(scores[(layer, head)]) / len(
                scores[(layer, head)]
            )
        frequency_rows.append(row)
    return {
        "rounds": rounds,
        "adjacent_overlap": adjacent_overlap,
        "head_frequency": frequency_rows,
        "stable_in_all_rounds": [
            {"layer": layer, "head": head}
            for (layer, head), count in sorted(frequency.items())
            if count == len(rounds)
        ],
    }


def _parse_selection(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, raw_path = value.split("=", 1)
        if not label or not raw_path:
            raise ValueError("selection must use non-empty LABEL=PATH")
        return label, Path(raw_path)
    path = Path(value)
    return _label_for_path(path), path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", nargs="+", required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    try:
        args.selection = [_parse_selection(value) for value in args.selection]
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rendered = json.dumps(analyze_manifests(args.selection), indent=2) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
