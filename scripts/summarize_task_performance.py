#!/usr/bin/env python3
"""Tabulate Table 1: test localization and clean-none accuracy per label setting.

(a) Injected trials: fraction naming the injected position's label among the
    ten labels and ``none``.
(b) Clean trials: fraction answering ``none``.

Sources, all on the test split:
  digits, ordered   ste_topk_sweep/top32/test_{off,on}/results.csv, the
                    unmodified injected and clean runs of the Stage 04f grid
  letters/words     label_accuracy/<template>/test_summary.json (Stage 09)
  shuffled labels   label_shuffle/shuffled_<label set>/test_summary.json (Stage 09)
Mean is the unweighted average over the six settings.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

MODELS = (
    ("qwen3-4b-instruct-2507", "Qwen3-4B-IT"),
    ("llama3.1-8b-instruct", "LLaMA-3.1-8B-IT"),
    ("gemma3-12b-it", "Gemma-3-12B-IT"),
)
TEMPLATE = "semantic_highinj_posref_gate_balanced_disrupts"
SETTINGS = (
    ("digits_ordered", None),
    ("letters_ordered", f"label_accuracy/{TEMPLATE}_letters_a_j"),
    ("words_ordered", f"label_accuracy/{TEMPLATE}_numwords_one_ten"),
    ("digits_shuffled", "label_shuffle/shuffled_tokens_0_9"),
    ("letters_shuffled", "label_shuffle/shuffled_letters_a_j"),
    ("words_shuffled", "label_shuffle/shuffled_numwords_one_ten"),
)


def baseline_row(path: Path, condition: str) -> dict:
    with path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["condition"] == condition and row["router"] == "native"]
    if len(rows) != 1:
        raise ValueError(f"{path} has no single native {condition} row")
    return rows[0]


def digit_rates(root: Path) -> tuple[float, float]:
    top32 = root / "ste_topk_sweep/top32"
    injected = baseline_row(top32 / "test_off/results.csv", "injected_baseline")
    clean = baseline_row(top32 / "test_on/results.csv", "clean_baseline")
    return float(injected["exact_target_accuracy"]), float(clean["none_rate"])


def label_rates(run_dir: Path) -> tuple[float, float]:
    summary = json.loads((run_dir / "test_summary.json").read_text())
    clean = summary["clean_predicted_label_counts"]
    return float(summary["accuracy"]), clean.get("none", 0) / sum(clean.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", type=Path, default=Path("results"))
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for slug, label in MODELS:
        root = args.results_root / slug
        for setting, run_dir in SETTINGS:
            injected, clean = digit_rates(root) if run_dir is None else label_rates(root / run_dir)
            rows.append(dict(model=label, setting=setting,
                             injected_accuracy=injected, clean_none_accuracy=clean))

    lines = []
    for column, title in (("injected_accuracy", "(a) Injected trials: localization accuracy"),
                          ("clean_none_accuracy", "(b) Clean trials: none-response accuracy")):
        lines += [f"% {title}"]
        for _, label in MODELS:
            values = [100 * r[column] for r in rows if r["model"] == label]
            cells = " & ".join(f"{value:.2f}" for value in values)
            lines.append(f"{label} & {cells} & \\textbf{{{sum(values) / len(values):.2f}}} \\\\")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "task_performance.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "task_performance.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
