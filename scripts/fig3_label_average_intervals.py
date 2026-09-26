#!/usr/bin/env python3
"""95% Wilson intervals for Figure 3 (the mean over six label settings).

Every label setting contributes the same number of trials to a bar, so the
unweighted mean over settings equals the rate pooled over them; the interval
is the Wilson score interval of the pooled counts. Per-setting trial counts:
30,000 for a matched cell (100 concepts x 30 prompts x 10 positions), 300,000
for the two cells that pool all 100 gate-router position pairs, and 30 for the
unmodified clean run, which depends only on the prompt.

Reads ``provenance.json`` written by ``plot_fig3_label_average.py`` and prints
the rows of the appendix table plus a JSON record.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

MATCHED, POOLED, CLEAN_PROMPTS = 30_000, 300_000, 30

# (panel, array, direction, cell index, row label, per-setting n, gate alone)
ROWS = (
    ("a", "response", 0, 0, "Injected run", MATCHED, False),
    ("a", "response", 0, 1, "Gate patch", MATCHED, True),
    ("a", "response", 0, 2, "Target (clean run)$^\\ast$", CLEAN_PROMPTS, False),
    ("b", "response", 1, 0, "Clean run$^\\ast$", CLEAN_PROMPTS, False),
    ("b", "response", 1, 1, "Gate patch", MATCHED, True),
    ("b", "response", 1, 2, "Target (injected run)", MATCHED, False),
    ("c", "factorial", 1, 0, "Clean run$^\\ast$", CLEAN_PROMPTS, False),
    ("c", "factorial", 1, 1, "Gate clean $\\times$ router injected", MATCHED, False),
    ("c", "factorial", 1, 2, "Gate injected $\\times$ router clean", MATCHED, True),
    ("c", "factorial", 1, 3, "Gate injected $\\times$ router injected$^\\dagger$", POOLED, False),
    ("d", "factorial", 0, 0, "Injected run", MATCHED, False),
    ("d", "factorial", 0, 1, "Gate injected $\\times$ router clean", MATCHED, False),
    ("d", "factorial", 0, 2, "Gate clean $\\times$ router injected$^\\dagger$", POOLED, True),
    ("d", "factorial", 0, 3, "Gate clean $\\times$ router clean", MATCHED, False),
)


def wilson(successes: float, n: float, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - half, center + half


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance", type=Path, required=True,
                        help="provenance.json from plot_fig3_label_average.py")
    parser.add_argument("--output_json", type=Path)
    args = parser.parse_args()

    provenance = json.loads(args.provenance.read_text())
    arms = list(range(len(provenance["arms"])))
    per_arm = {
        "response": np.asarray(provenance["per_arm_response_percentages"])[arms] / 100,
        "factorial": np.asarray(provenance["per_arm_factorial_percentages"])[arms] / 100,
    }
    records = []
    panel = None
    for name, array, direction, cell, label, n, gate_alone in ROWS:
        if name != panel:
            panel = name
            print(f"--- ({name})")
        cells = []
        for model in range(per_arm[array].shape[2]):
            rates = per_arm[array][:, direction, model, cell]
            # Rates are exact k/n per setting; recover the integer counts.
            successes = float(np.rint(rates * n).sum())
            total = n * len(arms)
            low, high = wilson(successes, total)
            rate = successes / total
            records.append(dict(panel=name, row=label, model=provenance["models"][model],
                                rate=rate, low=low, high=high, n=total))
            value = f"{100 * rate:.1f}"
            if gate_alone:
                value = f"\\textbf{{{value}}}"
            cells.append(f"{value} & \\ciint{{{100 * low:.1f}}}{{{100 * high:.1f}}}")
        print(f"    \\quad {label}\n      & " + " & ".join(cells) + " \\\\")
    if args.output_json:
        args.output_json.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
