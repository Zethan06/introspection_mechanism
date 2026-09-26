#!/usr/bin/env python3
"""Draw Figure 3 with every bar averaged over the six label settings.

The gate masks and router heads are selected under ordered digit labels
(run_sh/04b, 04f) and applied without re-selection to the letters, number-word
and shuffled-label prompts of Table 1 (run_sh/04e). Each bar is the unweighted
mean over the six settings of the rates tabulated by
summarize_gate_label_transfer.py (panels a, b) and
summarize_router_label_transfer.py (panels c, d).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from introspection_core.manuscript_figures import (  # noqa: E402
    configure_manuscript_style, draw_intervention_panels,
)

MODELS = (
    ("qwen3-4b-instruct-2507", "Qwen3-4B-IT"),
    ("llama3.1-8b-instruct", "LLaMA-3.1-8B-IT"),
    ("gemma3-12b-it", "Gemma-3-12B-IT"),
)
MODEL_LABELS = ["Qwen3\n4B", "LLaMA\n8B", "Gemma\n12B"]
ARMS = (
    "digits_identity",
    "letters_identity",
    "words_identity",
    "digits_shuffled",
    "letters_shuffled",
    "words_shuffled",
)


def arm_arrays(gate: dict, router: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return one arm's [direction, 3] panel (a,b) and [direction, 4] (c,d) bars.

    Direction 0 is the injected run (none rate), 1 the clean run (position
    rate), as ``draw_intervention_panels`` expects. The unmodified bars of
    (c,d) reuse the gate runs of (a,b).
    """

    values = np.array([
        [gate["off_unmodified_none"], gate["off_patch_none"], gate["off_target_none"]],
        [gate["on_unmodified_position"], gate["on_patch_position"], gate["on_target_position"]],
    ])
    factorial = np.array([
        [gate["off_unmodified_none"], router["d_gate_injected_router_clean"],
         router["d_gate_clean_router_injected"], router["d_gate_clean_router_clean"]],
        [gate["on_unmodified_position"], router["c_gate_clean_router_injected"],
         router["c_gate_injected_router_clean"], router["c_gate_injected_router_injected"]],
    ])
    return 100 * values, 100 * factorial


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate_summary", type=Path, required=True,
                        help="gate_label_transfer.json from summarize_gate_label_transfer.py")
    parser.add_argument("--router_summary", type=Path, required=True,
                        help="router_label_transfer.json from summarize_router_label_transfer.py")
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    gate_rows = json.loads(args.gate_summary.read_text())
    router_rows = json.loads(args.router_summary.read_text())
    gate_by = {(r["model"], r["arm"]): r for r in gate_rows}
    router_by = {(r["model"], r["arm"]): r for r in router_rows}

    # [arm, direction, model, cell]
    values = np.zeros((len(ARMS), 2, len(MODELS), 3))
    factorial = np.zeros((len(ARMS), 2, len(MODELS), 4))
    for a, arm in enumerate(ARMS):
        for m, (slug, label) in enumerate(MODELS):
            values[a, :, m], factorial[a, :, m] = arm_arrays(gate_by[(label, arm)], router_by[(label, arm)])

    configure_manuscript_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "arms": list(ARMS),
        "models": [label for _, label in MODELS],
        "response_order": ["before_patching", "gate_patch", "target"],
        "factorial_order": ["before_patching", "router_patch", "gate_patch", "gate_patch_router_patch"],
        "directions": ["off: injected recipient, none rate", "on: clean recipient, position rate"],
        "per_arm_response_percentages": values.tolist(),
        "per_arm_factorial_percentages": factorial.tolist(),
    }
    mean_values = values.mean(axis=0)
    mean_factorial = factorial.mean(axis=0)
    fig = draw_intervention_panels(MODEL_LABELS, mean_values, mean_factorial)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(args.output_dir / f"fig3_interventions_mean6.{extension}", dpi=250)
    plt.close(fig)
    provenance.update(response_percentages=mean_values.tolist(),
                      factorial_percentages=mean_factorial.tolist())
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(args.output_dir)


if __name__ == "__main__":
    main()
