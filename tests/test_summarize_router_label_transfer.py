"""Pure tests for the Fig 3c/d and Table 2 label-transfer summary."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from summarize_router_label_transfer import (  # noqa: E402
    cross_position,
    grid_rates,
    pin_rates,
)


def _grid(baseline: str, patch: str) -> list[dict[str, str]]:
    rows = []
    for intervention, diagonal, mismatch in (
        (baseline, "0.10", "0.20"),
        (patch, "0.80", "0.60"),
    ):
        for relation, rate, n in (("diagonal", diagonal, "30000"), ("mismatch", mismatch, "270000")):
            rows.append({
                "intervention": intervention,
                "position_relation": relation,
                "n_trials": n,
                "number_rate": rate,
                "none_rate": str(1 - float(rate)),
                "overall_donor_accuracy": "0.05",
                "overall_router_accuracy": "0.50",
                "other_number_rate": "0.04",
            })
    return rows


class RouterLabelTransferTests(unittest.TestCase):
    def test_pin_cells_follow_condition_and_router(self) -> None:
        rows = [
            {"condition": "clean_baseline", "router": "native", "number_rate": "0.1"},
            {"condition": "clean_with_injected_top32", "router": "native", "number_rate": "0.7"},
            {"condition": "clean_baseline", "router": "clean_patch", "number_rate": "0.1"},
            {"condition": "clean_with_injected_top32", "router": "clean_patch", "number_rate": "0.3"},
        ]
        rates = pin_rates(rows, column="number_rate")
        self.assertEqual(rates["unmodified"], 0.1)
        self.assertEqual(rates["gate_patch_router_native"], 0.7)
        self.assertEqual(rates["gate_patch_router_clean"], 0.3)

    def test_gate_router_cell_pools_all_hundred_position_pairs(self) -> None:
        rates = grid_rates(
            _grid("clean_baseline", "clean_with_injected_top32"), column="number_rate"
        )
        self.assertAlmostEqual(rates["router_patch_diagonal"], 0.10)
        self.assertAlmostEqual(rates["gate_patch_router_patch_pooled"], 0.1 * 0.8 + 0.9 * 0.6)

    def test_table2_reads_the_mismatched_patch_cells(self) -> None:
        table = cross_position(_grid("clean_baseline", "clean_with_injected_top32"))
        self.assertEqual(table["output_i"], 0.05)
        self.assertEqual(table["output_j"], 0.50)
        self.assertEqual(table["n_trials"], 270000.0)


if __name__ == "__main__":
    unittest.main()
