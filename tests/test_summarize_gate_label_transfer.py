"""Pure tests for the gate label-transfer summary."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from summarize_gate_label_transfer import gap_closed, transfer_row  # noqa: E402


def _direction(baseline_number: float, patch_number: float) -> dict:
    return {
        kind: {
            "n_trials": 30000.0,
            "none_rate": 1 - number,
            "number_rate": number,
            "exact_target_accuracy": number / 2,
        }
        for kind, number in (("baseline", baseline_number), ("patch", patch_number))
    }


class GateLabelTransferTests(unittest.TestCase):
    def test_gap_closed_is_fraction_of_distance_to_target(self) -> None:
        self.assertAlmostEqual(gap_closed(0.2, 0.6, 1.0), 0.5)
        self.assertTrue(math.isnan(gap_closed(0.5, 0.7, 0.5)))

    def test_targets_come_from_the_opposite_direction_baseline(self) -> None:
        off = _direction(baseline_number=0.8, patch_number=0.2)
        on = _direction(baseline_number=0.1, patch_number=0.45)
        row = transfer_row(model="m", arm="a", off=off, on=on, min_gap=0.1)
        self.assertAlmostEqual(row["off_target_none"], 0.9)
        self.assertAlmostEqual(row["off_gap_closed"], (0.8 - 0.2) / (0.9 - 0.2))
        self.assertAlmostEqual(row["on_target_position"], 0.8)
        self.assertAlmostEqual(row["on_gap_closed"], 0.5)

    def test_small_gap_is_not_scored(self) -> None:
        off = _direction(baseline_number=0.2, patch_number=0.1)
        on = _direction(baseline_number=0.15, patch_number=0.3)
        row = transfer_row(model="m", arm="a", off=off, on=on, min_gap=0.1)
        self.assertTrue(math.isnan(row["on_gap_closed"]))


if __name__ == "__main__":
    unittest.main()
