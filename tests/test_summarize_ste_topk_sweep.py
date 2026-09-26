"""Tests for STE Top-k sweep result aggregation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_ste_topk_sweep import summarize_sweep


class SummarizeSteTopKSweepTests(unittest.TestCase):
    def _write_result(
        self,
        root: Path,
        *,
        top_k: int,
        direction: str,
        conversion_rate: float,
        result_split: str = "test",
    ) -> None:
        transition = "none_to_number" if direction == "on" else "number_to_none"
        output = (
            root / f"top{top_k}" / f"{result_split}_{direction}" / "summary.json"
        )
        output.parent.mkdir(parents=True)
        output.write_text(
            json.dumps(
                {
                    "model": "model",
                    "top_k": top_k,
                    "selection_direction": direction,
                    "transitions": [
                        {
                            "transition": transition,
                            "router": "native",
                            "n_trials": 100,
                            "source_prediction_trials": 80,
                            "converted_trials": int(80 * conversion_rate),
                            "conversion_rate": conversion_rate,
                            "target_accuracy_before": 0.2,
                            "target_accuracy_after": 0.8,
                            "target_accuracy_delta": 0.6,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    def test_joins_both_directions_by_top_k(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for top_k in (1, 2):
                self._write_result(
                    root,
                    top_k=top_k,
                    direction="on",
                    conversion_rate=0.1 * top_k,
                )
                self._write_result(
                    root,
                    top_k=top_k,
                    direction="off",
                    conversion_rate=0.2 * top_k,
                )
            model, rows = summarize_sweep(root, top_k_values=(1, 2))
            self.assertEqual(model, "model")
            self.assertEqual([row["top_k"] for row in rows], [1, 2])
            self.assertAlmostEqual(
                float(rows[1]["none_to_number_conversion_rate"]), 0.2
            )
            self.assertAlmostEqual(
                float(rows[1]["number_to_none_conversion_rate"]), 0.4
            )

    def test_reads_validation_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for direction in ("on", "off"):
                self._write_result(
                    root,
                    top_k=32,
                    direction=direction,
                    conversion_rate=0.5,
                    result_split="validation",
                )
            model, rows = summarize_sweep(
                root,
                top_k_values=(32,),
                result_split="validation",
            )
            self.assertEqual(model, "model")
            self.assertEqual(rows[0]["top_k"], 32)


if __name__ == "__main__":
    unittest.main()
