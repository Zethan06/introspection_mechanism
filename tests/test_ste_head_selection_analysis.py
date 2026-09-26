"""Tests for train-only selected-head stability summaries."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.analyze_ste_head_selection import analyze_manifests


class SteHeadSelectionAnalysisTests(unittest.TestCase):
    def test_reports_overlap_frequency_and_stable_heads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for index, heads in enumerate(
                (
                    [(1, 2), (1, 3)],
                    [(1, 2), (1, 4), (2, 0)],
                )
            ):
                path = root / f"round{index}" / "train" / "selected_heads.json"
                path.parent.mkdir(parents=True)
                path.write_text(
                    json.dumps(
                        {
                            "top_k": len(heads),
                            "layers_searched": [1, 2],
                            "heads": [
                                {
                                    "layer": layer,
                                    "head": head,
                                    "selected": 1,
                                    "selection_rank": rank,
                                    "score": 0.5,
                                }
                                for rank, (layer, head) in enumerate(heads, 1)
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                paths.append(path)

            report = analyze_manifests((f"r{index}", path) for index, path in enumerate(paths))

        self.assertEqual(report["adjacent_overlap"][0]["intersection"], 1)
        self.assertEqual(report["stable_in_all_rounds"], [{"layer": 1, "head": 2}])
        frequency = {(row["layer"], row["head"]): row["count"] for row in report["head_frequency"]}
        self.assertEqual(frequency[(1, 2)], 2)
        self.assertEqual(frequency[(1, 3)], 1)


if __name__ == "__main__":
    unittest.main()
