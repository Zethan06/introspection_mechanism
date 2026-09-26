"""Tests for layer/strength localization sweep configuration."""

from __future__ import annotations

import unittest

from scripts.run_sweep_injection_parallel import (
    find_best,
    parse_launcher_args,
)
from scripts.sweep_injection_localization import parse_args, resolve_layer_end


class SweepInjectionLocalizationCliTests(unittest.TestCase):
    def _required_args(self) -> list[str]:
        return [
            "--model",
            "model",
            "--concepts_json",
            "concepts.json",
            "--cluster_csv",
            "clusters.csv",
            "--results_dir",
            "results",
        ]

    def test_manifest_baseline_remains_backward_compatible_default(self) -> None:
        args = parse_args(self._required_args())

        self.assertEqual(args.baseline_mode, "manifest")

    def test_accepts_full_english_baseline_for_exhaustive_sweep(self) -> None:
        args = parse_args(
            [
                *self._required_args(),
                "--max_concepts",
                "1000",
                "--baseline_mode",
                "full_english",
            ]
        )

        self.assertEqual(args.max_concepts, 1000)
        self.assertEqual(args.baseline_mode, "full_english")
        self.assertEqual(args.baseline_min_word_len, 1)
        self.assertEqual(args.baseline_max_word_len, 32)
        self.assertEqual(args.baseline_case_filter, "all")

    def test_rejects_nonpositive_concept_limit(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args([*self._required_args(), "--max_concepts", "0"])

    def test_accepts_evaluation_prompt(self) -> None:
        args = parse_args(
            [
                *self._required_args(),
                "--prompt_template",
                "semantic_highinj_posref_gate_balanced_disrupts",
            ]
        )

        self.assertEqual(
            args.prompt_template,
            "semantic_highinj_posref_gate_balanced_disrupts",
        )

    def test_rejects_unregistered_prompt(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    *self._required_args(),
                    "--prompt_template",
                    "semantic_highinj_posref_gate_balanced_disrupts_tokens_1_9",
                ]
            )

    def test_default_layer_end_is_the_last_layer(self) -> None:
        self.assertEqual(resolve_layer_end(36, None), 35)
        self.assertEqual(resolve_layer_end(1, None), 0)
        self.assertEqual(resolve_layer_end(36, 12), 12)


class ExhaustiveSweepLauncherTests(unittest.TestCase):
    @staticmethod
    def _grid_row(layer: int, strength: float, accuracy: float) -> dict:
        return {
            "injection_layer": layer,
            "extraction_layer": layer,
            "strength": strength,
            "n_trials": 250,
            "n_correct": round(250 * accuracy),
            "accuracy": accuracy,
            "mean_correct_prob": accuracy / 2,
            "clean_n_correct": 0,
            "clean_accuracy": 0.0,
            "clean_mean_correct_prob": 0.0,
        }

    def test_launcher_forwards_complete_grid_unchanged(self) -> None:
        launcher_args, remaining = parse_launcher_args(
            [
                "--work_dir",
                "tmp/calibration",
                "--model",
                "model",
                "--max_concepts",
                "1000",
                "--layer_step",
                "1",
                "--strengths",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
            ]
        )

        self.assertEqual(str(launcher_args.work_dir), "tmp/calibration")
        self.assertEqual(
            remaining,
            [
                "--model",
                "model",
                "--max_concepts",
                "1000",
                "--layer_step",
                "1",
                "--strengths",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
            ],
        )

    def test_best_is_selected_from_complete_grid(self) -> None:
        rows = [
            self._grid_row(3, 3.0, 0.8),
            self._grid_row(4, 7.0, 0.7),
            self._grid_row(10, 2.0, 0.9),
        ]

        best = find_best(rows)

        self.assertEqual(best["injection_layer"], 10)
        self.assertEqual(best["strength"], 2.0)


if __name__ == "__main__":
    unittest.main()
