"""CPU tests for the STE Top-k gate-effect plot."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

# `scripts` is not a package, and an installed `scripts` distribution can shadow
# it on sys.path, so load the module by file path as the other suites do.
_SPEC = importlib.util.spec_from_file_location(
    "plot_ste_topk_gate_effect",
    Path(__file__).resolve().parents[1] / "scripts/plot_ste_topk_gate_effect.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
# dataclasses resolves the defining module through sys.modules.
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
load_curve, main = _MODULE.load_curve, _MODULE.main
load_random_control = _MODULE.load_random_control


class PlotSteTopkGateEffectTests(unittest.TestCase):
    def _write_summary(
        self,
        path: Path,
        *,
        vary_baseline: bool = False,
        top_k_values: tuple[int, ...] = (1, 2, 4, 8, 16, 24, 32),
    ) -> None:
        fieldnames = [
            "top_k",
            "none_to_number_target_accuracy_before",
            "none_to_number_target_accuracy_after",
            "number_to_none_target_accuracy_before",
            "number_to_none_target_accuracy_after",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for index, top_k in enumerate(top_k_values):
                writer.writerow(
                    {
                        "top_k": top_k,
                        "none_to_number_target_accuracy_before": (
                            0.1 + 0.01 * index if vary_baseline else 0.1
                        ),
                        "none_to_number_target_accuracy_after": 0.2 + index / 10,
                        "number_to_none_target_accuracy_before": 0.25,
                        "number_to_none_target_accuracy_after": 0.7 + index / 20,
                    }
                )

    def _write_random_summary(
        self,
        path: Path,
        *,
        top_k_values: tuple[int, ...] = (1, 4, 16),
        n_draws: int = 10,
    ) -> None:
        statistics = ("mean", "std", "ci_low", "ci_high", "min", "max")
        fieldnames = ["top_k", "n_draws"] + [
            f"{transition}_{metric}_{statistic}"
            for transition in ("none_to_number", "number_to_none")
            for metric in (
                "conversion_rate",
                "target_accuracy_before",
                "target_accuracy_after",
                "target_accuracy_delta",
            )
            for statistic in statistics
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for index, top_k in enumerate(top_k_values):
                row = {"top_k": top_k, "n_draws": n_draws}
                for field in fieldnames[2:]:
                    if field.endswith("_ci_low"):
                        row[field] = 0.10 + index / 40
                    elif field.endswith("_ci_high"):
                        row[field] = 0.20 + index / 40
                    else:
                        row[field] = 0.15 + index / 40
                writer.writerow(row)

    def test_load_curve_maps_outcomes_and_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "transition_summary.csv"
            self._write_summary(summary)
            curve = load_curve("Model", summary)

            self.assertEqual(curve.top_k, (0, 1, 2, 4, 8, 16, 24, 32))
            self.assertAlmostEqual(curve.gate_off_none[0], 0.25)
            self.assertAlmostEqual(curve.gate_on_number[0], 0.1)
            self.assertAlmostEqual(curve.gate_off_none[-1], 1.0)
            self.assertAlmostEqual(curve.gate_on_number[-1], 0.8)
            self.assertAlmostEqual(curve.clean_none_baseline, 0.9)
            self.assertAlmostEqual(curve.injected_number_baseline, 0.75)

    def test_rejects_inconsistent_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "transition_summary.csv"
            self._write_summary(summary, vary_baseline=True)
            with self.assertRaisesRegex(ValueError, "inconsistent clean-none"):
                load_curve("Model", summary)

    def test_main_writes_vector_preview_and_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "transition_summary.csv"
            output = root / "plots"
            self._write_summary(summary)
            main(
                [
                    "--input",
                    f"Model={summary}",
                    "--output_dir",
                    str(output),
                ]
            )

            for stem in ("model_ste_topk_gate_effect", "all_models_ste_topk_gate_effect"):
                for suffix in (".pdf", ".svg", ".png"):
                    self.assertTrue((output / stem).with_suffix(suffix).is_file())
            self.assertTrue((output / "ste_topk_gate_effect_data.csv").is_file())

    def test_accepts_the_sweep_default_top_k_list(self) -> None:
        # run_sh/04b_ste_topk_sweep.sh defaults to this list, without 24.
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "transition_summary.csv"
            self._write_summary(summary, top_k_values=(1, 2, 4, 8, 16, 32))
            curve = load_curve("Model", summary)

            self.assertEqual(curve.top_k, (0, 1, 2, 4, 8, 16, 32))

    def test_rejects_models_with_different_top_k_grids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wide, narrow = root / "wide.csv", root / "narrow.csv"
            self._write_summary(wide)
            self._write_summary(narrow, top_k_values=(1, 2, 4, 8, 16, 32))
            with self.assertRaisesRegex(ValueError, "same top_k values"):
                main(
                    [
                        "--input",
                        f"Wide={wide}",
                        "--input",
                        f"Narrow={narrow}",
                        "--output_dir",
                        str(root / "plots"),
                    ]
                )

    def test_load_random_control_reads_mean_and_interval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "random_topk_transition_summary.csv"
            self._write_random_summary(summary)
            control = load_random_control("Model", summary)

            self.assertEqual(control.top_k, (1, 4, 16))
            self.assertEqual(control.n_draws, 10)
            self.assertAlmostEqual(control.gate_on_number[0], 0.15)
            self.assertAlmostEqual(control.gate_off_low[0], 0.10)
            self.assertAlmostEqual(control.gate_off_high[-1], 0.25)

    def test_random_control_may_cover_a_subset_of_the_sweep(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary, random_summary = root / "sweep.csv", root / "random.csv"
            self._write_summary(summary)
            self._write_random_summary(random_summary, top_k_values=(1, 4, 16))
            output = root / "plots"
            main(
                [
                    "--input",
                    f"Model={summary}",
                    "--random_input",
                    f"Model={random_summary}",
                    "--output_dir",
                    str(output),
                ]
            )

            data = output / "ste_topk_gate_effect_data.csv"
            with data.open(newline="", encoding="utf-8") as handle:
                rows = {int(row["top_k"]): row for row in csv.DictReader(handle)}
            self.assertEqual(rows[4]["random_n_draws"], "10")
            self.assertAlmostEqual(
                float(rows[4]["random_gate_on_number_rate_mean"]), 0.175
            )
            # k=2 is in the sweep but not in the control, and stays blank.
            self.assertEqual(rows[2]["random_gate_on_number_rate_mean"], "")

    def test_rejects_random_k_outside_the_sweep(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary, random_summary = root / "sweep.csv", root / "random.csv"
            self._write_summary(summary, top_k_values=(1, 2, 4))
            self._write_random_summary(random_summary, top_k_values=(1, 64))
            with self.assertRaisesRegex(ValueError, "outside the Model sweep"):
                main(
                    [
                        "--input",
                        f"Model={summary}",
                        "--random_input",
                        f"Model={random_summary}",
                        "--output_dir",
                        str(root / "plots"),
                    ]
                )

    def test_rejects_random_input_without_a_matching_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary, random_summary = root / "sweep.csv", root / "random.csv"
            self._write_summary(summary)
            self._write_random_summary(random_summary)
            with self.assertRaisesRegex(ValueError, "no matching --input"):
                main(
                    [
                        "--input",
                        f"Model={summary}",
                        "--random_input",
                        f"Other={random_summary}",
                        "--output_dir",
                        str(root / "plots"),
                    ]
                )


    def test_split_directions_writes_one_panel_per_direction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary, random_summary = root / "sweep.csv", root / "random.csv"
            self._write_summary(summary)
            self._write_random_summary(random_summary, top_k_values=(1, 4, 16))
            output = root / "plots"
            main(
                [
                    "--input",
                    f"Model={summary}",
                    "--random_input",
                    f"Model={random_summary}",
                    "--output_dir",
                    str(output),
                    "--split_directions",
                ]
            )

            for stem in (
                "model_ste_topk_gate_effect_gate_on",
                "model_ste_topk_gate_effect_gate_off",
                "all_models_ste_topk_gate_effect",
            ):
                for suffix in (".pdf", ".svg", ".png"):
                    self.assertTrue((output / stem).with_suffix(suffix).is_file())
            # The combined per-model panel is not written in this layout.
            self.assertFalse(
                (output / "model_ste_topk_gate_effect.pdf").is_file()
            )

    def test_split_overview_accepts_several_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "a.csv", root / "b.csv"
            random_summary = root / "random.csv"
            self._write_summary(first)
            self._write_summary(second)
            self._write_random_summary(random_summary, top_k_values=(1, 4, 16))
            output = root / "plots"
            main(
                [
                    "--input",
                    f"First={first}",
                    "--input",
                    f"Second={second}",
                    "--random_input",
                    f"First={random_summary}",
                    "--output_dir",
                    str(output),
                    "--split_directions",
                    "--hide_titles",
                ]
            )

            self.assertTrue(
                (output / "all_models_ste_topk_gate_effect.pdf").is_file()
            )
            self.assertTrue(
                (output / "second_ste_topk_gate_effect_gate_off.pdf").is_file()
            )


if __name__ == "__main__":
    unittest.main()
