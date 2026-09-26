"""Tests for the averaged attention-browser command line."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.build_position_averaged_attention_visualization import (
    parse_args,
    validate_calibration_selection,
)


def parse_visualization_args(argv):
    return parse_args(
        [
            *argv,
            "--cluster_file",
            "clusters.csv",
            "--results_dir",
            "results",
            "--concepts_json",
            "concepts.json",
            "--concept_csv",
            "concepts.csv",
            "--injection_layer",
            "4",
            "--strength",
            "3",
        ]
    )


class AttentionVisualizationCliTests(unittest.TestCase):
    def test_token_localization_defaults_to_zero_based_ids(self) -> None:
        args = parse_visualization_args(["--model", "model"])

        self.assertEqual(args.position_index_start, 0)
        self.assertEqual(args.choice_suffix, "")

    def test_accepts_registered_prompt_templates(self) -> None:
        for template_name in (
            "token_localization",
            "semantic_highinj_posref_gate_balanced_disrupts",
        ):
            with self.subTest(template_name=template_name):
                args = parse_visualization_args(
                    [
                        "--model",
                        "model",
                        "--task",
                        "token_localization",
                        "--prompt_template",
                        template_name,
                    ]
                )

                self.assertEqual(args.prompt_template, template_name)
                self.assertEqual(args.position_index_start, 0)

    def test_rejects_unregistered_template_and_one_based_ids(self) -> None:
        for extra in (
            ["--prompt_template", "natural_signal_question"],
            ["--position_index_start", "1"],
        ):
            with self.subTest(extra=extra):
                with self.assertRaises(SystemExit):
                    parse_visualization_args(["--model", "model", *extra])

    def test_calibration_selection_must_match_cli_layer_and_strength(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            path.write_text(json.dumps({"injection_layer": 4, "strength": 3.0}))

            selection = validate_calibration_selection(
                path, injection_layer=4, strength=3.0
            )
            self.assertEqual(selection["injection_layer"], 4)
            with self.assertRaisesRegex(ValueError, "do not match"):
                validate_calibration_selection(
                    path, injection_layer=5, strength=3.0
                )


if __name__ == "__main__":
    unittest.main()
