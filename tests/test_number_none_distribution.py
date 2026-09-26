"""Tests for held-out number-vs-none distribution helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from introspection_core.number_none_distribution import (
    load_number_none_test_scores,
    load_ste_plot_layers,
    normalize_density_ridge_scores,
)


class NumberNoneDistributionTests(unittest.TestCase):
    def test_ste_plot_layers_keep_range_and_two_layer_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "model.env"
            path.write_text("export STE_LAYERS=5-7\n", encoding="utf-8")
            ste_range, selected = load_ste_plot_layers(
                path, np.arange(2, 11, dtype=np.int64)
            )

        self.assertEqual(ste_range, [5, 7])
        self.assertEqual(selected, [3, 4, 5, 6, 7, 8, 9])

    def test_loads_frozen_final_token_number_none_scores(self) -> None:
        metadata = {
            "protocol": "train_direction_then_test_once",
            "capture_position": "final_token",
            "contrasts": ["any_number_vs_none"],
            "panels": ["all_boundaries"],
            "primitive_outcomes": ["none", "exact_number", "wrong_number"],
            "layers": [4, 5],
        }
        projections = np.arange(6, dtype=float).reshape(3, 1, 1, 2)
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "test_scores.npz"
            np.savez_compressed(
                path,
                metadata=np.array(json.dumps(metadata)),
                unit_contrast_projection=projections,
                primitive_outcome=np.array([0, 1, 2]),
            )
            layers, scores, labels, loaded_metadata = (
                load_number_none_test_scores(path)
            )

        np.testing.assert_array_equal(layers, [4, 5])
        np.testing.assert_array_equal(scores, projections[:, 0, 0, :])
        np.testing.assert_array_equal(labels, [False, True, True])
        self.assertEqual(loaded_metadata["capture_position"], "final_token")

    def test_rejects_non_final_token_artifact(self) -> None:
        metadata = {
            "protocol": "train_direction_then_test_once",
            "capture_position": "routing_boundary",
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "test_scores.npz"
            np.savez_compressed(path, metadata=np.array(json.dumps(metadata)))
            with self.assertRaisesRegex(ValueError, "final token"):
                load_number_none_test_scores(path)

    def test_loads_explicit_post_injection_token_scores(self) -> None:
        metadata = {
            "protocol": "train_direction_then_test_once",
            "capture_position": "post_injection_token",
            "contrasts": ["any_number_vs_none"],
            "panels": ["all_boundaries"],
            "primitive_outcomes": ["none", "exact_number", "wrong_number"],
            "layers": [4],
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "test_scores.npz"
            np.savez_compressed(
                path, metadata=np.array(json.dumps(metadata)),
                unit_contrast_projection=np.zeros((2, 1, 1, 1)),
                primitive_outcome=np.array([0, 1]),
            )
            _, _, labels, found = load_number_none_test_scores(
                path, capture_position="post_injection_token"
            )
        np.testing.assert_array_equal(labels, [False, True])
        self.assertEqual(found["capture_position"], "post_injection_token")

    def test_rejects_unknown_primitive_outcome_code(self) -> None:
        metadata = {
            "protocol": "train_direction_then_test_once",
            "capture_position": "final_token",
            "contrasts": ["any_number_vs_none"],
            "panels": ["all_boundaries"],
            "primitive_outcomes": ["none", "exact_number", "wrong_number"],
            "layers": [4],
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "test_scores.npz"
            np.savez_compressed(
                path,
                metadata=np.array(json.dumps(metadata)),
                unit_contrast_projection=np.zeros((2, 1, 1, 1)),
                primitive_outcome=np.array([0, 3]),
            )
            with self.assertRaisesRegex(ValueError, "outside"):
                load_number_none_test_scores(path)

    def test_density_ridge_uses_one_scale_for_every_layer(self) -> None:
        scores = np.array([[0.0, 10.0], [5.0, 20.0]])
        normalized, parameters = normalize_density_ridge_scores(scores)
        np.testing.assert_allclose(
            normalized,
            np.array([[0.0, 0.5], [0.25, 1.0]]),
        )
        self.assertEqual(parameters["minimum"], 0.0)
        self.assertEqual(parameters["maximum"], 20.0)

if __name__ == "__main__":
    unittest.main()
