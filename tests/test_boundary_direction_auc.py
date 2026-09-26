"""Tests for model-independent boundary-direction estimation and AUC."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from introspection_core.boundary_direction_auc import (
    DirectionBank,
    DirectionContrast,
    boundary_protocol,
    boundary_panel_indices,
    boundary_representation,
    classify_primitive_outcomes,
    complete_unique_clean_union_statistics,
    load_direction_bank,
    roc_auc,
    score_boundary_deltas,
    unique_clean_reference_components,
)
from introspection_core.boundary_direction_runtime import (
    BoundaryTask,
    capture_positions,
)


class BoundaryDirectionAucTest(unittest.TestCase):
    def test_boundary_representation_supports_delta_and_injected_latent(self) -> None:
        injected = torch.tensor([[3.0, 4.0]])
        clean = torch.tensor([[1.0, 1.0]])

        self.assertTrue(
            torch.equal(
                boundary_representation(injected, clean, "delta"),
                torch.tensor([[2.0, 3.0]]),
            )
        )
        self.assertIs(boundary_representation(injected, None, "injected"), injected)
        self.assertIs(
            boundary_representation(
                injected, None, "injected_number_vs_unique_clean"
            ),
            injected,
        )
        with self.assertRaisesRegex(ValueError, "clean states are required"):
            boundary_representation(injected, None, "delta")

    def test_unique_clean_protocol_maps_testing_to_injected_latents(self) -> None:
        protocol = boundary_protocol("injected_number_vs_unique_clean")
        self.assertEqual(protocol.test_representation, "injected")
        self.assertTrue(protocol.unique_clean_negative)
        self.assertEqual(protocol.required_capture_position, "final_token")

    def test_unique_clean_components_count_each_cluster_once(self) -> None:
        first_position = torch.tensor(
            [
                [[[3.0, 4.0]]],
                [[[0.0, 2.0]]],
            ]
        )
        clean_states = first_position.expand(-1, 10, -1, -1).clone()
        counts, sums, union_counts, union_sums = (
            unique_clean_reference_components(
                clean_states,
                source_panel_count=2,
                outcome_count=5,
                none_index=1,
            )
        )

        torch.testing.assert_close(counts[:, 1], torch.tensor([2, 2]))
        self.assertEqual(int(union_counts[1]), 2)
        expected_sum = torch.tensor([[0.6, 1.8]])
        torch.testing.assert_close(sums[0, 1], expected_sum)
        torch.testing.assert_close(union_sums[1], expected_sum)

        counts[:, 2] = torch.tensor([3, 4])
        sums[0, 2] = torch.tensor([[1.0, 2.0]])
        sums[1, 2] = torch.tensor([[3.0, 4.0]])
        completed_counts, completed_sums = (
            complete_unique_clean_union_statistics(
                counts,
                sums,
                union_counts,
                union_sums,
                none_index=1,
            )
        )
        self.assertEqual(int(completed_counts[1]), 2)
        self.assertEqual(int(completed_counts[2]), 7)
        torch.testing.assert_close(completed_sums[1], expected_sum)
        torch.testing.assert_close(completed_sums[2], torch.tensor([[4.0, 6.0]]))

    def test_final_token_capture_position_repeats_sequence_end(self) -> None:
        task = BoundaryTask(
            examples=(),
            position_count=2,
            base_tokens=torch.zeros(3, 7, dtype=torch.long),
            candidate_positions=torch.tensor([[1, 3], [1, 3], [1, 3]]),
            boundary_positions=torch.tensor([[2, 4], [2, 4], [2, 4]]),
            candidate_ids=torch.empty(3, 0, dtype=torch.long),
            boundary_text=(),
            position_panel_index=torch.tensor([0, 1]),
        )

        self.assertTrue(
            torch.equal(
                capture_positions(task, "final_token"),
                torch.full((3, 2), 6, dtype=torch.long),
            )
        )
        self.assertIs(
            capture_positions(task, "routing_boundary"), task.boundary_positions
        )

    def test_post_injection_capture_uses_immediate_successor(self) -> None:
        task = BoundaryTask(
            examples=(), position_count=2,
            base_tokens=torch.zeros(1, 8, dtype=torch.long),
            candidate_positions=torch.tensor([[1, 4]]),
            boundary_positions=torch.tensor([[3, 6]]),
            candidate_ids=torch.empty(1, 0, dtype=torch.long),
            boundary_text=(), position_panel_index=torch.tensor([0, 1]),
        )
        self.assertEqual(
            capture_positions(task, "post_injection_token").tolist(), [[2, 5]]
        )
        task.candidate_positions[0, 1] = 7
        with self.assertRaisesRegex(ValueError, "no following token"):
            capture_positions(task, "post_injection_token")

    def test_outcome_encoding_uses_runtime_candidate_count(self) -> None:
        predictions = torch.tensor([4, 2, 7, 4])
        targets = torch.tensor([0, 2, 1, 3])
        self.assertEqual(
            classify_primitive_outcomes(predictions, targets, none_index=4).tolist(),
            [0, 1, 2, 0],
        )

    def test_boundary_panels_support_variable_candidate_layouts(self) -> None:
        boundary_text = [
            [" TOKEN", " TOKEN", "\n", "\n\n"],
            [" TOKEN", " TOKEN", "\n", "\n\n"],
        ]
        indices = boundary_panel_indices(
            boundary_text, ("token_marker_mean", "literal_newline")
        )
        self.assertEqual(indices.tolist(), [0, 0, 1, 1])

    def test_direction_bank_uses_count_weighted_union_panel(self) -> None:
        layers = [1, 3]
        panels = ("token_marker_mean", "literal_newline")
        outcomes = ("all", "none", "any_number", "exact_number", "wrong_number")
        counts = torch.tensor(
            [
                [4, 2, 2, 1, 1],
                [2, 1, 1, 1, 1],
            ]
        )
        sums = torch.zeros(2, 5, 2, 2)
        # Marker: number rotates toward +x and none toward +y.
        sums[0, outcomes.index("none"), :, 1] = 2
        sums[0, outcomes.index("any_number"), :, 0] = 2
        sums[0, outcomes.index("exact_number"), :, 0] = 1
        sums[0, outcomes.index("wrong_number"), :, 0] = 1
        # Newline: number rotates toward +y and none toward -x.
        sums[1, outcomes.index("none"), :, 0] = -1
        sums[1, outcomes.index("any_number"), :, 1] = 1
        sums[1, outcomes.index("exact_number"), :, 1] = 1
        sums[1, outcomes.index("wrong_number"), :, 1] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "components.pt"
            torch.save(
                {
                    "layers": layers,
                    "panels": panels,
                    "outcomes": outcomes,
                    "pooled_counts": counts,
                    "pooled_unit_sums": sums,
                },
                path,
            )
            bank = load_direction_bank([path], layers)

        self.assertEqual(bank.panels, ("all_boundaries", *panels))
        counts_by_panel = bank.training_counts["any_number_vs_none"]
        self.assertEqual(counts_by_panel["all_boundaries"]["positive"], 3)
        self.assertEqual(counts_by_panel["token_marker_mean"]["positive"], 2)
        self.assertEqual(bank.contrast_directions.shape, (4, 3, 2, 2))

    def test_direction_bank_honors_explicit_unique_clean_union(self) -> None:
        layers = [1]
        panels = ("marker", "newline")
        outcomes = ("all", "none", "any_number", "exact_number", "wrong_number")
        counts = torch.ones(2, 5, dtype=torch.int64)
        sums = torch.zeros(2, 5, 1, 2)
        union_counts = torch.tensor([2, 1, 2, 1, 1])
        union_sums = torch.zeros(5, 1, 2)
        union_sums[outcomes.index("none"), 0] = torch.tensor([0.0, 1.0])
        union_sums[outcomes.index("any_number"), 0] = torch.tensor([2.0, 0.0])
        union_sums[outcomes.index("exact_number"), 0] = torch.tensor([1.0, 0.0])
        union_sums[outcomes.index("wrong_number"), 0] = torch.tensor([1.0, 0.0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "components.pt"
            torch.save(
                {
                    "layers": layers,
                    "panels": panels,
                    "outcomes": outcomes,
                    "pooled_counts": counts,
                    "pooled_unit_sums": sums,
                    "pooled_union_counts": union_counts,
                    "pooled_union_unit_sums": union_sums,
                },
                path,
            )
            bank = load_direction_bank([path], layers)

        training = bank.training_counts["any_number_vs_none"]["all_boundaries"]
        self.assertEqual(training, {"positive": 2, "negative": 1})
        expected = torch.tensor([1.0, -1.0]) / np.sqrt(2.0)
        torch.testing.assert_close(bank.contrast_directions[0, 0, 0], expected)

    def test_scoring_masks_non_applicable_source_panels(self) -> None:
        contrast = DirectionContrast(
            "positive_vs_negative", "positive", "negative", ("positive",), ("negative",)
        )
        directions = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]], [[0.0, 1.0]]]])
        bank = DirectionBank(
            layers=(4,),
            source_panels=("marker", "newline"),
            panels=("all_boundaries", "marker", "newline"),
            contrasts=(contrast,),
            contrast_directions=directions,
            positive_prototypes=directions,
            training_counts={},
        )
        deltas = torch.tensor([[[2.0, 0.0]], [[0.0, 3.0]]])
        scores = score_boundary_deltas(deltas, torch.tensor([0, 1]), bank)[
            "unit_contrast_projection"
        ]
        self.assertEqual(scores.shape, (2, 1, 3, 1))
        self.assertAlmostEqual(float(scores[0, 0, 0, 0]), 1.0)
        self.assertAlmostEqual(float(scores[1, 0, 2, 0]), 1.0)
        self.assertTrue(torch.isnan(scores[0, 0, 2, 0]))
        self.assertTrue(torch.isnan(scores[1, 0, 1, 0]))

    def test_roc_auc_handles_ties(self) -> None:
        labels = np.array([False, True, False, True])
        self.assertAlmostEqual(roc_auc(labels, np.array([0.0, 1.0, 0.0, 1.0])), 1.0)
        self.assertAlmostEqual(roc_auc(labels, np.ones(4)), 0.5)


if __name__ == "__main__":
    unittest.main()
