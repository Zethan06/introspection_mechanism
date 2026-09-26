"""Pure metric tests for the donor-position by forced-router grid."""

from __future__ import annotations

import unittest

import torch

from introspection_core.donor_router_mismatch import (
    DonorRouterAccumulator,
    aggregate_position_rows,
    router_source_indices,
    sharded_complete_position_batches,
)
from introspection_core.injected_trials import InjectedTrial


class DonorRouterAccumulatorTests(unittest.TestCase):
    @staticmethod
    def _trial(concept: int, cluster: int, position: int) -> InjectedTrial:
        return InjectedTrial(concept, cluster, position, False, False)

    def test_distinguishes_donor_router_none_and_other_number(self) -> None:
        logits = torch.full((4, 11), -5.0)
        logits[0, 2] = 5.0
        logits[1, 7] = 5.0
        logits[2, 10] = 5.0
        logits[3, 4] = 5.0
        stats = DonorRouterAccumulator(donor_position=2, router_position=7)
        stats.update(logits, gate_temperature=1.0)
        row = stats.row(intervention="patched")
        self.assertEqual(row["position_relation"], "mismatch")
        self.assertAlmostEqual(float(row["none_rate"]), 0.25)
        self.assertAlmostEqual(float(row["number_rate"]), 0.75)
        self.assertAlmostEqual(float(row["overall_donor_accuracy"]), 0.25)
        self.assertAlmostEqual(float(row["overall_router_accuracy"]), 0.25)
        self.assertAlmostEqual(float(row["other_number_rate"]), 0.25)
        self.assertNotIn("follow_donor_given_number", row)
        self.assertNotIn("follow_router_given_number", row)

    def test_tensor_round_trip_preserves_counts(self) -> None:
        logits = torch.full((2, 11), -5.0)
        logits[0, 1] = 5.0
        logits[1, 6] = 5.0
        original = DonorRouterAccumulator(donor_position=1, router_position=6)
        original.update(logits, gate_temperature=0.5)
        restored = DonorRouterAccumulator.from_tensor(
            original.tensor("cpu"), donor_position=1, router_position=6
        )
        self.assertEqual(restored.prediction_counts, original.prediction_counts)
        self.assertEqual(restored.n, original.n)

    def test_aggregate_is_trial_weighted(self) -> None:
        rows = [
            {
                "intervention": "patched",
                "position_relation": "mismatch",
                "n_trials": n,
                **{
                    metric: value
                    for metric, value in {
                        "none_rate": 0.0,
                        "number_rate": 1.0,
                        "overall_donor_accuracy": donor,
                        "overall_router_accuracy": 1.0 - donor,
                        "other_number_rate": 0.0,
                        "mean_none_probability": 0.0,
                        "mean_number_probability": 1.0,
                        "mean_donor_probability": donor,
                        "mean_router_probability": 1.0 - donor,
                        "mean_gate_score": 1.0,
                    }.items()
                },
            }
            for n, donor in ((1, 1.0), (3, 0.0))
        ]
        summary = aggregate_position_rows(rows)[0]
        self.assertAlmostEqual(float(summary["overall_donor_accuracy"]), 0.25)
        self.assertAlmostEqual(float(summary["overall_router_accuracy"]), 0.75)
        self.assertNotIn("follow_donor_given_number", summary)
        self.assertNotIn("follow_router_given_number", summary)

    def test_complete_batches_preserve_all_positions_per_coordinate(self) -> None:
        trials = [
            self._trial(concept, 0, position)
            for concept in range(5)
            for position in range(10)
        ]
        shards = [
            sharded_complete_position_batches(
                trials,
                batch_size=32,
                rank=rank,
                world_size=2,
            )
            for rank in range(2)
        ]
        flattened = [trial for shard in shards for batch in shard for trial in batch]
        self.assertEqual(len(flattened), len(trials))
        self.assertEqual(set(flattened), set(trials))
        for shard in shards:
            for batch in shard:
                by_coordinate: dict[tuple[int, int], set[int]] = {}
                for trial in batch:
                    key = (trial.concept_index, trial.cluster_index)
                    by_coordinate.setdefault(key, set()).add(trial.position)
                self.assertTrue(
                    all(
                        positions == set(range(10))
                        for positions in by_coordinate.values()
                    )
                )

    def test_router_source_indices_select_paired_position(self) -> None:
        trials = [
            self._trial(concept, 0, position)
            for concept in range(2)
            for position in range(10)
        ]
        indices = router_source_indices(trials, router_position=7)
        selected = [trials[index] for index in indices.tolist()]
        self.assertTrue(all(trial.position == 7 for trial in selected))
        self.assertEqual(
            [trial.concept_index for trial in selected],
            [trial.concept_index for trial in trials],
        )


if __name__ == "__main__":
    unittest.main()
