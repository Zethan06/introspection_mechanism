import unittest

import numpy as np

from introspection_core.prompt_search_sampling import (
    contiguous_shard_bounds,
    geometric_coreset_order,
    geometric_panel_weights,
    nested_stratified_order,
)


class PromptSearchSamplingTests(unittest.TestCase):
    def test_contiguous_shards_cover_without_overlap(self) -> None:
        bounds = [contiguous_shard_bounds(10, index, 3) for index in range(3)]
        self.assertEqual(bounds, [(0, 3), (3, 6), (6, 10)])
        covered = [item for start, end in bounds for item in range(start, end)]
        self.assertEqual(covered, list(range(10)))

    def test_order_is_deterministic_complete_and_nested(self):
        rows = [
            {
                "token_id": index * 3 + 1,
                "word": ("Word" if index % 5 == 0 else "word") + str(index),
                "word_len": index % 12 + 1,
                "baseline_difficulty": (index % 17) / 16,
            }
            for index in range(200)
        ]
        first, annotations = nested_stratified_order(rows, seed=7)
        second, _ = nested_stratified_order(rows, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(sorted(first), list(range(len(rows))))
        self.assertEqual(len(annotations), len(rows))
        self.assertTrue(set(first[:32]).issubset(first[:64]))

    def test_rejects_duplicate_token_ids(self):
        rows = [
            {"token_id": 1, "word": "one", "baseline_difficulty": 0.0},
            {"token_id": 1, "word": "two", "baseline_difficulty": 1.0},
        ]
        with self.assertRaisesRegex(ValueError, "unique"):
            nested_stratified_order(rows)

    def test_geometric_panels_are_nested_and_cover_all_weight(self):
        rng = np.random.default_rng(7)
        vectors = np.concatenate(
            [
                rng.normal(loc=(-3.0, 0.0), scale=0.2, size=(12, 2)),
                rng.normal(loc=(0.0, 3.0), scale=0.2, size=(12, 2)),
                rng.normal(loc=(3.0, 0.0), scale=0.2, size=(12, 2)),
            ],
            axis=0,
        ).astype(np.float32)
        coreset = geometric_coreset_order(
            vectors, max_size=8, projection_dim=2, seed=11, batch_size=16
        )
        medoids = coreset["medoid_indices"]
        order = coreset["center_order"]
        panel_four = medoids[order[:4]]
        panel_eight = medoids[order[:8]]
        self.assertEqual(len(set(panel_eight.tolist())), 8)
        self.assertTrue(
            set(panel_four.tolist()).issubset(panel_eight.tolist())
        )

        weights, coverage = geometric_panel_weights(
            coreset["fine_centers"],
            coreset["fine_counts"],
            order[:4],
        )
        self.assertEqual(int(weights.sum()), vectors.shape[0])
        self.assertEqual(weights.shape, (4,))
        self.assertEqual(coverage.shape, (8,))
        self.assertTrue(np.all(coverage >= 0))


if __name__ == "__main__":
    unittest.main()
