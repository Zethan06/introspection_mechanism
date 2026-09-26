"""CPU tests for the random-k head control sampling and aggregation."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

from introspection_core.random_head_controls import (
    candidate_components,
    components_to_mask,
    draw_seed,
    sample_random_components,
    summarize_draws,
)

# `scripts` is not a package, so load the evaluation module by file path.
_SPEC = importlib.util.spec_from_file_location(
    "evaluate_random_topk_head_controls",
    Path(__file__).resolve().parents[1]
    / "scripts/evaluate_random_topk_head_controls.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
build_draws, aggregate_draw_rows = _MODULE.build_draws, _MODULE.aggregate_draw_rows


class SampleRandomComponentsTests(unittest.TestCase):
    def test_pool_covers_every_candidate_layer_and_head(self) -> None:
        pool = candidate_components(layers=[13, 14], n_heads=3)

        self.assertEqual(len(pool), 6)
        self.assertEqual(pool[0], (13, 0))
        self.assertEqual(pool[-1], (14, 2))

    def test_draw_is_reproducible_and_distinct(self) -> None:
        first = sample_random_components(
            layers=[17, 18, 19], n_heads=8, top_k=6, seed=7
        )
        again = sample_random_components(
            layers=[17, 18, 19], n_heads=8, top_k=6, seed=7
        )
        other = sample_random_components(
            layers=[17, 18, 19], n_heads=8, top_k=6, seed=8
        )

        self.assertEqual(first, again)
        self.assertNotEqual(first, other)
        self.assertEqual(len(set(first)), 6)
        self.assertEqual(first, sorted(first))

    def test_draw_can_use_the_whole_pool_but_not_more(self) -> None:
        full = sample_random_components(layers=[5], n_heads=4, top_k=4, seed=1)

        self.assertEqual(len(full), 4)
        with self.assertRaises(ValueError):
            sample_random_components(layers=[5], n_heads=4, top_k=5, seed=1)

    def test_sampling_is_not_restricted_to_one_layer(self) -> None:
        # A uniform pool draw should reach every candidate layer across seeds,
        # which a per-layer matched scheme would also do but a pool bug would
        # not.
        layers = set()
        for seed in range(20):
            layers.update(
                layer
                for layer, _ in sample_random_components(
                    layers=[17, 18, 19], n_heads=4, top_k=2, seed=seed
                )
            )

        self.assertEqual(layers, {17, 18, 19})

    def test_mask_marks_exactly_the_sampled_heads(self) -> None:
        components = [(17, 1), (19, 3)]
        mask = components_to_mask(components, layers=[17, 18, 19], n_heads=4)

        self.assertEqual(tuple(mask.shape), (3, 4))
        self.assertEqual(int(mask.sum()), 2)
        self.assertTrue(bool(mask[0, 1]))
        self.assertTrue(bool(mask[2, 3]))

    def test_mask_rejects_components_outside_the_pool(self) -> None:
        with self.assertRaises(ValueError):
            components_to_mask([(20, 0)], layers=[17, 18], n_heads=4)
        with self.assertRaises(ValueError):
            components_to_mask([(17, 9)], layers=[17, 18], n_heads=4)
        with self.assertRaises(ValueError):
            components_to_mask([(17, 0), (17, 0)], layers=[17, 18], n_heads=4)

    def test_draw_seed_ignores_direction_but_separates_k_and_draw(self) -> None:
        self.assertNotEqual(
            draw_seed(base_seed=42, top_k=4, draw=0),
            draw_seed(base_seed=42, top_k=8, draw=0),
        )
        self.assertNotEqual(
            draw_seed(base_seed=42, top_k=4, draw=0),
            draw_seed(base_seed=42, top_k=4, draw=1),
        )


class SummarizeDrawsTests(unittest.TestCase):
    def test_interval_brackets_the_mean(self) -> None:
        statistics = summarize_draws([0.2, 0.4, 0.6, 0.8])

        self.assertAlmostEqual(statistics.mean, 0.5)
        self.assertEqual(statistics.n, 4)
        self.assertLess(statistics.ci_low, statistics.mean)
        self.assertGreater(statistics.ci_high, statistics.mean)
        self.assertAlmostEqual(statistics.minimum, 0.2)
        self.assertAlmostEqual(statistics.maximum, 0.8)

    def test_single_draw_reports_a_degenerate_interval(self) -> None:
        statistics = summarize_draws([0.3])

        self.assertAlmostEqual(statistics.ci_low, 0.3)
        self.assertAlmostEqual(statistics.ci_high, 0.3)
        self.assertAlmostEqual(statistics.std, 0.0)

    def test_identical_draws_give_a_zero_width_interval(self) -> None:
        statistics = summarize_draws([0.25] * 5)

        self.assertAlmostEqual(statistics.ci_low, 0.25)
        self.assertAlmostEqual(statistics.ci_high, 0.25)


class BuildDrawsTests(unittest.TestCase):
    def test_enumerates_every_cardinality_and_repeat(self) -> None:
        draws = build_draws(
            layers=[17, 18, 19],
            n_heads=8,
            top_k_values=[1, 4],
            repeats=3,
            base_seed=42,
        )

        self.assertEqual(len(draws), 6)
        self.assertEqual({item.top_k for item in draws}, {1, 4})
        self.assertEqual(
            sorted(item.draw for item in draws if item.top_k == 4), [0, 1, 2]
        )
        for item in draws:
            self.assertEqual(len(item.components), item.top_k)

    def test_rejects_a_cardinality_larger_than_the_pool(self) -> None:
        with self.assertRaisesRegex(ValueError, "candidate pool"):
            build_draws(
                layers=[17],
                n_heads=4,
                top_k_values=[8],
                repeats=2,
                base_seed=42,
            )

    def test_rejects_a_pool_with_too_few_distinct_sets(self) -> None:
        # Two draws of the whole pool are necessarily the same head set, so the
        # "independent draws" the interval assumes would not exist.
        with self.assertRaisesRegex(ValueError, "fewer distinct Top4 sets"):
            build_draws(
                layers=[17],
                n_heads=4,
                top_k_values=[4],
                repeats=2,
                base_seed=42,
            )

    def test_keeps_draws_that_coincide_by_chance(self) -> None:
        # Single-head draws from a small pool collide often. Dropping or
        # resampling them would bias the control away from uniform sampling,
        # so identical draws are kept.
        draws = build_draws(
            layers=[17, 18],
            n_heads=4,
            top_k_values=[1],
            repeats=8,
            base_seed=3,
        )

        self.assertEqual(len(draws), 8)
        self.assertLess(len({item.components for item in draws}), 8)


class AggregateDrawRowsTests(unittest.TestCase):
    def _rows(self, *, values: dict[int, list[float]]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for top_k, rates in values.items():
            for draw, rate in enumerate(rates):
                for transition in ("none_to_number", "number_to_none"):
                    rows.append(
                        {
                            "top_k": top_k,
                            "draw": draw,
                            "transition": transition,
                            "conversion_rate": rate,
                            "target_accuracy_before": 0.1,
                            "target_accuracy_after": rate,
                            "target_accuracy_delta": rate - 0.1,
                        }
                    )
        return rows

    def test_summarizes_both_directions_per_cardinality(self) -> None:
        rows = self._rows(values={4: [0.2, 0.4, 0.6], 8: [0.5, 0.5, 0.5]})

        summary = aggregate_draw_rows(rows, top_k_values=[4, 8], repeats=3)

        self.assertEqual([row["top_k"] for row in summary], [4, 8])
        self.assertAlmostEqual(
            summary[0]["none_to_number_target_accuracy_after_mean"], 0.4
        )
        self.assertAlmostEqual(
            summary[1]["number_to_none_target_accuracy_after_mean"], 0.5
        )
        self.assertAlmostEqual(
            summary[1]["number_to_none_target_accuracy_after_ci_low"], 0.5
        )
        self.assertEqual(summary[0]["n_draws"], 3)

    def test_rejects_a_missing_draw(self) -> None:
        rows = self._rows(values={4: [0.2, 0.4]})

        with self.assertRaisesRegex(ValueError, "expected 3"):
            aggregate_draw_rows(rows, top_k_values=[4], repeats=3)


if __name__ == "__main__":
    unittest.main()
