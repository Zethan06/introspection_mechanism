"""Tests for the shuffled-label localization control."""

from __future__ import annotations

import random
import unittest

from introspection_core.attention_inputs import _derangement
from introspection_core.injected_trials import InjectedTrial
from introspection_core.label_accuracy import (
    LabelAccuracyCounts,
    LabelAccuracyEvaluation,
    LabelAccuracyRunMetadata,
    LabelCandidateLayout,
    build_label_accuracy_summary,
    label_accuracy_cluster_rows,
    label_accuracy_position_rows,
)
from introspection_core.prompts import (
    LETTER_POSITION_LABELS,
    SEMANTIC_HIGHINJ_POSREF_GATE_BALANCED_DISRUPTS_LETTERS_SYSTEM_PROMPT,
    build_labeled_disrupts_template,
)


LABELS = ("A", "B", "C")
LAYOUT = LabelCandidateLayout(
    labels=(*LABELS, "none"), token_ids=(10, 11, 12, 13), position_count=3
)


def _metadata(permutation: str) -> LabelAccuracyRunMetadata:
    from pathlib import Path

    return LabelAccuracyRunMetadata(
        model="m",
        split="test",
        prompt_template="t",
        cluster_csv=Path("c.csv"),
        concepts_json=Path("c.json"),
        concept_vectors=Path("v.pt"),
        injection_layer=0,
        strength=1.0,
        scale_mode="unit",
        seed=42,
        prompt_preamble="system",
        choice_suffix="",
        dtype="bfloat16",
        max_concepts=None,
        max_clusters=None,
        label_permutation=permutation,
    )


class DerangementTests(unittest.TestCase):
    def test_is_a_permutation_without_fixed_points(self):
        rng = random.Random(0)
        labels = tuple("ABCDEFGHIJ")
        for _ in range(200):
            permuted = _derangement(labels, rng, key="k")
            self.assertEqual(sorted(permuted), sorted(labels))
            self.assertTrue(
                all(new != old for new, old in zip(permuted, labels, strict=True))
            )

    def test_seed_determines_the_permutation_bank(self):
        labels = tuple("ABCDEFGHIJ")
        first = [_derangement(labels, random.Random(7), key="k") for _ in range(3)]
        second = [_derangement(labels, random.Random(7), key="k") for _ in range(3)]
        self.assertEqual(first, second)

    def test_rejects_degenerate_label_sets(self):
        with self.assertRaises(ValueError):
            _derangement(("A",), random.Random(0), key="k")


class TemplateFactoryTests(unittest.TestCase):
    def test_display_labels_must_permute_the_canonical_set(self):
        with self.assertRaises(ValueError):
            build_labeled_disrupts_template(
                tuple("ABCDEFGHIK"), name="bad"
            )

    def test_rejects_a_system_prompt_for_a_different_label_set(self):
        # canonical_labels and system_prompt default independently, so a caller
        # overriding one and not the other must not get a prompt whose response
        # contract names a different label set than the candidate list.
        with self.assertRaises(ValueError) as caught:
            build_labeled_disrupts_template(
                tuple("1234567890"),
                name="digits-with-letter-contract",
                canonical_labels=tuple("0123456789"),
            )
        self.assertIn("never offers", str(caught.exception))

    def test_accepts_a_matching_label_set_and_system_prompt(self):
        digits = tuple("0123456789")
        template = build_labeled_disrupts_template(
            tuple("1234567890"),
            name="digits",
            canonical_labels=digits,
            system_prompt="Answer one of: "
            + ", ".join(f"`{d}`" for d in digits)
            + ", or `none`.",
        )
        self.assertEqual(template.candidate_labels(10), [*digits, "none"])

    def test_scored_candidates_stay_canonical_under_a_permutation(self):
        permuted = tuple("HIDACJBEFG")
        template = build_labeled_disrupts_template(permuted, name="shuffled")
        self.assertEqual(template.item_labels, permuted)
        # Answers are scored in canonical order regardless of display order, so
        # every cluster shares one candidate layout.
        self.assertEqual(
            template.candidate_labels(10), [*LETTER_POSITION_LABELS, "none"]
        )

    def test_permuting_moves_labels_but_not_wording_or_items(self):
        items = [f"w{index}" for index in range(10)]
        ascending = build_labeled_disrupts_template(
            LETTER_POSITION_LABELS, name="ascending"
        )
        shuffled = build_labeled_disrupts_template(
            tuple("HIDACJBEFG"), name="shuffled"
        )
        ascending_turns = ascending.turns(items, "system", "")
        shuffled_turns = shuffled.turns(items, "system", "")

        self.assertEqual(
            ascending_turns[0]["content"],
            SEMANTIC_HIGHINJ_POSREF_GATE_BALANCED_DISRUPTS_LETTERS_SYSTEM_PROMPT,
        )
        self.assertEqual(
            ascending_turns[0]["content"], shuffled_turns[0]["content"]
        )
        self.assertIn("TOKEN A: w0", ascending_turns[-1]["content"])
        self.assertIn("TOKEN H: w0", shuffled_turns[-1]["content"])
        # Items keep their slots; only the label beside them moves.
        for index, item in enumerate(items):
            ascending_slot = ascending_turns[-1]["content"].index(f": {item}")
            shuffled_slot = shuffled_turns[-1]["content"].index(f": {item}")
            self.assertEqual(
                ascending_turns[-1]["content"][:ascending_slot].count("TOKEN "),
                shuffled_turns[-1]["content"][:shuffled_slot].count("TOKEN "),
                f"item {index} changed slots",
            )


class PermutedScoringTests(unittest.TestCase):
    def test_correctness_follows_the_per_cluster_answer_key(self):
        counts = LabelAccuracyCounts()
        # Cluster 0 is the identity; cluster 1 answers slot 0 with "C".
        expected = ((0, 1, 2), (2, 0, 1))
        trials = [
            InjectedTrial(0, 0, 0, False, False),
            InjectedTrial(0, 1, 0, False, False),
            InjectedTrial(0, 1, 0, False, False),
        ]
        # Predict "A" for cluster 0 slot 0 (right), then "A" and "C" for
        # cluster 1 slot 0 (wrong, then right).
        counts.update(trials, [0, 0, 2], LAYOUT, expected_index_by_cluster=expected)
        self.assertEqual(counts.n_correct, 2)
        self.assertEqual(counts.correct_by_cluster[0], 1)
        self.assertEqual(counts.correct_by_cluster[1], 1)
        self.assertEqual(counts.trials_by_cluster[1], 2)

    def test_identity_key_matches_the_default_behaviour(self):
        trials = [InjectedTrial(0, 0, 1, False, False)]
        without = LabelAccuracyCounts()
        without.update(trials, [1], LAYOUT)
        with_key = LabelAccuracyCounts()
        with_key.update(trials, [1], LAYOUT, expected_index_by_cluster=((0, 1, 2),))
        self.assertEqual(without.n_correct, with_key.n_correct)

    def test_counting_answers_are_tallied_separately(self):
        counts = LabelAccuracyCounts()
        # Cluster 0 answers slot 0 with "C"; a counting strategy would say "A".
        expected = ((2, 0, 1),)
        counts.update(
            [
                InjectedTrial(0, 0, 0, False, False),
                InjectedTrial(0, 0, 0, False, False),
            ],
            [0, 2],
            LAYOUT,
            expected_index_by_cluster=expected,
        )
        self.assertEqual(counts.n_counting, 1)
        self.assertEqual(counts.n_correct, 1)
        self.assertEqual(counts.counting_by_position[0], 1)

    def test_counting_and_correct_coincide_under_the_identity_key(self):
        counts = LabelAccuracyCounts()
        counts.update(
            [InjectedTrial(0, 0, 1, False, False)],
            [1],
            LAYOUT,
            expected_index_by_cluster=((0, 1, 2),),
        )
        self.assertEqual(counts.n_counting, counts.n_correct)

    def test_none_is_tallied_against_the_permuted_key(self):
        counts = LabelAccuracyCounts()
        counts.update(
            [InjectedTrial(0, 0, 0, False, False)],
            [LAYOUT.none_index],
            LAYOUT,
            expected_index_by_cluster=((2, 0, 1),),
        )
        self.assertEqual(counts.n_correct, 0)
        self.assertEqual(counts.n_none, 1)


class ReportingTests(unittest.TestCase):
    def _evaluation(self, expected, clean_correct=2) -> LabelAccuracyEvaluation:
        counts = LabelAccuracyCounts()
        for cluster, key in enumerate(expected):
            for position in range(3):
                counts.update(
                    [InjectedTrial(0, cluster, position, False, False)],
                    [key[position]],
                    LAYOUT,
                    expected_index_by_cluster=expected,
                )
        return LabelAccuracyEvaluation(
            layout=LAYOUT,
            clean_predicted_labels={"none": clean_correct},
            counts=counts,
            n_trials=len(expected) * 3,
            n_clusters=len(expected),
            expected_index_by_cluster=expected,
            clean_correct=clean_correct,
        )

    def test_is_permuted_detects_a_non_identity_key(self):
        self.assertFalse(self._evaluation(((0, 1, 2), (0, 1, 2))).is_permuted)
        self.assertTrue(self._evaluation(((0, 1, 2), (2, 0, 1))).is_permuted)

    def test_position_rows_refuse_to_name_a_label_when_permuted(self):
        rows = label_accuracy_position_rows(self._evaluation(((2, 0, 1), (1, 2, 0))))
        self.assertEqual([row[1] for row in rows], ["*", "*", "*"])
        rows = label_accuracy_position_rows(self._evaluation(((0, 1, 2), (0, 1, 2))))
        self.assertEqual([row[1] for row in rows], ["A", "B", "C"])

    def test_position_rows_carry_the_counting_rate(self):
        # Every prediction follows the permuted key, so nothing looks like a
        # counting answer.
        rows = label_accuracy_position_rows(self._evaluation(((2, 0, 1), (1, 2, 0))))
        self.assertEqual([row[-1] for row in rows], [0.0, 0.0, 0.0])
        # Under the identity key the two coincide by construction.
        rows = label_accuracy_position_rows(self._evaluation(((0, 1, 2), (0, 1, 2))))
        self.assertEqual([row[-1] for row in rows], [1.0, 1.0, 1.0])

    def test_cluster_rows_record_each_display_order(self):
        rows = label_accuracy_cluster_rows(self._evaluation(((2, 0, 1), (1, 2, 0))))
        # Separated rather than concatenated, so multi-character labels such as
        # the number words stay parseable.
        self.assertEqual([row[1] for row in rows], ["C|A|B", "B|C|A"])

    def test_clean_arm_is_reported_on_the_cluster_denominator(self):
        evaluation = self._evaluation(((2, 0, 1), (1, 2, 0)), clean_correct=1)
        summary = build_label_accuracy_summary(
            evaluation, metadata=_metadata("shuffled"), concept_count=1
        )
        self.assertEqual(summary["n_clean_trials"], 2)
        self.assertEqual(summary["n_clean_correct"], 1)
        self.assertAlmostEqual(summary["clean_none_rate"], 0.5)
        # The injected arm keeps its own, much larger denominator.
        self.assertEqual(summary["n_trials"], 6)

    def test_summary_rejects_a_mislabelled_run(self):
        evaluation = self._evaluation(((2, 0, 1), (1, 2, 0)))
        with self.assertRaises(ValueError):
            build_label_accuracy_summary(
                evaluation, metadata=_metadata("identity"), concept_count=1
            )


if __name__ == "__main__":
    unittest.main()
