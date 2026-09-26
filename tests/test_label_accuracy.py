"""CPU-only tests for label-agnostic localization scoring and artifacts."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

from introspection_core import (
    InjectedTrial,
    LabelCandidateLayout,
    LabelAccuracyCounts,
    label_accuracy_outputs_match,
    resolve_label_candidate_layout,
    write_label_accuracy_outputs,
)


class CandidateLayoutTests(unittest.TestCase):
    def test_resolves_ordered_labels_and_token_ids(self) -> None:
        examples = [
            SimpleNamespace(
                positions=(0, 1),
                candidate_token_ids={"one": 11, "two": 12, "none": 13},
            ),
            SimpleNamespace(
                positions=(0, 1),
                candidate_token_ids={"one": 11, "two": 12, "none": 13},
            ),
        ]

        layout = resolve_label_candidate_layout(examples)

        self.assertEqual(layout.labels, ("one", "two", "none"))
        self.assertEqual(layout.token_ids, (11, 12, 13))
        self.assertEqual(layout.none_index, 2)

    def test_rejects_empty_or_inconsistent_examples(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            resolve_label_candidate_layout([])

        first = SimpleNamespace(
            positions=(0, 1),
            candidate_token_ids={"one": 11, "two": 12, "none": 13},
        )
        changed_token = SimpleNamespace(
            positions=(0, 1),
            candidate_token_ids={"one": 11, "two": 99, "none": 13},
        )
        with self.assertRaisesRegex(ValueError, "disagree"):
            resolve_label_candidate_layout([first, changed_token])

        duplicate = SimpleNamespace(
            positions=(0, 1),
            candidate_token_ids={"one": 11, "two": 11, "none": 13},
        )
        with self.assertRaisesRegex(ValueError, "distinct"):
            resolve_label_candidate_layout([duplicate])


class LabelAccuracyCountsTests(unittest.TestCase):
    def test_separates_correct_none_and_wrong_labels(self) -> None:
        layout = LabelCandidateLayout(("one", "two", "none"), (11, 12, 13), 2)
        trials = [
            InjectedTrial(0, 0, 0, False, False),
            InjectedTrial(0, 0, 1, False, False),
            InjectedTrial(1, 0, 0, False, False),
            InjectedTrial(1, 0, 1, False, False),
        ]
        counts = LabelAccuracyCounts()

        counts.update(trials, [0, 2, 1, 1], layout)

        self.assertEqual(counts.n_correct, 2)
        self.assertEqual(counts.n_none, 1)
        self.assertEqual(counts.predicted_labels, {"one": 1, "two": 2, "none": 1})
        self.assertEqual(counts.correct_by_concept, {0: 1, 1: 1})
        self.assertEqual(counts.none_by_position, {1: 1})

    def test_rejects_mismatched_batch_lengths_and_invalid_predictions(self) -> None:
        layout = LabelCandidateLayout(("one", "none"), (11, 13), 1)
        trial = InjectedTrial(0, 0, 0, False, False)
        counts = LabelAccuracyCounts()

        with self.assertRaisesRegex(ValueError, "equal length"):
            counts.update([trial], [], layout)
        with self.assertRaisesRegex(ValueError, "out of range"):
            counts.update([trial], [2], layout)


class ExistingOutputTests(unittest.TestCase):
    def test_requires_complete_matching_artifact_set(self) -> None:
        expected = {
            "model": "model-id",
            "split": "test",
            "prompt_template": "number-words",
            "cluster_csv": "clusters.csv",
            "concepts_json": "concepts.json",
            "concept_vectors": "vectors.pt",
            "injection_layer": 3,
            "strength": 2.0,
            "scale_mode": "relative_hidden_norm",
            "seed": 42,
        }
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            write_label_accuracy_outputs(
                output_dir,
                split_name="test",
                summary=expected,
                position_rows=[[0, "one", 1, 1, 0, 1.0]],
                concept_rows=[[0, "concept", 1, 1, 0, 1.0]],
            )

            self.assertTrue(
                label_accuracy_outputs_match(
                    output_dir,
                    split_name="test",
                    expected=expected,
                )
            )
            self.assertFalse(
                label_accuracy_outputs_match(
                    output_dir,
                    split_name="test",
                    expected={**expected, "strength": 3.0},
                )
            )
            (output_dir / "test_per_position.csv").unlink()
            self.assertFalse(
                label_accuracy_outputs_match(
                    output_dir,
                    split_name="test",
                    expected=expected,
                )
            )


if __name__ == "__main__":
    unittest.main()
