"""Tests for the helpers that rebuild paper inputs: gate heads, C_nonintro, lexical plans."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import torch

from introspection_core.attention_inputs import _derangement
from scripts import run_lexical_replacement_control as lexical
from scripts.export_gate_heads import head_rows


class ExportGateHeadsTests(unittest.TestCase):
    def test_marks_exactly_the_selected_heads_of_the_window(self) -> None:
        hard = torch.zeros(2, 4)
        hard[0, 1] = hard[1, 3] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "head_mask.pt"
            torch.save({"layers": [7, 8], "n_heads": 4, "top_k": 2, "hard_mask": hard}, path)
            rows = head_rows("model", path)

        self.assertEqual(len(rows), 8)
        self.assertEqual(
            [(row["layer"], row["head"]) for row in rows if row["is_ste"]],
            [(7, 1), (8, 3)],
        )

    def test_rejects_a_mask_that_does_not_select_top_k(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "head_mask.pt"
            torch.save({"layers": [0], "n_heads": 2, "top_k": 2, "hard_mask": torch.ones(1, 1)}, path)
            with self.assertRaises(ValueError):
                head_rows("model", path)


class SampleLowAccuracyConceptsTests(unittest.TestCase):
    def test_bottom_pool_and_seeded_sample(self) -> None:
        metrics = pd.DataFrame(
            {
                "concept": [f"c{index}" for index in range(8)],
                "injected_argmax_accuracy": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
                "injected_mean_correct_prob": [0.5] * 8,
                "accuracy_gain_over_clean": [0.0] * 8,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            concepts = root / "dataset/concepts"
            concepts.mkdir(parents=True)
            (concepts / "selected.json").write_text(
                json.dumps(
                    {"concept_vector_words": ["c0", "c1"], "baseline_words": ["w"], "baseline_mode": "full_english"}
                )
            )
            pd.DataFrame({"concept": [f"c{i}" for i in range(8)]}).to_csv(
                concepts / "shortlist.csv", index=False
            )
            # Lower-scoring vocabulary outside the shortlist must never enter C_nonintro.
            metrics.loc[len(metrics)] = ["outside", 0.0, 0.0, 0.0]
            metrics_path = root / "metrics.csv"
            metrics.sample(frac=1, random_state=0).to_csv(metrics_path, index=False)
            argv = [
                "sample_low_accuracy_concepts.py",
                "--screening_metrics", str(metrics_path),
                "--dataset_dir", str(root / "dataset"),
                "--expected_candidate_count", "8",
                "--pool_size", "4",
                "--sample_size", "2",
                "--seed", "42",
            ]
            from scripts import sample_low_accuracy_concepts

            with patch("sys.argv", argv):
                sample_low_accuracy_concepts.main()
            with patch("sys.argv", argv + ["--expected_candidate_count", "3000"]):
                with self.assertRaisesRegex(ValueError, "shortlist must contain 3000"):
                    sample_low_accuracy_concepts.main()
            metrics[metrics["concept"] != "c7"].to_csv(metrics_path, index=False)
            with patch("sys.argv", argv):
                with self.assertRaisesRegex(ValueError, "missing 1 shortlisted"):
                    sample_low_accuracy_concepts.main()
            pool = pd.read_csv(concepts / "bottom4.csv")["concept"].tolist()
            sample = pd.read_csv(concepts / "bottom2_seed42.csv")["concept"].tolist()

        self.assertEqual(pool, ["c4", "c5", "c6", "c7"])
        self.assertEqual(sample, random.Random(42).sample(pool, 2))


class LexicalReplacementControlTests(unittest.TestCase):
    def test_identity_arm_prints_canonical_labels(self) -> None:
        _, display = lexical.arm_layout("letters_identity", 3)
        self.assertEqual(display, tuple("ABCDEFGHIJ"))

    def test_shuffled_arm_matches_the_shuffled_accuracy_derangement(self) -> None:
        canonical = tuple(str(index) for index in range(10))
        rng = random.Random(42)
        expected = [_derangement(canonical, rng, key=str(index)) for index in range(3)]
        for cluster_id in range(3):
            _, display = lexical.arm_layout("digits_shuffled", cluster_id)
            self.assertEqual(display, expected[cluster_id])
            self.assertTrue(all(new != old for new, old in zip(display, canonical)))

    def test_plan_pairs_concept_and_random_words_and_skips_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in ("concepts", "clusters", "vocabulary"):
                (root / folder).mkdir()
            (root / "concepts/validation.json").write_text(
                json.dumps({"concept_vector_words": ["alpha", "w1"]})
            )
            with (root / "clusters/test.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["cluster_key", "choices"])
                writer.writeheader()
                writer.writerow({"cluster_key": "k", "choices": json.dumps([f"w{i}" for i in range(10)])})
            with (root / "vocabulary/english.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["word"])
                writer.writeheader()
                writer.writerows({"word": word} for word in ["x", "y", "z", "w0"])
            args = type("Args", (), {"dataset_dir": root, "output_dir": root / "out", "seed": 42})
            lexical.build_plan(args)
            plan = json.loads((root / "out/plan.json").read_text())
            protocol = json.loads((root / "out/protocol.json").read_text())

        # "w1" equals the original word at slot 1, so that pair is skipped.
        self.assertEqual(protocol["skipped_unchanged"], [[1, 0, 1]])
        self.assertEqual(len(plan), 2 * (10 + 9))
        for concept_row, random_row in zip(plan[::2], plan[1::2]):
            self.assertEqual(concept_row["condition"], "concept_word")
            self.assertEqual(random_row["condition"], "random_word")
            self.assertNotIn(random_row["replacement_word"], concept_row["choices"])



class PaperTableTests(unittest.TestCase):
    def test_transition_layer_is_the_largest_rise(self) -> None:
        import numpy as np

        from scripts.plot_manuscript_figures import transition_layer

        layers = np.array([2, 0, 1, 3])
        accuracy = np.array([80.0, 10.0, 30.0, 85.0])
        self.assertEqual(transition_layer(layers, accuracy), 2)
        with self.assertRaises(ValueError):
            transition_layer(np.array([0, 2]), np.array([1.0, 2.0]))

    def test_task_performance_reads_grid_baselines_and_label_runs(self) -> None:
        from scripts import summarize_task_performance as table

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            top32 = root / "ste_topk_sweep/top32"
            for direction, condition, value in (
                ("off", "injected_baseline", "0.25"),
                ("on", "clean_baseline", "0.9"),
            ):
                (top32 / f"test_{direction}").mkdir(parents=True)
                with (top32 / f"test_{direction}/results.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(
                        handle, fieldnames=["condition", "router", "none_rate", "exact_target_accuracy"])
                    writer.writeheader()
                    writer.writerow({"condition": condition, "router": "native",
                                     "none_rate": value, "exact_target_accuracy": value})
                    writer.writerow({"condition": condition, "router": "clean_patch",
                                     "none_rate": "0", "exact_target_accuracy": "0"})
            run = root / "label_shuffle/shuffled_letters_a_j"
            run.mkdir(parents=True)
            (run / "test_summary.json").write_text(json.dumps(
                {"accuracy": 0.5, "clean_predicted_label_counts": {"none": 3, "A": 1}}))

            self.assertEqual(table.digit_rates(root), (0.25, 0.9))
            self.assertEqual(table.label_rates(run), (0.5, 0.75))


if __name__ == "__main__":
    unittest.main()
