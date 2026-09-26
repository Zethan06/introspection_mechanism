"""Tests for full-English-vocabulary injection word search orchestration."""

from __future__ import annotations

import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "search_vocab_injection_words",
    ROOT / "scripts" / "search_vocab_injection_words.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def parse_search_args(argv):
    stage = "fine" if "--candidate_words_csv" in argv else "coarse"
    results_value = argv[argv.index("--results_dir") + 1]
    results_path = Path(results_value)
    canonical = [
        "--dataset_dir",
        str(results_path.parent / "dataset"),
        "--screening_stage",
        stage,
        "--work_dir",
        results_value,
    ]
    if "--prompt_templates" not in argv:
        canonical.extend(
            ["--prompt_templates", "semantic_highinj_posref_gate_balanced_disrupts"]
        )
    return MODULE.parse_args([*argv, *canonical])


class VocabInjectionSearchCliTests(unittest.TestCase):
    def test_merge_source_csv_preserves_na_like_concept_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "worker.csv"
            path.write_text("concept,token_id\nNA,1\nNone,2\nnull,3\nnan,4\n")

            frame = MODULE._read_csv_preserving_strings(path)

        self.assertEqual(
            frame["concept"].tolist(),
            ["NA", "None", "null", "nan"],
        )

    def test_canonical_cli_uses_one_prompt_template(self) -> None:
        args = parse_search_args(
            [
                "--model",
                "model",
                "--cluster_csv",
                "clusters.csv",
                "--results_dir",
                "results",
            ]
        )

        self.assertEqual(args.layer, 4)
        self.assertEqual(args.strength, 3.0)
        self.assertEqual(args.batch_size, 64)
        self.assertEqual(args.gpus, "0,1,2,3,4,5")
        self.assertEqual(args.baseline_mode, "full_english")
        self.assertIsNone(args.baseline_words_json)
        self.assertIsNone(args.calibration_concepts_json)
        self.assertIsNone(args.candidate_words_csv)
        self.assertEqual(args.candidate_column, "concept")
        self.assertEqual(args.expected_candidate_count, 3000)
        self.assertEqual(args.prompt_templates, ["semantic_highinj_posref_gate_balanced_disrupts"])

    def test_runner_forwards_prompt_and_worker_count(self) -> None:
        args = parse_search_args(
            [
                "--model",
                "model",
                "--cluster_csv",
                "clusters.csv",
                "--results_dir",
                "results",
                "--num_workers",
                "6",
                "--max_clusters",
                "3",
            ]
        )

        forwarded = MODULE._forwarded_args(args)

        self.assertIn("--prompt_templates", forwarded)
        self.assertEqual(forwarded[forwarded.index("--num_workers") + 1], "6")
        self.assertEqual(forwarded[forwarded.index("--max_clusters") + 1], "3")
        self.assertIn("semantic_highinj_posref_gate_balanced_disrupts", forwarded)

    def test_rejects_unregistered_prompt(self) -> None:
        with self.assertRaises(SystemExit):
            parse_search_args(
                [
                    "--model",
                    "model",
                    "--cluster_csv",
                    "clusters.csv",
                    "--results_dir",
                    "results",
                    "--prompt_templates",
                    "semantic_highinj_posref_gate_balanced_disrupts_tokens_1_9",
                ]
            )

    def test_runner_forwards_explicit_baseline_manifest(self) -> None:
        args = parse_search_args(
            [
                "--model",
                "model",
                "--cluster_csv",
                "clusters.csv",
                "--results_dir",
                "results",
                "--baseline_words_json",
                "concepts.json",
            ]
        )

        forwarded = MODULE._forwarded_args(args)

        self.assertEqual(args.baseline_mode, "manifest")
        mode_index = forwarded.index("--baseline_mode")
        self.assertEqual(forwarded[mode_index + 1], "manifest")
        index = forwarded.index("--baseline_words_json")
        self.assertEqual(forwarded[index + 1], "concepts.json")

    def test_runner_forwards_evaluation_only_candidate_filter(self) -> None:
        args = parse_search_args(
            [
                "--model",
                "model",
                "--cluster_csv",
                "clusters.csv",
                "--results_dir",
                "results",
                "--calibration_concepts_json",
                "calibration.json",
                "--candidate_words_csv",
                "top3000.csv",
                "--candidate_column",
                "word",
            ]
        )

        forwarded = MODULE._forwarded_args(args)

        self.assertEqual(
            forwarded[forwarded.index("--calibration_concepts_json") + 1],
            "calibration.json",
        )
        self.assertEqual(
            forwarded[forwarded.index("--candidate_words_csv") + 1],
            "top3000.csv",
        )
        self.assertEqual(
            forwarded[forwarded.index("--candidate_column") + 1],
            "word",
        )
        self.assertEqual(
            forwarded[forwarded.index("--expected_candidate_count") + 1],
            "3000",
        )

    def test_candidate_filter_requires_calibration_panel(self) -> None:
        with self.assertRaises(SystemExit):
            parse_search_args(
                [
                    "--model",
                    "model",
                    "--cluster_csv",
                    "clusters.csv",
                    "--results_dir",
                    "results",
                    "--candidate_words_csv",
                    "top3000.csv",
                ]
            )

    def test_rejects_manifest_path_with_explicit_full_english_mode(self) -> None:
        with self.assertRaises(SystemExit):
            parse_search_args(
                [
                    "--model",
                    "model",
                    "--cluster_csv",
                    "clusters.csv",
                    "--results_dir",
                    "results",
                    "--baseline_mode",
                    "full_english",
                    "--baseline_words_json",
                    "concepts.json",
                ]
            )

    def test_requires_manifest_path_for_manifest_mode(self) -> None:
        with self.assertRaises(SystemExit):
            parse_search_args(
                [
                    "--model",
                    "model",
                    "--cluster_csv",
                    "clusters.csv",
                    "--results_dir",
                    "results",
                    "--baseline_mode",
                    "manifest",
                ]
            )

    def test_resume_rejects_cached_manifest_baseline_for_full_english(self) -> None:
        args = parse_search_args(
            [
                "--model",
                "model",
                "--cluster_csv",
                "clusters.csv",
                "--results_dir",
                "results",
            ]
        )
        args.num_workers = 1
        cached = {
            "model": "model",
            "layer": 4,
            "num_workers": 1,
            "case_filter": "all",
            "min_word_len": 1,
            "max_word_len": 32,
            "baseline_mode": "manifest",
            "baseline_source_path": "concepts.json",
        }

        with self.assertRaisesRegex(
            ValueError,
            "never reuse vectors across baseline definitions",
        ):
            MODULE._validate_resume_manifest(cached, args)

    def test_resume_rejects_legacy_manifest_without_truncation_tracking(self) -> None:
        args = parse_search_args(
            [
                "--model",
                "model",
                "--cluster_csv",
                "clusters.csv",
                "--results_dir",
                "results",
            ]
        )
        args.num_workers = 1
        cached = {
            "model": "model",
            "layer": 4,
            "num_workers": 1,
            "case_filter": "all",
            "min_word_len": 1,
            "max_word_len": 32,
            "baseline": (
                "mean activation over the complete eligible English vocabulary"
            ),
        }

        with self.assertRaisesRegex(
            ValueError,
            "cannot prove complete-vocabulary membership",
        ):
            MODULE._validate_resume_manifest(cached, args)

    def test_resume_accepts_current_untruncated_full_english_manifest(self) -> None:
        args = parse_search_args(
            [
                "--model",
                "model",
                "--cluster_csv",
                "clusters.csv",
                "--results_dir",
                "results",
            ]
        )
        args.num_workers = 1
        cached = {
            "model": "model",
            "layer": 4,
            "num_workers": 1,
            "case_filter": "all",
            "min_word_len": 1,
            "max_word_len": 32,
            "baseline_mode": "full_english",
            "baseline_source_path": None,
            "max_tokens": None,
        }

        MODULE._validate_resume_manifest(cached, args)

    def test_resume_rejects_incomplete_full_english_baseline(self) -> None:
        args = parse_search_args(
            [
                "--model",
                "model",
                "--cluster_csv",
                "clusters.csv",
                "--results_dir",
                "results",
            ]
        )
        args.num_workers = 1
        cached = {
            "model": "model",
            "layer": 4,
            "num_workers": 1,
            "case_filter": "all",
            "min_word_len": 1,
            "max_word_len": 32,
            "baseline_mode": "full_english",
            "baseline_word_count": 100,
            "english_word_count": 40_000,
            "baseline_source_path": None,
        }

        with self.assertRaisesRegex(
            ValueError,
            "full-English baseline count differs",
        ):
            MODULE._validate_resume_manifest(cached, args)

    def test_loads_explicit_baseline_words_without_reordering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "concepts.json"
            path.write_text(json.dumps({"baseline_words": ["Desks", "Sand"]}))

            words = MODULE._load_baseline_words(path)

        self.assertEqual(words, ["Desks", "Sand"])
        self.assertEqual(
            MODULE._baseline_words_sha256(words),
            "769763b47e20bd8139316125c5fd23e812ab528297bd2488315f31075aaa5686",
        )

    def test_registered_concept_ranking_uses_gain_then_name(self) -> None:
        rows = [
            {
                "concept": "zeta",
                "token_id": 1,
                "injected_argmax_accuracy": 0.8,
                "injected_mean_correct_prob": 0.6,
                "accuracy_gain_over_clean": 0.2,
            },
            {
                "concept": "beta",
                "token_id": 2,
                "injected_argmax_accuracy": 0.8,
                "injected_mean_correct_prob": 0.6,
                "accuracy_gain_over_clean": 0.3,
            },
            {
                "concept": "alpha",
                "token_id": 3,
                "injected_argmax_accuracy": 0.8,
                "injected_mean_correct_prob": 0.6,
                "accuracy_gain_over_clean": 0.3,
            },
        ]

        worker_order = [
            row["concept"] for row in sorted(rows, key=MODULE._concept_rank_key)
        ]
        merged_order = MODULE._sort_concept_metric_frame(
            pd.DataFrame(rows)
        )["concept"].tolist()

        self.assertEqual(worker_order, ["alpha", "beta", "zeta"])
        self.assertEqual(merged_order, worker_order)

    def test_rejects_empty_explicit_baseline_words(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "concepts.json"
            path.write_text(json.dumps({"baseline_words": []}))

            with self.assertRaisesRegex(ValueError, "non-empty baseline_words"):
                MODULE._load_baseline_words(path)

    def test_screening_vocabulary_excludes_calibration_words_case_insensitively(
        self,
    ) -> None:
        rows = [
            {"word": "Alpha", "token_id": 1},
            {"word": "beta", "token_id": 2},
            {"word": "Gamma", "token_id": 3},
        ]

        screening, excluded = MODULE._screening_vocabulary(
            rows,
            ["ALPHA", "Beta"],
        )

        self.assertEqual(screening, [{"word": "Gamma", "token_id": 3}])
        self.assertEqual(excluded, 2)

    def test_candidate_rows_follow_shortlist_order_and_require_exact_membership(
        self,
    ) -> None:
        rows = [
            {"word": "alpha", "token_id": 1},
            {"word": "beta", "token_id": 2},
            {"word": "gamma", "token_id": 3},
        ]

        selected = MODULE._select_evaluation_rows(rows, ["gamma", "alpha"])

        self.assertEqual([row["word"] for row in selected], ["gamma", "alpha"])
        with self.assertRaisesRegex(ValueError, "occurs 0 times in V_screen"):
            MODULE._select_evaluation_rows(rows, ["missing"])

    def test_candidate_csv_requires_fixed_count_and_unique_words(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidates.csv"
            path.write_text("concept\nNA\nNone\n")
            words = MODULE._load_candidate_words(
                path,
                column="concept",
                expected_count=2,
            )
            self.assertEqual(words, ["NA", "None"])

            path.write_text("concept\nalpha\nalpha\n")
            with self.assertRaisesRegex(ValueError, "must be unique"):
                MODULE._load_candidate_words(
                    path,
                    column="concept",
                    expected_count=2,
                )

    def test_prepare_keeps_full_vocab_baseline_when_filtering_candidates(self) -> None:
        class Tokenizer:
            def __len__(self) -> int:
                return 100

        vocab_rows = [
            {
                "word": "alpha",
                "word_lower": "alpha",
                "word_len": 5,
                "token_id": 1,
                "token_text": " alpha",
            },
            {
                "word": "beta",
                "word_lower": "beta",
                "word_len": 4,
                "token_id": 2,
                "token_text": " beta",
            },
            {
                "word": "gamma",
                "word_lower": "gamma",
                "word_len": 5,
                "token_id": 3,
                "token_text": " gamma",
            },
        ]
        activations = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration_path = root / "calibration.json"
            calibration_path.write_text(
                json.dumps({"concept_vector_words": ["BETA"]})
            )
            candidate_path = root / "top.csv"
            candidate_path.write_text("concept\ngamma\nalpha\n")
            results_dir = root / "results"
            args = parse_search_args(
                [
                    "--role",
                    "prepare",
                    "--model",
                    "model",
                    "--cluster_csv",
                    "clusters.csv",
                    "--results_dir",
                    str(results_dir),
                    "--calibration_concepts_json",
                    str(calibration_path),
                    "--candidate_words_csv",
                    str(candidate_path),
                    "--expected_candidate_count",
                    "2",
                ]
            )
            with (
                patch.object(
                    MODULE,
                    "_model",
                    return_value=SimpleNamespace(tokenizer=Tokenizer()),
                ),
                patch.object(MODULE, "_english_vocab", return_value=vocab_rows),
                patch.object(
                    MODULE,
                    "extract_last_token_residuals",
                    return_value=activations,
                ) as extract,
            ):
                MODULE.prepare(args)

            manifest = json.loads(
                (results_dir / "prepared_vectors.json").read_text()
            )
            evaluation = pd.read_csv(results_dir / "evaluation.csv")

        extract.assert_called_once()
        self.assertEqual(extract.call_args.args[1], ["alpha", "beta", "gamma"])
        self.assertEqual(manifest["baseline_word_count"], 3)
        self.assertEqual(manifest["english_word_count"], 3)
        self.assertEqual(manifest["screening_word_count"], 2)
        self.assertEqual(manifest["calibration_lexical_exclusion_count"], 1)
        self.assertEqual(manifest["evaluation_word_count"], 2)
        self.assertEqual(evaluation["word"].tolist(), ["gamma", "alpha"])

    def test_rejects_more_workers_than_requested_tokens(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "evaluation vocabulary has 2 words",
        ):
            MODULE._validate_worker_capacity(2, 3)

    def test_accepts_one_or_more_tokens_per_worker(self) -> None:
        MODULE._validate_worker_capacity(3, 3)

    def test_runner_rejects_empty_shards_before_launching(self) -> None:
        with self.assertRaises(SystemExit):
            parse_search_args(
                [
                    "--model",
                    "model",
                    "--cluster_csv",
                    "clusters.csv",
                    "--results_dir",
                    "results",
                    "--gpus",
                    "0,1,2",
                    "--max_tokens",
                    "2",
                ]
            )

    def test_runner_counts_multi_gpu_groups_as_workers(self) -> None:
        args = parse_search_args(
            [
                "--model",
                "model",
                "--cluster_csv",
                "clusters.csv",
                "--results_dir",
                "results",
                "--gpus",
                "0,1;2,3",
                "--max_tokens",
                "2",
            ]
        )
        self.assertEqual(args.gpus, "0,1;2,3")

    def test_loads_existing_resume_summaries_by_template(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.csv"
            path.write_text(
                "worker_id,prompt_template,n_trials\n"
                "0,semantic_highinj_posref_gate_balanced_disrupts,100\n"
            )

            summaries = MODULE._load_summary_rows(path)

        self.assertEqual(
            summaries["semantic_highinj_posref_gate_balanced_disrupts"]["n_trials"],
            "100",
        )

    def test_empty_resume_summary_is_not_considered_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.csv"
            path.touch()

            self.assertEqual(MODULE._load_summary_rows(path), {})

    def test_resume_preserves_summary_for_completed_template(self) -> None:
        template = "semantic_highinj_posref_gate_balanced_disrupts"
        with tempfile.TemporaryDirectory() as directory:
            results_dir = Path(directory)
            shard_path = results_dir / "vector_shards" / "worker_0.pt"
            shard_path.parent.mkdir(parents=True)
            shard_path.touch()
            workers_dir = results_dir / "workers"
            workers_dir.mkdir()
            pd.DataFrame([{"word": "cat"}]).to_csv(
                workers_dir / f"worker_0_{template}.csv",
                index=False,
            )
            original_summary = {
                "worker_id": 0,
                "prompt_template": template,
                "n_words": 1,
                "n_trials": 10,
            }
            pd.DataFrame([original_summary]).to_csv(
                workers_dir / "worker_0_summary.csv",
                index=False,
            )
            args = parse_search_args(
                [
                    "--role",
                    "worker",
                    "--model",
                    "model",
                    "--cluster_csv",
                    "clusters.csv",
                    "--results_dir",
                    str(results_dir),
                    "--prompt_templates",
                    template,
                    "--resume",
                ]
            )
            payload = {
                "rows": [{"word": "cat"}],
                "vectors": torch.zeros(1, 2),
            }
            with (
                patch.object(MODULE.torch, "load", return_value=payload),
                patch.object(
                    MODULE,
                    "_model",
                    return_value=SimpleNamespace(tokenizer=object()),
                ),
                patch.object(MODULE, "PromptManager", return_value=object()),
                patch.object(MODULE, "evaluate_cluster_localization") as evaluate,
            ):
                MODULE.worker(args)

            resumed = pd.read_csv(workers_dir / "worker_0_summary.csv")

        evaluate.assert_not_called()
        self.assertEqual(resumed.to_dict("records"), [original_summary])


if __name__ == "__main__":
    unittest.main()
