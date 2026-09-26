from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from introspection_core.balanced_cluster_search import (
    CANDIDATE_BANKS,
    build_task,
    consume_scores,
    init_accumulator,
    load_candidate_tokens,
    output_filenames,
    select_choice_token_disjoint_clusters,
    select_disjoint_clusters_with_full_pool_fallback,
    split_candidate_banks,
)
from introspection_core.cluster_split import (
    build_cluster_split_manifest,
    require_valid_cluster_split,
    write_cluster_split_manifest,
)


class CandidateBankSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokens = [
            {
                "word": f"word{index}",
                "word_lower": f"word{index}",
                "token_id": 1000 + index,
            }
            for index in range(20)
        ]

    def test_split_is_deterministic_disjoint_and_balanced(self) -> None:
        first = split_candidate_banks(
            self.tokens, candidate_token_count=12, seed=42
        )
        second = split_candidate_banks(
            self.tokens, candidate_token_count=12, seed=42
        )

        self.assertEqual(first, second)
        self.assertEqual(set(first), set(CANDIDATE_BANKS))
        self.assertEqual(
            [len(first[name]) for name in CANDIDATE_BANKS],
            [3, 3, 3, 3],
        )

        id_sets = {
            name: {row["token_id"] for row in first[name]}
            for name in CANDIDATE_BANKS
        }
        for index, left in enumerate(CANDIDATE_BANKS):
            for right in CANDIDATE_BANKS[index + 1 :]:
                self.assertTrue(id_sets[left].isdisjoint(id_sets[right]))
        self.assertEqual(
            set().union(*id_sets.values()),
            {row["token_id"] for row in self.tokens[:12]},
        )

        for block_index in range(3):
            rows = [
                row
                for name in CANDIDATE_BANKS
                for row in first[name]
                if row["candidate_block"] == block_index
            ]
            self.assertEqual(len(rows), 4)
            self.assertEqual(
                {row["candidate_rank"] for row in rows},
                {
                    block_index * 4 + 1,
                    block_index * 4 + 2,
                    block_index * 4 + 3,
                    block_index * 4 + 4,
                },
            )

        self.assertNotIn("candidate_bank", self.tokens[0])

    def test_split_rejects_invalid_or_insufficient_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "multiple of four"):
            split_candidate_banks(
                self.tokens, candidate_token_count=10, seed=42
            )
        with self.assertRaisesRegex(ValueError, "need 24"):
            split_candidate_banks(
                self.tokens, candidate_token_count=24, seed=42
            )

    def test_bank_specific_output_names(self) -> None:
        args = SimpleNamespace(candidate_bank="validation", dataset_clusters=30)
        names = output_filenames(args)
        self.assertEqual(names["clusters"], "clusters/validation.csv")
        self.assertEqual(
            names["candidate_tokens"], "candidates/validation.csv"
        )
        self.assertEqual(
            names["summary"], "manifests/cluster_search/validation.json"
        )

    def test_cluster_search_uses_clean_position_prior_prompt(self) -> None:
        choices = [f"word{index}" for index in range(10)]
        token_rows = [
            {
                "word_lower": word,
                "token_id": 100 + index,
            }
            for index, word in enumerate(choices)
        ]

        class PromptManager:
            def render(self, template_name, items, *, preamble):
                self.template_name = template_name
                self.items = items
                self.preamble = preamble
                return SimpleNamespace(
                    input_ids="tokens",
                    answer_token_by_choice={
                        str(index): 200 + index for index in range(10)
                    },
                    records=[
                        {"text": word, "token_ids": [100 + index]}
                        for index, word in enumerate(choices)
                    ],
                )

        manager = PromptManager()
        task = build_task(
            manager,
            {"cluster_key": "cluster", "token_rows": token_rows},
            choices,
            0,
            SimpleNamespace(
                num_choices=10,
                position_index_start=0,
                prompt_preamble="system",
            ),
        )

        self.assertEqual(manager.template_name, "token_localization")
        self.assertEqual(manager.items, choices)
        self.assertEqual(sorted(task["number_tokens"]), list(range(10)))

    def test_cluster_scores_use_one_based_position_labels(self) -> None:
        choices = [f"word{index}" for index in range(1, 10)]
        cluster = {"cluster_key": "cluster", "choices": choices}
        args = SimpleNamespace(num_choices=9, position_index_start=1)
        accumulators = {"cluster": init_accumulator(cluster, args)}
        pending = [
            {
                "cluster_key": "cluster",
                "choices": choices,
                "permutation_idx": 0,
                "number_tokens": {
                    index: index - 1 for index in range(1, 10)
                },
            }
        ]
        logits = torch.tensor(
            [[9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        )

        consume_scores(pending, logits, accumulators, args)

        accumulator = accumulators["cluster"]
        self.assertEqual(
            sorted(accumulator["position_sums"]),
            list(range(1, 10)),
        )
        self.assertEqual(accumulator["best_prompt"]["argmax_position"], 1)

    def test_legacy_unnamed_bank_is_rejected(self) -> None:
        args = SimpleNamespace(candidate_bank=None, dataset_clusters=30)
        with self.assertRaisesRegex(ValueError, "candidate_bank"):
            output_filenames(args)

    def test_candidate_loader_applies_ranked_bank_split(self) -> None:
        words = [
            "alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
            "golf", "hotel", "india", "juliet", "kilo", "lima",
        ]

        class Tokenizer:
            all_special_ids: list[int] = []

            def __init__(self) -> None:
                self.ids = {word: index + 100 for index, word in enumerate(words)}
                self.words = {token_id: word for word, token_id in self.ids.items()}

            def encode(self, text: str, add_special_tokens: bool) -> list[int]:
                self.assert_no_special_tokens(add_special_tokens)
                return [self.ids[text.removeprefix(" ")]]

            @staticmethod
            def assert_no_special_tokens(add_special_tokens: bool) -> None:
                if add_special_tokens:
                    raise AssertionError("candidate encoding must omit special tokens")

            def decode(
                self, token_ids: list[int], clean_up_tokenization_spaces: bool
            ) -> str:
                if clean_up_tokenization_spaces:
                    raise AssertionError("round-trip decoding must not clean spaces")
                return " " + self.words[token_ids[0]]

        with tempfile.TemporaryDirectory() as tmp:
            prior = Path(tmp) / "summary_by_token.csv"
            with prior.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "word", "word_lower", "word_len", "mean_number_softmax"
                    ],
                )
                writer.writeheader()
                for word in reversed(words):
                    writer.writerow(
                        {
                            "word": word,
                            "word_lower": word,
                            "word_len": len(word),
                            "mean_number_softmax": 0.1,
                        }
                    )

            args = SimpleNamespace(
                prior_summary=prior,
                candidate_min_len=3,
                candidate_max_len=12,
                candidate_min_mean_prior=0.0,
                candidate_max_mean_prior=0.25,
                candidate_max_argmax_rate=0.35,
                candidate_max_position_range=999.0,
                candidate_token_count=12,
                num_choices=10,
                seed=42,
            )
            loaded = {}
            for bank in CANDIDATE_BANKS:
                args.candidate_bank = bank
                loaded[bank] = load_candidate_tokens(args, Tokenizer())

        token_sets = {
            bank: {row["token_id"] for row in rows}
            for bank, rows in loaded.items()
        }
        self.assertEqual(
            [len(loaded[bank]) for bank in CANDIDATE_BANKS],
            [3, 3, 3, 3],
        )
        self.assertEqual(len(set().union(*token_sets.values())), 12)
        for index, left in enumerate(CANDIDATE_BANKS):
            for right in CANDIDATE_BANKS[index + 1 :]:
                self.assertTrue(token_sets[left].isdisjoint(token_sets[right]))

    def test_cluster_selection_greedily_skips_reused_choice_tokens(self) -> None:
        clusters = pd.DataFrame(
            [
                {"cluster_key": "best"},
                {"cluster_key": "overlap"},
                {"cluster_key": "next"},
            ]
        )
        token_rows = pd.DataFrame(
            [
                *(
                    {"cluster_key": "best", "token_id": token_id}
                    for token_id in (1, 2, 3)
                ),
                *(
                    {"cluster_key": "overlap", "token_id": token_id}
                    for token_id in (3, 4, 5)
                ),
                *(
                    {"cluster_key": "next", "token_id": token_id}
                    for token_id in (6, 7, 8)
                ),
            ]
        )

        selected = select_choice_token_disjoint_clusters(
            clusters,
            token_rows,
            count=2,
            num_choices=3,
        )

        self.assertEqual(selected["cluster_key"].tolist(), ["best", "next"])

    def test_cluster_selection_fails_when_disjoint_pool_is_too_small(self) -> None:
        clusters = pd.DataFrame(
            [{"cluster_key": "first"}, {"cluster_key": "overlap"}]
        )
        token_rows = pd.DataFrame(
            [
                {"cluster_key": "first", "token_id": 1},
                {"cluster_key": "first", "token_id": 2},
                {"cluster_key": "overlap", "token_id": 2},
                {"cluster_key": "overlap", "token_id": 3},
            ]
        )

        with self.assertRaisesRegex(ValueError, "Expand the cluster search pool"):
            select_choice_token_disjoint_clusters(
                clusters,
                token_rows,
                count=2,
                num_choices=2,
            )

    def test_cluster_selection_retries_complete_ranked_pool(self) -> None:
        clusters = pd.DataFrame(
            [
                {"cluster_key": "best"},
                {"cluster_key": "qualified_overlap"},
                {"cluster_key": "fallback"},
            ]
        )
        qualified = clusters.iloc[:2]
        token_rows = pd.DataFrame(
            [
                {"cluster_key": "best", "token_id": 1},
                {"cluster_key": "best", "token_id": 2},
                {"cluster_key": "qualified_overlap", "token_id": 2},
                {"cluster_key": "qualified_overlap", "token_id": 3},
                {"cluster_key": "fallback", "token_id": 4},
                {"cluster_key": "fallback", "token_id": 5},
            ]
        )

        selected = select_disjoint_clusters_with_full_pool_fallback(
            clusters,
            qualified,
            token_rows,
            count=2,
            num_choices=2,
        )

        self.assertEqual(selected["cluster_key"].tolist(), ["best", "fallback"])

    def test_candidate_loader_accepts_sentencepiece_decode_without_space(self) -> None:
        words = [
            "alpha", "bravo", "charlie", "delta", "echo",
            "foxtrot", "golf", "hotel", "india", "juliet",
        ]

        class Tokenizer:
            all_special_ids: list[int] = []

            def __init__(self) -> None:
                self.ids = {word: index + 100 for index, word in enumerate(words)}
                self.words = {token_id: word for word, token_id in self.ids.items()}

            def encode(self, text: str, add_special_tokens: bool) -> list[int]:
                self.assert_no_special_tokens(add_special_tokens)
                return [self.ids[text.removeprefix(" ")]]

            @staticmethod
            def assert_no_special_tokens(add_special_tokens: bool) -> None:
                if add_special_tokens:
                    raise AssertionError("candidate encoding must omit special tokens")

            def decode(
                self, token_ids: list[int], clean_up_tokenization_spaces: bool
            ) -> str:
                if clean_up_tokenization_spaces:
                    raise AssertionError("round-trip decoding must not clean spaces")
                return self.words[token_ids[0]]

            def convert_ids_to_tokens(self, token_id: int) -> str:
                return "▁" + self.words[token_id]

        with tempfile.TemporaryDirectory() as tmp:
            prior = Path(tmp) / "summary_by_token.csv"
            with prior.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "word", "word_lower", "word_len", "mean_number_softmax"
                    ],
                )
                writer.writeheader()
                for word in words:
                    writer.writerow(
                        {
                            "word": word,
                            "word_lower": word,
                            "word_len": len(word),
                            "mean_number_softmax": 0.1,
                        }
                    )

            args = SimpleNamespace(
                prior_summary=prior,
                candidate_min_len=3,
                candidate_max_len=12,
                candidate_min_mean_prior=0.0,
                candidate_max_mean_prior=0.25,
                candidate_max_argmax_rate=0.35,
                candidate_max_position_range=999.0,
                candidate_token_count=3000,
                candidate_bank=None,
                num_choices=10,
                seed=42,
            )
            loaded = load_candidate_tokens(args, Tokenizer())

        self.assertEqual(len(loaded), 10)
        self.assertEqual(loaded[0]["token_text"], "alpha")


class ClusterSplitManifestTests(unittest.TestCase):
    def _write_split(self, directory: Path) -> None:
        settings = {
            "model": "model",
            "source_prior_sha256": "prior-hash",
            "seed": 42,
            "candidate_token_count": 8,
            "candidate_ranking": ["registered"],
            "candidate_filters": {"min_len": 3},
            "num_choices": 2,
            "dataset_clusters": 1,
            "cluster_search": {"neighbor_pool": 10},
        }
        for bank_index, bank in enumerate(CANDIDATE_BANKS):
            first_rank = bank_index * 2 + 1
            words = [f"{bank}a", f"{bank}b"]
            candidates = pd.DataFrame(
                [
                    {
                        "candidate_bank": bank,
                        "candidate_rank": first_rank + offset,
                        "token_id": first_rank + offset + 100,
                        "word_lower": word,
                    }
                    for offset, word in enumerate(words)
                ]
            )
            candidate_dir = directory / "candidates"
            cluster_dir = directory / "clusters"
            summary_dir = directory / "manifests" / "cluster_search"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            cluster_dir.mkdir(parents=True, exist_ok=True)
            summary_dir.mkdir(parents=True, exist_ok=True)
            candidates.to_csv(candidate_dir / f"{bank}.csv", index=False)
            pd.DataFrame(
                [{"cluster_key": f"{bank}-cluster", "choices": json.dumps(words)}]
            ).to_csv(cluster_dir / f"{bank}.csv", index=False)
            (summary_dir / f"{bank}.json").write_text(
                json.dumps(
                    {"candidate_bank": bank, "split_settings": settings}
                )
            )

    def test_writes_and_rechecks_joint_four_bank_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._write_split(directory)

            manifest_path = write_cluster_split_manifest(directory)
            payload = json.loads(manifest_path.read_text())
            require_valid_cluster_split(directory / "clusters" / "calibration.csv")

        self.assertTrue(payload["valid"])
        self.assertEqual(payload["bank_names"], list(CANDIDATE_BANKS))
        self.assertEqual(len(payload["candidate_file_sha256"]), 4)
        self.assertTrue(
            all(payload["pairwise_disjoint_choice_token_ids"].values())
        )

    def test_rejects_cross_bank_choice_token_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._write_split(directory)
            path = directory / "candidates" / "train.csv"
            frame = pd.read_csv(path)
            frame.loc[0, "token_id"] = 101
            frame.to_csv(path, index=False)

            with self.assertRaisesRegex(ValueError, "token IDs overlap"):
                build_cluster_split_manifest(directory)


if __name__ == "__main__":
    unittest.main()
