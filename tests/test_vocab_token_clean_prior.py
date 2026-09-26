"""Tests for the local clean token-prior generator."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import torch

from introspection_core.vocab_token_clean_prior import (
    SCORE_FIELDS,
    combine_worker_outputs,
    extract_single_word_tokens,
    group_tasks_by_length,
    make_groups,
    parse_args,
    score_candidate_numbers,
)


def parse_prior_args(argv):
    return parse_args([*argv, "--results_dir", "results"])


class FakeTokenizer:
    def __init__(self, raw_tokens: list[str], decoded_tokens: list[str]) -> None:
        self.raw_tokens = raw_tokens
        self.decoded_tokens = decoded_tokens
        self.all_special_ids: list[int] = []

    def __len__(self) -> int:
        return len(self.raw_tokens)

    def decode(self, token_ids, clean_up_tokenization_spaces=False) -> str:
        del clean_up_tokenization_spaces
        return self.decoded_tokens[token_ids[0]]

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self.raw_tokens[token_id]

    def encode(self, text: str, add_special_tokens=False) -> list[int]:
        del add_special_tokens
        word = text.lstrip()
        for token_id, raw_token in enumerate(self.raw_tokens):
            if raw_token.lstrip("▁Ġ") == word:
                return [token_id]
        return []


class TokenPriorCliTests(unittest.TestCase):
    def test_accepts_paired_gpu_groups(self) -> None:
        args = parse_prior_args(
            [
                "--model",
                "model",
                "--gpus",
                "2,3;5,6",
                "--max_tokens",
                "10",
            ]
        )

        self.assertEqual(args.gpus, "2,3;5,6")
        self.assertEqual(args.position_index_start, 0)

    def test_rejects_too_few_max_tokens(self) -> None:
        with self.assertRaises(SystemExit):
            parse_prior_args(
                [
                    "--model",
                    "model",
                    "--num_choices",
                    "10",
                    "--max_tokens",
                    "9",
                ]
            )


class TokenGroupingTests(unittest.TestCase):
    def _args(self):
        return parse_prior_args(["--model", "model", "--min_word_len", "1"])

    def test_extracts_literal_space_word_start_token(self) -> None:
        tokenizer = FakeTokenizer(["Ġalpha"], [" alpha"])

        tokens = extract_single_word_tokens(tokenizer, self._args())

        self.assertEqual([token["word"] for token in tokens], ["alpha"])

    def test_extracts_sentencepiece_word_start_token_without_decoded_space(self) -> None:
        tokenizer = FakeTokenizer(["▁alpha"], ["alpha"])

        tokens = extract_single_word_tokens(tokenizer, self._args())

        self.assertEqual([token["word"] for token in tokens], ["alpha"])
        self.assertEqual(tokens[0]["token_text"], "alpha")

    def test_extracts_markerless_standalone_token(self) -> None:
        tokenizer = FakeTokenizer(["alpha"], ["alpha"])

        tokens = extract_single_word_tokens(tokenizer, self._args())

        self.assertEqual([token["word"] for token in tokens], ["alpha"])

    def test_rejects_continuation_token_that_does_not_round_trip(self) -> None:
        tokenizer = FakeTokenizer(["##alpha"], ["alpha"])

        tokens = extract_single_word_tokens(tokenizer, self._args())

        self.assertEqual(tokens, [])

    def test_final_partial_group_uses_non_primary_fillers(self) -> None:
        tokens = [
            {
                "token_id": index,
                "token_text": f" word{index}",
                "word": f"word{index}",
                "word_lower": f"word{index}",
                "word_len": 5,
            }
            for index in range(5)
        ]

        groups = make_groups(tokens, num_choices=3)

        self.assertEqual(len(groups), 2)
        self.assertEqual(len(groups[1]["entries"]), 3)
        self.assertEqual(
            [entry["is_primary"] for entry in groups[1]["entries"]],
            [True, True, False],
        )


class TransformerLensScoringTests(unittest.TestCase):
    def test_groups_tasks_by_exact_prompt_length(self) -> None:
        tasks = [
            {"input_ids": torch.zeros(1, 3, dtype=torch.long), "id": 0},
            {"input_ids": torch.zeros(1, 5, dtype=torch.long), "id": 1},
            {"input_ids": torch.zeros(1, 3, dtype=torch.long), "id": 2},
        ]

        groups = group_tasks_by_length(tasks)

        self.assertEqual([[task["id"] for task in group] for group in groups], [[0, 2], [1]])

    def test_scores_restricted_transformer_bridge_outputs(self) -> None:
        logits = torch.tensor([[1.0, 2.0], [3.0, 1.0]])
        log_probs = torch.log(torch.tensor([[0.1, 0.2], [0.3, 0.05]]))

        _, candidate_probs, full_probs, predictions = score_candidate_numbers(
            logits,
            log_probs,
            [0, 1],
        )

        self.assertEqual(predictions, [1, 0])
        torch.testing.assert_close(full_probs, log_probs.exp())
        torch.testing.assert_close(candidate_probs.sum(dim=1), torch.ones(2))


class TokenPriorAggregationTests(unittest.TestCase):
    def _write_worker_scores(self, worker_dir: Path) -> None:
        rows = []
        specifications = [
            (0, "alpha", 0, 0.2, 0.02, 0),
            (0, "alpha", 1, 0.4, 0.04, 0),
            (1, "beta", 0, 0.8, 0.08, 1),
            (1, "beta", 1, 0.6, 0.06, 1),
        ]
        for prompt_id, (
            token_id,
            word,
            position,
            score,
            full_prob,
            is_argmax,
        ) in enumerate(specifications):
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "group_idx": 0,
                    "rotation": position,
                    "position": position,
                    "token_id": token_id,
                    "token_text": f" {word}",
                    "word": word,
                    "word_lower": word,
                    "word_len": len(word),
                    "number_logit": score,
                    "number_softmax": score,
                    "full_vocab_prob": full_prob,
                    "clean_prediction": 1,
                    "is_argmax": is_argmax,
                }
            )
        path = worker_dir / "worker_0_token_position_scores.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SCORE_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def test_writes_cluster_search_compatible_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results_dir = Path(directory)
            worker_dir = results_dir / "workers"
            worker_dir.mkdir()
            self._write_worker_scores(worker_dir)

            result = combine_worker_outputs(
                results_dir,
                num_workers=1,
                num_choices=2,
                position_index_start=0,
                worker_dir=worker_dir,
            )

            with (results_dir / "summary_by_token.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(result["token_count"], 2)
        self.assertEqual(result["token_position_rows"], 4)
        self.assertEqual(rows[0]["word"], "beta")
        self.assertAlmostEqual(float(rows[0]["mean_number_softmax"]), 0.7)
        self.assertAlmostEqual(float(rows[1]["position_score_range"]), 0.2)
        self.assertEqual(int(rows[0]["argmax_count"]), 2)
        self.assertTrue(
            {
                "word",
                "word_lower",
                "word_len",
                "mean_number_softmax",
                "position_score_range",
                "argmax_count",
                "argmax_rate",
            }.issubset(rows[0])
        )


if __name__ == "__main__":
    unittest.main()
