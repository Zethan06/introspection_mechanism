"""Tests for the evaluation prompt and its label variants."""

from __future__ import annotations

from contextlib import nullcontext
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from introspection_core.localization_evaluation import (
    ClusterPrompt,
    evaluate_cluster_localization,
)
from introspection_core.attention_aggregation import _clean_expected_label
from introspection_core.attention_inputs import TokenLocalizationCsvTask
from introspection_core.prompts import (
    SEMANTIC_HIGHINJ_POSREF_GATE_BALANCED_DISRUPTS_QUESTION,
    SEMANTIC_HIGHINJ_POSREF_GATE_BALANCED_DISRUPTS_LETTERS_SYSTEM_PROMPT,
    SEMANTIC_HIGHINJ_POSREF_GATE_BALANCED_DISRUPTS_NUMWORDS_SYSTEM_PROMPT,
    SEMANTIC_HIGHINJ_POSREF_GATE_BALANCED_DISRUPTS_SYSTEM_PROMPT,
    PromptManager,
    REGISTRY,
)
from scripts.sweep_injection_localization import parse_args


class _RecordingTokenizer:
    eos_token = "<eos>"

    def __init__(self) -> None:
        self.chat_template_calls: list[dict] = []

    def apply_chat_template(self, messages, **kwargs) -> str:
        self.chat_template_calls.append(kwargs)
        return "\n".join(message["content"] for message in messages)

    def __call__(self, text: str, **kwargs) -> dict:
        del kwargs
        return {
            "input_ids": torch.arange(len(text)).unsqueeze(0),
            "offset_mapping": torch.tensor(
                [[[index, index + 1] for index in range(len(text))]]
            ),
        }

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        return [100 + sum(ord(character) for character in text)]

    def decode(self, token_ids: list[int]) -> str:
        return f"<{','.join(str(token_id) for token_id in token_ids)}>"


class _AlternatingRoleTokenizer(_RecordingTokenizer):
    def __init__(self) -> None:
        super().__init__()
        self.message_roles: list[list[str]] = []

    def apply_chat_template(self, messages, **kwargs) -> str:
        roles = [message["role"] for message in messages]
        self.message_roles.append(roles)
        if roles[:3] == ["system", "assistant", "user"]:
            raise RuntimeError(
                "Conversation roles must alternate user/assistant/user/assistant/..."
            )
        return super().apply_chat_template(messages, **kwargs)


class EvaluationPromptTests(unittest.TestCase):
    def test_attention_task_preserves_clean_none_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cluster_path = Path(directory) / "clusters.csv"
            with cluster_path.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["cluster_key", "choices"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "cluster_key": "disrupts-cluster",
                        "choices": json.dumps(list("abcdefghij")),
                    }
                )

            examples = TokenLocalizationCsvTask(
                path=cluster_path,
                position_index_start=0,
                template_name="semantic_highinj_posref_gate_balanced_disrupts",
            ).build_examples(PromptManager(_RecordingTokenizer()))

        self.assertEqual(examples[0].clean_target_label, "none")
        self.assertEqual(
            list(examples[0].candidate_token_ids),
            [*[str(index) for index in range(10)], "none"],
        )
        self.assertEqual(_clean_expected_label(examples[0], 4), "none")

    def test_letter_labels_replace_digits_in_prompt_and_candidates(self) -> None:
        template = REGISTRY[
            "semantic_highinj_posref_gate_balanced_disrupts_letters_a_j"
        ]
        items = list("abcdefghij")

        messages = template.turns(items, "system", "")
        rendered = PromptManager(_RecordingTokenizer()).render(
            "semantic_highinj_posref_gate_balanced_disrupts_letters_a_j",
            items,
            preamble="system",
        )

        self.assertEqual(
            template.candidate_labels(10),
            [*"ABCDEFGHIJ", "none"],
        )
        self.assertEqual(
            messages[0]["content"],
            SEMANTIC_HIGHINJ_POSREF_GATE_BALANCED_DISRUPTS_LETTERS_SYSTEM_PROMPT,
        )
        self.assertIn("TOKEN A: a", messages[2]["content"])
        self.assertIn("TOKEN J: j", messages[2]["content"])
        self.assertNotIn("TOKEN 0:", messages[2]["content"])
        self.assertIn("`A`, `B`, `C`", messages[0]["content"])
        self.assertNotIn("`0`", messages[0]["content"])
        self.assertEqual(list(rendered.answer_token_by_choice), [*"ABCDEFGHIJ", "none"])
        self.assertEqual(
            [record["choice"] for record in rendered.records],
            list("ABCDEFGHIJ"),
        )

    def test_number_word_labels_replace_digits_in_prompt_and_candidates(self) -> None:
        template = REGISTRY[
            "semantic_highinj_posref_gate_balanced_disrupts_numwords_one_ten"
        ]
        items = list("abcdefghij")
        labels = [
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
        ]

        messages = template.turns(items, "system", "")
        rendered = PromptManager(_RecordingTokenizer()).render(
            "semantic_highinj_posref_gate_balanced_disrupts_numwords_one_ten",
            items,
            preamble="system",
        )

        self.assertEqual(template.candidate_labels(10), [*labels, "none"])
        self.assertEqual(
            messages[0]["content"],
            SEMANTIC_HIGHINJ_POSREF_GATE_BALANCED_DISRUPTS_NUMWORDS_SYSTEM_PROMPT,
        )
        self.assertIn("TOKEN one: a", messages[2]["content"])
        self.assertIn("TOKEN ten: j", messages[2]["content"])
        self.assertNotIn("TOKEN 0:", messages[2]["content"])
        self.assertIn("`one`, `two`, `three`", messages[0]["content"])
        self.assertNotIn("`0`", messages[0]["content"])
        self.assertEqual(list(rendered.answer_token_by_choice), [*labels, "none"])
        self.assertEqual(
            [record["choice"] for record in rendered.records],
            labels,
        )

    def test_number_word_labels_require_ten_items_and_system_preamble(self) -> None:
        template = REGISTRY[
            "semantic_highinj_posref_gate_balanced_disrupts_numwords_one_ten"
        ]

        with self.assertRaises(ValueError):
            template.turns(list("abcdefghij"), "user", "")
        with self.assertRaises(ValueError):
            template.turns(list("abcde"), "system", "")

    def test_number_word_task_maps_positions_to_word_labels(self) -> None:
        labels = [
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
        ]
        with tempfile.TemporaryDirectory() as directory:
            cluster_path = Path(directory) / "clusters.csv"
            with cluster_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["cluster_key", "choices"])
                writer.writeheader()
                writer.writerow(
                    {
                        "cluster_key": "number-words",
                        "choices": json.dumps(list("abcdefghij")),
                    }
                )
            examples = TokenLocalizationCsvTask(
                path=cluster_path,
                preamble="system",
                template_name=(
                    "semantic_highinj_posref_gate_balanced_disrupts_numwords_one_ten"
                ),
            ).build_examples(PromptManager(_RecordingTokenizer()))

        self.assertEqual(
            examples[0].expected_candidate_by_position,
            dict(enumerate(labels)),
        )


    def test_keeps_acknowledgement_and_has_no_assistant_prefill(self) -> None:
        template = REGISTRY["semantic_highinj_posref_gate_balanced_disrupts"]
        items = list("abcdefghij")

        messages = template.turns(items, "system", "")

        self.assertTrue(template.add_generation_prompt)
        self.assertEqual(
            [message["role"] for message in messages],
            ["system", "assistant", "user"],
        )
        self.assertEqual(
            messages[0]["content"],
            SEMANTIC_HIGHINJ_POSREF_GATE_BALANCED_DISRUPTS_SYSTEM_PROMPT,
        )
        self.assertEqual(messages[1]["content"], "Understood.")
        self.assertEqual(
            messages[2]["content"],
            " ".join(
                f"TOKEN {index}: {item}"
                for index, item in enumerate(items)
            )
            + "\n\n"
            + SEMANTIC_HIGHINJ_POSREF_GATE_BALANCED_DISRUPTS_QUESTION,
        )
        self.assertEqual(
            template.candidate_labels(10),
            [*[str(index) for index in range(10)], "none"],
        )

    def test_disables_thinking_for_fresh_answer_turn(self) -> None:
        tokenizer = _RecordingTokenizer()

        PromptManager(tokenizer).render(
            "semantic_highinj_posref_gate_balanced_disrupts",
            list("abcdefghij"),
            preamble="system",
        )

        self.assertFalse(
            tokenizer.chat_template_calls[0]["enable_thinking"]
        )
        self.assertTrue(
            tokenizer.chat_template_calls[0]["add_generation_prompt"]
        )

    def test_maps_leading_system_role_for_alternating_chat_templates(self) -> None:
        tokenizer = _AlternatingRoleTokenizer()

        PromptManager(tokenizer).render(
            "semantic_highinj_posref_gate_balanced_disrupts",
            list("abcdefghij"),
            preamble="system",
        )

        self.assertEqual(
            tokenizer.message_roles,
            [
                ["system", "assistant", "user"],
                ["user", "assistant", "user"],
            ],
        )

    def test_requires_system_preamble(self) -> None:
        template = REGISTRY["semantic_highinj_posref_gate_balanced_disrupts"]

        with self.assertRaisesRegex(ValueError, "requires preamble='system'"):
            template.turns(["cat"], "none", "")

    def test_requires_exactly_ten_items(self) -> None:
        template = REGISTRY["semantic_highinj_posref_gate_balanced_disrupts"]

        with self.assertRaisesRegex(ValueError, "exactly 10 items"):
            template.turns(["cat", "dog"], "system", "")

    def test_sweep_cli_selects_disrupts_template(self) -> None:
        args = parse_args(
            [
                "--model",
                "model",
                "--concepts_json",
                "concepts.json",
                "--cluster_csv",
                "clusters.csv",
                "--run_dir",
                "results",
                "--prompt_template",
                "semantic_highinj_posref_gate_balanced_disrupts",
            ]
        )

        self.assertEqual(
            args.prompt_template,
            "semantic_highinj_posref_gate_balanced_disrupts",
        )



class _NoneEvaluationModel:
    def __init__(self) -> None:
        self.bridge = SimpleNamespace(
            cfg=SimpleNamespace(device="cpu", dtype=torch.float32)
        )
        self.call_count = 0

    def last_token_candidate_stats(
        self, tokens: torch.Tensor, *, candidate_token_ids
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del candidate_token_ids
        self.call_count += 1
        if self.call_count == 1:
            logits = torch.tensor([[0.0, 1.0, 4.0]])
        else:
            self.assert_batch_size = int(tokens.shape[0])
            logits = torch.tensor(
                [[4.0, 1.0, 0.0], [1.0, 4.0, 0.0]]
            )
        return logits, torch.log_softmax(logits, dim=-1)


class NoneEvaluationTests(unittest.TestCase):
    def test_clean_runs_are_scored_against_none(self) -> None:
        example = ClusterPrompt(
            key="cluster",
            rank=1,
            choices=("cat", "dog"),
            input_ids=torch.tensor([[10, 11, 12]]),
            span_starts=(0, 1),
            span_ends=(1, 2),
            candidate_labels=("0", "1", "none"),
            candidate_token_ids=(20, 21, 22),
        )
        model = _NoneEvaluationModel()

        with patch(
            "introspection_core.localization_evaluation.inject",
            return_value=nullcontext(),
        ):
            result = evaluate_cluster_localization(
                model,
                examples=[example],
                concept_names=["concept"],
                unit_vectors=torch.tensor([[1.0, 0.0]]),
                injection_layer=1,
                strength=2.0,
                scale_mode="unit",
                batch_size=2,
                include_rows=True,
            )

        self.assertEqual(int(result.clean_correct[0]), 2)
        expected_none_prob = float(
            torch.softmax(torch.tensor([0.0, 1.0, 4.0]), dim=-1)[2]
        )
        self.assertAlmostEqual(
            float(result.clean_correct_prob_sum[0]),
            2 * expected_none_prob,
        )
        self.assertTrue(all(row["clean_correct"] == 1 for row in result.rows))
        self.assertTrue(
            all(row["clean_argmax_choice"] == "none" for row in result.rows)
        )
        self.assertEqual(int(result.injected_correct[0]), 2)

if __name__ == "__main__":
    unittest.main()
