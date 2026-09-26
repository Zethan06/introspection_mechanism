"""Tests for the direct token-localization question."""

import unittest

from introspection_core.prompts import REGISTRY


class TokenLocalizationPromptTests(unittest.TestCase):
    def test_zero_based_prompt_uses_direct_question_and_answer(self) -> None:
        messages = REGISTRY["token_localization"].turns(
            ["alpha", "beta"], "system", ""
        )

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn(
            "The researcher can inject neural activity patterns",
            messages[0]["content"],
        )
        self.assertEqual(
            messages[1]["content"],
            "TOKEN 0: alpha TOKEN 1: beta\n\n"
            "Which token is the injection located in?",
        )
        self.assertEqual(
            messages[2],
            {"role": "assistant", "content": "It is located in TOKEN "},
        )

    def test_registry_holds_only_the_paper_prompts(self) -> None:
        self.assertEqual(
            sorted(REGISTRY),
            [
                "semantic_highinj_posref_gate_balanced_disrupts",
                "semantic_highinj_posref_gate_balanced_disrupts_letters_a_j",
                "semantic_highinj_posref_gate_balanced_disrupts_numwords_one_ten",
                "token_localization",
            ],
        )


if __name__ == "__main__":
    unittest.main()
