"""Tests for rendered-text to token-span resolution."""

import unittest

from introspection_core.localization import resolve_text_span_to_tokens


class BrokenOffsetTokenizer:
    """Tiny tokenizer whose IDs are right while its offsets are shifted."""

    vocabulary = {"prefix": 1, "target": 2, "suffix": 3}

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {
            "input_ids": [
                token_id
                for token, token_id in self.vocabulary.items()
                if token in text
            ]
        }

    def decode(self, token_ids):
        reverse = {token_id: token for token, token_id in self.vocabulary.items()}
        return " ".join(reverse[token_id] for token_id in token_ids)


class RepeatedWordTokenizer:
    vocabulary = {"target": 1, "middle": 2}

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {
            "input_ids": [self.vocabulary[word] for word in text.split()]
        }

    def decode(self, token_ids):
        reverse = {token_id: word for word, token_id in self.vocabulary.items()}
        return " ".join(reverse[token_id] for token_id in token_ids)


class TextSpanResolutionTests(unittest.TestCase):
    def test_falls_back_to_prefix_alignment_for_invalid_offsets(self) -> None:
        tokenizer = BrokenOffsetTokenizer()
        text = "prefix target suffix"
        input_ids = [1, 2, 3]
        shifted_offsets = [(0, 1), (1, 2), (2, 3)]

        span = resolve_text_span_to_tokens(
            text,
            shifted_offsets,
            input_ids,
            tokenizer,
            text.index("target"),
            text.index("target") + len("target"),
            single_token=True,
        )

        self.assertEqual((span.start, span.end), (1, 2))
        self.assertEqual(span.token_ids, [2])
        self.assertEqual(span.text, "target")

    def test_rejects_same_text_at_the_wrong_token_position(self) -> None:
        tokenizer = RepeatedWordTokenizer()
        text = "target middle target"
        char_start = text.rindex("target")
        input_ids = [1, 2, 1]
        offsets_pointing_to_wrong_occurrence = [
            (char_start, len(text)),
            (7, 13),
            (0, 6),
        ]

        span = resolve_text_span_to_tokens(
            text,
            offsets_pointing_to_wrong_occurrence,
            input_ids,
            tokenizer,
            char_start,
            len(text),
            single_token=True,
        )

        self.assertEqual((span.start, span.end), (2, 3))
        self.assertEqual(span.token_ids, [1])


if __name__ == "__main__":
    unittest.main()
