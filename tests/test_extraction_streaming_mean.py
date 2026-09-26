"""Tests for memory-bounded vocabulary baseline extraction."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from introspection_core.extraction import (
    extract_mean_last_token_residuals_by_layer,
    load_concept_vector_payload,
)


class _LengthTokenizer:
    eos_token = "<eos>"
    pad_token = None
    chat_template = "available"

    def apply_chat_template(self, messages, **kwargs) -> str:
        del kwargs
        return messages[0]["content"]

    def __call__(self, prompts, **kwargs):
        del kwargs
        lengths = torch.tensor([[len(prompt)] for prompt in prompts])
        return _Encoding(
            input_ids=lengths,
            attention_mask=torch.ones_like(lengths),
        )


class _Encoding(dict):
    def __init__(self, **values):
        super().__init__(values)
        self.__dict__.update(values)

    def to(self, device):
        return _Encoding(
            **{key: value.to(device) for key, value in self.items()}
        )


class _FakeModel:
    def __init__(self) -> None:
        self.tokenizer = _LengthTokenizer()
        self.bridge = SimpleNamespace(cfg=SimpleNamespace(device="cpu"))

    def resid_hook_name(self, layer: int) -> str:
        return f"layer_{layer}"

    def run_with_cache(self, input_ids, *, names, attention_mask):
        del attention_mask
        cache = {}
        for layer in (1, 3):
            name = self.resid_hook_name(layer)
            if names(name):
                values = input_ids.to(torch.float32) + layer
                cache[name] = torch.stack((values, values * 2), dim=-1)
        return None, cache


class StreamingBaselineTests(unittest.TestCase):
    def test_matches_direct_mean_across_batches(self) -> None:
        words = ["a", "alphabet", "cat"]
        means = extract_mean_last_token_residuals_by_layer(
            _FakeModel(), words, layers=[1, 3], batch_size=2
        )
        prompt_lengths = torch.tensor(
            [len(f"Tell me about {word}.") for word in words],
            dtype=torch.float32,
        )

        for layer in (1, 3):
            expected_first = (prompt_lengths + layer).mean()
            self.assertTrue(
                torch.allclose(
                    means[layer],
                    torch.tensor([expected_first, expected_first * 2]),
                )
            )

    def test_loads_validated_cached_concept_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "vectors.pt"
            torch.save(
                {
                    "layer": 3,
                    "rows": [{"word": "alpha"}, {"word": "beta"}],
                    "vectors": torch.tensor([[3.0, 4.0], [0.0, 2.0]]),
                },
                path,
            )

            vectors = load_concept_vector_payload(
                path,
                concepts=["alpha", "beta"],
                layer=3,
            )

        torch.testing.assert_close(vectors.norm(dim=-1), torch.ones(2))

    def test_selects_a_split_out_of_a_population_payload(self) -> None:
        # The canonical payload covers every concept; a split asks for a subset
        # in its own order and must get those exact rows back.
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "vectors.pt"
            torch.save(
                {
                    "layer": 3,
                    "concepts": ["alpha", "beta", "gamma"],
                    "unit_vectors": torch.tensor(
                        [[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]]
                    ),
                },
                path,
            )

            vectors = load_concept_vector_payload(
                path,
                concepts=["gamma", "alpha"],
                layer=3,
            )

        torch.testing.assert_close(
            vectors, torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        )

    def test_rejects_concepts_absent_from_the_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "vectors.pt"
            torch.save(
                {
                    "layer": 3,
                    "rows": [{"word": "beta"}, {"word": "alpha"}],
                    "vectors": torch.eye(2),
                },
                path,
            )

            with self.assertRaisesRegex(ValueError, "missing 1 of 2"):
                load_concept_vector_payload(
                    path,
                    concepts=["alpha", "delta"],
                    layer=3,
                )

    def test_rejects_cached_vectors_from_another_layer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "vectors.pt"
            torch.save(
                {
                    "layer": 3,
                    "rows": [{"word": "alpha"}, {"word": "beta"}],
                    "vectors": torch.eye(2),
                },
                path,
            )

            with self.assertRaisesRegex(ValueError, "different layer"):
                load_concept_vector_payload(
                    path,
                    concepts=["alpha", "beta"],
                    layer=7,
                )


if __name__ == "__main__":
    unittest.main()
