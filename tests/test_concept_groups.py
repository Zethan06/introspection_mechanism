"""Concept-group loading and the model helpers the Section 4 captures rely on."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from introspection_core.concept_groups import load_populations
from introspection_core.model import HookedModel


class ConceptGroupTests(unittest.TestCase):
    def test_cached_scoring_retains_inference_mode_without_outer_context(self):
        model = object.__new__(HookedModel)
        observed = []

        def capture(*args, **kwargs):
            observed.append(torch.is_inference_mode_enabled())
            return torch.zeros(1, 2), torch.zeros(1, 4)

        model._incremental_last_token_candidate_logits = capture
        model.incremental_last_token_candidate_stats(torch.zeros(1, 1, dtype=torch.long),
            prefix_kv_cache=None, prefix_length=2, candidate_token_ids=[0, 1])
        self.assertEqual(observed, [True])

    def test_output_weights_match_actual_linear_for_square_and_rectangular_shapes(self):
        for width in (4, 3):
            with self.subTest(width=width):
                linear = torch.nn.Linear(4, width, bias=False)
                linear.weight.data.copy_(torch.arange(width * 4).reshape(width, 4).float())
                model = object.__new__(HookedModel)
                model.cfg = SimpleNamespace(n_layers=1, n_heads=2, d_head=2, d_model=width)
                model.bridge = SimpleNamespace(blocks=[SimpleNamespace(attn=SimpleNamespace(o=SimpleNamespace(original_component=linear)))])
                z = torch.tensor([[1., 2.], [3., 4.]])
                per_head = torch.einsum("hd,hdm->hm", z, model.attention_output_weights(0))
                torch.testing.assert_close(per_head.sum(0), linear(z.flatten()))
                if width == 4:
                    wrong = torch.einsum("hd,hdm->m", z, linear.weight.reshape(2, 2, 4))
                    self.assertFalse(torch.equal(wrong, linear(z.flatten())))

    def test_populations_preserve_literal_names_and_reject_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            high, low = Path(directory) / "high.csv", Path(directory) / "low.csv"
            high.write_text("concept\nNA\nNULL\n")
            low.write_text("concept\nHugo\nPSP\n")
            names, groups = load_populations(high, low, 2)
            self.assertEqual(names, ["NA", "NULL", "Hugo", "PSP"])
            self.assertEqual(groups, ["validation100", "validation100", "bottom100", "bottom100"])
            low.write_text("concept\nNA\nPSP\n")
            with self.assertRaises(ValueError):
                load_populations(high, low, 2)


if __name__ == "__main__":
    unittest.main()
