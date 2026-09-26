"""Check selected-head, final-row QK score ablation with GQA."""
import unittest
import json
from pathlib import Path
import tempfile

import torch

from introspection_core.qk_score_ablation import (
    rotated_capture, score_term, subtract_score_term, validate_frozen_prompt,
)


class QKScoreAblationTests(unittest.TestCase):
    def test_rotated_capture_preserves_values_and_layer_mapping(self):
        parts = {}
        hooks = rotated_capture([2, 7], parts, 'k')
        for layer, (name, callback) in zip((2, 7), hooks):
            value = torch.full((1, 3, 2, 4), float(layer), dtype=torch.bfloat16)
            self.assertEqual(name, f'blocks.{layer}.attn.hook_rot_k')
            self.assertIs(callback(value, None), value)
            torch.testing.assert_close(parts[layer], value.float())

    def test_frozen_prompt_rejects_changed_tokens_or_positions(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / 'complete.json'
            reference.write_text(json.dumps(dict(input_token_ids=[1, 2, 3], positions=[1])))
            validate_frozen_prompt(torch.tensor([[1, 2, 3]]), [1], reference)
            with self.assertRaisesRegex(RuntimeError, 'frozen prompt'):
                validate_frozen_prompt(torch.tensor([[1, 4, 3]]), [1], reference)
            with self.assertRaisesRegex(RuntimeError, 'frozen prompt'):
                validate_frozen_prompt(torch.tensor([[1, 2, 3]]), [2], reference)

    def test_terms_reconstruct_clean_and_injected_scores(self):
        torch.manual_seed(9)
        k0 = torch.randn(1, 5, 2, 4, dtype=torch.float64)
        k1 = k0 + torch.randn(3, 5, 2, 4, dtype=torch.float64) * 0.2
        q0 = torch.randn(1, 6, 4, dtype=torch.float64)
        q1 = q0 + torch.randn(3, 6, 4, dtype=torch.float64) * 0.1
        heads = [0, 4]
        query = score_term(k0, k1, q0, q1, heads, "query")
        key = score_term(k0, k1, q0, q1, heads, "key")
        for batch in range(3):
            for column, head in enumerate(heads):
                group = head // 3
                clean = k0[0, :, group] @ q0[0, head] / 2
                injected = k1[batch, :, group] @ q1[batch, head] / 2
                self.assertTrue(torch.allclose(query[batch, column] + key[batch, column],
                                               (injected - clean).float(), atol=1e-6))
                self.assertTrue(torch.allclose(injected.float() - query[batch, column],
                                               (k1[batch, :, group] @ q1[batch, head] / 2
                                                - k0[0, :, group] @ (q1[batch, head] - q0[0, head]) / 2).float(), atol=1e-6))

    def test_patch_only_final_row_and_selected_heads(self):
        scores = torch.randn(2, 6, 3, 5)
        original = scores.clone()
        term = torch.randn(2, 2, 5)
        patched = subtract_score_term(scores, term, [0, 4])
        self.assertTrue(torch.equal(patched[:, :, :-1], scores[:, :, :-1]))
        self.assertTrue(torch.equal(patched[:, [1, 2, 3, 5], -1], scores[:, [1, 2, 3, 5], -1]))
        self.assertTrue(torch.allclose(patched[:, [0, 4], -1], scores[:, [0, 4], -1] - term))
        self.assertTrue(torch.equal(scores, original))


if __name__ == "__main__":
    unittest.main()
