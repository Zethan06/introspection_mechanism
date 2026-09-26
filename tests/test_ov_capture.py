"""CPU checks of the OV capture: GQA indexing and the value observer."""
import unittest
from types import SimpleNamespace

import torch

from introspection_core.ov_capture import capture_attention_values, reconstruct_z


class OVCaptureTests(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(7)
        self.v0 = torch.randn(2, 5, 2, 3, generator=g, dtype=torch.float64)
        self.vi = torch.randn(2, 5, 2, 3, generator=g, dtype=torch.float64)
        self.a0 = torch.randn(2, 3, 5, generator=g, dtype=torch.float64).softmax(-1)
        self.ai = torch.randn(2, 3, 5, generator=g, dtype=torch.float64).softmax(-1)
        self.heads = [3, 0, 2]

    def test_reconstructed_z_matches_explicit_gqa_readout(self):
        z0, zi = reconstruct_z(self.a0, self.ai, self.v0, self.vi, heads=self.heads, n_heads=4)
        for j, h in enumerate(self.heads):
            torch.testing.assert_close(z0[:, j], (self.a0[:, j, :, None] * self.v0[:, :, h // 2]).sum(1))
            torch.testing.assert_close(zi[:, j], (self.ai[:, j, :, None] * self.vi[:, :, h // 2]).sum(1))

    def test_reject_invalid_gqa(self):
        with self.assertRaises(ValueError):
            reconstruct_z(self.a0, self.ai, self.v0, self.vi, heads=self.heads, n_heads=3)

    def test_capture_includes_final_value_and_removes_observer(self):
        linear = torch.nn.Linear(4, 6, bias=True)
        inputs = torch.arange(40).reshape(2, 5, 4).float()
        block = SimpleNamespace(attn=SimpleNamespace(v=SimpleNamespace(original_component=linear)))

        def forward(tokens, components, **kwargs):
            linear(inputs[:, :-1])
            linear(inputs[:, -1:])
            kwargs['attention_output'][0] = {'pattern': torch.ones(2, 2, 5) / 5}
            kwargs['residual_output'][0] = inputs[:, -1]
            return torch.zeros(2, 2, 3), None

        model = SimpleNamespace(bridge=SimpleNamespace(blocks=[block]),
                                cfg=SimpleNamespace(d_head=3), final_head_ov_inputs=forward)
        _, captured = capture_attention_values(model, torch.zeros(2, 5), [(0, 0), (0, 1)])
        torch.testing.assert_close(captured[0]['v'], linear(inputs).reshape(2, 5, 2, 3))
        self.assertEqual(len(linear._forward_hooks), 0)


if __name__ == '__main__':
    unittest.main()
