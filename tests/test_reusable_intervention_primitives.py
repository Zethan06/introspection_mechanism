from __future__ import annotations

import contextlib
import unittest
from types import SimpleNamespace

import torch

from introspection_core.injection import make_injection_hook
from introspection_core.model import HookedModel


class InjectionHookBuilderTest(unittest.TestCase):
    def test_builds_composable_per_row_unit_injection(self) -> None:
        model = SimpleNamespace(resid_hook_name=lambda layer: f"resid_{layer}")
        vectors = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        name, hook = make_injection_hook(
            model,
            layer=3,
            positions=[(0, 1), (2, 3)],
            vector=vectors,
            strength=2.0,
            scale="unit",
        )
        activation = torch.zeros(2, 3, 2)
        output = hook(activation, None)
        self.assertEqual(name, "resid_3")
        torch.testing.assert_close(output[0, 0], torch.tensor([2.0, 4.0]))
        torch.testing.assert_close(output[1, 2], torch.tensor([6.0, 8.0]))
        torch.testing.assert_close(output[0, 1], torch.zeros(2))


class _FakeBridge:
    def __init__(self) -> None:
        self.cfg = SimpleNamespace(device="cpu")
        self._hooks = []
        self.weight = torch.tensor(
            [[1.0, 0.0, 1.0], [0.0, 1.0, -1.0]],
        )

    @contextlib.contextmanager
    def hooks(self, *, fwd_hooks):
        previous = self._hooks
        self._hooks = list(fwd_hooks)
        try:
            yield
        finally:
            self._hooks = previous

    def __call__(self, tokens, return_type=None, **kwargs):
        del return_type, kwargs
        residual = torch.stack((tokens.float(), -tokens.float()), dim=-1)
        for name, hook in self._hooks:
            if name != "ln_final.hook_out":
                residual = hook(residual, None)
        return residual

    def unembed(self, residual):
        return residual @ self.weight


class DifferentiableCandidateLogitsTest(unittest.TestCase):
    def _model(self) -> HookedModel:
        model = HookedModel.__new__(HookedModel)
        model.bridge = _FakeBridge()
        model.cfg = SimpleNamespace(n_layers=1, output_logits_soft_cap=0.0)
        model._hook_names = {
            "blocks.0.hook_resid_post",
            "ln_final.hook_out",
        }
        return model

    def test_retains_gradient_through_intervention_hook(self) -> None:
        model = self._model()
        alpha = torch.tensor(0.5, requires_grad=True)

        def intervention(residual, hook):
            del hook
            output = residual.clone()
            output[:, -1, 0] += alpha
            return output

        logits = model.differentiable_last_token_candidate_logits(
            torch.tensor([[1, 2]]),
            candidate_token_ids=[0, 2],
            fwd_hooks=[("blocks.0.hook_resid_post", intervention)],
        )
        logits.sum().backward()
        self.assertIsNotNone(alpha.grad)
        self.assertNotEqual(float(alpha.grad), 0.0)

    def test_cached_final_token_path_retains_hook_gradient(self) -> None:
        model = self._model()
        alpha = torch.tensor(0.5, requires_grad=True)

        def intervention(residual, hook):
            del hook
            output = residual.clone()
            output[:, 0, 0] += alpha
            return output

        logits = model.differentiable_incremental_last_token_candidate_logits(
            torch.tensor([[2]]),
            prefix_kv_cache=((torch.zeros(1, 1), torch.zeros(1, 1)),),
            prefix_length=3,
            candidate_token_ids=[0, 2],
            fwd_hooks=[("blocks.0.hook_resid_post", intervention)],
        )
        logits.sum().backward()
        self.assertIsNotNone(alpha.grad)
        self.assertNotEqual(float(alpha.grad), 0.0)


if __name__ == "__main__":
    unittest.main()
