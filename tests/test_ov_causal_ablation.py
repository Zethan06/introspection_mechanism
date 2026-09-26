"""OV intervention algebra and output-hook boundaries."""
import unittest
from types import SimpleNamespace

import torch

from introspection_core.ov_causal_ablation import ov_terms, intervention_vectors, subtract_ov_writes
from scripts.run_ov_causal_ablation import write_summary
import csv
import tempfile
from pathlib import Path


class OVCausalTests(unittest.TestCase):
    def test_full_terms_preserve_interaction_and_sparse_gqa(self):
        g = torch.Generator().manual_seed(29)
        v0, vi = [torch.randn(2, 5, 2, 3, generator=g, dtype=torch.float64) for _ in range(2)]
        a0, ai = [torch.randn(2, 3, 5, generator=g, dtype=torch.float64).softmax(-1) for _ in range(2)]
        wo = torch.randn(3, 3, 7, generator=g, dtype=torch.float64)
        heads = [3, 0, 2]
        terms = ov_terms(a0, ai, v0, vi, wo, heads=heads, n_heads=4)
        for j,h in enumerate(heads):
            clean = (a0[:,j,:,None]*v0[:,:,h//2]).sum(1) @ wo[j]
            injected = (ai[:,j,:,None]*vi[:,:,h//2]).sum(1) @ wo[j]
            torch.testing.assert_close(terms['R'][:,j]+terms['C'][:,j], injected-clean)
            # Removing C equals injected attention reading clean values.
            torch.testing.assert_close(injected-terms['C'][:,j], (ai[:,j,:,None]*v0[:,:,h//2]).sum(1) @ wo[j])
            # Removing R retains the interaction (not simply clean attention).
            expected = clean + (ai[:,j,:,None]*(vi-v0)[:,:,h//2]).sum(1) @ wo[j]
            torch.testing.assert_close(injected-terms['R'][:,j], expected)
        no_r = ov_terms(ai, ai, v0, vi, wo, heads=heads, n_heads=4)
        self.assertEqual(no_r['R'].abs().max().item(), 0)
        no_c = ov_terms(a0, ai, v0, v0, wo, heads=heads, n_heads=4)
        self.assertEqual(no_c['C'].abs().max().item(), 0)
        summed = intervention_vectors({4: terms})[4]
        torch.testing.assert_close(summed['R'], terms['R'].sum(1))
        torch.testing.assert_close(summed['RC'], terms['R'].sum(1) + terms['C'].sum(1))

    def test_hook_boundary_and_cleanup(self):
        linear = torch.nn.Linear(3, 4)
        model = SimpleNamespace(bridge=SimpleNamespace(blocks=[SimpleNamespace(attn=SimpleNamespace(o=SimpleNamespace(original_component=linear)))]))
        x = torch.randn(2,1,3); delta = torch.randn(2,4)
        native = linear(x)
        with subtract_ov_writes(model, {0: delta}):
            torch.testing.assert_close(linear(x), native-delta[:,None])
        self.assertEqual(len(linear._forward_hooks),0)
        with self.assertRaises(ValueError):
            with subtract_ov_writes(model, {0: delta}):
                linear(x.expand(-1,2,-1))
        self.assertEqual(len(linear._forward_hooks),0)
        with self.assertRaises(RuntimeError):
            with subtract_ov_writes(model, {0: delta}):
                pass

    def test_summary_keeps_none_in_accuracy_denominator(self):
        rows = [dict(group='g',cluster=0,concept_index=0,position=i,condition=c,
                     prediction=p,correct=int(p==i))
                for c,preds in [('native',[0,1]),('full_minus_C',[0,10])]
                for i,p in enumerate(preds)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'summary.csv'; write_summary(path,rows)
            with path.open() as handle:
                summary = {r['condition']: r for r in csv.DictReader(handle)}
        self.assertEqual(float(summary['full_minus_C']['accuracy']),.5)
        self.assertEqual(float(summary['full_minus_C']['delta_accuracy_pp']),-50)
        self.assertEqual(float(summary['full_minus_C']['accuracy_given_number']),1)


class OVProbabilitySummaryTests(unittest.TestCase):
    def test_paired_probability_drop_without_argmax_change(self):
        rows=[]
        for condition,probs in [('native',[.8,.6]),('full_minus_C_top5',[.5,.4])]:
            for position,p in enumerate(probs):
                rows.append(dict(group='g',cluster=0,concept_index=0,position=position,
                    condition=condition,prediction=position,correct=1,
                    target_probability=p,target_candidate_probability=p+.05,
                    none_probability=.05,none_candidate_probability=.06))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'summary.csv'
            write_summary(path,rows)
            with path.open() as handle:
                summary={r['condition']:r for r in csv.DictReader(handle)}
        self.assertEqual(float(summary['full_minus_C_top5']['accuracy']),1.)
        self.assertAlmostEqual(float(summary['full_minus_C_top5']['mean_target_probability']),.45)
        self.assertAlmostEqual(float(summary['full_minus_C_top5']['delta_target_probability_pp']),-25.)


if __name__ == '__main__':
    unittest.main()
