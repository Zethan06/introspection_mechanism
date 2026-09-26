"""Dense-reference proof for exact reduced OV SVD and source subsets."""
import unittest
import torch
from introspection_core.ov_output_svd import output_qr, output_svd_metrics


class OVOutputSVDTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.v0=torch.randn(2,12,2,4,dtype=torch.float64)
        self.vi=torch.randn_like(self.v0)
        self.a0=torch.randn(2,3,12,dtype=torch.float64).softmax(-1)
        self.ai=torch.randn_like(self.a0).softmax(-1)
        self.weights=torch.randn(3,4,7,dtype=torch.float64)
        self.heads=[3,0,2]

    def test_dense_equivalence_full_and_ten_sources(self):
        r=output_qr(self.weights)
        for positions in (None,list(range(1,11))):
            metrics=output_svd_metrics(self.a0,self.ai,self.v0,self.vi,r,heads=self.heads,n_heads=4,positions=positions)
            subset=list(range(12)) if positions is None else positions
            for b in range(2):
                for j,h in enumerate(self.heads):
                    dv=(self.vi-self.v0)[b,subset,h//2]
                    m=self.weights[j].T @ dv.T
                    left,s,vh=torch.linalg.svd(m,full_matrices=False)
                    a=self.ai[b,j,subset]
                    reading=vh @ a
                    coeff=s*reading
                    torch.testing.assert_close(metrics['sigma1'][b,j],s[0])
                    torch.testing.assert_close(metrics['reading1_abs'][b,j],reading[0].abs())
                    torch.testing.assert_close(metrics['reading1_cos_abs'][b,j],reading[0].abs()/a.norm())
                    torch.testing.assert_close(metrics['response_norm'][b,j],(m@a).norm())
                    torch.testing.assert_close(metrics['first_mode_norm'][b,j],coeff[0].abs())
                    torch.testing.assert_close(metrics['response_energy_top1'][b,j],coeff[0].square()/coeff.square().sum())
                    torch.testing.assert_close(metrics['attention_mass'][b,j],a.sum())
                    torch.testing.assert_close(metrics['energy_top10'][b,j],torch.tensor(1.,dtype=torch.float64))

    def test_fixed_attention_and_values(self):
        r=output_qr(self.weights)
        fixed=output_svd_metrics(self.ai,self.ai,self.v0,self.vi,r,heads=self.heads,n_heads=4)
        self.assertEqual(fixed['interaction_response_norm'].max().item(),0)
        zero=output_svd_metrics(self.a0,self.ai,self.v0,self.v0,r,heads=self.heads,n_heads=4)
        for values in zero.values():
            self.assertTrue(torch.isfinite(values).all())
        for key in ('sigma1','reading1_abs','response_norm','matrix_nonzero','response_energy_top1'):
            self.assertEqual(zero[key].max().item(),0)

    def test_invalid_inputs(self):
        r=output_qr(self.weights)
        with self.assertRaises(ValueError):
            output_svd_metrics(self.a0,self.ai,self.v0,self.vi,r,heads=self.heads,n_heads=3)
        with self.assertRaises(ValueError):
            output_svd_metrics(self.a0,self.ai,self.v0,self.vi,r,heads=self.heads,n_heads=4,positions=[1,1])


class OVModalWriteTests(unittest.TestCase):
    def test_dense_topk_subtraction_and_remainder(self):
        from introspection_core.ov_output_svd import topk_content_writes
        torch.manual_seed(31)
        v0=torch.randn(2,9,2,6,dtype=torch.float64)
        vi=torch.randn_like(v0)
        a=torch.randn(2,3,9,dtype=torch.float64).softmax(-1)
        weights=torch.randn(3,6,11,dtype=torch.float64)
        q,r=torch.linalg.qr(weights.transpose(-1,-2),mode='reduced')
        heads=[3,0,2]
        from unittest.mock import patch
        with patch('torch.linalg.svd', wraps=torch.linalg.svd) as svd:
            sweep=topk_content_writes(a,v0,vi,q,r,heads=heads,n_heads=4,top_k=6,top_ks=[1,2,5,6])
            self.assertEqual(svd.call_count,1)
        for k in (1,2,5,6):
            out=topk_content_writes(a,v0,vi,q,r,heads=heads,n_heads=4,top_k=k)
            torch.testing.assert_close(sweep[f'top{k}'],out['top'])
            for b in range(2):
                for j,h in enumerate(heads):
                    m=weights[j].T @ (vi-v0)[b,:,h//2].T
                    left,s,vh=torch.linalg.svd(m,full_matrices=False)
                    expected=(left[:,:k]*s[:k]) @ vh[:k] @ a[b,j]
                    torch.testing.assert_close(out['top'][b,j],expected)
                    torch.testing.assert_close(out['full'][b,j],m @ a[b,j])
            torch.testing.assert_close(out['top']+out['rest'],out['full'])
            torch.testing.assert_close((out['top']*out['rest']).sum(-1),torch.zeros(2,3,dtype=torch.float64))
        torch.testing.assert_close(out['rest'],torch.zeros_like(out['rest']),atol=1e-12,rtol=0)
        zero=topk_content_writes(a,v0,v0,q,r,heads=heads,n_heads=4,top_k=5)
        self.assertEqual(zero['top'].abs().max().item(),0)


if __name__=='__main__':
    unittest.main()
