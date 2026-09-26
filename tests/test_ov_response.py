import unittest

import numpy as np
import torch

from introspection_core.ov_response import comparison_weights, response_norms, summarize_responses


class OVResponseTests(unittest.TestCase):
    def test_norm_matches_explicit_output_projection(self):
        generator = torch.Generator().manual_seed(3)
        z = torch.randn(7, 2, 3, generator=generator)
        weights = [torch.randn(8, 3, generator=generator) for _ in range(2)]
        actual = response_norms(z, weights)
        expected = torch.stack([(z[:, j].double() @ o.double().T).norm(dim=-1)
                                for j, o in enumerate(weights)], dim=1).numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_layer_weights_and_zero_selected_layer(self):
        heads = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)]
        weights = comparison_weights(heads, [(0, 0), (0, 1), (1, 0)])
        np.testing.assert_allclose(weights['non_STE_layer_matched'], [0, 0, 2/3, 0, 1/3, 0])
        for w in weights.values():
            self.assertAlmostEqual(w.sum(), 1)

    def test_reject_missing_control_duplicate_and_nonfinite(self):
        with self.assertRaises(ValueError):
            comparison_weights([(0, 0), (1, 0)], [(0, 0)])
        with self.assertRaises(ValueError):
            comparison_weights([(0, 0), (0, 1)], [(0, 0), (0, 0)])
        with self.assertRaises(ValueError):
            response_norms(torch.full((2, 1, 3), float('nan')), [torch.ones(4, 3)])

    def test_paired_contrast_and_exact_point_estimate(self):
        # Both heads share concept variation. Their group contrasts are identical,
        # so synchronous bootstrap must yield exactly zero difference, not noise.
        x = np.array([8., 6., 4., 1.])[:, None, None] * np.ones((1, 2, 2))
        x[:, :, 0] += 2
        weights = comparison_weights([(0, 0), (0, 1)], [(0, 0)])
        rows = summarize_responses(x, weights, valid_count=2, repeats=17)
        row = next(r for r in rows if r['group'] == 'STE_minus_non_STE_layer_matched')
        self.assertEqual(row['difference'], 0)
        self.assertEqual(row['difference_ci_low'], 0)
        self.assertEqual(row['difference_ci_high'], 0)
        ste = next(r for r in rows if r['group'] == 'STE')
        self.assertEqual(ste['difference'], 4.5)

    def test_norm_precedes_averaging_and_zero_ratio(self):
        z = torch.tensor([[[1.]], [[-1.]], [[0.]], [[0.]]])
        norms = response_norms(z, [torch.ones(1, 1)]).reshape(2, 2, 1)
        result = summarize_responses(norms, {'STE': np.ones(1)}, valid_count=1, repeats=3)[0]
        self.assertEqual(result['valid_mean'], 1)
        self.assertIsNone(result['valid_bottom_ratio'])
