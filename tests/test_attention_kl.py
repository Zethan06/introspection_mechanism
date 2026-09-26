import unittest

import torch

from introspection_core.attention_kl import attention_kl_metrics


class AttentionKlTests(unittest.TestCase):
    def test_constant_shift_is_invisible(self):
        scores = torch.tensor([[[1., -2., 3., 0.]]], dtype=torch.float64)
        values = attention_kl_metrics(scores, torch.full_like(scores, 100), torch.tensor([0, 2]))
        for key in ('full_kl', 'subset_conditional_kl', 'subset_coarse_kl', 'full_tv'):
            self.assertAlmostEqual(values[key].item(), 0., places=12)

    def test_subset_mass_change_survives_coarsening(self):
        scores = torch.zeros(1, 1, 4, dtype=torch.float64)
        removed = torch.tensor([[[2., 2., 0., 0.]]], dtype=torch.float64)
        values = attention_kl_metrics(scores, removed, torch.tensor([0, 1]))
        self.assertAlmostEqual(values['subset_conditional_kl'].item(), 0.)
        self.assertGreater(values['subset_coarse_kl'].item(), 0.4)
        self.assertLess(values['subset_mass_delta'].item(), -0.3)
        torch.testing.assert_close(values['full_kl'], values['subset_coarse_kl'])

    def test_matches_distribution_kl_and_data_processing(self):
        generator = torch.Generator().manual_seed(9)
        scores = torch.randn(3, 2, 11, generator=generator, dtype=torch.float64)
        removed = torch.randn(3, 2, 11, generator=generator, dtype=torch.float64)
        values = attention_kl_metrics(scores, removed, torch.tensor([1, 3, 8]))
        expected = torch.distributions.kl_divergence(
            torch.distributions.Categorical(logits=scores),
            torch.distributions.Categorical(logits=scores - removed))
        torch.testing.assert_close(values['full_kl'], expected)
        self.assertTrue((values['subset_coarse_kl'] <= values['full_kl'] + 1e-12).all())

    def test_rejects_invalid_subset(self):
        scores = torch.zeros(1, 1, 4)
        for indices in ([1, 1], [4], [], [0, 1, 2, 3]):
            with self.assertRaises(ValueError):
                attention_kl_metrics(scores, scores, torch.tensor(indices))
