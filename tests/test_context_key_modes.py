"""CPU checks for native QK capture, spectral identities, and aggregation."""
import contextlib
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch

from introspection_core.context_key_modes import (
    capture_context_qk, context_key_mode_metrics, load_context_mode_captures, summarize_context_modes,
)


class ContextKeyModeTests(unittest.TestCase):
    def test_first_mode_rows_preserve_sign_and_residual_energy(self):
        generator = torch.Generator().manual_seed(29)
        for positions, width in ((3, 8), (8, 3)):
            key = torch.randn(2, 2, positions, width, generator=generator, dtype=torch.float64)
            query = torch.randn(2, 6, width, generator=generator, dtype=torch.float64)
            result = context_key_mode_metrics(key, query, include_first_mode_rows=True)
            u, singular, vh = torch.linalg.svd(key, full_matrices=False)
            coefficient = (vh[:, :, 0].repeat_interleave(3, 1) * query).sum(-1)
            expected = (u[..., 0].repeat_interleave(3, 1)
                        * singular[..., 0].repeat_interleave(3, 1)[..., None]
                        * coefficient[..., None] / width**0.5)
            first = result['first_mode_score_rows']
            torch.testing.assert_close(first, expected)
            full = torch.einsum('bhtd,bhd->bht', key.repeat_interleave(3, 1), query) / width**0.5
            residual = full - first
            torch.testing.assert_close(residual.norm(dim=-1), result['remaining_modes_score_norm'])
            torch.testing.assert_close(first.square().sum(-1) / full.square().sum(-1),
                                       result['first_read_energy_fraction'])
            reverse = context_key_mode_metrics(key, -query, include_first_mode_rows=True)
            torch.testing.assert_close(reverse['first_mode_score_rows'], -first)

    def test_target_rows_from_full_context_svd_preserve_mode_energy(self):
        generator = torch.Generator().manual_seed(37)
        key = torch.randn(2, 2, 9, 4, generator=generator, dtype=torch.float64)
        query = torch.randn(2, 6, 4, generator=generator, dtype=torch.float64)
        selected = torch.tensor([1, 5, 7], dtype=torch.long)
        result = context_key_mode_metrics(key, query, target_row_indices=selected)
        u, singular, vh = torch.linalg.svd(key, full_matrices=False)
        coefficient = (vh.repeat_interleave(3, 1) * query[:, :, None]).sum(-1)
        modes = (u.index_select(2, selected).repeat_interleave(3, 1)
                 * singular.repeat_interleave(3, 1)[..., None, :]
                 * coefficient[:, :, None, :] / key.shape[-1]**0.5)
        full = torch.einsum('bhtd,bhd->bht', key.repeat_interleave(3, 1).index_select(2, selected),
                            query) / key.shape[-1]**0.5
        torch.testing.assert_close(result['target_score_rows'], full)
        torch.testing.assert_close(result['target_first_mode_score_rows'], modes[..., 0])
        torch.testing.assert_close(result['target_modal_energy'], modes.square().sum(-1))
        torch.testing.assert_close(result['target_first_modal_energy_fraction'],
                                   modes[..., 0].square() / modes.square().sum(-1))

    def test_modes_match_direct_svd_with_gqa(self):
        generator = torch.Generator().manual_seed(8)
        key = torch.randn(2, 2, 7, 4, generator=generator, dtype=torch.float64)
        query = torch.randn(2, 6, 4, generator=generator, dtype=torch.float64)
        result = context_key_mode_metrics(key, query)
        _, singular, vh = torch.linalg.svd(key, full_matrices=False)
        expanded = key.repeat_interleave(3, 1)
        direct = torch.einsum('bhtd,bhd->bht', expanded, query).norm(dim=-1) / 2
        projection = (vh[:, :, 0].repeat_interleave(3, 1) * query).sum(-1).abs()
        expected_first = singular[:, :, 0].repeat_interleave(3, 1) * projection / 2
        torch.testing.assert_close(result['all_modes_score_norm'], direct)
        torch.testing.assert_close(result['first_mode_score_norm'], expected_first)
        torch.testing.assert_close(result['first_query_projection_absolute'], projection)
        torch.testing.assert_close(result['all_modes_score_norm'].square(),
                                   result['first_mode_score_norm'].square() + result['remaining_modes_score_norm'].square())

    def test_ten_position_modes_match_direct_svd(self):
        generator = torch.Generator().manual_seed(19)
        key = torch.randn(2, 2, 10, 16, generator=generator, dtype=torch.float64)
        query = torch.randn(2, 4, 16, generator=generator, dtype=torch.float64)
        result = context_key_mode_metrics(key, query)
        _, singular, vh = torch.linalg.svd(key, full_matrices=False)
        projection = (vh[:, :, 0].repeat_interleave(2, 1) * query).sum(-1).abs()
        torch.testing.assert_close(result['sigma1'], singular[:, :, 0].repeat_interleave(2, 1))
        torch.testing.assert_close(result['first_query_projection_absolute'], projection)
        expected = torch.einsum('bhtd,bhd->bht', key.repeat_interleave(2, 1), query).norm(dim=-1) / 4
        torch.testing.assert_close(result['all_modes_score_norm'], expected)

    def test_query_state_changes_response_but_not_spectrum(self):
        key = torch.diag(torch.tensor([4., 1.], dtype=torch.float64))[None, None]
        clean = context_key_mode_metrics(key, torch.tensor([[[0., 1.]]], dtype=torch.float64))
        injected = context_key_mode_metrics(key, torch.tensor([[[1., 0.]]], dtype=torch.float64))
        torch.testing.assert_close(clean['sigma1'], injected['sigma1'])
        self.assertEqual(float(clean['first_mode_score_norm']), 0)
        self.assertAlmostEqual(float(injected['all_modes_score_norm'] / clean['all_modes_score_norm']), 4)

    def test_zero_rank_deficient_and_repeated_singular_values(self):
        query = torch.ones(1, 1, 4, dtype=torch.float64)
        for key in (torch.zeros(1, 1, 2, 4, dtype=torch.float64),
                    torch.ones(1, 1, 2, 4, dtype=torch.float64),
                    torch.eye(4, dtype=torch.float64)[None, None]):
            metrics = context_key_mode_metrics(key, query)
            self.assertTrue(all(torch.isfinite(value).all() for value in metrics.values()))
        repeated = context_key_mode_metrics(torch.eye(4)[None, None], query)
        self.assertEqual(float(repeated['first_spectral_gap_fraction']), 0)
        zero_query = context_key_mode_metrics(torch.ones(1, 1, 2, 4), torch.zeros(1, 1, 4))
        self.assertEqual(float(zero_query['all_modes_score_norm']), 0)

    def test_invalid_gqa_and_nonfinite(self):
        with self.assertRaises(ValueError):
            context_key_mode_metrics(torch.ones(1, 2, 3, 4), torch.ones(1, 3, 4))
        with self.assertRaises(ValueError):
            context_key_mode_metrics(torch.full((1, 1, 3, 4), torch.nan), torch.ones(1, 1, 4))

    def test_ratio_of_means_and_layer_matching(self):
        heads = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)]
        selected = [(0, 0), (1, 0), (1, 1)]
        values = np.array([[[10, 2, 20, 30, 8], [30, 4, 40, 50, 10]],
                           [[6, 1, 12, 18, 4], [10, 3, 16, 22, 6]],
                           [[1, 1, 2, 3, 2], [3, 1, 4, 5, 2]],
                           [[4, 1, 5, 6, 2], [6, 1, 7, 8, 2]]], dtype=float)
        rows, contrasts = summarize_context_modes(values, heads, selected, valid_count=2, repeats=40)
        lookup = {row['group']: row for row in rows}
        means = values.mean(1)
        valid = means[:2, [0, 2, 3]].mean()
        bottom = means[2:, [0, 2, 3]].mean()
        self.assertAlmostEqual(lookup['STE']['ratio'], valid / bottom)
        matched_valid = means[:2, 1].mean() / 3 + means[:2, 4].mean() * 2 / 3
        self.assertAlmostEqual(lookup['non_STE_layer_matched']['valid'], matched_valid)
        self.assertEqual(len(contrasts), 2)
        zero, _ = summarize_context_modes(np.zeros_like(values), heads, selected, valid_count=2, repeats=2)
        self.assertIsNone(zero[0]['ratio'])
        with self.assertRaises(ValueError):
            summarize_context_modes(values, heads, [(0, 0), (0, 1)], valid_count=2)

    def test_capture_prefix_and_final_and_hook_cleanup(self):
        class Bridge:
            active = []

            @contextlib.contextmanager
            def hooks(self, fwd_hooks):
                self.active = fwd_hooks
                try:
                    yield
                finally:
                    self.active = []

        class Model:
            bridge = Bridge()

            def final_head_ov_inputs(self, tokens, components, fwd_hooks):
                for sequence, fill in [(tokens.shape[1]-1, 1.), (1, 2.)]:
                    for _, hook in self.bridge.active:
                        value = torch.full((len(tokens), sequence, 2, 4), fill)
                        self_outer.assertIs(hook(value, None), value)
                return torch.zeros(len(tokens), len(components), 4), None

        self_outer = self
        model = Model()
        _, result = capture_context_qk(model, torch.ones(2, 6), [(3, 0)])
        self.assertEqual(result[3]['k'].shape, (2, 2, 6, 4))
        self.assertTrue((result[3]['q'] == 2).all())
        self.assertTrue((result[3]['k'][:, :, :5] == 1).all())
        self.assertEqual(model.bridge.active, [])

    def test_cli_shards_and_query_provenance(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/analyze_ste_context_modes.py'
        spec = importlib.util.spec_from_file_location('analyze_context_modes', path)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            components = [[0, 0], [0, 1], [1, 0], [1, 1]]
            heads_csv = root / 'heads.csv'
            with heads_csv.open('w') as stream:
                writer = csv.writer(stream)
                writer.writerow(['model', 'layer', 'head', 'is_ste'])
                writer.writerows(['fixture', l, h, h == 0] for l, h in components)
            shards = []
            for start in (0, 4):
                shard = root / str(start)
                shard.mkdir()
                meta = dict(schema_version=1, status='complete', centered=False, query_state='clean',
                            components=components, concepts=['a', 'b', 'c', 'd'],
                            concept_groups=['validation100']*2+['bottom100']*2,
                            positions=[1, 2], prompt_index=0, final_query_position=5, head_width=2,
                            model='fixture', trial_start=start, trial_end=start+4)
                (shard / 'complete.json').write_text(json.dumps(meta))
                (shard / 'sources.json').write_text('{}')
                torch.save(dict(trial_indices=torch.arange(start, start+4),
                                metrics={'first_mode_score_norm': torch.ones(4, 4) * (2 if start == 0 else 1)}),
                           shard / 'metrics.pt')
                shards.append(str(shard))
            argv = ['analyze', '--captures', *shards, '--on-heads', str(heads_csv), '--off-heads', str(heads_csv),
                    '--model-id', 'fixture', '--results_dir', str(root / 'out'), '--bootstrap', '10']
            with patch.object(sys, 'argv', argv):
                cli.main()
            with (root / 'out/summary.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue(all(float(row['ratio']) == 2 for row in rows))
            self.assertTrue(all(row['query_state'] == 'clean' for row in rows))
            meta['query_state'] = 'injected'
            (Path(shards[1]) / 'complete.json').write_text(json.dumps(meta))
            argv[-3] = str(root / 'mixed')
            with patch.object(sys, 'argv', argv), self.assertRaisesRegex(ValueError, 'query_state'):
                cli.main()
            meta['query_state'] = 'clean'
            (Path(shards[1]) / 'complete.json').write_text(json.dumps(meta))
            argv[3] = shards[0]
            with patch.object(sys, 'argv', argv), self.assertRaisesRegex(ValueError, 'duplicate trials'):
                cli.main()

            second = root / 'cluster1'
            second.mkdir()
            meta.update(prompt_index=1, positions=[3, 4], final_query_position=7, trial_start=0, trial_end=8)
            (second / 'complete.json').write_text(json.dumps(meta))
            (second / 'sources.json').write_text('{}')
            torch.save(dict(trial_indices=torch.arange(8), metrics={'first_mode_score_norm': torch.full((8, 4), 4.)}),
                       second / 'metrics.pt')
            paths = [*shards, second]
            combined_meta, combined = load_context_mode_captures(paths, average_clusters=True, expected_clusters=2)
            self.assertEqual(combined_meta['cluster_count'], 2)
            self.assertTrue((combined['first_mode_score_norm'][:2] == 3).all())
            self.assertTrue((combined['first_mode_score_norm'][2:] == 2.5).all())
            with self.assertRaisesRegex(ValueError, 'multiple clusters'):
                load_context_mode_captures(paths)
            with self.assertRaisesRegex(ValueError, 'expected all 3'):
                load_context_mode_captures(paths, average_clusters=True, expected_clusters=3)
            argv = ['analyze', '--captures', *map(str, paths), '--on-heads', str(heads_csv),
                    '--off-heads', str(heads_csv), '--model-id', 'fixture', '--results_dir', str(root / 'average'),
                    '--bootstrap', '10', '--average-clusters', '--expected-clusters', '2']
            with patch.object(sys, 'argv', argv):
                cli.main()
            with (root / 'average/summary.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            # Ratio of cluster-averaged responses = 3/2.5, not mean([2/1,4/4]).
            self.assertTrue(all(float(row['ratio']) == 1.2 for row in rows))

    def test_shared_query_capture_partial_batch_and_resume(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/capture_ste_context_modes.py'
        spec = importlib.util.spec_from_file_location('capture_context_modes', path)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        model = SimpleNamespace(cfg=SimpleNamespace(d_head=2),
                                final_head_ov_inputs=lambda tokens, components: (torch.zeros(len(tokens), 1, 2), None))
        ex = SimpleNamespace(input_ids=torch.ones(1, 12, dtype=torch.long),
                             injection_spans=[SimpleNamespace(start=i+1) for i in range(10)])
        source = {'metadata': {'concepts': ['a', 'b'], 'concept_groups': ['validation100', 'bottom100'],
                              'frozen_prompt_config': {'scale_mode': 'relative_hidden_norm'}, 'model': 'fixture',
                              'args': {'injection_layer': 0, 'strength': 3}}}
        args = SimpleNamespace(trial_start=0, trial_end=3, batch_size=2, model=None, device='cpu', resume=False)
        calls = []

        def fake_capture(model, tokens, components, fwd_hooks=()):
            calls.append(len(tokens))
            injected = bool(fwd_hooks)
            return torch.zeros(len(tokens), 1, 2), {0: {
                'q': torch.ones(len(tokens), 1, 2) * (2 if injected else 1),
                'k': torch.ones(len(tokens), 1, 12, 2) * (1 if injected else 0)}}

        def fake_batch(trials, **kwargs):
            return kwargs['base_tokens'].repeat(len(trials), 1), object()

        with tempfile.TemporaryDirectory() as temporary:
            outputs = {state: Path(temporary) / state for state in ('clean', 'injected')}
            with patch.object(cli, 'capture_context_qk', fake_capture), \
                 patch('introspection_core.injected_trials.build_injected_batch', fake_batch):
                cli.capture_cluster(model, ex, None, [(0, 0)], source, args, 0, outputs)
                self.assertEqual(calls, [2, 2, 1, 1])
                clean = torch.load(outputs['clean'] / 'metrics.pt', weights_only=True)['metrics']
                injected = torch.load(outputs['injected'] / 'metrics.pt', weights_only=True)['metrics']
                torch.testing.assert_close(injected['all_modes_score_norm'], clean['all_modes_score_norm'] * 2)
                self.assertEqual(clean['all_modes_score_norm'].shape, (3, 1))
                args.resume = True
                cli.capture_cluster(model, ex, None, [(0, 0)], source, args, 0, outputs)
                self.assertEqual(len(calls), 4)
                # Batch size only chunks the loop, so an OOM retry at a smaller
                # batch must still skip the clusters that already completed.
                args.batch_size = 1
                cli.capture_cluster(model, ex, None, [(0, 0)], source, args, 0, outputs)
                self.assertEqual(len(calls), 4)
                args.trial_end = 2
                with self.assertRaisesRegex(ValueError, 'incompatible completed'):
                    cli.capture_cluster(model, ex, None, [(0, 0)], source, args, 0, outputs)
                target_args = SimpleNamespace(trial_start=0, trial_end=3, batch_size=2, model=None,
                                              device='cpu', resume=False, key_positions='context',
                                              save_score_rows=False, save_first_mode_rows=False,
                                              target_positions='injection')
                target_outputs = {state: Path(temporary) / f'target_{state}'
                                  for state in ('clean', 'injected')}
                cli.capture_cluster(model, ex, None, [(0, 0)], source, target_args, 0, target_outputs)
                manifest = json.loads((target_outputs['injected'] / 'complete.json').read_text())
                self.assertEqual(manifest['target_position_selection'], 'injection')
                self.assertEqual(manifest['target_positions'], list(range(1, 11)))
                full = torch.load(target_outputs['injected'] / 'score_rows.pt', weights_only=True)['score_rows']
                first = torch.load(target_outputs['injected'] / 'first_mode_score_rows.pt', weights_only=True)
                self.assertEqual(full.shape, (3, 1, 10))
                self.assertEqual(first['score_rows'].shape, full.shape)
                self.assertEqual(first['first_modal_energy_fraction'].shape, full.shape)


if __name__ == '__main__':
    unittest.main()
