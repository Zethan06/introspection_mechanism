"""Check target indexing and averaging of first-mode response fractions."""
import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch


class SuccessorFirstModeTests(unittest.TestCase):
    def test_signed_target_and_mean_of_per_sample_energy_fractions(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/analyze_successor_first_mode.py'
        spec = importlib.util.spec_from_file_location('analyze_successor_first_mode', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = root / 'capture/injected/cluster_00'
            capture.mkdir(parents=True)
            groups = ['validation100'] * 100 + ['bottom100'] * 100
            meta = dict(status='complete', query_state='injected', key_position_selection='successor',
                        save_first_mode_rows=True, prompt_index=0, trial_start=0, trial_end=2000,
                        components=[[0, h] for h in range(32)], concepts=list(map(str, range(200))),
                        concept_groups=groups, head_width=16, model='fixture',
                        positions=list(range(10)), key_positions=list(range(1, 11)))
            (capture / 'complete.json').write_text(json.dumps(meta))
            trial = torch.arange(2000)
            position = (trial % 10)[:, None, None].expand(-1, 32, 1)
            amplitude = (1 + 2 * (trial % 2)).float()
            amplitude[1000:] *= -1
            first = torch.zeros(2000, 32, 10).scatter_(-1, position,
                                                     amplitude[:, None, None].expand(-1, 32, 1))
            residual = torch.zeros_like(first).scatter_(-1, (position + 1) % 10, 1)
            full = first + residual
            metrics = dict(first_mode_score_norm=first.norm(dim=-1),
                           sigma1=torch.ones(2000, 32),
                           first_query_projection_absolute=amplitude.abs()[:, None].expand(-1, 32),
                           all_modes_score_norm=full.norm(dim=-1),
                           remaining_modes_score_norm=residual.norm(dim=-1),
                           first_read_energy_fraction=first.square().sum(-1) / full.square().sum(-1))
            torch.save(dict(trial_indices=trial, metrics=metrics), capture / 'metrics.pt')
            for name, rows in [('score_rows.pt', full), ('first_mode_score_rows.pt', first)]:
                torch.save(dict(trial_indices=trial, score_rows=rows), capture / name)
            heads = root / 'heads.csv'
            heads.write_text('model,layer,head,is_ste\n' +
                             ''.join(f'fixture,0,{h},true\n' for h in range(32)))
            argv = ['analyze', '--capture-root', str(root / 'capture'), '--on-heads', str(heads),
                    '--model-id', 'fixture', '--expected-clusters', '1', '--results_dir', str(root / 'out')]
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                module.main()
            with (root / 'out/summary.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            for row, sign in zip(rows, (1, -1)):
                self.assertAlmostEqual(float(row['first_mean']), sign * 2)
                self.assertEqual(float(row['first_positive']), float(sign > 0))
                self.assertEqual(float(row['residual_mean']), 0)
                self.assertAlmostEqual(float(row['energy_fraction_mean']), 0.7)
                self.assertEqual(int(row['sample_count']), 32000)


if __name__ == '__main__':
    unittest.main()
