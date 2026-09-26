"""CPU checks for portable, protocol-consistent behavior summaries."""
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


MODELS = ("qwen3-4b-instruct-2507", "llama3.1-8b-instruct", "gemma3-12b-it")


def load_cli():
    path = Path(__file__).resolve().parents[1] / 'scripts/summarize_ste_qk_score_ablation.py'
    spec = importlib.util.spec_from_file_location('qk_score_summary', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class QKScoreSummaryTests(unittest.TestCase):
    def make_captures(self, root, cli):
        for model in MODELS:
            path = root / model / 'cluster_00'
            path.mkdir(parents=True)
            manifest = dict(cluster=0, trial_end=2000, source='frozen/source', selection='heads.json',
                            model=model, batch_size=8, selected_heads=[[0, 0]], conditions=list(cli.CONDITIONS))
            (path / 'manifest.json').write_text(json.dumps(manifest))
            (path / 'predictions.pt').touch()
            rows = [dict(group=group, condition=condition, n_trials=1000, number_count=600,
                         exact_count=500, none_count=400, native_number_to_none=0, native_none_to_number=0)
                    for group in cli.GROUPS for condition in cli.CONDITIONS]
            with (path / 'summary.csv').open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    def test_summary_requires_no_historical_run(self):
        cli = load_cli()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_captures(root, cli)
            with patch.object(sys, 'argv', ['summary', '--results_dir', str(root), '--expected-clusters', '1']), \
                    contextlib.redirect_stdout(io.StringIO()):
                cli.main()
            with (root / 'summary/summary.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 6 * len(MODELS))
            self.assertTrue(all(float(row['exact_rate']) == .5 for row in rows))
            report = (root / 'summary/README.md').read_text()
            self.assertIn('1 frozen clusters', report)
            self.assertIn('audit not requested', report)

    def test_baseline_is_validated_before_writing_output(self):
        cli = load_cli()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_captures(root, cli)
            baseline = root / 'baseline.csv'
            baseline.write_text('cluster_index,group,condition,number_count,exact_count,n_trials\n')
            args = ['summary', '--results_dir', str(root), '--expected-clusters', '1',
                    '--baseline-csv', MODELS[0], str(baseline)]
            with patch.object(sys, 'argv', args), self.assertRaisesRegex(ValueError, 'baseline rows'):
                cli.main()
            self.assertFalse((root / 'summary').exists())

    def test_rejects_mixed_head_selection(self):
        cli = load_cli()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_captures(root, cli)
            path = root / MODELS[0] / 'cluster_00'
            reference = json.loads((path / 'manifest.json').read_text())
            reference['selected_heads'] = [[1, 1]]
            with self.assertRaisesRegex(ValueError, 'selected_heads'):
                cli.read_cluster(path, 0, reference)


if __name__ == '__main__':
    unittest.main()
