#!/usr/bin/env python3
"""Summarize signed first-mode target contributions from frozen captures."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-root', type=Path, required=True)
    parser.add_argument('--on-heads', type=Path, required=True)
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--expected-clusters', type=int, default=30)
    parser.add_argument('--results_dir', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    with args.on_heads.open() as stream:
        selected = {(int(r['layer']), int(r['head'])) for r in csv.DictReader(stream)
                    if r['model'] == args.model_id and r['is_ste'].lower() == 'true'}
    if len(selected) != 32:
        raise ValueError('expected 32 fixed gate-on STE heads')
    samples = {name: [] for name in ('first', 'residual', 'energy_fraction', 'target')}
    scalar_metrics = ('sigma1', 'first_query_projection_absolute', 'first_mode_score_norm')
    samples.update({name: [] for name in scalar_metrics})
    reference = None
    maximum_norm_error = 0.0
    for cluster in range(args.expected_clusters):
        path = args.capture_root / 'injected' / f'cluster_{cluster:02d}'
        meta = json.loads((path / 'complete.json').read_text())
        if (meta['status'] != 'complete' or meta['query_state'] != 'injected'
                or meta['key_position_selection'] != 'successor'
                or not meta.get('save_first_mode_rows')
                or meta['prompt_index'] != cluster
                or meta['trial_start'] != 0 or meta['trial_end'] != 2000):
            raise ValueError(f'incomplete or incompatible capture: {path}')
        if meta['key_positions'] != [p + 1 for p in meta['positions']]:
            raise ValueError('incorrect successor positions')
        if reference is None:
            reference = meta
        for key in ('components', 'concepts', 'concept_groups', 'head_width', 'model'):
            if meta[key] != reference[key]:
                raise ValueError(f'cluster mismatch: {key}')
        indices = [i for i, head in enumerate(meta['components']) if tuple(head) in selected]
        if len(indices) != 32:
            raise ValueError('missing or duplicate selected heads')
        loaded = [torch.load(path / name, map_location='cpu', weights_only=True)
                  for name in ('metrics.pt', 'score_rows.pt', 'first_mode_score_rows.pt')]
        for payload in loaded:
            if not torch.equal(payload['trial_indices'], torch.arange(2000)):
                raise ValueError('trial ordering mismatch')
        metrics = {k: v[:, indices].double() for k, v in loaded[0]['metrics'].items()}
        full, first = [p['score_rows'][:, indices].double() for p in loaded[1:]]
        if full.shape != (2000, 32, 10) or first.shape != full.shape:
            raise ValueError('incorrect score-row shape')
        if not torch.isfinite(full).all() or not torch.isfinite(first).all():
            raise ValueError('nonfinite score rows')
        residual = full - first
        for rows, metric in ((full, 'all_modes_score_norm'), (first, 'first_mode_score_norm'),
                             (residual, 'remaining_modes_score_norm')):
            error = (rows.norm(dim=-1) - metrics[metric]).abs()
            maximum_norm_error = max(maximum_norm_error, float(error.max()))
            # Cancellation in the residual is judged against the full response.
            tolerance = 2e-4 * metrics['all_modes_score_norm'] + 1e-5
            if (error > tolerance).any():
                raise ValueError(f'row/spectral norm mismatch: {metric}')
        # Use the captured spectral norms for R1^2/Rall^2. Direct rows have
        # cancellation error for tiny responses; the norm checks above are scaled.
        energy = metrics['all_modes_score_norm'].square()
        fraction = torch.where(energy > 0, metrics['first_mode_score_norm'].square() / energy, torch.nan)
        nonzero = energy > 0
        torch.testing.assert_close(fraction[nonzero], metrics['first_read_energy_fraction'][nonzero],
                                   rtol=3e-4, atol=1e-5)
        target = (torch.arange(2000) % 10)[:, None, None].expand(-1, 32, 1)
        for name, rows in (('first', first), ('residual', residual), ('target', full)):
            samples[name].append(rows.gather(-1, target)[..., 0].numpy())
        samples['energy_fraction'].append(fraction.numpy())
        for name in scalar_metrics:
            samples[name].append(metrics[name].numpy())
    groups = np.repeat(reference['concept_groups'], 10)
    if list(reference['concept_groups']).count('validation100') != 100 or len(groups) != 2000:
        raise ValueError('expected 100 valid and 100 bottom concepts')
    stacked = {k: np.stack(v) for k, v in samples.items()}
    rows = []
    for group in ('validation100', 'bottom100'):
        chosen = {k: v[:, groups == group].reshape(-1) for k, v in stacked.items()}
        row = dict(model=args.model_id, concept_group=group, clusters=args.expected_clusters,
                   heads=32, sample_count=len(chosen['first']))
        for name in ('first', 'residual', 'target'):
            value = chosen[name]
            row.update({f'{name}_mean': float(value.mean()),
                        f'{name}_std': float(value.std()),
                        f'{name}_positive': float((value > 0).mean())})
            for percentile in (5, 25, 50, 75, 95):
                row[f'{name}_p{percentile}'] = float(np.percentile(value, percentile))
        fraction = chosen['energy_fraction']
        row['energy_fraction_mean'] = float(np.nanmean(fraction))
        row['zero_energy_count'] = int(np.isnan(fraction).sum())
        for name in scalar_metrics:
            row[f'{name}_mean'] = float(chosen[name].mean())
        np.testing.assert_allclose(row['first_mean'] + row['residual_mean'], row['target_mean'], atol=1e-10)
        rows.append(row)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    with (args.results_dir / 'summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.results_dir / 'protocol.json').write_text(json.dumps(dict(
        capture_root=str(args.capture_root.resolve()), on_heads=str(args.on_heads.resolve()),
        model=args.model_id, clusters=args.expected_clusters,
        aggregation='equal cluster, concept, injection-position and selected-head weights',
        distribution='pooled trial/head percentiles; not confidence intervals',
        energy_fraction='mean of per-trial/head R1^2 / Rall^2; zero energy excluded and counted',
        maximum_norm_error=maximum_norm_error,
    ), indent=2))
    print(json.dumps(rows), flush=True)


if __name__ == '__main__':
    main()
