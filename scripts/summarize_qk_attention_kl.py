#!/usr/bin/env python3
"""Summarize paired fixed-state attention KL captures without pooling heads first."""

import argparse
import csv
import json
from pathlib import Path

import torch


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures', nargs='+', type=Path, required=True)
    parser.add_argument('--results_dir', type=Path, required=True)
    parser.add_argument('--expected-clusters', nargs='+', type=int, default=[0, 1, 2])
    args = parser.parse_args()
    torch.set_num_threads(4)
    summaries, cluster_rows, head_rows, comparisons, provenance = [], [], [], [], []
    for model_dir in args.captures:
        paths = sorted(model_dir.glob('cluster_*'))
        manifests = [json.loads((path / 'complete.json').read_text()) for path in paths]
        if [m['cluster'] for m in manifests] != sorted(args.expected_clusters):
            raise ValueError(f'incomplete cluster set: {model_dir}')
        reference = manifests[0]
        invariant = ('model', 'components', 'concepts', 'concept_groups', 'trial_end',
                     'query_term', 'key_term', 'baseline', 'intervention', 'subset', 'kl_direction')
        if any(any(m[k] != reference[k] for k in invariant) for m in manifests):
            raise ValueError('incompatible capture protocol')
        if reference['trial_end'] != 2000:
            raise ValueError('summary requires all 200 concepts and ten positions')
        payloads = [torch.load(path / 'metrics.pt', weights_only=True) for path in paths]
        for payload in payloads:
            if not torch.equal(payload['trial_indices'], torch.arange(2000)):
                raise ValueError('incomplete or reordered trials')
            if any(v.shape != (2000, 32) or not torch.isfinite(v).all() for v in payload['metrics'].values()):
                raise ValueError('invalid per-trial/head metrics')
        model = model_dir.name
        provenance.append(dict(model=model, captures=[str(p.resolve()) for p in paths],
                               score_reconstruction_max_error=max(m['score_reconstruction_max_error'] for m in manifests),
                               native_prediction_disagreements=[m['behavior_native_prediction_disagreements'] for m in manifests]))
        for group in ('validation100', 'bottom100'):
            mask = torch.tensor([g == group for g in reference['concept_groups']]).repeat_interleave(10)
            combined = {}
            for key in payloads[0]['metrics']:
                mode, metric = key.split('/')
                values = torch.stack([payload['metrics'][key][mask].double() for payload in payloads])
                combined[key] = values
                common = dict(model=model, concept_group=group, term=mode, metric=metric)
                means = values.mean((1, 2))
                summaries.append(dict(**common, n_clusters=len(paths), n_trials=values.shape[0]*values.shape[1],
                                      n_heads=32, mean=float(values.mean()),
                                      median=float(values.flatten().quantile(.5)),
                                      p90=float(values.flatten().quantile(.9)),
                                      p99=float(values.flatten().quantile(.99)),
                                      cluster_mean_min=float(means.min()), cluster_mean_max=float(means.max())))
                for index, manifest in enumerate(manifests):
                    cluster_rows.append(dict(**common, cluster=manifest['cluster'], mean=float(means[index])))
                for index, (layer, head) in enumerate(reference['components']):
                    per_head = values[:, :, index].flatten()
                    head_rows.append(dict(**common, layer=layer, head=head, mean=float(per_head.mean()),
                                          p90=float(per_head.quantile(.9))))
            for metric in ('full_kl', 'subset_conditional_kl', 'subset_coarse_kl'):
                q, k = combined[f'query/{metric}'], combined[f'key/{metric}']
                comparisons.append(dict(model=model, concept_group=group, metric=metric,
                                        query_mean=float(q.mean()), key_mean=float(k.mean()),
                                        query_over_key_mean=float(q.mean()/k.mean()) if k.mean() > 0 else None,
                                        paired_query_minus_key_mean=float((q-k).mean()),
                                        fraction_trial_heads_query_lt_key=float((q<k).double().mean())))
    args.results_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in [('summary', summaries), ('cluster_means', cluster_rows),
                       ('head_summary', head_rows), ('paired_comparison', comparisons)]:
        write_csv(args.results_dir / f'{name}.csv', rows)
    (args.results_dir / 'protocol.json').write_text(json.dumps(dict(
        aggregation='compute metric per trial/head, then equal average over 32 heads, 10 positions, 100 concepts and clusters',
        percentiles='pooled individual trial-head metric values, not head-averaged values',
        uncertainty=f'cluster mean range is descriptive, not a confidence interval; {len(args.expected_clusters)} prompt clusters',
        subset='ten successor tokens; conditional KL plus coarse ten-token/outside KL and full-attention subset mass',
        units='nats for KL; probabilities for mass and TV', captures=provenance), indent=2))
    print(json.dumps(comparisons, indent=2))


if __name__ == '__main__':
    main()
