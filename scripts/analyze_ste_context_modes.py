#!/usr/bin/env python3
"""Compare STE and non-STE context-mode responses, preserving query provenance."""
import argparse
import csv
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from introspection_core.context_key_modes import (
    context_key_alignment_metrics, load_context_mode_captures, summarize_context_modes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures', type=Path, nargs='+', required=True, help='Complete capture or nonoverlapping shards')
    parser.add_argument('--on-heads', type=Path, required=True, help='Reviewed on metrics/heads.csv')
    parser.add_argument('--off-heads', type=Path, required=True, help='Reviewed off metrics/heads.csv')
    parser.add_argument('--model-id', required=True, help='Model column in reviewed head CSVs')
    parser.add_argument('--results_dir', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--average-clusters', action='store_true', help='Equal mean of per-trial metrics across clusters before ratios')
    parser.add_argument('--expected-clusters', type=int, help='Require all cluster indices 0 through N-1')
    parser.add_argument('--derive-alignment', action='store_true',
                        help='Add per-trial query-normalized alignment metrics to the captured ones')
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error('bootstrap must be positive')
    if args.results_dir.exists() and any(args.results_dir.iterdir()):
        raise FileExistsError('results_dir must be empty')
    torch.set_num_threads(4)
    meta, metrics = load_context_mode_captures(
        args.captures, average_clusters=args.average_clusters, expected_clusters=args.expected_clusters,
        derived=context_key_alignment_metrics if args.derive_alignment else None,
    )
    nc = len(meta['concepts'])
    heads = list(map(tuple, meta['components']))
    groups = meta['concept_groups']
    if len(groups) != nc or set(groups) != {'validation100', 'bottom100'}:
        raise ValueError('invalid explicit concept groups')
    concept_order = [i for label in ('validation100', 'bottom100') for i, group in enumerate(groups) if group == label]
    valid_count = groups.count('validation100')
    selections = {}
    for name, path in [('on', args.on_heads), ('off', args.off_heads)]:
        with path.open() as stream:
            records = [r for r in csv.DictReader(stream) if r['model'] == args.model_id]
        csv_heads = [(int(r['layer']), int(r['head'])) for r in records]
        if len(csv_heads) != len(set(csv_heads)) or set(csv_heads) != set(heads):
            raise ValueError(f'{name} reviewed head universe differs from capture')
        if any(r['is_ste'].lower() not in ('true', 'false') for r in records):
            raise ValueError('invalid is_ste flag')
        selections[name] = [h for h, r in zip(csv_heads, records) if r['is_ste'].lower() == 'true']
    rows, contrasts, headrows = [], [], []
    for metric, metric_values in sorted(metrics.items()):
        values = metric_values[concept_order]
        for selection, chosen in selections.items():
            summary, comparisons = summarize_context_modes(values, heads, chosen, valid_count=valid_count,
                                                            repeats=args.bootstrap, seed=args.seed)
            common = dict(model=args.model_id, query_state=meta['query_state'], cluster_count=meta['cluster_count'], selection=selection, metric=metric)
            rows.extend(dict(**common, **row) for row in summary)
            contrasts.extend(dict(**common, **row) for row in comparisons)
            mv, mb = values[:valid_count].mean((0, 1)), values[valid_count:].mean((0, 1))
            headrows.extend(dict(**common, layer=l, head=h, is_ste=(l, h) in chosen, valid=mv[j], bottom=mb[j])
                            for j, (l, h) in enumerate(heads))
    args.results_dir.mkdir(parents=True, exist_ok=True)
    for name, data in [('summary.csv', rows), ('contrasts.csv', contrasts), ('heads.csv', headrows)]:
        with (args.results_dir / name).open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    protocol = dict(query_state=meta['query_state'], centered=False, selections=selections,
                    key_position_selection=meta.get('key_position_selection', 'context'),
                    derive_alignment=args.derive_alignment,
                    model_id=args.model_id, captures=[str(p.resolve()) for p in args.captures],
                    on_heads=str(args.on_heads.resolve()), off_heads=str(args.off_heads.resolve()),
                    bootstrap=args.bootstrap, seed=args.seed, valid_count=valid_count, bottom_count=nc-valid_count,
                    aggregation='per-trial metric -> equal clusters -> positions -> weighted heads -> concepts -> valid/bottom',
                    cluster_count=meta['cluster_count'], cluster_indices=meta['cluster_indices'],
                    capture_manifest=meta)
    (args.results_dir / 'analysis_protocol.json').write_text(json.dumps(protocol, indent=2))
    print(f'COMPLETE query={meta["query_state"]} results={args.results_dir}', flush=True)


if __name__ == '__main__':
    main()
