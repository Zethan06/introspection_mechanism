#!/usr/bin/env python3
"""Summarize the OV output-SVD statistics per concept group (Section 4 OV tables).

Each metric is averaged over positions, heads and clusters within a concept,
then over the 100 concepts of each group; intervals bootstrap concepts within
each group (5,000 resamples), conditional on the 30 evaluation clusters.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def summarize(root: Path, bootstrap: int = 5000) -> None:
    plan = json.loads((root/'jobs.json').read_text())
    rows, provenance = [], {}
    rng = np.random.default_rng(42)
    for model in sorted({j['model'] for j in plan}):
        jobs = [j for j in plan if j['model'] == model]
        clusters, reference = {}, None
        for job in jobs:
            folder = Path(job['output'])
            complete = json.loads((folder/'complete.json').read_text())
            protocol = json.loads((folder/'protocol.json').read_text())
            if complete['status'] != 'complete' or complete['trials_per_cluster'] != 2000:
                raise ValueError('incomplete capture')
            if complete['clusters'] != job['clusters']:
                raise ValueError('shard cluster mismatch')
            if reference is None:
                reference = protocol
            for key in ('source_sha256','components','concepts','concept_groups',
                        'output_svd_code_sha256','head_selection_sha256','output_svd_definition'):
                if protocol[key] != reference[key]:
                    raise ValueError(f'incompatible {key}')
            for cluster in job['clusters']:
                if cluster in clusters:
                    raise ValueError('duplicate cluster shard')
                clusters[cluster] = folder/f'cluster_{cluster:02d}'
        if set(clusters) != set(range(30)):
            raise ValueError('requires exactly all 30 evaluation clusters')
        components = list(map(tuple, reference['components']))
        if len(components) != 32:
            raise ValueError('requires the Top32 gate-on heads')
        sums, checks = {}, dict(closure_scaled_max=0.)
        for cluster in range(30):
            p = torch.load(clusters[cluster]/'metrics.pt',weights_only=True,mmap=True)
            if list(map(tuple,p['components'])) != components or not torch.equal(p['trial_indices'],torch.arange(2000)):
                raise ValueError('trial/head order mismatch')
            selected={key:value for key,value in p['metrics'].items() if key.startswith('output_svd_')}
            for key,value in selected.items():
                if value.shape != (2000,32) or not torch.isfinite(value).all():
                    raise ValueError(f'invalid metric {key}: {value.shape}')
                # Positions, selected heads and clusters are equally weighted within concept.
                score=value.double().reshape(200,10,32).mean((1,2)).numpy()
                sums[key]=sums.get(key,np.zeros(200))+score/30
                if key.endswith('closure_scaled'):
                    checks['closure_scaled_max']=max(checks['closure_scaled_max'],float(value.max()))
        groups=np.asarray(reference['concept_groups'])
        high=np.flatnonzero(groups=='validation100'); low=np.flatnonzero(groups=='bottom100')
        if len(high)!=100 or len(low)!=100:
            raise ValueError('requires both frozen 100-concept groups')
        hb=rng.integers(100,size=(bootstrap,100)); lb=rng.integers(100,size=(bootstrap,100))
        for key,score in sums.items():
            h,l=score[high],score[low]
            ci=np.quantile(h[hb].mean(1)-l[lb].mean(1),[.025,.975])
            rows.append(dict(model=model,metric=key,high_n=100,low_n=100,high_mean=float(h.mean()),
                low_mean=float(l.mean()),high_over_low=float(h.mean()/l.mean()) if l.mean()!=0 else None,
                gap=float(h.mean()-l.mean()),ci_low=float(ci[0]),ci_high=float(ci[1])))
        target=root/'summaries'/model; target.mkdir(parents=True,exist_ok=True)
        torch.save(dict(metrics={key:torch.from_numpy(value) for key,value in sums.items()},
                        concepts=reference['concepts'],groups=reference['concept_groups']),target/'concept_metrics.pt')
        provenance[model]=dict(checks=checks,clusters=30,trials=60000,heads=32,
                               components=components,source_sha256=reference['source_sha256'])
        print('SUMMARIZED',model,checks,flush=True)
    with (root/'summary.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (root/'verification.json').write_text(json.dumps(provenance,indent=2))
    # The quantities of the OV tables in Section 4 and its appendix.
    metrics=(['attention_norm','delta_attention_norm','frobenius','energy_top5','response_norm',
              'response_norm_top5','response_energy_top5']
             +[f'mode{k}_sigma' for k in range(1,6)]+[f'mode{k}_reading_cos_abs' for k in range(1,6)])
    lines=['# OV output SVD, C_intro / C_nonintro','',
           'M = W_O dV^T over the full context, gate-on Top-32 heads, 100 concepts per group, 30 evaluation clusters, ten positions. '
           'Cells: group means (intro / non-intro) and their ratio. Every metric is in summary.csv with its concept-bootstrap interval.']
    models=list(provenance)
    lines += ['','| Metric | '+' | '.join(models)+' |','|---|'+'---:|'*len(models)]
    for metric in metrics:
        cells=[]
        for model in models:
            row=next(r for r in rows if r['model']==model and r['metric']==f'output_svd_full_{metric}')
            ratio='' if row['high_over_low'] is None else f" ({row['high_over_low']:.2f}x)"
            cells.append(f"{row['high_mean']:.5g} / {row['low_mean']:.5g}{ratio}")
        lines.append('| '+metric+' | '+' | '.join(cells)+' |')
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    (root/'complete.json').write_text(json.dumps(dict(status='complete',models=models,trials=60000*len(models),bootstrap=bootstrap),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--bootstrap',type=int,default=5000)
    args=parser.parse_args()
    if args.bootstrap<1: parser.error('bootstrap must be positive')
    torch.set_num_threads(2)
    summarize(args.root,args.bootstrap)
