#!/usr/bin/env python3
"""Capture the SVD statistics of M = W_O dV^T in the gate-on heads on the evaluation clusters.

For every trial and selected head, M is formed over all context positions and
factorized exactly through the QR factor of W_O; ov_output_svd.output_svd_metrics
lists the statistics.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from introspection_core.head_output_patch import GATE_MASK_DIR
from introspection_core.ov_capture import capture_attention_values, reconstruct_z
from introspection_core.ov_output_svd import output_qr, output_svd_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--results_dir', type=Path, required=True)
    parser.add_argument('--head-selection', type=Path, required=True, help='heads.csv; its is_ste Top-32 heads are captured')
    parser.add_argument('--model-slug', required=True, help='Model key in --head-selection')
    parser.add_argument('--training-config', type=Path)
    parser.add_argument('--clusters', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--trial-end', type=int)
    parser.add_argument('--svd-device', default='cpu')
    parser.add_argument('--svd-dtype', choices=['float32', 'float64'], default='float64')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.batch_size < 1 or len(set(args.clusters)) != len(args.clusters):
        parser.error('invalid batch size or duplicate clusters')
    out = args.results_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    source = json.loads((args.source / 'sources.json').read_text())
    meta = source['metadata']
    concepts, groups = meta['concepts'], meta['concept_groups']
    with args.head_selection.open() as handle:
        components = [(int(r['layer']), int(r['head'])) for r in csv.DictReader(handle)
                      if r['model'] == args.model_slug and r['is_ste'].lower() == 'true']
    if len(set(components)) != 32 or len(components) != 32 or not set(components) <= set(map(tuple, meta['components'])):
        raise ValueError('expected 32 unique frozen heads within source universe')
    layers = sorted({l for l, _ in components})
    if len(concepts) != 200 or groups.count('validation100') != 100 or groups.count('bottom100') != 100:
        raise ValueError('expected frozen validation100/bottom100 groups')
    protocol = dict(
        status='running',
        source_sha256=hashlib.sha256((args.source / 'sources.json').read_bytes()).hexdigest(),
        output_svd_definition='M=W_O deltaV.T; W_O=QR; exact SVD of (R deltaV.T).T; native subset attention',
        output_svd_code_sha256=hashlib.sha256((Path(__file__).resolve().parents[1]/'introspection_core/ov_output_svd.py').read_bytes()).hexdigest(),
        head_selection_sha256=hashlib.sha256(args.head_selection.read_bytes()).hexdigest(),
        components=components, concepts=concepts, concept_groups=groups,
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    )
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    from introspection_core.attention_inputs import TokenLocalizationCsvTask
    from introspection_core.extraction import load_concept_vector_payload
    from introspection_core.injected_trials import InjectedTrial, build_injected_batch, validate_token0_9_candidate_layout
    from introspection_core.model import HookedModel, ModelConfig
    from introspection_core.prompts import PromptManager
    model = HookedModel(ModelConfig(name=meta['model'], device=args.device, dtype='bfloat16'))
    frozen = meta['frozen_prompt_config']
    training_config = args.training_config or Path(meta['args']['model_results']) / GATE_MASK_DIR / 'train_on/configuration.json'
    original = json.loads(training_config.read_text())
    examples = TokenLocalizationCsvTask(
        path=Path(meta['args']['cluster_csv']), preamble=frozen['prompt_preamble'],
        choice_suffix=original.get('choice_suffix', ''), position_index_start=0,
        template_name=frozen['prompt_template'], name=frozen['prompt_template'],
    ).build_examples(PromptManager(model.tokenizer))
    validate_token0_9_candidate_layout(examples)
    if any(c < 0 or c >= len(examples) for c in args.clusters):
        raise ValueError('cluster index outside source')
    vectors = load_concept_vector_payload(Path(meta['args']['concept_vectors']), concepts=concepts,
                                         layer=int(meta['args']['injection_layer']))
    output_factors = {}
    for layer in layers:
        heads = [h for l, h in components if l == layer]
        wo = model.attention_output_weights(layer).cpu().double()
        output_factors[layer] = output_qr(wo[heads].to(device=args.svd_device, dtype=getattr(torch, args.svd_dtype)))
    end = args.trial_end or len(concepts) * 10
    if not 0 < end <= len(concepts) * 10:
        raise ValueError('invalid trial-end')
    for cluster in args.clusters:
        ex = examples[cluster]
        positions = torch.tensor([[ex.injection_spans[p].start for p in range(10)]])
        successors = positions[0] + 1
        if positions.min() <= 0 or successors.max() >= ex.input_ids.shape[1] - 1:
            raise ValueError('injection/successors must precede final query')
        target = out / f'cluster_{cluster:02d}'
        target.mkdir()
        clean_cache, pieces = {}, []
        verification = dict(native_z_relative_error_max=0., read_only_z_max_abs=0.)
        start = time.monotonic()
        for offset in range(0, end, args.batch_size):
            stop = min(offset + args.batch_size, end)
            trials = [InjectedTrial(i // 10, 0, i % 10, False, False) for i in range(offset, stop)]
            tokens, hook = build_injected_batch(
                trials, base_tokens=ex.input_ids, injection_token_positions=positions,
                concept_vectors=vectors, model=model, injection_layer=int(meta['args']['injection_layer']),
                strength=float(meta['args']['strength']), scale_mode=frozen['scale_mode'])
            batch = len(trials)
            if batch not in clean_cache:
                clean_cache[batch] = capture_attention_values(model, tokens, components)
                reference, _ = model.final_head_ov_inputs(tokens, components)
                err = float((reference - clean_cache[batch][0]).abs().max())
                verification['read_only_z_max_abs'] = max(verification['read_only_z_max_abs'], err)
                if err > 1e-6:
                    raise RuntimeError(f'read-only hooks changed native clean z: {err}')
            z0, clean = clean_cache[batch]
            zi, injected = capture_attention_values(model, tokens, components, fwd_hooks=[hook])
            if offset == 0:
                reference, _ = model.final_head_ov_inputs(tokens, components, fwd_hooks=[hook])
                err = float((reference - zi).abs().max())
                verification['read_only_z_max_abs'] = max(verification['read_only_z_max_abs'], err)
                if err > 1e-6:
                    raise RuntimeError(f'read-only hooks changed injected z: {err}')
            batch_metrics = {}
            for layer in layers:
                cols = [j for j, (l, h) in enumerate(components) if l == layer]
                heads = [components[j][1] for j in cols]
                c, inj = clean[layer], injected[layer]
                rz0, rzi = reconstruct_z(c['a'], inj['a'], c['v'], inj['v'],
                                         heads=heads, n_heads=int(model.cfg.n_heads))
                for native, rebuilt in ((z0[:, cols].double(), rz0), (zi[:, cols].double(), rzi)):
                    err = float((native - rebuilt).norm() / native.norm().clamp_min(1e-12))
                    verification['native_z_relative_error_max'] = max(verification['native_z_relative_error_max'], err)
                    if err > .01:
                        raise RuntimeError(f'AV reconstruction differs from native z: {err}')
                output_metrics = output_svd_metrics(c['a'], inj['a'], c['v'], inj['v'],
                    output_factors[layer], heads=heads, n_heads=int(model.cfg.n_heads))
                batch_metrics[layer] = {f'output_svd_full_{name}': value for name, value in output_metrics.items()}
            pieces.append({name: torch.stack([batch_metrics[l][name][:, [h2 for l2, h2 in components if l2 == l].index(h)]
                                              for l, h in components], 1)
                           for name in batch_metrics[layers[0]]})
            if offset == 0 or stop % (args.batch_size * 10) == 0 or stop == end:
                print(f'cluster={cluster} trials={stop}/{end} elapsed_s={time.monotonic()-start:.1f}', flush=True)
        payload = dict(metrics={name: torch.cat([p[name] for p in pieces]) for name in pieces[0]},
                       trial_indices=torch.arange(end), components=components,
                       cluster=cluster, input_token_ids=ex.input_ids[0].tolist(),
                       injection_positions=positions[0].tolist(), successor_positions=successors.tolist())
        torch.save(payload, target / 'metrics.pt')
        (target / 'verification.json').write_text(json.dumps(verification, indent=2))
        (target / 'complete.json').write_text(json.dumps(dict(status='complete', trials=end, elapsed_seconds=time.monotonic()-start), indent=2))
        print('VERIFIED', cluster, verification, flush=True)
    (out / 'complete.json').write_text(json.dumps(dict(status='complete', clusters=args.clusters, trials_per_cluster=end), indent=2))
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
