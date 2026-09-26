#!/usr/bin/env python3
"""Subtract parts of the gate-head output change in the injected run (OV causal ablations).

Without --top-k/--top-ks: remove R = W_O V0^T da (-da), C = W_O dV^T aI (-dV), or
both. With --top-k k: remove the k leading modes of M = W_O dV^T, the remaining
modes, or all of C. With --top-ks: remove the leading k modes for each k.
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
from introspection_core.ov_causal_ablation import ov_terms, intervention_vectors, subtract_ov_writes
from introspection_core.ov_capture import capture_attention_values, reconstruct_z


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--heads', type=Path, required=True, help='Frozen heads.csv with is_ste selection')
    parser.add_argument('--model-slug', required=True)
    parser.add_argument('--training-config', type=Path, required=True)
    parser.add_argument('--results_dir', type=Path, required=True)
    parser.add_argument('--clusters', type=int, nargs='+', required=True)
    parser.add_argument('--concepts-per-group', type=int, help='Balanced prefix for preflight only; omit for full evaluation')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--top-k', type=int, help='Ablate full-context content modes: top-k, remainder, and full C')
    parser.add_argument('--top-ks', type=int, nargs='+', help='Shared-SVD sweep; only leading-mode removals plus native')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    if args.batch_size < 1 or len(set(args.clusters)) != len(args.clusters):
        parser.error('positive batch size and unique clusters required')
    if args.concepts_per_group is not None and args.concepts_per_group < 1:
        parser.error('concepts-per-group must be positive')
    if args.top_k is not None and args.top_k < 1:
        parser.error('top-k must be positive')
    if args.top_ks is not None and (args.top_k is not None or not args.top_ks or min(args.top_ks) < 1 or len(set(args.top_ks)) != len(args.top_ks)):
        parser.error("top-ks must be unique positive ranks and cannot be combined with top-k")
    return args


def write_summary(path, rows):
    """Preserve the unconditional denominator and paired trial comparisons."""
    result = []
    for group in sorted({r['group'] for r in rows}):
        subset = [r for r in rows if r['group'] == group]
        native = {(r['cluster'], r['concept_index'], r['position']): r for r in subset if r['condition'] == 'native'}
        for condition in sorted({r['condition'] for r in subset}):
            sample = [r for r in subset if r['condition'] == condition]
            correct = sum(r['correct'] for r in sample)
            numbers = sum(r['prediction'] < 10 for r in sample)
            before = sum(native[(r['cluster'], r['concept_index'], r['position'])]['correct'] for r in sample)
            result.append(dict(group=group, condition=condition, n=len(sample),
                accuracy=correct / len(sample), delta_accuracy_pp=100 * (correct-before) / len(sample),
                number_rate=numbers / len(sample),
                accuracy_given_number=correct / numbers if numbers else None))
            if 'target_probability' in sample[0]:
                for key in ('target_probability', 'target_candidate_probability', 'none_probability', 'none_candidate_probability'):
                    value = sum(r[key] for r in sample) / len(sample)
                    baseline = sum(native[(r['cluster'], r['concept_index'], r['position'])][key] for r in sample) / len(sample)
                    result[-1]['mean_' + key] = value
                    result[-1]['delta_' + key + '_pp'] = 100 * (value-baseline)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result[0]))
        writer.writeheader(); writer.writerows(result)


@torch.inference_mode()
def main(argv=None):
    args = parse_args(argv)
    modal = bool(args.top_k or args.top_ks)
    source = json.loads((args.source / 'sources.json').read_text())
    meta = source['metadata']
    with args.heads.open() as handle:
        components = [(int(r['layer']), int(r['head'])) for r in csv.DictReader(handle)
                      if r['model'] == args.model_slug and r['is_ste'].lower() == 'true']
    if len(components) != 32 or len(set(components)) != 32:
        raise ValueError('expected 32 unique frozen selected heads')
    if not set(components) <= set(map(tuple, meta['components'])):
        raise ValueError('selected heads outside source component universe')
    layers = sorted({l for l, _ in components})
    concepts, groups = meta['concepts'], meta['concept_groups']
    if len(concepts) != len(groups) or len(set(concepts)) != len(concepts):
        raise ValueError('invalid concept identities/groups')
    concept_indices = []
    for group in sorted(set(groups)):
        indices = [i for i, g in enumerate(groups) if g == group]
        if args.concepts_per_group and args.concepts_per_group > len(indices):
            raise ValueError('concepts-per-group exceeds frozen group')
        concept_indices.extend(indices[:args.concepts_per_group])
    trial_ids = [10 * i + p for i in sorted(concept_indices) for p in range(10)]
    out = args.results_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(status='running', args={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        components=components, concept_indices=sorted(concept_indices),
        conditions=((['native'] + [f'full_minus_C_top{k}' for k in args.top_ks]) if args.top_ks else (['native', f'full_minus_C_top{args.top_k}', f'full_minus_C_rest_after{args.top_k}', 'full_minus_C']
                    if args.top_k else ['native', 'full_minus_R', 'full_minus_C', 'full_minus_RC'])),
        modal_definition='Full-context M=W_O deltaV.T, uncentered SVD, largest singular values; frozen native per-trial modes' if modal else None,
        probability_definition='Full-vocabulary softmax plus separately normalized eleven-candidate softmax' if modal else None,
        code_sha256={name: hashlib.sha256((Path(__file__).resolve().parents[1]/name).read_bytes()).hexdigest() for name in ('scripts/run_ov_causal_ablation.py', 'introspection_core/ov_output_svd.py', 'introspection_core/ov_causal_ablation.py')},
        intervention='Subtract frozen clean/injected per-trial OV writes at final query, after W_O before output normalization. Upstream effects propagate; later terms remain frozen.',
        scoring='argmax over ten position tokens plus none; all injected trials retained',
        hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.source/'sources.json', args.heads, args.training_config)},
        preflight=args.concepts_per_group is not None)
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    from introspection_core.attention_inputs import TokenLocalizationCsvTask
    from introspection_core.extraction import load_concept_vector_payload
    from introspection_core.injected_trials import InjectedTrial, build_injected_batch, validate_token0_9_candidate_layout
    from introspection_core.model import HookedModel, ModelConfig
    from introspection_core.prompts import PromptManager
    torch.set_num_threads(4)
    model = HookedModel(ModelConfig(name=meta['model'], device=args.device, dtype='bfloat16'))
    frozen = meta['frozen_prompt_config']
    original = json.loads(args.training_config.read_text())
    examples = TokenLocalizationCsvTask(path=Path(meta['args']['cluster_csv']),
        preamble=frozen['prompt_preamble'], choice_suffix=original.get('choice_suffix',''), position_index_start=0,
        template_name=frozen['prompt_template'], name=frozen['prompt_template']).build_examples(PromptManager(model.tokenizer))
    validate_token0_9_candidate_layout(examples)
    if any(c < 0 or c >= len(examples) for c in args.clusters):
        raise ValueError('cluster outside frozen evaluation bank')
    vectors = load_concept_vector_payload(Path(meta['args']['concept_vectors']), concepts=concepts,
                                         layer=int(meta['args']['injection_layer']))
    weights = {l: model.attention_output_weights(l).cpu().double()[[h for ll,h in components if ll == l]] for l in layers}
    output_factors = {l: torch.linalg.qr(w.transpose(-1, -2), mode='reduced') for l,w in weights.items()} if modal else {}
    rows, verification = [], {'max_native_z_relative_error': 0., 'zero_patch_logit_error': 0.}
    if modal:
        verification.update(modal_closure_scaled_max=0., modal_full_vs_direct_relative_error_max=0.)
    started = time.monotonic()
    with (out/'trials.csv').open('w', newline='') as handle:
        writer = None
        for cluster in args.clusters:
            ex = examples[cluster]
            ids = [ex.candidate_token_ids[str(p)] for p in range(10)]
            other = [v for k,v in ex.candidate_token_ids.items() if k not in [str(p) for p in range(10)]]
            if len(other) != 1:
                raise ValueError('expected one none candidate')
            ids += other
            positions = torch.tensor([[ex.injection_spans[p].start for p in range(10)]])
            if positions.min() <= 0 or positions.max() >= ex.input_ids.shape[1]-1:
                raise ValueError('injection must lie within prefix')
            clean_by_batch = {}
            for offset in range(0, len(trial_ids), args.batch_size):
                indices = trial_ids[offset:offset+args.batch_size]
                trials = [InjectedTrial(i//10, 0, i%10, False, False) for i in indices]
                tokens, hook = build_injected_batch(trials, base_tokens=ex.input_ids,
                    injection_token_positions=positions, concept_vectors=vectors, model=model,
                    injection_layer=int(meta['args']['injection_layer']), strength=float(meta['args']['strength']),
                    scale_mode=frozen['scale_mode'])
                if len(indices) not in clean_by_batch:
                    clean_by_batch[len(indices)] = capture_attention_values(model, tokens, components)
                z0, clean = clean_by_batch[len(indices)]
                zi, injected = capture_attention_values(model, tokens, components, fwd_hooks=[hook])
                terms, modal_writes = {}, {}
                for layer in layers:
                    heads = [h for l,h in components if l == layer]
                    c, inj = clean[layer], injected[layer]
                    terms[layer] = ov_terms(c['a'], inj['a'], c['v'], inj['v'], weights[layer], heads=heads, n_heads=int(model.cfg.n_heads))
                    if modal:
                        from introspection_core.ov_output_svd import topk_content_writes
                        q,r = output_factors[layer]
                        modal_writes[layer] = topk_content_writes(inj['a'], c['v'], inj['v'], q,r,
                            heads=heads, n_heads=int(model.cfg.n_heads), top_k=args.top_k or max(args.top_ks), top_ks=args.top_ks)
                        error = float(modal_writes[layer]['closure_scaled'].max())
                        verification['modal_closure_scaled_max'] = max(verification['modal_closure_scaled_max'], error)
                        error = float((modal_writes[layer]['full']-terms[layer]['C']).norm()/terms[layer]['C'].norm().clamp_min(1e-12))
                        verification['modal_full_vs_direct_relative_error_max'] = max(verification['modal_full_vs_direct_relative_error_max'], error)
                        if error > 1e-9:
                            raise RuntimeError(f'modal full write disagrees with direct C: {error}')
                    # Native AV reconstruction catches wrong GQA mapping/capture paths.
                    rebuilt0, rebuilti = reconstruct_z(c['a'], inj['a'], c['v'], inj['v'], heads=heads, n_heads=int(model.cfg.n_heads))
                    cols = [j for j,(l,h) in enumerate(components) if l == layer]
                    for native, rebuilt in ((z0[:,cols].double(), rebuilt0), (zi[:,cols].double(), rebuilti)):
                        err = float((native-rebuilt).norm()/native.norm().clamp_min(1e-12))
                        verification['max_native_z_relative_error'] = max(verification['max_native_z_relative_error'], err)
                        if err > .01:
                            raise RuntimeError(f'native AV reconstruction failed: {err}')
                prefix_hooks = [hook] + [(model.attn_hook_name(l,'v'), lambda value, hook: value) for l in layers]
                cache = model.build_prefix_kv_cache(tokens, fwd_hooks=prefix_hooks)
                probabilities = {}
                def score(condition):
                    kwargs = dict(prefix_kv_cache=cache, prefix_length=tokens.shape[1]-1, candidate_token_ids=ids)
                    if modal:
                        logits, log_probs = model.incremental_last_token_candidate_stats(tokens[:,-1:], **kwargs)
                        if not torch.isfinite(logits).all() or not torch.isfinite(log_probs).all():
                            raise RuntimeError('nonfinite probability/logit output')
                        probabilities[condition] = log_probs.exp()
                        return logits.cpu().float()
                    return model.incremental_last_token_candidate_logits_only(tokens[:,-1:], **kwargs).cpu().float()
                outputs = {'native': score('native')}
                if offset == 0:
                    with subtract_ov_writes(model, {l: torch.zeros(len(indices), weights[l].shape[-1]) for l in layers}):
                        sham = score('sham')
                    err = float((sham-outputs['native']).abs().max())
                    verification['zero_patch_logit_error'] = max(verification['zero_patch_logit_error'], err)
                    if err != 0:
                        raise RuntimeError(f'zero intervention changes logits: {err}')
                    if modal and not torch.equal(probabilities['sham'], probabilities['native']):
                        raise RuntimeError('zero intervention changes full-vocabulary probabilities')
                if modal:
                    interventions = ([(f'top{k}', f'full_minus_C_top{k}') for k in args.top_ks] if args.top_ks else [('top', f'full_minus_C_top{args.top_k}'),
                                            ('rest', f'full_minus_C_rest_after{args.top_k}'),
                                            ('full', 'full_minus_C')])
                    for part, condition in interventions:
                        with subtract_ov_writes(model, {l: modal_writes[l][part].sum(1) for l in layers}):
                            outputs[condition] = score(condition)
                else:
                    writes = intervention_vectors(terms)
                    for term in ('R','C','RC'):
                        condition = f'full_minus_{term}'
                        with subtract_ov_writes(model, {l: writes[l][term] for l in layers}):
                            outputs[condition] = score(condition)
                for condition, logits in outputs.items():
                    for j, i in enumerate(indices):
                        prediction = int(logits[j].argmax())
                        row = dict(cluster=cluster, concept_index=i//10, concept=concepts[i//10], group=groups[i//10],
                            position=i%10, condition=condition, prediction=prediction, correct=int(prediction == i%10))
                        if modal:
                            candidate_probs = logits[j].softmax(-1)
                            row.update(target_probability=float(probabilities[condition][j,i%10]),
                                target_candidate_probability=float(candidate_probs[i%10]),
                                none_probability=float(probabilities[condition][j,10]),
                                none_candidate_probability=float(candidate_probs[10]))
                        if writer is None:
                            writer = csv.DictWriter(handle, fieldnames=list(row)); writer.writeheader()
                        writer.writerow(row); rows.append(row)
                handle.flush()
                print(f'cluster={cluster} trials={min(offset+len(indices),len(trial_ids))}/{len(trial_ids)} elapsed={time.monotonic()-started:.1f}s', flush=True)
            write_summary(out/'summary.csv', rows)
    protocol.update(status='complete', verification=verification, elapsed_seconds=time.monotonic()-started)
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))


if __name__ == '__main__':
    main()
