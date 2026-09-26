#!/usr/bin/env python3
"""Capture fixed-state Q/K removal KL on full context and ten successors."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from introspection_core.attention_kl import attention_kl_metrics
from introspection_core.qk_score_ablation import rotated_capture, score_term, validate_frozen_prompt


@torch.inference_mode()
def capture(model, tokens, selected, candidate_ids, injection_hooks=()):
    """Read actual final-row scores and rotated Q/K without patching scores."""
    prefix, last, queries, scores = {}, {}, {}, {}
    cache = model.build_prefix_kv_cache(
        tokens, fwd_hooks=[*injection_hooks, *rotated_capture(selected, prefix, 'k')])
    hooks = [*rotated_capture(selected, last, 'k'),
             *rotated_capture(selected, queries, 'q')]
    for layer, heads in selected.items():
        def save(value, hook, layer=layer, heads=heads):
            del hook
            scores[layer] = value[:, heads, -1].detach().float().cpu()
            return value
        hooks.append((model.attn_hook_name(layer, 'qk_scores'), save))
    logits, _ = model.incremental_last_token_candidate_stats(
        tokens[:, -1:], prefix_kv_cache=cache, prefix_length=tokens.shape[1] - 1,
        candidate_token_ids=candidate_ids, fwd_hooks=hooks)
    if any(set(values) != set(selected) for values in (prefix, last, queries, scores)):
        raise RuntimeError('incomplete Q/K/score capture')
    states = {layer: dict(k=torch.cat((prefix[layer], last[layer]), 1),
                          q=queries[layer][:, -1], scores=scores[layer]) for layer in selected}
    return states, logits.argmax(-1).cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--results_dir', type=Path, required=True)
    parser.add_argument('--clusters', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--trial-end', type=int, default=2000)
    parser.add_argument('--behavior-reference', type=Path)
    parser.add_argument(
        '--prompt-reference-root', type=Path,
        help='Directory containing cluster_XX/complete.json files used to verify frozen prompts. '
             'Defaults to the source capture directory.',
    )
    args = parser.parse_args()
    if args.batch_size < 1 or not 1 <= args.trial_end <= 2000 or len(set(args.clusters)) != len(args.clusters):
        parser.error('invalid batch, trial count, or clusters')
    from introspection_core.attention_inputs import TokenLocalizationCsvTask
    from introspection_core.extraction import load_concept_vector_payload
    from introspection_core.injected_trials import InjectedTrial, build_injected_batch, validate_token0_9_candidate_layout
    from introspection_core.head_output_patch import GATE_MASK_DIR, load_head_selection, parse_head_group_spec
    from introspection_core.label_accuracy import resolve_label_candidate_layout
    from introspection_core.model import HookedModel, ModelConfig
    from introspection_core.prompts import PromptManager

    source = json.loads((args.source / 'sources.json').read_text())
    meta = source['metadata']
    prompt_reference_root = args.prompt_reference_root or args.source.parent
    if len(meta['concepts']) != 200 or any(meta['concept_groups'].count(g) != 100 for g in ('validation100', 'bottom100')):
        raise ValueError('expected frozen valid100 and bottom100')
    selection_path = Path(meta['args']['model_results']) / GATE_MASK_DIR / 'train_on/selected_heads.json'
    selection = load_head_selection(selection_path)
    if selection['selection_direction'] != 'on' or Path(selection['model']).resolve() != Path(meta['model']).resolve():
        raise ValueError('selection/model mismatch')
    components = parse_head_group_spec(selection['gate_group'])[1]
    if len(components) != 32 or len(set(components)) != 32:
        raise ValueError('expected 32 unique STE heads')
    selected = {}
    for layer, head in components:
        selected.setdefault(layer, []).append(head)
    for cluster in args.clusters:
        if (args.results_dir / f'cluster_{cluster:02d}').exists():
            raise FileExistsError(f'cluster already exists: {cluster}')
    torch.set_num_threads(4)
    model = HookedModel(ModelConfig(name=meta['model'], device=args.device, dtype='bfloat16'))
    config = json.loads(selection_path.with_name('configuration.json').read_text())
    frozen = meta['frozen_prompt_config']
    examples = TokenLocalizationCsvTask(
        path=Path(meta['args']['cluster_csv']), preamble=frozen['prompt_preamble'],
        choice_suffix=config.get('choice_suffix', ''), position_index_start=0,
        template_name=frozen['prompt_template'], name=frozen['prompt_template'],
    ).build_examples(PromptManager(model.tokenizer))
    validate_token0_9_candidate_layout(examples)
    if any(not 0 <= c < len(examples) for c in args.clusters):
        raise ValueError('invalid cluster')
    vectors = load_concept_vector_payload(Path(meta['args']['concept_vectors']),
                                         concepts=meta['concepts'], layer=int(meta['args']['injection_layer']))
    ordered = [(layer, head) for layer, heads in selected.items() for head in heads]
    for cluster in args.clusters:
        started = time.monotonic()
        ex = examples[cluster]
        positions = torch.tensor([[ex.injection_spans[p].start for p in range(10)]])
        successors = positions[0] + 1
        if positions.min() <= 0 or successors.max() >= ex.input_ids.shape[1]:
            raise ValueError('invalid prefix injection/successor positions')
        validate_frozen_prompt(ex.input_ids, positions[0].tolist(),
                               prompt_reference_root / f'cluster_{cluster:02d}' / 'complete.json')
        layout = resolve_label_candidate_layout([ex])
        clean_by_batch, chunks, predictions = {}, [], []
        reconstruction_error = 0.0
        for offset in range(0, args.trial_end, args.batch_size):
            stop = min(offset + args.batch_size, args.trial_end)
            trials = [InjectedTrial(i // 10, 0, i % 10, False, False) for i in range(offset, stop)]
            tokens, hook = build_injected_batch(
                trials, base_tokens=ex.input_ids, injection_token_positions=positions,
                concept_vectors=vectors, model=model, injection_layer=int(meta['args']['injection_layer']),
                strength=float(meta['args']['strength']), scale_mode=frozen['scale_mode'])
            batch = len(trials)
            if batch not in clean_by_batch:
                clean_by_batch[batch], _ = capture(model, tokens, selected, layout.token_ids)
            clean = clean_by_batch[batch]
            injected, prediction = capture(model, tokens, selected, layout.token_ids, [hook])
            predictions.append(prediction)
            layer_values = []
            for layer, heads in selected.items():
                k0, q0 = clean[layer]['k'], clean[layer]['q']
                k1, q1, scores = injected[layer]['k'], injected[layer]['q'], injected[layer]['scores']
                kv = torch.tensor(heads) // (q1.shape[1] // k1.shape[2])
                direct = torch.einsum('bthd,bhd->bht', k1[:, :, kv].double(), q1[:, heads].double()) / q1.shape[-1]**0.5
                reconstruction_error = max(reconstruction_error, float((scores - direct).abs().max()))
                values = {}
                for mode in ('query', 'key'):
                    term = score_term(k0, k1, q0, q1, heads, mode)
                    for name, value in attention_kl_metrics(scores, term, successors).items():
                        values[f'{mode}/{name}'] = value.float()
                layer_values.append(values)
            chunks.append({name: torch.cat([values[name] for values in layer_values], 1) for name in layer_values[0]})
            if offset == 0 or stop == args.trial_end or stop % (args.batch_size * 10) == 0:
                print(f'cluster={cluster} trials={stop}/{args.trial_end} elapsed_s={time.monotonic()-started:.1f}', flush=True)
        native = torch.cat(predictions)
        disagreements = None
        if args.behavior_reference:
            old = torch.load(args.behavior_reference / f'cluster_{cluster:02d}' / 'predictions.pt', weights_only=True)
            disagreements = int((native != old['predictions']['native'][:args.trial_end]).sum())
        out = args.results_dir / f'cluster_{cluster:02d}'
        out.mkdir(parents=True)
        torch.save(dict(metrics={name: torch.cat([part[name] for part in chunks]) for name in chunks[0]},
                        trial_indices=torch.arange(args.trial_end), native_predictions=native), out / 'metrics.pt')
        (out / 'sources.json').write_text(json.dumps(source, indent=2))
        (out / 'complete.json').write_text(json.dumps(dict(
            status='complete', model=meta['model'], cluster=cluster, components=ordered,
            concepts=meta['concepts'], concept_groups=meta['concept_groups'],
            positions=positions[0].tolist(), successor_positions=successors.tolist(),
            input_token_ids=ex.input_ids[0].tolist(), trial_end=args.trial_end, batch_size=args.batch_size,
            query_term='K0 (qI-q0) / sqrt(d)', key_term='(KI-K0) qI / sqrt(d)',
            baseline='native hooked pre-softmax scores, original injection forward',
            intervention='offline subtraction in float64, no cross-layer feedback or bf16 re-rounding',
            subset='conditional ten-successor distribution; also ten bins plus outside bin',
            kl_direction='native || removed', log_base='e', gate='on', heads='STE',
            score_reconstruction_max_error=reconstruction_error,
            behavior_native_prediction_disagreements=disagreements,
            source=str(args.source), prompt_reference_root=str(prompt_reference_root),
            selection=str(selection_path),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            metrics_sha256=hashlib.sha256(Path(__file__).resolve().parents[1].joinpath('introspection_core/attention_kl.py').read_bytes()).hexdigest(),
            elapsed_seconds=time.monotonic()-started), indent=2))
        print(f'COMPLETE cluster={cluster} native_disagreements={disagreements}', flush=True)


if __name__ == '__main__':
    main()
