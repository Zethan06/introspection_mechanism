#!/usr/bin/env python3
"""Capture uncentered DeltaK singular responses across frozen prompt clusters."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from introspection_core.head_output_patch import GATE_MASK_DIR
from introspection_core.context_key_modes import capture_context_qk, context_key_mode_metrics


def capture_cluster(model, ex, vectors, components, source, args, cluster: int, outputs: dict) -> None:
    """Capture one cluster with a shared forward pass for both query states."""
    from introspection_core.injected_trials import InjectedTrial, build_injected_batch

    meta = source['metadata']
    concepts, groups = meta['concepts'], meta['concept_groups']
    frozen = meta['frozen_prompt_config']
    positions = torch.tensor([[ex.injection_spans[p].start for p in range(10)]], dtype=torch.long)
    if not (positions.min() > 0 and positions.max() < ex.input_ids.shape[1] - 1):
        raise ValueError('injection must be in the prefix, before final input token')
    key_selection = getattr(args, 'key_positions', 'context')
    save_score_rows = getattr(args, 'save_score_rows', False)
    save_first_mode_rows = getattr(args, 'save_first_mode_rows', False)
    target_selection = getattr(args, 'target_positions', None)
    save_target_rows = target_selection is not None
    save_score_rows = save_score_rows or save_target_rows
    save_first_mode_rows = save_first_mode_rows or save_target_rows
    key_positions = positions[0] + (1 if key_selection == 'successor' else 0)
    target_positions = positions[0] + (1 if target_selection == 'successor' else 0)
    if key_selection != 'context' and (
        len(set(key_positions.tolist())) != 10 or key_positions.max() >= ex.input_ids.shape[1] - 1
    ):
        raise ValueError('selected key positions must be ten distinct prefix tokens')
    if save_target_rows:
        if key_selection != 'context':
            raise ValueError('target positions require a full-context SVD')
        if (len(set(target_positions.tolist())) != 10
                or target_positions.min() < 0 or target_positions.max() >= ex.input_ids.shape[1] - 1):
            raise ValueError('target positions must be ten distinct prefix tokens')
    npositions = positions.shape[1]
    total = len(concepts) * npositions
    end = total if args.trial_end is None else args.trial_end
    if not args.trial_start < end <= total:
        raise ValueError('invalid trial interval')
    layers = sorted({layer for layer, _ in components})
    common = dict(status='complete', schema_version=1, centered=False,
                  components=components, concepts=concepts, concept_groups=groups,
                  positions=positions[0].tolist(), prompt_index=cluster,
                  key_position_selection=key_selection,
                  key_positions=(key_positions.tolist() if key_selection != 'context' else None),
                  input_token_ids=ex.input_ids[0].tolist(),
                  final_query_position=ex.input_ids.shape[1]-1,
                  head_width=int(model.cfg.d_head), trial_start=args.trial_start, trial_end=end,
                  model=args.model or meta['model'])
    if save_target_rows:
        common.update(target_position_selection=target_selection,
                      target_positions=target_positions.tolist())
    if save_score_rows:
        common['save_score_rows'] = True
    if save_first_mode_rows:
        common['save_first_mode_rows'] = True
    pending = {}
    for state, out in outputs.items():
        if (out / 'complete.json').exists() and args.resume:
            previous = json.loads((out / 'complete.json').read_text())
            expected = dict(**common, query_state=state)
            # JSON converts tuple head coordinates to lists.
            expected = json.loads(json.dumps(expected))
            if any(previous.get(k) != v for k, v in expected.items()):
                raise ValueError(f'incompatible completed capture at {out}')
            if (json.loads((out / 'sources.json').read_text()) != source or
                not (out / 'metrics.pt').exists() or
                (save_score_rows and not (out / 'score_rows.pt').exists()) or
                (save_first_mode_rows and not (out / 'first_mode_score_rows.pt').exists())):
                raise ValueError(f'incomplete/different source at {out}')
            print(f'SKIP cluster={cluster} query={state}', flush=True)
            continue
        if out.exists() and any(out.iterdir()):
            # An interrupted cluster has no completion marker. Only replace files
            # from this capture format after confirming the same frozen source.
            saved_source = out / 'sources.json'
            if not args.resume or (out / 'complete.json').exists() or not saved_source.exists():
                raise FileExistsError(f'refusing overwrite: {out}')
            if json.loads(saved_source.read_text()) != source:
                raise ValueError(f'different interrupted source at {out}')
        out.mkdir(parents=True, exist_ok=True)
        (out / 'sources.json').write_text(json.dumps(source, indent=2))
        pending[state] = out
    if not pending:
        return
    clean_by_batch, parts = {}, {state: [] for state in pending}
    score_parts = {state: [] for state in pending} if save_score_rows else None
    first_parts = {state: [] for state in pending} if save_first_mode_rows else None
    local_energy_parts = {state: [] for state in pending} if save_target_rows else None
    start = time.monotonic()
    clean_error = 0.0
    for offset in range(args.trial_start, end, args.batch_size):
        stop = min(offset + args.batch_size, end)
        trials = [InjectedTrial(i // npositions, 0, i % npositions, False, False) for i in range(offset, stop)]
        tokens, hook = build_injected_batch(
            trials, base_tokens=ex.input_ids, injection_token_positions=positions,
            concept_vectors=vectors, model=model, injection_layer=int(meta['args']['injection_layer']),
            strength=float(meta['args']['strength']), scale_mode=frozen['scale_mode'],
        )
        batch = len(trials)
        if batch not in clean_by_batch:
            cz, clean = capture_context_qk(model, tokens, components)
            native_z, _ = model.final_head_ov_inputs(tokens, components)
            error = float((cz - native_z).abs().max())
            clean_error = max(clean_error, error)
            if error > 1e-6:
                raise RuntimeError(f'read-only hooks changed native z: {error}')
            clean_by_batch[batch] = clean
        clean = clean_by_batch[batch]
        _, injected = capture_context_qk(model, tokens, components, fwd_hooks=[hook])
        by_state = {state: {} for state in pending}
        for layer in layers:
            delta_key = (injected[layer]['k'] - clean[layer]['k']).to(args.device)
            if key_selection != 'context':
                delta_key = delta_key.index_select(2, key_positions.to(args.device))
            for state in pending:
                query = clean[layer]['q'] if state == 'clean' else injected[layer]['q']
                query = query.to(args.device)
                by_state[state][layer] = context_key_mode_metrics(
                    delta_key, query, include_first_mode_rows=save_first_mode_rows and not save_target_rows,
                    target_row_indices=target_positions.to(args.device) if save_target_rows else None)
                if save_score_rows:
                    if save_target_rows:
                        scores = by_state[state][layer]['target_score_rows']
                    else:
                        n, groups, rows, width = delta_key.shape
                        repeats = query.shape[1] // groups
                        scores = torch.einsum('bgtd,bgrd->bgrt', delta_key.float(),
                                              query.float().reshape(n, groups, repeats, width))
                        scores = scores.reshape(n, query.shape[1], rows) / width**0.5
                    by_state[state][layer]['score_rows'] = scores
        for state, by_layer in by_state.items():
            # Preserve metadata head ordering, including sparse/noncontiguous sets.
            parts[state].append({key: torch.stack([by_layer[l][key][:, h] for l, h in components], 1).cpu()
                                 for key in by_layer[layers[0]]
                                 if key not in ('score_rows', 'first_mode_score_rows', 'target_score_rows',
                                                'target_first_mode_score_rows', 'target_modal_energy',
                                                'target_first_modal_energy_fraction')})
            if save_first_mode_rows:
                first_key = 'target_first_mode_score_rows' if save_target_rows else 'first_mode_score_rows'
                first_parts[state].append(torch.stack(
                    [by_layer[l][first_key][:, h] for l, h in components], 1).cpu())
            if save_score_rows:
                score_parts[state].append(torch.stack(
                    [by_layer[l]['score_rows'][:, h] for l, h in components], 1).cpu())
            if save_target_rows:
                local_energy_parts[state].append(torch.stack(
                    [by_layer[l]['target_first_modal_energy_fraction'][:, h] for l, h in components], 1).cpu())
        if offset == args.trial_start or stop % (args.batch_size * 10) == 0 or stop == end:
            print(f'cluster={cluster} trials={stop}/{end} query={",".join(pending)} '
                  f'elapsed_s={time.monotonic()-start:.1f}', flush=True)
    for state, out in pending.items():
        payload = {'metrics': {key: torch.cat([part[key] for part in parts[state]]) for key in parts[state][0]},
                   'trial_indices': torch.arange(args.trial_start, end)}
        temporary = out / 'metrics.partial.pt'
        torch.save(payload, temporary)
        temporary.replace(out / 'metrics.pt')
        if save_first_mode_rows:
            first_rows = torch.cat(first_parts[state])
            if not save_target_rows and not torch.allclose(
                first_rows.norm(dim=-1), payload['metrics']['first_mode_score_norm'], rtol=1e-4, atol=1e-5,
            ):
                raise RuntimeError('first-mode rows disagree with first-mode response norm')
            temporary = out / 'first_mode_score_rows.partial.pt'
            first_payload = {'trial_indices': payload['trial_indices'], 'score_rows': first_rows}
            if save_target_rows:
                first_payload['first_modal_energy_fraction'] = torch.cat(local_energy_parts[state])
            torch.save(first_payload, temporary)
            temporary.replace(out / 'first_mode_score_rows.pt')
        if save_score_rows:
            score_rows = torch.cat(score_parts[state])
            norms = score_rows.norm(dim=-1)
            if not save_target_rows and not torch.allclose(
                norms, payload['metrics']['all_modes_score_norm'], rtol=1e-4, atol=1e-5,
            ):
                raise RuntimeError('position scores disagree with all-mode response')
            temporary = out / 'score_rows.partial.pt'
            torch.save({'trial_indices': payload['trial_indices'], 'score_rows': score_rows}, temporary)
            temporary.replace(out / 'score_rows.pt')
        # batch_size only chunks the trial loop, so it is recorded but excluded
        # from `common`: resuming after an OOM at a smaller batch must not
        # invalidate the clusters that already completed.
        manifest = dict(**common, query_state=state, batch_size=args.batch_size,
                        clean_native_z_max_error=clean_error,
                        elapsed_seconds=time.monotonic()-start)
        temporary = out / 'complete.partial.json'
        temporary.write_text(json.dumps(manifest, indent=2))
        temporary.replace(out / 'complete.json')
    print(f'CLUSTER_COMPLETE cluster={cluster}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Reviewed capture directory with sources.json')
    parser.add_argument('--results_dir', type=Path, required=True)
    parser.add_argument('--query-state', choices=('clean', 'injected', 'both'), required=True)
    parser.add_argument('--key-positions', choices=('context', 'injection', 'successor'), default='context',
                        help='Rows of DeltaK: full context, ten injection tokens, or their ten successors')
    parser.add_argument('--save-score-rows', action='store_true',
                        help='Save signed per-token DeltaK @ query / sqrt(d) alongside scalar metrics')
    parser.add_argument('--save-first-mode-rows', action='store_true',
                        help='Also save signed first-mode contributions at the selected positions')
    parser.add_argument('--target-positions', choices=('injection', 'successor'),
                        help='With a full-context SVD, save score rows at these ten positions')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--model', help='Override checkpoint location recorded in source')
    parser.add_argument('--training-config', type=Path, help='Frozen training configuration if its recorded directory moved')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--trial-start', type=int, default=0)
    parser.add_argument('--trial-end', type=int, help='Exclusive; defaults to all concepts and positions')
    cluster_options = parser.add_mutually_exclusive_group()
    cluster_options.add_argument('--clusters', nargs='+', type=int, help='Zero-based prompt cluster indices')
    cluster_options.add_argument('--all-clusters', action='store_true')
    parser.add_argument('--expected-clusters', type=int, help='Require this many examples in the source CSV')
    parser.add_argument('--resume', action='store_true', help='Skip matching completed clusters; rerun interrupted ones')
    args = parser.parse_args()
    if args.save_first_mode_rows:
        args.save_score_rows = True
    if args.batch_size < 1 or args.trial_start < 0:
        parser.error('batch_size must be positive and trial-start nonnegative')
    if args.target_positions is not None and args.key_positions != 'context':
        parser.error('--target-positions requires --key-positions context')
    if args.save_score_rows and args.key_positions == 'context' and args.target_positions is None:
        parser.error('--save-score-rows requires ten selected key positions or --target-positions')
    source = json.loads((args.source / 'sources.json').read_text())
    meta = source['metadata']
    concepts, groups = meta['concepts'], meta['concept_groups']
    if len(concepts) != len(groups) or set(groups) != {'validation100', 'bottom100'}:
        raise ValueError('source must explicitly identify valid and bottom concepts')
    from introspection_core.attention_inputs import TokenLocalizationCsvTask
    from introspection_core.extraction import load_concept_vector_payload
    from introspection_core.injected_trials import validate_token0_9_candidate_layout
    from introspection_core.model import HookedModel, ModelConfig
    from introspection_core.prompts import PromptManager

    torch.set_num_threads(4)
    model = HookedModel(ModelConfig(name=args.model or meta['model'], device=args.device, dtype='bfloat16'))
    frozen = meta['frozen_prompt_config']
    training_config = args.training_config or Path(meta['args']['model_results']) / GATE_MASK_DIR / 'train_on/configuration.json'
    original = json.loads(training_config.read_text())
    examples = TokenLocalizationCsvTask(
        path=Path(meta['args']['cluster_csv']), preamble=frozen['prompt_preamble'],
        choice_suffix=original.get('choice_suffix', ''), position_index_start=0,
        template_name=frozen['prompt_template'], name=frozen['prompt_template'],
    ).build_examples(PromptManager(model.tokenizer))
    validate_token0_9_candidate_layout(examples)
    if args.expected_clusters is not None and len(examples) != args.expected_clusters:
        raise ValueError(f'expected {args.expected_clusters} clusters, found {len(examples)}')
    clusters = list(range(len(examples))) if args.all_clusters else (args.clusters or [0])
    if len(set(clusters)) != len(clusters) or any(not 0 <= c < len(examples) for c in clusters):
        raise ValueError('invalid/duplicate cluster indices')
    vectors = load_concept_vector_payload(Path(meta['args']['concept_vectors']), concepts=concepts,
                                         layer=int(meta['args']['injection_layer']))
    components = list(map(tuple, meta['components']))
    if len(set(components)) != len(components) or not components:
        raise ValueError('components must be nonempty and unique')
    states = ['clean', 'injected'] if args.query_state == 'both' else [args.query_state]
    nested = args.all_clusters or args.clusters is not None or args.query_state == 'both'
    for cluster in clusters:
        outputs = {state: args.results_dir / state / f'cluster_{cluster:02d}' if nested else args.results_dir
                   for state in states}
        capture_cluster(model, examples[cluster], vectors, components, source, args, cluster, outputs)
    if nested:
        run = dict(status='complete', clusters=clusters, query_states=states,
                   concepts=len(concepts), positions_per_cluster=10,
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (args.results_dir / 'run_complete.json').write_text(json.dumps(run, indent=2))
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
