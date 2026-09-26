"""Uncentered context-key singular responses and fixed-head comparisons."""
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

from .ov_response import comparison_weights

if TYPE_CHECKING:
    from .model import HookedModel


def context_key_mode_metrics(
    delta_key: torch.Tensor, query: torch.Tensor, *, include_first_mode_rows: bool = False,
    target_row_indices: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Measure DeltaK @ q / sqrt(d), with GQA keys and an unnormalized query.

    Inputs: keys [batch, kv_head, context, width], queries [batch, head, width].
    Eigenvectors of DeltaK.T @ DeltaK are right singular vectors. First-mode
    results are basis-dependent when the largest singular value is repeated;
    ``first_spectral_gap_fraction`` exposes that degeneracy.
    """
    if delta_key.ndim != 4 or query.ndim != 3:
        raise ValueError('expected 4D keys and 3D queries')
    batch, groups, context, width = delta_key.shape
    if min(batch, groups, context, width) < 1 or query.shape[1] < 1:
        raise ValueError('empty key/query axes')
    if query.shape[0] != batch or query.shape[2] != width or query.shape[1] % groups:
        raise ValueError('incompatible key/query or GQA dimensions')
    if delta_key.device != query.device:
        raise ValueError('key/query devices differ')
    if not torch.isfinite(delta_key).all() or not torch.isfinite(query).all():
        raise ValueError('nonfinite key/query')
    if target_row_indices is not None:
        if target_row_indices.ndim != 1 or target_row_indices.numel() < 1:
            raise ValueError('target row indices must be a nonempty vector')
        if target_row_indices.device != delta_key.device:
            target_row_indices = target_row_indices.to(delta_key.device)
        if target_row_indices.dtype != torch.long:
            raise ValueError('target row indices must use torch.long')
        if target_row_indices.min() < 0 or target_row_indices.max() >= context:
            raise ValueError('target row index outside key context')
    dtype = torch.float64 if delta_key.dtype == torch.float64 or query.dtype == torch.float64 else torch.float32
    key, q = delta_key.to(dtype), query.to(dtype)
    repeats = q.shape[1] // groups
    if context < width:
        # A ten-position matrix is rank at most ten. A width-by-width Gram
        # matrix creates spurious near-zero modes in finite precision.
        _, singular, right_h = torch.linalg.svd(key, full_matrices=False)
        eigenvalues = singular.square()
        vectors = right_h.transpose(-1, -2)
    else:
        eigenvalues, vectors = torch.linalg.eigh(key.transpose(-1, -2) @ key)
        eigenvalues = eigenvalues.flip(-1).clamp_min(0)
        vectors = vectors.flip(-1)
    modes = eigenvalues.shape[-1]
    coefficients = torch.einsum(
        'bgdi,bgrd->bgri', vectors, q.reshape(batch, groups, repeats, width),
    ).reshape(batch, q.shape[1], modes)
    spectrum = eigenvalues.repeat_interleave(repeats, dim=1)
    energy = spectrum * coefficients.square() / width
    total = energy.sum(-1)
    direct = torch.einsum(
        'bgtd,bgrd->bgrt', key, q.reshape(batch, groups, repeats, width),
    ).reshape(batch, q.shape[1], context).norm(dim=-1) / width**0.5
    # A scale-relative check remains well-defined for zero and tiny responses.
    bound = key.norm(dim=(-1, -2)).repeat_interleave(repeats, 1) * q.norm(dim=-1) / width**0.5
    tolerance = (1e-10 if dtype == torch.float64 else 1e-4) * bound + torch.finfo(dtype).tiny
    if ((total.sqrt() - direct).abs() > tolerance).any():
        raise RuntimeError('spectral response disagrees with direct DeltaK @ query')
    first = spectrum[..., 0]
    second = spectrum[..., 1] if modes > 1 else torch.zeros_like(first)
    result = {
        'sigma1': first.sqrt(),
        'all_singular_rss': spectrum.sum(-1).sqrt(),
        'query_norm': q.norm(dim=-1),
        'first_query_projection_absolute': coefficients[..., 0].abs(),
        'first_mode_score_norm': energy[..., 0].sqrt(),
        'all_modes_score_norm': total.sqrt(),
        'remaining_modes_score_norm': energy[..., 1:].sum(-1).sqrt(),
        'first_read_energy_fraction': torch.where(total > 0, energy[..., 0] / total, 0),
        'top4_read_energy_fraction': torch.where(total > 0, energy[..., :4].sum(-1) / total, 0),
        'first_spectral_gap_fraction': torch.where(first > 0, (first - second) / first, 0),
        'direct_score_norm': direct,
    }
    if include_first_mode_rows:
        # K v1 = sigma1 u1; paired signs cancel without orienting either vector.
        first_axis = torch.einsum('bgtd,bgd->bgt', key, vectors[..., 0])
        result['first_mode_score_rows'] = (
            first_axis.repeat_interleave(repeats, 1) * coefficients[..., 0, None] / width**0.5
        )
    if target_row_indices is not None:
        target_key = key.index_select(2, target_row_indices)
        target_basis = torch.einsum('bgtd,bgdm->bgtm', target_key, vectors)
        target_modes = (target_basis.repeat_interleave(repeats, 1)
                        * coefficients[:, :, None, :] / width**0.5)
        target_score = torch.einsum(
            'bgtd,bgrd->bgrt', target_key, q.reshape(batch, groups, repeats, width),
        ).reshape(batch, q.shape[1], target_row_indices.numel()) / width**0.5
        reconstruction_error = (target_modes.sum(-1) - target_score).abs()
        target_bound = (target_key.norm(dim=-1).repeat_interleave(repeats, 1)
                        * q.norm(dim=-1, keepdim=True) / width**0.5)
        if (reconstruction_error > 1e-4 * target_bound + 1e-5).any():
            raise RuntimeError('target mode scores disagree with direct DeltaK @ query: '
                               f'max_error={reconstruction_error.max().item():.6g}, '
                               f'max_bound={target_bound.max().item():.6g}')
        target_modal_energy = target_modes.square().sum(-1)
        result.update(
            target_score_rows=target_score,
            target_first_mode_score_rows=target_modes[..., 0],
            target_modal_energy=target_modal_energy,
            target_first_modal_energy_fraction=torch.where(
                target_modal_energy > 0,
                target_modes[..., 0].square() / target_modal_energy,
                torch.zeros_like(target_modal_energy),
            ),
        )
    return result


def context_key_alignment_metrics(
    values: dict[str, np.ndarray], head_width: int,
) -> dict[str, np.ndarray]:
    """Rebuild query-normalized alignment from the captured per-trial scalars.

    ``all_modes_score_norm`` is ||DeltaK q|| / sqrt(d) and ``all_singular_rss``
    is ||DeltaK||_F, so dividing out the query length and the matrix size gives
    alignment alone: how much of DeltaK the query direction actually reads, and
    how close that read comes to the spectral bound sigma1. These are functions
    of the same trial, so they must be formed before any averaging.
    """
    if head_width < 1:
        raise ValueError('head_width must be positive')
    required = ('all_modes_score_norm', 'all_singular_rss', 'sigma1', 'query_norm',
                'first_query_projection_absolute')
    if any(key not in values for key in required):
        raise ValueError(f'captured metrics must include {required}')
    read = values['all_modes_score_norm'] * head_width**0.5
    query = values['query_norm']

    def divide(numerator, denominator):
        # Degenerate trials (a zero query or an unchanged key matrix) have no
        # defined alignment; they are reported as zero rather than dropped,
        # which would silently unbalance the concept groups.
        return np.where(denominator > 0, numerator / np.where(denominator > 0, denominator, 1), 0.0)

    unit_projection = divide(values['first_query_projection_absolute'], query)
    return {
        'normalized_alignment': divide(read, values['all_singular_rss'] * query),
        'spectral_bound_utilization': divide(read, values['sigma1'] * query),
        'unit_first_query_projection': unit_projection,
        'unit_first_query_projection_squared': unit_projection**2,
    }


@torch.inference_mode()
def capture_context_qk(
    model: 'HookedModel', tokens: torch.Tensor, components: Sequence[tuple[int, int]], *, fwd_hooks=(),
) -> tuple[torch.Tensor, dict[int, dict[str, torch.Tensor]]]:
    """Read native post-normalization/RoPE Q/K on the prefix-plus-final path.

    Requires bridge hook_rot_q/k tensors [batch, sequence, head, width].
    Does not patch activations or require optional diagnostic model changes.
    """
    layers = sorted({layer for layer, _ in components})
    parts = {layer: {'q': [], 'k': []} for layer in layers}
    hooks = []
    for layer in layers:
        for name in ('q', 'k'):
            def save(value, hook, layer=layer, name=name):
                del hook
                parts[layer][name].append(value.detach().float().transpose(1, 2).cpu())
                return value
            hooks.append((f'blocks.{layer}.attn.hook_rot_{name}', save))
    with model.bridge.hooks(fwd_hooks=hooks):
        z, _ = model.final_head_ov_inputs(tokens, components, fwd_hooks=fwd_hooks)
    captured = {}
    for layer in layers:
        q, k = parts[layer]['q'], parts[layer]['k']
        if not q or not k:
            raise RuntimeError(f'missing rotated Q/K hooks at layer {layer}')
        keys = torch.cat(k, dim=2)
        if keys.shape[2] != tokens.shape[1] or q[-1].shape[2] != 1:
            raise RuntimeError('expected complete prefix keys plus a single final query')
        captured[layer] = {'q': q[-1][:, :, -1], 'k': keys}
    return z, captured


def summarize_context_modes(
    values: np.ndarray, heads: Sequence[tuple[int, int]], selected: Sequence[tuple[int, int]], *,
    valid_count: int, repeats: int = 3000, seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Average positions, then heads, then concepts; bootstrap concepts only.

    Input axes are concept, position, head; valid concepts precede bottom.
    Both controls use the complement of this selection (not on/off union).
    Reuse the same concept resamples across STE and both controls.
    Undefined ratios, including CIs with zero denominators, are returned as None.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3 or min(values.shape) < 1 or values.shape[2] != len(heads):
        raise ValueError('expected nonempty concept × position × head values')
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('expected finite nonnegative values')
    if not 0 < valid_count < len(values) or repeats < 1:
        raise ValueError('empty groups or bootstrap')
    weights = comparison_weights(heads, selected)
    x = values.mean(1)
    rng = np.random.default_rng(seed)
    counts = (valid_count, len(x) - valid_count)
    bootstrap = []
    for n in counts:
        draws = rng.integers(n, size=(repeats, n))
        bootstrap.append(np.stack([np.bincount(row, minlength=n) / n for row in draws]))
    bv, bb = bootstrap[0] @ x[:valid_count], bootstrap[1] @ x[valid_count:]
    mv, mb = x[:valid_count].mean(0), x[valid_count:].mean(0)

    def ratio(a, b):
        return float(a / b) if b > 0 else None

    def interval(a):
        return tuple(map(float, np.quantile(a, [.025, .975])))

    rows, contrasts, samples, means = [], [], {}, {}
    for group, weight in weights.items():
        v, b = float(mv @ weight), float(mb @ weight)
        vb, bbt = bv @ weight, bb @ weight
        lo, hi = interval(vb / bbt) if np.all(bbt > 0) else (None, None)
        rows.append(dict(group=group, valid=v, bottom=b, ratio=ratio(v, b),
                         ratio_ci_low=lo, ratio_ci_high=hi, gap=v-b,
                         heads=int((weight > 0).sum()),
                         valid_higher_heads=int(((mv > mb) & (weight > 0)).sum())))
        samples[group], means[group] = (vb, bbt), (v, b)
    for control in ('non_STE_all', 'non_STE_layer_matched'):
        sv, sb = means['STE']
        cv, cb = means[control]
        svb, sbb = samples['STE']
        cvb, cbb = samples[control]
        glo, ghi = interval((svb-sbb)-(cvb-cbb))
        rlo, rhi = interval((svb*cbb)/(sbb*cvb)) if np.all(sbb*cvb > 0) else (None, None)
        contrasts.append(dict(control=control, difference_of_gaps=(sv-sb)-(cv-cb),
                              gap_ci_low=glo, gap_ci_high=ghi,
                              ratio_of_ratios=ratio(sv*cb, sb*cv),
                              ratio_of_ratios_ci_low=rlo, ratio_of_ratios_ci_high=rhi))
    return rows, contrasts


def load_context_mode_captures(captures, *, average_clusters: bool = False,
                               expected_clusters: int | None = None,
                               derived=None) -> tuple[dict, dict[str, np.ndarray]]:
    """Validate complete per-cluster trials and equally average scalar metrics.

    Returns concept × position × head arrays. Clusters remain paired observations
    of the same concepts; they are not promoted to independent concept samples.
    Shards may split a cluster, but every cluster must cover every trial once.
    ``derived`` adds metrics built from the captured ones per trial, before any
    averaging: ratios of cluster means are not means of per-trial ratios.
    """
    import json
    from pathlib import Path

    if not captures:
        raise ValueError('no captures')
    entries = []
    for path in map(Path, captures):
        entries.append((path, json.loads((path / 'complete.json').read_text()),
                        json.loads((path / 'sources.json').read_text())))
    meta, source = entries[0][1:]
    if meta.get('schema_version') != 1 or meta.get('centered') is not False:
        raise ValueError('expected uncentered context-mode schema 1')
    if meta.get('query_state') not in ('clean', 'injected', 'paired'):
        raise ValueError('invalid query state')
    by_cluster = {}
    for path, other, other_source in entries:
        if other.get('status') != 'complete' or other_source != source:
            raise ValueError('incomplete or different capture sources')
        for key in ('schema_version', 'query_state', 'centered', 'components', 'concepts',
                    'concept_groups', 'head_width', 'model', 'key_position_selection'):
            if other.get(key) != meta.get(key):
                raise ValueError(f'incompatible capture shards: {key}')
        by_cluster.setdefault(other['prompt_index'], []).append((path, other))
    clusters = sorted(by_cluster)
    if not average_clusters and len(clusters) != 1:
        raise ValueError('multiple clusters require --average-clusters')
    if expected_clusters is not None and clusters != list(range(expected_clusters)):
        raise ValueError(f'expected all {expected_clusters} clusters indexed from zero')
    nc, npositions, nheads = len(meta['concepts']), len(meta['positions']), len(meta['components'])
    if min(nc, npositions, nheads) < 1:
        raise ValueError('empty capture dimensions')
    accumulated = {}
    cluster_metadata = []
    for cluster in clusters:
        shards = by_cluster[cluster]
        first = shards[0][1]
        if len(first['positions']) != npositions:
            raise ValueError('different number of positions across clusters')
        payloads = []
        for path, other in shards:
            for key in ('positions', 'key_position_selection', 'key_positions',
                        'final_query_position', 'input_token_ids'):
                if other.get(key) != first.get(key):
                    raise ValueError(f'incompatible within-cluster shards: {key}')
            payload = torch.load(path / 'metrics.pt', map_location='cpu', weights_only=True)
            expected = torch.arange(other['trial_start'], other['trial_end'])
            if not torch.equal(payload['trial_indices'], expected):
                raise ValueError('trial indices disagree with capture manifest')
            if payloads and set(payload['metrics']) != set(payloads[0]['metrics']):
                raise ValueError('incompatible metric sets')
            for value in payload['metrics'].values():
                if value.shape != (len(expected), nheads):
                    raise ValueError('metric shape disagrees with trial/head metadata')
                if not torch.isfinite(value).all() or (value < 0).any():
                    raise ValueError('expected finite nonnegative metrics')
            payloads.append(payload)
        indices = torch.cat([p['trial_indices'] for p in payloads])
        order = indices.argsort()
        if not torch.equal(indices[order], torch.arange(nc * npositions)):
            raise ValueError('missing or duplicate trials: supply every concept and position exactly once')
        trial_values = {}
        for key in payloads[0]['metrics']:
            values = torch.cat([p['metrics'][key] for p in payloads])[order].double().numpy()
            trial_values[key] = values.reshape(nc, npositions, nheads)
        if derived is not None:
            extra = derived(trial_values, int(meta['head_width']))
            if set(extra) & set(trial_values):
                raise ValueError('derived metrics must not shadow captured ones')
            trial_values.update(extra)
        if not trial_values or (accumulated and set(accumulated) != set(trial_values)):
            raise ValueError('incompatible/empty metric sets across clusters')
        for key, values in trial_values.items():
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f'expected finite nonnegative values: {key}')
            if key not in accumulated:
                accumulated[key] = np.zeros_like(values)
            accumulated[key] += values / len(clusters)
        cluster_metadata.append(dict(prompt_index=cluster, positions=first['positions'],
                                     key_positions=first.get('key_positions'),
                                     final_query_position=first['final_query_position'],
                                     input_token_ids=first.get('input_token_ids')))
    result_meta = dict(meta, cluster_indices=clusters, cluster_count=len(clusters),
                       cluster_metadata=cluster_metadata)
    return result_meta, accumulated
