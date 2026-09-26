"""Direction-free final-token OV response and layer-matched STE comparison."""
from collections.abc import Mapping, Sequence

import numpy as np
import torch


def response_norms(delta_z: torch.Tensor, output_weights: Sequence[torch.Tensor]) -> np.ndarray:
    """Return ||O_h delta_z_h||, preserving trial/head axes, in float64."""
    if delta_z.ndim != 3 or delta_z.shape[1] != len(output_weights):
        raise ValueError('expected trial × head × head_width and one O per head')
    if not torch.isfinite(delta_z).all():
        raise ValueError('nonfinite delta_z')
    norms = []
    for j, weight in enumerate(output_weights):
        if weight.ndim != 2 or weight.shape[1] != delta_z.shape[2] or not torch.isfinite(weight).all():
            raise ValueError('invalid O shape or values')
        z, o = delta_z[:, j].double(), weight.double()
        # Gram form avoids materializing trial × model_width for every head.
        norms.append(((z @ (o.T @ o)) * z).sum(-1).clamp_min(0).sqrt())
    return torch.stack(norms, dim=1).cpu().numpy()


def comparison_weights(heads: Sequence[tuple[int, int]], selected: Sequence[tuple[int, int]]) -> dict[str, np.ndarray]:
    """Equal head weights and control weights matching the STE layer counts."""
    heads, selected = list(heads), list(selected)
    if len(set(heads)) != len(heads) or len(set(selected)) != len(selected):
        raise ValueError('duplicate heads')
    if not selected or not set(selected) < set(heads):
        raise ValueError('selected must be a nonempty proper subset of heads')
    mask = np.array([h in set(selected) for h in heads])
    matched = np.zeros(len(heads))
    for layer in sorted({h[0] for h in selected}):
        control = np.array([h[0] == layer and not chosen for h, chosen in zip(heads, mask)])
        if not control.any():
            raise ValueError(f'no same-layer controls for layer {layer}')
        count = sum(h[0] == layer for h in selected)
        matched[control] = count / len(selected) / control.sum()
    return {'STE': mask / mask.sum(), 'non_STE_all': (~mask) / (~mask).sum(),
            'non_STE_layer_matched': matched}


def summarize_responses(values: np.ndarray, weights: Mapping[str, np.ndarray], *,
                        valid_count: int, repeats: int = 5000, seed: int = 42) -> list[dict]:
    """Average positions/heads; bootstrap independent concepts, paired across heads.

    Input axes: concept, injection position, head. Valid concepts precede bottom.
    Point estimates always use observed data; bootstrap draws only determine CIs.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('expected finite nonnegative concept × position × head responses')
    if min(values.shape) < 1 or not 0 < valid_count < len(values) or repeats < 1:
        raise ValueError('empty groups, axes, or bootstrap')
    rng = np.random.default_rng(seed)
    iv = rng.integers(valid_count, size=(repeats, valid_count))
    nb = len(values) - valid_count
    ib = rng.integers(nb, size=(repeats, nb))
    rows, means, boot = [], {}, {}
    for name, weight in weights.items():
        weight = np.asarray(weight)
        if weight.shape != (values.shape[2],) or not np.isfinite(weight).all() or np.any(weight < 0) or not np.isclose(weight.sum(), 1):
            raise ValueError('invalid head weights')
        scores = values.mean(1) @ weight
        v, b = scores[:valid_count], scores[valid_count:]
        vm, bm = float(v.mean()), float(b.mean())
        means[name] = (vm, bm)
        boot[name] = v[iv].mean(1) - b[ib].mean(1)
        lo, hi = np.quantile(boot[name], [.025, .975])
        rows.append(dict(group=name, valid_mean=vm, bottom_mean=bm, difference=vm-bm,
                         valid_bottom_ratio=vm/bm if bm else None,
                         difference_ci_low=float(lo), difference_ci_high=float(hi)))
    for name in weights:
        if name == 'STE':
            continue
        vm, bm = means['STE'][0]-means[name][0], means['STE'][1]-means[name][1]
        lo, hi = np.quantile(boot['STE']-boot[name], [.025, .975])
        rows.append(dict(group='STE_minus_'+name, valid_mean=vm, bottom_mean=bm,
                         difference=vm-bm, valid_bottom_ratio=None,
                         difference_ci_low=float(lo), difference_ci_high=float(hi)))
    return rows
