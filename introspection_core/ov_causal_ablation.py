"""Paired OV interventions at the output projection, before output normalization."""
from contextlib import contextmanager

import torch


def ov_terms(
    a0: torch.Tensor, ai: torch.Tensor, v0: torch.Tensor, vi: torch.Tensor,
    weights: torch.Tensor, *, heads: list[int], n_heads: int,
) -> dict[str, torch.Tensor]:
    """Return R = W_O V0^T da and C = W_O dV^T aI per head, [batch, head, width], float64.

    R and C are the two terms of the output change: do = R + C.
    """
    if a0.shape != ai.shape or v0.shape != vi.shape or v0.ndim != 4:
        raise ValueError('incompatible clean/injected shapes')
    batch, length, kv, width = v0.shape
    if (n_heads % kv or a0.shape != (batch, len(heads), length)
            or weights.ndim != 3 or weights.shape[:2] != (len(heads), width)
            or len(set(heads)) != len(heads)
            or any(h < 0 or h >= n_heads for h in heads)):
        raise ValueError('incompatible attention/GQA/output weights')
    if not all(torch.isfinite(x).all() for x in (a0, ai, v0, vi, weights)):
        raise ValueError('nonfinite OV input')
    a0, ai, v0, vi, weights = [x.double() for x in (a0, ai, v0, vi, weights)]
    index = torch.tensor(heads, device=v0.device) // (n_heads // kv)
    v0, vi = [x.index_select(2, index).transpose(1, 2) for x in (v0, vi)]
    routing = torch.einsum('bhn,bhnd->bhd', ai - a0, v0)
    content = torch.einsum('bhn,bhnd->bhd', ai, vi - v0)
    return {name: torch.einsum('bhd,hdm->bhm', z, weights)
            for name, z in [('R', routing), ('C', content)]}


def intervention_vectors(terms: dict[int, dict[str, torch.Tensor]]) -> dict[int, dict[str, torch.Tensor]]:
    """Sum the selected heads of each layer: R (the -da term), C (-dV) and RC (both)."""
    result = {}
    for layer, pair in terms.items():
        values = {name: pair[name].sum(1) for name in ('R', 'C')}
        result[layer] = {**values, 'RC': values['R'] + values['C']}
    return result


@contextmanager
def subtract_ov_writes(model, vectors):
    """Subtract frozen per-trial writes only on a one-token final-query pass.

    Native downstream computation remains active. Simultaneous multilayer
    interventions can change later heads, so removing RC is a total-write
    subtraction control, not necessarily equivalent to clean-head replacement.
    """
    handles, calls = [], {layer: 0 for layer in vectors}
    try:
        for layer, delta in vectors.items():
            linear = model.bridge.blocks[layer].attn.o.original_component
            if not isinstance(linear, torch.nn.Linear):
                raise TypeError('attention output projection must be Linear')

            def subtract(module, inputs, output, layer=layer, delta=delta):
                del module, inputs
                if output.ndim != 3 or output.shape[1] != 1 or output[:, 0].shape != delta.shape:
                    raise ValueError('OV subtraction requires a matching final-token batch')
                calls[layer] += 1
                return (output.float() - delta.to(output.device, torch.float32)[:, None]).to(output.dtype)

            handles.append(linear.register_forward_hook(subtract))
        yield
        if any(count != 1 for count in calls.values()):
            raise RuntimeError(f'incomplete or repeated OV intervention: {calls}')
    finally:
        for handle in handles:
            handle.remove()
