"""Capture the attention weights and value vectors of selected heads at the final query."""
import torch


@torch.inference_mode()
def capture_attention_values(model, tokens, components, *, fwd_hooks=()):
    """Capture actual projected V, final attention, z, and post-block residuals.

    Uses the existing prefix-cache/final-token path. V includes biases and native
    projection rounding; no reconstruction from pre-normalization inputs is used.
    """
    layers = sorted({layer for layer, _ in components})
    parts = {layer: [] for layer in layers}
    diagnostics, residuals = {}, {}
    handles = []
    try:
        for layer in layers:
            linear = model.bridge.blocks[layer].attn.v.original_component
            if not isinstance(linear, torch.nn.Linear):
                raise TypeError('V projection must be torch.nn.Linear')

            def save(module, inputs, value, layer=layer):
                del module, inputs
                value = value.detach().reshape(*value.shape[:2], -1, int(model.cfg.d_head))
                parts[layer].append(value.float().cpu())

            # PyTorch module hooks survive the bridge's nested prefix/final
            # hook scopes, which otherwise remove a duplicated hook_v observer.
            handles.append(linear.register_forward_hook(save))
        z, _ = model.final_head_ov_inputs(
            tokens, components, fwd_hooks=fwd_hooks,
            attention_output=diagnostics, residual_output=residuals)
    finally:
        for handle in handles:
            handle.remove()
    result = {}
    for layer in layers:
        v = torch.cat(parts[layer], dim=1)
        a = diagnostics[layer]['pattern']
        if v.ndim != 4 or v.shape[1] != tokens.shape[1] or a.shape[-1] != v.shape[1]:
            raise ValueError(f'V/cache/context mismatch at layer {layer}: {v.shape}, {a.shape}')
        result[layer] = {'v': v, 'a': a, 'residual': residuals[layer]}
    return z, result


def reconstruct_z(a0, ai, v0, vi, *, heads, n_heads):
    """Rebuild per-head z = V^T a for the clean and injected runs, in float64.

    A is [batch, selected_head, source]; V is [batch, source, kv_head, width].
    GQA maps each query head to its contiguous KV group. Comparing the result
    with the model's own z catches a wrong capture path or head mapping.
    """
    if a0.shape != ai.shape or v0.shape != vi.shape or a0.ndim != 3 or v0.ndim != 4:
        raise ValueError('incompatible A/V shapes')
    b, n, kv, _width = v0.shape
    if (n_heads % kv or a0.shape != (b, len(heads), n)
            or any(h < 0 or h >= n_heads for h in heads)):
        raise ValueError('incompatible head/GQA dimensions')
    a0, ai, v0, vi = [x.double() for x in (a0, ai, v0, vi)]
    index = torch.tensor(heads, device=v0.device) // (n_heads // kv)
    v0 = v0.index_select(2, index).transpose(1, 2)
    vi = vi.index_select(2, index).transpose(1, 2)
    return torch.einsum('bhn,bhnd->bhd', a0, v0), torch.einsum('bhn,bhnd->bhd', ai, vi)
