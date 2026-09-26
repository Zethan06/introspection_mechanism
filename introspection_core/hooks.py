"""Low-level hook-closure builders shared by injection.py and patching.py.

TransformerLens hook functions all share the signature
``(tensor, hook) -> tensor`` but the tensor being intercepted has a
different shape depending on the hook point:

    resid_post / resid_pre   [batch, pos, d_model]
    hook_z                   [batch, pos, n_heads, d_head]
    hook_pattern / hook_attn_scores
                              [batch, n_heads, query_pos, key_pos]
    hook_v / hook_k / hook_q [batch, pos, n_heads_or_kv_heads, d_head]

Rather than force one signature to cover all of these (which the original
design sketch proposed but which produces a leaky, over-parameterized
function once you account for the extra head axis and pattern's transposed
position axis), this module provides two focused primitives:

- :func:`write_span` — writes into a [batch, pos, ...] shaped tensor at a
  per-row (start, end) position span, optionally selecting a single
  trailing "head" index first. Used by injection.py (no head) and by
  patching.py for hook_z / hook_v (with a head).
- :func:`write_query_rows` — writes into a [batch, n_heads, query_pos,
  key_pos] shaped tensor (hook_pattern or hook_attn_scores) at specific
  query rows.

Both support ``mode="add"`` (used by injection: steer the residual stream)
and ``mode="overwrite"`` (used by patching: replace activations with a
value captured from another run).
"""

from __future__ import annotations

from typing import Sequence

import torch

Span = tuple[int, int]


def _resolve_scale(scale: str, base_slice: torch.Tensor) -> torch.Tensor:
    """Return a per-position multiplier for ``value`` before it's combined
    with ``base_slice``.

    scale="unit": multiplier is 1 (value is used as-is; callers are
        expected to have already L2-normalized it if that's the intent).
    scale="relative_hidden_norm": multiplier is the L2 norm of the
        existing activation at that position, so an added unit vector's
        magnitude scales with the residual stream's own scale at that
        point (matches the old `NormScaledAdditionIntervention` /
        `make_batch_injection_hook` "relative_hidden_norm" behavior).
    """
    if scale == "unit":
        return torch.ones(
            base_slice.shape[:-1] + (1,), device=base_slice.device, dtype=base_slice.dtype
        )
    if scale == "relative_hidden_norm":
        return torch.linalg.vector_norm(base_slice.float(), dim=-1, keepdim=True).to(
            dtype=base_slice.dtype
        )
    raise ValueError(f"Unknown scale mode: {scale!r}; expected 'unit' or 'relative_hidden_norm'")


def write_span(
    tensor: torch.Tensor,
    *,
    spans: Sequence[Span],
    value: torch.Tensor,
    mode: str = "add",
    strength: float = 1.0,
    scale: str = "unit",
    head: int | None = None,
) -> torch.Tensor:
    """Write ``value`` into ``tensor`` at a per-batch-row position span.

    Args:
        tensor: activation of shape [batch, pos, feature] (e.g.
            hook_resid_post), or [batch, pos, n_heads, d_head] (e.g.
            hook_z / hook_v) when ``head`` is given.
        spans: one (start, end) pair per batch row. ``end`` is exclusive.
            A row whose span is entirely out of range (start >= seq_len)
            is left untouched.
        value: [batch, feature] (mode="add") or [batch, span_len, feature]
            (mode="overwrite", span lengths must all be equal — pass one
            captured tensor per component/patching call). For mode="add"
            the same row vector is broadcast across every position in
            that row's span.
        mode: "add" — ``strength * scale_multiplier * value`` is added
            in-place. "overwrite" — the slice is replaced by ``value``
            directly (``strength``/``scale`` are ignored).
        strength: scalar multiplier, only used for mode="add".
        scale: "unit" or "relative_hidden_norm", only used for mode="add".
        head: if given, index into the second-to-last axis first (for
            hook_z / hook_v's per-head layout) before writing, and put it
            back. Ignored for 3D tensors.

    Returns:
        A new tensor (input is not mutated in place) with the same shape
        as ``tensor``.
    """
    if len(spans) != tensor.shape[0]:
        raise ValueError(f"Expected one span per batch row ({tensor.shape[0]}), got {len(spans)}")
    out = tensor.clone()
    seq_len = tensor.shape[1]

    working = out if head is None else out[:, :, head, :]

    for row, (start, end) in enumerate(spans):
        if start >= seq_len:
            continue
        end_clamped = min(end, seq_len)
        if end_clamped <= start:
            continue
        if mode == "add":
            base_slice = working[row, start:end_clamped, :]
            row_value = value[row].to(device=tensor.device, dtype=tensor.dtype)
            multiplier = _resolve_scale(scale, base_slice)
            working[row, start:end_clamped, :] = base_slice + strength * multiplier * row_value
        elif mode == "overwrite":
            span_len = end_clamped - start
            row_value = value[row, :span_len, :].to(device=tensor.device, dtype=tensor.dtype)
            working[row, start:end_clamped, :] = row_value
        else:
            raise ValueError(f"Unknown mode: {mode!r}; expected 'add' or 'overwrite'")

    if head is not None:
        out[:, :, head, :] = working
    return out


def write_query_rows(
    tensor: torch.Tensor,
    *,
    query_positions: Sequence[int],
    value: torch.Tensor,
    heads: Sequence[int] | None = None,
) -> torch.Tensor:
    """Overwrite specific query rows of an attention row tensor.

    Args:
        tensor: hook_pattern or hook_attn_scores activation, shape
            [batch, n_heads, query_pos, key_pos].
        query_positions: which query-position rows to overwrite (same set
            applied to every batch row — matches how boundary positions
            are shared across a batch).
        value: replacement tensor of shape
            [batch, len(heads or all n_heads), len(query_positions), key_pos].
        heads: if given, restrict the overwrite to these head indices;
            otherwise all heads are overwritten.

    Returns:
        A new tensor (input is not mutated in place).
    """
    out = tensor.clone()
    head_index = torch.tensor(heads, device=tensor.device) if heads is not None else torch.arange(
        tensor.shape[1], device=tensor.device
    )
    pos_index = torch.tensor(list(query_positions), device=tensor.device, dtype=torch.long)
    value = value.to(device=tensor.device, dtype=tensor.dtype)
    out[:, head_index[:, None], pos_index[None, :], :] = value
    return out
