"""Attention patching over attention-internal hook points.

One ``collect``/``patch`` pair for ``hook_z``, ``hook_v`` and ``hook_pattern``.
Writing ``hook_pattern`` (post-softmax weights) needs no manual recomputation
of ``pattern @ V``: ``hook_z`` is computed downstream of it in the same forward
pass. GQA needs no special handling, since these hooks are exposed at full
``n_heads`` width.
"""

from __future__ import annotations

import contextlib

import torch

from .hooks import write_query_rows


def collect(model, tokens: torch.Tensor, *, layer: int, kind: str) -> torch.Tensor:
    """Run a forward pass and return one attention-internals activation.

    Args:
        model: a HookedModel.
        tokens: LongTensor [batch, seq].
        layer: which layer to read from.
        kind: "z" (per-head output, pre-o_proj, [batch,pos,n_heads,d_head]),
            "pattern" (post-softmax weights, [batch,n_heads,q_pos,k_pos]),
            "qk_scores" (masked pre-softmax scores with the same layout),
            or "k"/"v" (per-head key/value vectors,
            [batch,pos,n_kv_heads_or_n_heads,d_head]).

    Returns:
        The requested activation tensor, detached, on the model's device.
    """
    hook_name = model.attn_hook_name(layer, kind)
    _logits, cache = model.run_with_cache(tokens, names=lambda n: n == hook_name)
    return cache[hook_name].detach()


@contextlib.contextmanager
def patch(
    model,
    *,
    layer: int,
    kind: str,
    source: torch.Tensor,
    heads: list[int] | None = None,
    query_spans=None,
    mode: str = "overwrite",
):
    """Context manager: while active, overwrite (or add to) one attention
    activation at ``layer`` with ``source``, restricted to specific heads
    and/or specific query positions.

    Args:
        model: a HookedModel.
        layer: which layer to patch.
        kind: "z", "pattern", "qk_scores", "k", or "v" — see
            :func:`collect`.
        source: the replacement tensor, as captured by a previous
            :func:`collect` call (or a modification thereof). Shape must
            match what the tensor at that hook looks like — batch
            dimension is handled per-row.
        heads: for kind in {"z","k","v"}: which head index (patches all
            positions for that head, unless ``query_spans`` is also
            given — see below). For kind in {"pattern","qk_scores"}:
            which head indices to restrict the patch to (None = all heads).
        query_spans: for kind in {"z","k","v"}: one (start,end) span per
            batch row (see hooks.write_span) — scopes the patch to those
            query positions instead of the whole sequence. For kind in
            {"pattern","qk_scores"}: a flat list of query-position indices
            (applied to every batch row identically, matching the
            "boundary_positions shared across batch" convention) — pass via
            ``query_spans`` as a list[int] in this case, not spans.
        mode: "overwrite" (default; replace activations with ``source``)
            or "add" (add ``source`` — rare, provided for parity with
            injection-style patches at the attention level; most patching
            experiments want "overwrite" since ``source`` is a captured
            clean/injected activation, not a direction to add).

    Example — patch one head's clean output into an injected run, only at
    the boundary positions:
        clean_z = collect(model, clean_tokens, layer=6, kind="z")
        with patch(model, layer=6, kind="z", source=clean_z,
                    heads=[3], query_spans=[(10, 11)] * batch):
            logits = model.forward_logits(injected_tokens)
    """
    row_kinds = ("pattern", "qk_scores")
    if kind not in ("z", "k", "v", *row_kinds):
        raise ValueError(
            f"Unknown kind: {kind!r}; expected 'z', 'k', 'v', 'pattern', "
            "or 'qk_scores'"
        )

    hook_name = model.attn_hook_name(layer, kind)

    if kind in row_kinds:
        def hook_fn(activation, hook):
            if query_spans is None:
                # No position restriction: overwrite/add the whole tensor.
                if mode == "overwrite":
                    return source.to(device=activation.device, dtype=activation.dtype)
                if mode == "add":
                    return activation + source.to(device=activation.device, dtype=activation.dtype)
                raise ValueError(f"Unknown mode: {mode!r}")
            if mode != "overwrite":
                raise ValueError(
                    f"mode='add' is not supported for kind={kind!r} with "
                    "query_spans scoping"
                )
            return write_query_rows(
                activation, query_positions=query_spans, value=source, heads=heads
            )
    else:
        def hook_fn(activation, hook):
            batch = activation.shape[0]
            seq_len = activation.shape[1]
            n_heads = activation.shape[2]
            if query_spans is None:
                spans = [(0, seq_len)] * batch
            else:
                spans = list(query_spans)

            head_list = list(heads) if heads is not None else list(range(n_heads))

            out = activation.clone()
            for row, (start, end) in enumerate(spans):
                start_clamped = max(0, min(start, seq_len))
                end_clamped = max(0, min(end, seq_len))
                if end_clamped <= start_clamped:
                    continue
                # `source` (from collect()) is the full
                # [batch, pos, n_heads, d_head] sequence tensor; slice out
                # this row's span and restrict to the selected heads so it
                # lines up with the destination slice being written.
                # index_select keeps this correct regardless of whether
                # head_list is contiguous.
                row_value = source[row, start_clamped:end_clamped, :, :].to(
                    device=activation.device, dtype=activation.dtype
                )
                for head in head_list:
                    dest = out[row, start_clamped:end_clamped, head, :]
                    src = row_value[:, head, :]
                    if mode == "overwrite":
                        out[row, start_clamped:end_clamped, head, :] = src
                    elif mode == "add":
                        out[row, start_clamped:end_clamped, head, :] = dest + src
                    else:
                        raise ValueError(f"Unknown mode: {mode!r}; expected 'overwrite' or 'add'")
            return out

    with model.bridge.hooks(fwd_hooks=[(hook_name, hook_fn)]):
        yield
