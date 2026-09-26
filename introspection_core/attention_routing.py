"""One-hot attention redirection for the router heads.

The intervention rewrites one query row of selected heads after softmax so
that all mass sits on one key token: inject a concept at candidate ``i`` and
make the final prompt position attend only to the successor token ``t_j`` of
candidate ``j``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def force_attention_to_keys(
    pattern: torch.Tensor,
    *,
    heads: Sequence[int],
    key_positions: int | Sequence[int] | torch.Tensor,
    query_position: int = -1,
) -> torch.Tensor:
    """Return ``pattern`` with selected head/query rows made one-hot.

    Args:
        pattern: Post-softmax attention, shaped ``[batch, head, query, key]``.
        heads: Zero-based head indices to intervene on.
        key_positions: One destination key per batch row, or one scalar key
            shared by the batch.
        query_position: Query row to replace. ``-1`` selects the final query,
            which is the row that determines next-token localization logits.

    The function rejects future keys instead of silently creating non-causal
    attention.  Unselected heads and query rows are unchanged.
    """
    if pattern.dim() != 4:
        raise ValueError(
            "pattern must have shape [batch, head, query, key], got "
            f"{tuple(pattern.shape)}"
        )
    if not heads:
        raise ValueError("heads must be non-empty")

    batch, n_heads, n_queries, n_keys = pattern.shape
    head_list = [int(head) for head in heads]
    if len(set(head_list)) != len(head_list):
        raise ValueError(f"heads contains duplicates: {head_list}")
    bad_heads = [head for head in head_list if not 0 <= head < n_heads]
    if bad_heads:
        raise ValueError(
            f"head indices out of range for {n_heads} heads: {bad_heads}"
        )

    query = int(query_position)
    if query < 0:
        query += n_queries
    if not 0 <= query < n_queries:
        raise ValueError(
            f"query_position={query_position} is out of range for {n_queries} queries"
        )

    keys = torch.as_tensor(key_positions, dtype=torch.long, device=pattern.device)
    if keys.dim() == 0:
        keys = keys.expand(batch)
    elif keys.dim() != 1 or keys.numel() != batch:
        raise ValueError(
            "key_positions must be a scalar or one value per batch row; "
            f"got shape {tuple(keys.shape)} for batch={batch}"
        )
    if bool(((keys < 0) | (keys >= n_keys)).any()):
        raise ValueError(
            f"key_positions must be in [0, {n_keys - 1}], got {keys.tolist()}"
        )
    if bool((keys > query).any()):
        raise ValueError(
            "one-hot routing would create future attention: "
            f"query={query}, keys={keys.tolist()}"
        )

    out = pattern.clone()
    head_index = torch.tensor(head_list, dtype=torch.long, device=pattern.device)
    out[:, head_index, query, :] = 0
    row_index = torch.arange(batch, device=pattern.device)[:, None]
    out[row_index, head_index[None, :], query, keys[:, None]] = 1
    return out


def one_hot_attention_hook(
    model,
    *,
    layer: int,
    heads: Sequence[int],
    key_positions: int | Sequence[int] | torch.Tensor,
    query_position: int = -1,
):
    """Build a TransformerLens forward-hook tuple for one-hot routing."""
    hook_name = model.attn_hook_name(layer, "pattern")

    def hook_fn(pattern, hook):
        del hook
        return force_attention_to_keys(
            pattern,
            heads=heads,
            key_positions=key_positions,
            query_position=query_position,
        )

    return hook_name, hook_fn


def trailing_newline_key_positions(tokenizer, example) -> dict[int, int]:
    """Map each localization position to its first trailing newline token.

    The search starts at the exclusive end of each injection span and ends
    before the next item.  Searching by decoded content handles the final
    item's Llama token, which can represent several adjacent newlines rather
    than the single-newline token used by preceding items.
    """
    token_ids = example.input_ids[0].tolist()
    positions = list(example.positions)
    result: dict[int, int] = {}
    for offset, position in enumerate(positions):
        span = example.injection_spans[position]
        search_start = int(span.end)
        search_end = (
            int(example.injection_spans[positions[offset + 1]].start)
            if offset + 1 < len(positions)
            else len(token_ids)
        )
        match = None
        for token_index in range(search_start, search_end):
            text = tokenizer.decode(
                [int(token_ids[token_index])],
                clean_up_tokenization_spaces=False,
            )
            if "\n" in text:
                match = token_index
                break
        if match is None:
            raise ValueError(
                f"No trailing newline token found for position {position}; "
                f"searched token indices [{search_start}, {search_end})"
            )
        result[int(position)] = match
    return result


def candidate_routing_key_positions(tokenizer, example) -> dict[int, int]:
    """Map each candidate to its successor token t_j, the redirection target.

    Candidates 0 through N-2 are followed by the leading-space ``" TOKEN"``
    that starts the next entry; the last candidate is followed by a newline.
    A trailing newline directly after a candidate is used when present.
    """
    token_ids = example.input_ids[0].tolist()
    positions = list(example.positions)
    result: dict[int, int] = {}
    for offset, position in enumerate(positions):
        span = example.injection_spans[position]
        search_start = int(span.end)
        has_next_candidate = offset + 1 < len(positions)
        search_end = (
            int(example.injection_spans[positions[offset + 1]].start)
            if has_next_candidate
            else len(token_ids)
        )
        match = None
        for token_index in range(search_start, search_end):
            text = tokenizer.decode(
                [int(token_ids[token_index])],
                clean_up_tokenization_spaces=False,
            )
            if "\n" in text:
                match = token_index
                break
            if (
                has_next_candidate
                and text.startswith(" ")
                and text.strip() == "TOKEN"
            ):
                match = token_index
                break
        if match is None:
            expected = (
                'a trailing newline or the next " TOKEN" token'
                if has_next_candidate
                else "a newline token"
            )
            raise ValueError(
                f"No candidate routing key ({expected}) found for position "
                f"{position}; searched token indices [{search_start}, {search_end})"
            )
        result[int(position)] = match
    return result
