"""Remove one term of the final-query QK score change in selected heads."""

from collections.abc import Callable, Iterable, Sequence
import json
from pathlib import Path

import torch


def rotated_capture(
    layers: Iterable[int], parts: dict[int, torch.Tensor], name: str,
) -> list[tuple[str, Callable]]:
    """Collect native post-RoPE Q/K tensors without changing hook values."""
    if name not in ('q', 'k'):
        raise ValueError('expected q or k capture')
    hooks = []
    for layer in layers:
        def save(value, hook, layer=layer):
            del hook
            parts[layer] = value.detach().float().cpu()
            return value
        hooks.append((f'blocks.{layer}.attn.hook_rot_{name}', save))
    return hooks


def validate_frozen_prompt(
    tokens: torch.Tensor, positions: Sequence[int], reference_path: Path,
) -> None:
    """Reject prompts or injection positions that differ from the frozen capture."""
    reference = json.loads(reference_path.read_text())
    if (tokens.ndim != 2 or tokens.shape[0] != 1
            or reference['input_token_ids'] != tokens[0].tolist()
            or reference['positions'] != list(positions)):
        raise RuntimeError('frozen prompt differs from reference')


def score_term(
    clean_key: torch.Tensor,
    current_key: torch.Tensor,
    clean_query: torch.Tensor,
    current_query: torch.Tensor,
    heads: Sequence[int],
    mode: str,
) -> torch.Tensor:
    """Return [batch, selected_head, key_pos] scores to subtract.

    Keys are post-RoPE [batch, key_pos, kv_head, width]; queries are
    post-RoPE [batch, query_head, width]. Clean tensors may have batch one.
    """
    if mode not in ("query", "key"):
        raise ValueError(f"unknown ablation mode: {mode}")
    if clean_key.ndim != 4 or current_key.ndim != 4 or clean_query.ndim != 3 or current_query.ndim != 3:
        raise ValueError("invalid key/query ranks")
    batch, context, groups, width = current_key.shape
    nheads = current_query.shape[1]
    if (clean_key.shape[1:] != current_key.shape[1:] or clean_query.shape[1:] != current_query.shape[1:]
            or clean_key.shape[0] not in (1, batch) or clean_query.shape[0] not in (1, batch)
            or current_query.shape != (batch, nheads, width) or nheads % groups):
        raise ValueError("incompatible key/query shapes or GQA grouping")
    if not heads or len(set(heads)) != len(heads) or min(heads) < 0 or max(heads) >= nheads:
        raise ValueError("invalid selected heads")
    device = current_key.device
    k0 = clean_key.to(device=device, dtype=torch.float32)
    k1 = current_key.float()
    q0 = clean_query.to(device=device, dtype=torch.float32)
    q1 = current_query.float()
    head_index = torch.as_tensor(heads, device=device)
    kv_index = head_index // (nheads // groups)
    if mode == "query":
        keys = k0[:, :, kv_index, :]
        queries = q1[:, head_index, :] - q0[:, head_index, :]
    else:
        keys = k1[:, :, kv_index, :] - k0[:, :, kv_index, :]
        queries = q1[:, head_index, :]
    return torch.einsum("bthd,bhd->bht", keys, queries) / width**0.5


def subtract_score_term(scores: torch.Tensor, term: torch.Tensor, heads: Sequence[int]) -> torch.Tensor:
    """Patch only selected heads' final-query score row, before softmax."""
    if scores.ndim != 4 or term.shape != (scores.shape[0], len(heads), scores.shape[-1]):
        raise ValueError("score/term shape mismatch")
    if not torch.isfinite(term).all():
        raise ValueError("nonfinite score term")
    patched = scores.clone()
    patched[:, list(heads), -1, :] = (
        scores[:, list(heads), -1, :].float() - term
    ).to(scores.dtype)
    return patched
