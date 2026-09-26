"""KV-cached, final-token, single-head output patching primitives."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from .injected_trials import gate_logit

# The Top-32 gate masks (Stage 04b, k=32), relative to results/<model>; every
# gate-head analysis reads its train_on/ and train_off/ masks from here.
GATE_MASK_DIR = Path("ste_topk_sweep") / "top32"


DIRECTIONS = ("clean_from_injected", "injected_from_clean")
SELECTION_RULES = ("bidirectional_rank_sum", "clean_only", "injected_only")


def parse_layer_spec(spec: str) -> list[int]:
    """Parse comma-separated layers and inclusive ranges."""

    layers: set[int] = set()
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(value) for value in part.split("-", 1))
            if end < start:
                raise ValueError(f"Invalid descending layer range: {part}")
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))
    if not layers:
        raise ValueError("At least one patch layer is required")
    return sorted(layers)


def parse_head_group_spec(spec: str) -> tuple[str, list[tuple[int, int]]]:
    """Parse ``name=L21H19,L22H5`` into a named component group."""

    if "=" not in spec:
        raise ValueError("Head group must have the form name=L21H19,L22H5")
    name, raw_components = (part.strip() for part in spec.split("=", 1))
    if not name:
        raise ValueError("Head group name cannot be empty")
    components: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for raw_component in raw_components.split(","):
        component = raw_component.strip().upper()
        if not component.startswith("L") or "H" not in component:
            raise ValueError(f"Invalid head component: {raw_component!r}")
        raw_layer, raw_head = component[1:].split("H", 1)
        pair = (int(raw_layer), int(raw_head))
        if pair in seen:
            raise ValueError(f"Duplicate head component in {name}: L{pair[0]}H{pair[1]}")
        seen.add(pair)
        components.append(pair)
    if not components:
        raise ValueError(f"Head group {name!r} cannot be empty")
    return name, components


def final_token_head_patch_hook(
    source_z: torch.Tensor,
    *,
    head: int,
):
    """Return a hook replacing one head at the sole incremental token."""

    if source_z.dim() != 4 or source_z.shape[1] != 1:
        raise ValueError("source_z must have shape [batch, 1, heads, d_head]")

    def patch(activation: torch.Tensor, hook) -> torch.Tensor:
        del hook
        if activation.dim() != 4 or activation.shape[1] != 1:
            raise ValueError(
                "KV-cached head patch expected [batch, 1, heads, d_head], "
                f"got {tuple(activation.shape)}"
            )
        if not 0 <= head < activation.shape[2]:
            raise IndexError(f"head {head} outside [0, {activation.shape[2]})")
        if source_z.shape[0] != activation.shape[0]:
            raise ValueError("source and destination batch sizes differ")
        output = activation.clone()
        output[:, 0, head, :] = source_z[:, 0, head, :].to(
            device=activation.device, dtype=activation.dtype
        )
        return output

    return patch


def capture_final_token_heads(
    model,
    *,
    last_tokens: torch.Tensor,
    prefix_kv_cache,
    prefix_length: int,
    candidate_token_ids: Sequence[int],
    layers: Sequence[int],
    candidate_only: bool = False,
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """Run one incremental forward and capture every head at selected layers."""

    captured: dict[int, torch.Tensor] = {}
    hooks: list[tuple[str, object]] = []
    for layer in layers:
        layer = int(layer)

        def capture(activation, hook, *, layer_index=layer):
            del hook
            captured[layer_index] = activation.detach().cpu()
            return activation

        hooks.append((model.attn_hook_name(layer, "z"), capture))
    if candidate_only:
        logits = model.incremental_last_token_candidate_logits_only(
            last_tokens,
            prefix_kv_cache=prefix_kv_cache,
            prefix_length=prefix_length,
            candidate_token_ids=candidate_token_ids,
            fwd_hooks=hooks,
        )
    else:
        logits, _ = model.incremental_last_token_candidate_stats(
            last_tokens,
            prefix_kv_cache=prefix_kv_cache,
            prefix_length=prefix_length,
            candidate_token_ids=candidate_token_ids,
            fwd_hooks=hooks,
        )
    missing = sorted(set(int(layer) for layer in layers) - set(captured))
    if missing:
        raise RuntimeError(f"Head-output hooks did not fire for layers: {missing}")
    return captured, logits.float()


def patched_final_token_logits(
    model,
    *,
    last_tokens: torch.Tensor,
    prefix_kv_cache,
    prefix_length: int,
    candidate_token_ids: Sequence[int],
    layer: int,
    head: int,
    source_z: torch.Tensor,
) -> torch.Tensor:
    """Run exactly one incremental forward with one final-token head patched."""

    hook = final_token_head_patch_hook(source_z, head=head)
    logits, _ = model.incremental_last_token_candidate_stats(
        last_tokens,
        prefix_kv_cache=prefix_kv_cache,
        prefix_length=prefix_length,
        candidate_token_ids=candidate_token_ids,
        fwd_hooks=[(model.attn_hook_name(int(layer), "z"), hook)],
    )
    return logits.float()


def final_token_head_group_patch_hooks(
    model,
    *,
    components: Sequence[tuple[int, int]],
    source_z_by_layer: dict[int, torch.Tensor],
) -> list[tuple[str, object]]:
    """Build hooks that jointly patch several final-token attention heads."""

    heads_by_layer: dict[int, list[int]] = {}
    for raw_layer, raw_head in components:
        layer, head = int(raw_layer), int(raw_head)
        heads_by_layer.setdefault(layer, []).append(head)
    hooks: list[tuple[str, object]] = []
    for layer, heads in heads_by_layer.items():
        if layer not in source_z_by_layer:
            raise ValueError(f"Missing source head outputs for layer {layer}")
        source_z = source_z_by_layer[layer]
        if source_z.dim() != 4 or source_z.shape[1] != 1:
            raise ValueError("source_z must have shape [batch, 1, heads, d_head]")
        if len(set(heads)) != len(heads):
            raise ValueError(f"Layer {layer} contains duplicate heads")

        def patch(activation, hook, *, selected=tuple(heads), source=source_z):
            del hook
            if activation.dim() != 4 or activation.shape[1] != 1:
                raise ValueError(
                    "KV-cached head patch expected [batch, 1, heads, d_head], "
                    f"got {tuple(activation.shape)}"
                )
            if source.shape[0] != activation.shape[0]:
                raise ValueError("source and destination batch sizes differ")
            invalid = [head for head in selected if not 0 <= head < activation.shape[2]]
            if invalid:
                raise IndexError(f"Heads outside [0, {activation.shape[2]}): {invalid}")
            output = activation.clone()
            output[:, 0, list(selected), :] = source[:, 0, list(selected), :].to(
                device=activation.device, dtype=activation.dtype
            )
            return output

        hooks.append((model.attn_hook_name(layer, "z"), patch))
    return hooks


def full_sequence_final_token_head_group_patch_hooks(
    model,
    *,
    components: Sequence[tuple[int, int]],
    source_z_by_layer: dict[int, torch.Tensor],
) -> list[tuple[str, object]]:
    """Patch selected heads only at the last token of a full-sequence run.

    ``source_z_by_layer`` contains paired clean donors with shape
    ``[batch, heads, d_head]``.  This variant complements the KV-cached helper
    above, whose activations contain only the one newly evaluated token.
    """

    heads_by_layer: dict[int, list[int]] = {}
    for raw_layer, raw_head in components:
        layer, head = int(raw_layer), int(raw_head)
        heads_by_layer.setdefault(layer, []).append(head)
    hooks: list[tuple[str, object]] = []
    for layer, heads in heads_by_layer.items():
        if layer not in source_z_by_layer:
            raise ValueError(f"Missing source head outputs for layer {layer}")
        source_z = source_z_by_layer[layer]
        if source_z.dim() != 3:
            raise ValueError("source_z must have shape [batch, heads, d_head]")
        if len(set(heads)) != len(heads):
            raise ValueError(f"Layer {layer} contains duplicate heads")

        def patch(activation, hook, *, selected=tuple(heads), source=source_z):
            del hook
            if activation.dim() != 4:
                raise ValueError(
                    "full-sequence head patch expected [batch, tokens, heads, "
                    f"d_head], got {tuple(activation.shape)}"
                )
            if source.shape[0] != activation.shape[0]:
                raise ValueError("source and destination batch sizes differ")
            if source.shape[1:] != activation.shape[2:]:
                raise ValueError("source and destination head shapes differ")
            invalid = [head for head in selected if not 0 <= head < activation.shape[2]]
            if invalid:
                raise IndexError(f"Heads outside [0, {activation.shape[2]}): {invalid}")
            output = activation.clone()
            output[:, -1, list(selected), :] = source[:, list(selected), :].to(
                device=activation.device, dtype=activation.dtype
            )
            return output

        hooks.append((model.attn_hook_name(layer, "z"), patch))
    return hooks


def patched_final_token_head_group_logits(
    model,
    *,
    last_tokens: torch.Tensor,
    prefix_kv_cache,
    prefix_length: int,
    candidate_token_ids: Sequence[int],
    components: Sequence[tuple[int, int]],
    source_z_by_layer: dict[int, torch.Tensor],
    extra_fwd_hooks: Sequence[tuple[str, object]] = (),
    candidate_only: bool = False,
) -> torch.Tensor:
    """Run one incremental forward with several final-token heads patched."""

    hooks = final_token_head_group_patch_hooks(
        model,
        components=components,
        source_z_by_layer=source_z_by_layer,
    )
    if candidate_only:
        logits = model.incremental_last_token_candidate_logits_only(
            last_tokens,
            prefix_kv_cache=prefix_kv_cache,
            prefix_length=prefix_length,
            candidate_token_ids=candidate_token_ids,
            fwd_hooks=[*hooks, *extra_fwd_hooks],
        )
    else:
        logits, _ = model.incremental_last_token_candidate_stats(
            last_tokens,
            prefix_kv_cache=prefix_kv_cache,
            prefix_length=prefix_length,
            candidate_token_ids=candidate_token_ids,
            fwd_hooks=[*hooks, *extra_fwd_hooks],
        )
    return logits.float()


def mode_shift(
    patched_logits: torch.Tensor,
    base_logits: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Per-example patched-minus-base number-vs-none gate-logit shift."""

    return gate_logit(patched_logits, temperature=temperature) - gate_logit(
        base_logits, temperature=temperature
    )


@dataclass
class HeadEffectAccumulator:
    """Streaming statistics for one layer/head/direction cell."""

    n: int = 0
    shift_sum: float = 0.0
    positive_count: int = 0
    favorable_count: int = 0
    target_count: int = 0
    exact_count: int = 0

    def update(
        self,
        shifts: torch.Tensor,
        predictions: torch.Tensor,
        *,
        direction: str,
        positions: torch.Tensor,
    ) -> None:
        if direction not in DIRECTIONS:
            raise ValueError(f"Unknown direction: {direction}")
        shifts = shifts.detach().cpu().float()
        predictions = predictions.detach().cpu().long()
        positions = positions.detach().cpu().long()
        if shifts.shape != predictions.shape or shifts.shape != positions.shape:
            raise ValueError("shifts, predictions, and positions must have equal shape")
        self.n += int(shifts.numel())
        self.shift_sum += float(shifts.sum())
        self.positive_count += int(shifts.gt(0).sum())
        if direction == "clean_from_injected":
            self.favorable_count += int(shifts.gt(0).sum())
            self.target_count += int(predictions.lt(10).sum())
            self.exact_count += int(predictions.eq(positions).sum())
        else:
            self.favorable_count += int(shifts.lt(0).sum())
            self.target_count += int(predictions.eq(10).sum())
            self.exact_count += int(predictions.eq(positions).sum())

    def row(self) -> dict:
        if self.n <= 0:
            raise RuntimeError("cannot summarize an empty head effect")
        return {
            "n_trials": self.n,
            "mean_delta_mode": self.shift_sum / self.n,
            "positive_delta_rate": self.positive_count / self.n,
            "favorable_delta_rate": self.favorable_count / self.n,
            "target_success_rate": self.target_count / self.n,
            "exact_number_rate": self.exact_count / self.n,
        }


def assign_direction_ranks(rows: list[dict]) -> None:
    """Add descending clean and ascending injected ranks in place."""

    clean_order = sorted(
        range(len(rows)),
        key=lambda index: rows[index]["clean_from_injected_mean_delta_mode"],
        reverse=True,
    )
    injected_order = sorted(
        range(len(rows)),
        key=lambda index: rows[index]["injected_from_clean_mean_delta_mode"],
    )
    for rank, index in enumerate(clean_order, 1):
        rows[index]["clean_from_injected_rank_desc"] = rank
    for rank, index in enumerate(injected_order, 1):
        rows[index]["injected_from_clean_rank_asc"] = rank


def select_top_heads(
    rows: Sequence[dict],
    *,
    top_k: int,
    rule: str = "bidirectional_rank_sum",
) -> list[dict]:
    """Select and rank single heads using a declared, reproducible rule."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if rule not in SELECTION_RULES:
        raise ValueError(f"Unknown selection rule {rule!r}; expected {SELECTION_RULES}")
    required = {
        "layer",
        "head",
        "clean_from_injected_rank_desc",
        "injected_from_clean_rank_asc",
        "clean_from_injected_mean_delta_mode",
        "injected_from_clean_mean_delta_mode",
    }
    selected_rows: list[dict] = []
    for row in rows:
        missing = required - set(row)
        if missing:
            raise ValueError(f"Head-effect row is missing fields: {sorted(missing)}")
        clean_rank = int(row["clean_from_injected_rank_desc"])
        injected_rank = int(row["injected_from_clean_rank_asc"])
        clean_effect = float(row["clean_from_injected_mean_delta_mode"])
        injected_effect = float(row["injected_from_clean_mean_delta_mode"])
        if rule == "clean_only":
            if clean_effect <= 0:
                continue
            score = clean_rank
            tie_break = (injected_rank, int(row["layer"]), int(row["head"]))
        elif rule == "injected_only":
            if injected_effect >= 0:
                continue
            score = injected_rank
            tie_break = (clean_rank, int(row["layer"]), int(row["head"]))
        else:
            if clean_effect <= 0 or injected_effect >= 0:
                continue
            score = clean_rank + injected_rank
            tie_break = (
                max(clean_rank, injected_rank),
                clean_rank,
                int(row["layer"]),
                int(row["head"]),
            )
        copied = dict(row)
        copied["selection_score"] = score
        copied["_selection_tie_break"] = tie_break
        selected_rows.append(copied)
    if not selected_rows:
        raise ValueError(
            f"No heads have effects in the required direction for rule {rule!r}"
        )
    selected_rows.sort(
        key=lambda row: (row["selection_score"], row["_selection_tie_break"])
    )
    result: list[dict] = []
    for rank, row in enumerate(selected_rows[: min(top_k, len(selected_rows))], 1):
        row.pop("_selection_tie_break")
        row["selection_rank"] = rank
        result.append(row)
    return result


def head_group_spec(name: str, rows: Sequence[dict]) -> str:
    """Serialize selected single-head rows as ``name=LxHy,...``."""

    if not name or "=" in name or "," in name:
        raise ValueError("Head-group name must be non-empty and contain no '=' or ','")
    if not rows:
        raise ValueError("Cannot serialize an empty head group")
    return name + "=" + ",".join(
        f"L{int(row['layer'])}H{int(row['head'])}" for row in rows
    )


def load_head_selection(path: Path) -> dict:
    """Load and minimally validate a frozen head-selection manifest."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported head-selection schema_version")
    if not isinstance(payload.get("model"), str) or not payload["model"]:
        raise ValueError("Head-selection manifest must record a model")
    if payload.get("source_split") not in {
        "validation",
        "test",
        "train",
        "exploratory",
    }:
        raise ValueError("Head-selection manifest has an invalid source_split")
    heads = payload.get("heads")
    if not isinstance(heads, list) or not heads:
        raise ValueError("Head-selection manifest must contain a non-empty heads list")
    pairs: list[tuple[int, int]] = []
    for row in heads:
        if not isinstance(row, dict) or "layer" not in row or "head" not in row:
            raise ValueError("Every selected head must contain layer and head")
        pairs.append((int(row["layer"]), int(row["head"])))
    if len(set(pairs)) != len(pairs):
        raise ValueError("Head-selection manifest contains duplicate heads")
    if int(payload.get("top_k", -1)) != len(heads):
        raise ValueError("Head-selection top_k does not match the heads list")
    expected_spec = head_group_spec(str(payload.get("group_name", "top_heads")), heads)
    if payload.get("gate_group") != expected_spec:
        raise ValueError("gate_group does not match the manifest heads")
    return payload
