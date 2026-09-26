"""Multi-layer residual-stream capture and patch-then-forward.

  - :func:`capture_resid_layers`: residual stream at several layers.
  - :func:`run_injected_and_capture`: the same under an injection hook.
  - :func:`patch_resid_and_forward`: replace one layer's output and finish
    the forward pass.

Naming convention: "layer L" always refers to ``blocks.{L}.hook_resid_post``,
the residual stream after block L, matching :meth:`HookedModel.resid_hook_name`,
extraction.py and injection.py.
"""

from __future__ import annotations

from typing import Literal, Sequence

import torch

from .injection import inject


ResidualHookPoint = Literal["resid_post", "resid_pre"]


def resolve_resid_hook_name(
    model,
    layer: int,
    hook_point: ResidualHookPoint = "resid_post",
) -> str:
    """Resolve a residual hook while keeping ``resid_post`` as the default."""
    if hook_point == "resid_post":
        return model.resid_hook_name(layer)
    if hook_point == "resid_pre":
        return model.resid_pre_hook_name(layer)
    raise ValueError(
        f"Unsupported residual hook point {hook_point!r}; "
        "expected 'resid_post' or 'resid_pre'"
    )


# ---------------------------------------------------------------------------
# Clean multi-layer capture
# ---------------------------------------------------------------------------


def capture_resid_layers(
    model,
    tokens: torch.Tensor,
    layers: Sequence[int],
    *,
    hook_point: ResidualHookPoint = "resid_post",
) -> dict[int, torch.Tensor]:
    """Run a clean forward pass and return selected residual activations.

    Args:
        model: a HookedModel.
        tokens: [batch, seq] integer token ids.
        layers: which layers to capture.

    Returns:
        ``{layer: Tensor[batch, seq, d_model]}`` float32, on CPU.
        Each tensor is the full sequence's residual stream at ``hook_point``.
    """
    hook_names = {
        resolve_resid_hook_name(model, l, hook_point): l for l in layers
    }

    _logits, cache = model.run_with_cache(
        tokens, names=lambda n: n in hook_names
    )
    return {
        layer: cache[hook_name].detach().cpu().float()
        for hook_name, layer in hook_names.items()
    }


# ---------------------------------------------------------------------------
# Injected forward pass + multi-layer capture
# ---------------------------------------------------------------------------


def run_injected_and_capture(
    model,
    tokens: torch.Tensor,
    *,
    injection_layer: int,
    spans: list[tuple[int, int]],
    vector: torch.Tensor,
    strength: float,
    scale: str = "relative_hidden_norm",
    capture_layers: Sequence[int],
    capture_hook_point: ResidualHookPoint = "resid_post",
    capture_last_hooks: dict[str, torch.Tensor] | None = None,
    capture_token_index: int = -1,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Run an injected forward pass and capture requested residual hook points.

    The injection hook fires on ``injection_layer``'s resid_post; the capture
    hooks fire at ``capture_hook_point`` on every layer in ``capture_layers``.
    Because TL runs hooks in registration order and the injection modifies the
    residual stream *before* downstream blocks see it, downstream capture
    layers will see the already-injected representation.

    Args:
        model: a HookedModel.
        tokens: [batch, seq] integer token ids.
        injection_layer: which layer's resid_post to write to.
        spans: one (start, end) span per batch row (exclusive end).
        vector: [d_model] or [batch, d_model] unit vector(s) to inject.
        strength: injection scalar multiplier.
        scale: "unit" or "relative_hidden_norm".
        capture_layers: layers at which to record the selected residual hook.
        capture_hook_point: ``resid_post`` or ``resid_pre`` capture location.
        capture_last_hooks: optional mutable mapping whose keys are additional
            hook names to capture. Each value is replaced with a float32 CPU
            tensor after the forward pass.
        capture_token_index: sequence index captured for
            ``capture_last_hooks``; defaults to the final token.

    Returns:
        ``(logits[:, -1, :], {layer: Tensor[batch, seq, d_model]})``
        Logits are the last-position logits (float32 CPU). Captured tensors
        are float32, CPU.
    """
    captured: dict[int, torch.Tensor] = {}

    # Build the injection hook function (mirrors injection.py)
    inj_hook_name = model.resid_hook_name(injection_layer)
    value = vector if vector.dim() == 2 else vector.unsqueeze(0)

    from .hooks import write_span

    def inj_fn(activation, hook):
        rows = activation.shape[0]
        row_value = value if value.shape[0] == rows else value.expand(rows, -1)
        # broadcast a single span to every batch row so callers can pass one
        # span and use a batched input_ids without mismatch errors
        row_spans = spans * rows if len(spans) == 1 else list(spans)
        return write_span(
            activation,
            spans=row_spans,
            value=row_value,
            mode="add",
            strength=strength,
            scale=scale,
        )

    # Build capture hook functions for each requested layer
    def make_capture(layer: int):
        cap_name = resolve_resid_hook_name(
            model, layer, capture_hook_point
        )

        def cap_fn(activation, hook):
            captured[layer] = activation.detach().cpu().float()
            return activation  # pass-through

        return cap_name, cap_fn

    fwd_hooks = [(inj_hook_name, inj_fn)]
    for l in capture_layers:
        name, fn = make_capture(l)
        fwd_hooks.append((name, fn))

    if capture_last_hooks is not None:
        for extra_name in list(capture_last_hooks):
            def make_extra_capture(name: str):
                def cap_fn(activation, hook):
                    capture_last_hooks[name] = (
                        activation[:, capture_token_index]
                        .detach()
                        .cpu()
                        .float()
                    )
                    return activation

                return cap_fn

            fwd_hooks.append((extra_name, make_extra_capture(extra_name)))

    tokens = tokens.to(model.bridge.cfg.device)
    with torch.inference_mode():
        with model.bridge.hooks(fwd_hooks=fwd_hooks):
            logits = model.bridge(tokens)

    last_logits = logits[:, -1, :].detach().cpu().float()
    return last_logits, captured


# ---------------------------------------------------------------------------
# Patch resid_post at one layer, run rest of forward
# ---------------------------------------------------------------------------


def patch_resid_and_forward(
    model,
    tokens: torch.Tensor,
    *,
    layer: int,
    replacement: torch.Tensor,
) -> torch.Tensor:
    """Overwrite resid_post at *layer* with *replacement* and run downstream.

    All blocks after *layer* will see the patched representation (the hook
    fires at ``hook_resid_post`` so earlier blocks run normally). This is the
    TL equivalent of ``downstream_logits_from_layer_output`` — no
    ``model.model.layers[i]`` accessor needed.

    Args:
        model: a HookedModel.
        tokens: [1, seq] integer token ids (single sequence; the batch
            dimension of *replacement* is expected to match).
        layer: which layer's resid_post to overwrite.
        replacement: [batch, seq, d_model] patched hidden state. Each batch
            row is a separate condition to evaluate in one forward pass.

    Returns:
        [batch, vocab] last-position logits, float32 CPU.
    """
    hook_name = model.resid_hook_name(layer)
    batch = replacement.shape[0]

    def hook_fn(activation, hook):
        return replacement.to(device=activation.device, dtype=activation.dtype)

    # Repeat the single-sequence token ids to match the batch of replacements
    repeated = tokens.expand(batch, -1)
    tokens_dev = repeated.to(model.bridge.cfg.device)

    with torch.inference_mode():
        with model.bridge.hooks(fwd_hooks=[(hook_name, hook_fn)]):
            logits = model.bridge(tokens_dev)

    return logits[:, -1, :].detach().cpu().float()


# ---------------------------------------------------------------------------
# Gather target-token activations from a captured layer tensor
# ---------------------------------------------------------------------------


def gather_target_tokens(
    hidden: torch.Tensor,
    starts: torch.Tensor,
) -> torch.Tensor:
    """Index hidden states at per-row token positions.

    Args:
        hidden: [batch, seq, d_model] or [1, seq, d_model] (clean baseline).
        starts: [batch] integer positions (one per row).

    Returns:
        [batch, d_model].
    """
    if hidden.shape[0] == 1:
        # Clean baseline: same sequence for every row
        return hidden[0, starts.cpu(), :].float()
    rows = torch.arange(starts.shape[0])
    return hidden[rows, starts.cpu(), :].float()
