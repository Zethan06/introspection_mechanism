"""Positional residual-stream injection.

Adds ``alpha * ||h_p|| * v`` to the residual stream at the output of the
injection layer (``HookedModel.resid_hook_name(layer)``) at the requested
token positions. ``inject(...)`` is a context manager that wraps whatever
forward call is made inside it.
"""

from __future__ import annotations

import contextlib

import torch

from .hooks import write_span


def normalize_unit_vector(vector: torch.Tensor) -> torch.Tensor:
    """Validate finiteness/nonzero-norm and L2-normalize a vector to unit length.

    Raises if the vector is non-finite or has (numerically) zero norm —
    an injection with an invalid vector would silently do nothing or
    produce NaNs downstream, which research code should never mask.

    Shared by :func:`load_unit_vector` (vectors loaded from disk) and any
    caller with an in-memory vector (e.g. ``extraction.ConceptVector.vector``,
    which is a raw difference-of-means, not yet unit-normalized) — both need
    the same validation before injection.
    """
    vector = vector.detach().to(dtype=torch.float32)
    if not bool(torch.isfinite(vector).all()):
        raise ValueError("Invalid vector: contains non-finite values")
    norm = float(torch.linalg.vector_norm(vector))
    if not torch.isfinite(torch.tensor(norm)) or norm <= 0.0:
        raise ValueError(f"Invalid vector: norm={norm}")
    return vector / norm


def load_unit_vector(path, *, device, dtype) -> torch.Tensor:
    """Load a saved vector (as produced by extraction.py or the old
    vector_extraction.py) and normalize it via :func:`normalize_unit_vector`."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    vector = data["vector"] if isinstance(data, dict) else data
    try:
        unit = normalize_unit_vector(vector)
    except ValueError as exc:
        raise ValueError(f"Invalid vector at {path}: {exc}") from exc
    return unit.to(device=device, dtype=dtype)


def make_injection_hook(
    model,
    *,
    layer: int,
    positions,
    vector: torch.Tensor,
    strength: float,
    scale: str = "relative_hidden_norm",
):
    """Return one reusable residual-addition hook tuple.

    Unlike :func:`inject`, this does not install the hook.  It is intended for
    workflows that need to compose injection with capture, patching, or stop
    hooks in one forward pass.
    """

    hook_name = model.resid_hook_name(layer)
    spans = list(positions)
    value = vector if vector.dim() == 2 else vector.unsqueeze(0)

    def hook_fn(activation, hook):
        del hook
        rows = activation.shape[0]
        row_value = value if value.shape[0] == rows else value.expand(rows, -1)
        return write_span(
            activation,
            spans=spans,
            value=row_value.to(
                device=activation.device,
                dtype=activation.dtype,
            ),
            mode="add",
            strength=strength,
            scale=scale,
        )

    return hook_name, hook_fn


@contextlib.contextmanager
def inject(
    model,
    *,
    layer: int,
    positions,
    vector: torch.Tensor,
    strength: float,
    scale: str = "relative_hidden_norm",
):
    """Context manager: while active, every forward pass through ``model``
    adds ``strength * scale_multiplier * vector`` to the residual stream
    at ``layer``, at the given positions.

    Args:
        model: a HookedModel.
        layer: which layer's resid_post to write to.
        positions: one (start, end) span per batch row (see
            hooks.write_span) — the *same* span shape the batch's forward
            call will use. If you only ever run a single example at a
            time, pass a length-1 list.
        vector: [d_model] (broadcast to every row) or [batch, d_model]
            (one vector per row) — both accepted; a 1D vector is expanded.
        strength: scalar multiplier.
        scale: "unit" (add ``strength * vector`` as-is) or
            "relative_hidden_norm" (scale by the existing activation's
            norm at each position first — matches the default
            steering-vector convention).

    Example:
        with inject(model, layer=8, positions=[(5, 6)], vector=v, strength=6.0):
            logits = model.forward_logits(tokens)

    Implementation note: built on ``TransformerBridge.hooks(...)``, TL's
    own context manager, rather than manual ``add_hook``/``reset_hooks``
    bookkeeping — hooks are guaranteed removed on exit even if the wrapped
    forward pass raises.
    """
    hook = make_injection_hook(
        model,
        layer=layer,
        positions=positions,
        vector=vector,
        strength=strength,
        scale=scale,
    )
    with model.bridge.hooks(fwd_hooks=[hook]):
        yield


def run_injected(
    model,
    tokens: torch.Tensor,
    *,
    layer: int,
    spans,
    vector: torch.Tensor,
    strength: float,
    scale: str = "relative_hidden_norm",
) -> torch.Tensor:
    """Convenience one-shot: run a single teacher-forced forward pass with
    injection active and return the full logits tensor.

    Equivalent to opening :func:`inject` and calling
    ``model.forward_logits(tokens)`` inside it — provided for the common
    case where the caller doesn't need the context manager's flexibility.
    """
    with inject(model, layer=layer, positions=spans, vector=vector, strength=strength, scale=scale):
        return model.forward_logits(tokens)
