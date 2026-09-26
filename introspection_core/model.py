"""Model loading and hook-point resolution.

This is the only module that is allowed to know about TransformerLens's
``TransformerBridge`` boot signature and hook-name conventions. Every other
module in this package talks to a :class:`HookedModel` instance and never
imports ``transformer_lens`` directly, which is what keeps the rest of the
package architecture-agnostic.

Tokenization convention (load-bearing, do not change casually):
    Chat templates (``tokenizer.apply_chat_template``) already emit the
    model's BOS token as part of the template text. We therefore tokenize
    with ``add_special_tokens=False`` and feed the resulting ``input_ids``
    straight into the bridge, rather than using ``bridge.to_tokens`` (whose
    default ``prepend_bos=True`` would add a *second* BOS on top of the one
    already in the chat-templated text). This mirrors the old codebase's
    ``add_special_tokens=False`` convention and was verified empirically:
    ``tokenizer(text, add_special_tokens=False).input_ids`` is bit-identical
    to ``bridge.to_tokens(text, prepend_bos=False)``.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import os
from typing import Iterable, Literal, Sequence

import torch
import torch.nn.functional as F

AttnHookKind = Literal["z", "pattern", "qk_scores", "v", "q", "k"]

_ATTN_HOOK_SUFFIX: dict[str, str] = {
    "z": "attn.hook_z",
    "pattern": "attn.hook_pattern",
    "qk_scores": "attn.hook_attn_scores",
    "v": "attn.hook_v",
    "q": "attn.hook_q",
    "k": "attn.hook_k",
}


@dataclass
class ModelConfig:
    """Everything needed to load one model.

    Attributes:
        name: HF model id or local checkpoint path (passed to
            ``TransformerBridge.boot_transformers``).
        device: torch device string, e.g. ``"cuda"`` or ``"cuda:0"``.
        dtype: one of the ``torch`` dtype attribute names, e.g. ``"bfloat16"``.
        trust_remote_code: forwarded to the HF loader; only needed for
            architectures that ship custom modeling code.
    """

    name: str
    device: str = "cuda"
    dtype: str = "bfloat16"
    trust_remote_code: bool = False
    n_devices: int | None = None


class HookedModel:
    """Thin wrapper around a TransformerLens ``TransformerBridge``.

    Owns the bridge, the underlying HF tokenizer, and all hook-point name
    resolution. Downstream modules (injection, patching, scoring, ...)
    interact with the model exclusively through this class so that no other
    module needs to know TransformerLens hook-naming conventions or the
    HF-vs-bridge tokenization subtlety documented at module level.
    """

    def __init__(self, config: ModelConfig):
        # Imported lazily so importing this module (e.g. for type checking
        # or from a process that never loads a model) doesn't require
        # transformer_lens to be installed.
        from transformer_lens.model_bridge import TransformerBridge

        self.config = config
        dtype = getattr(torch, config.dtype)
        n_devices = config.n_devices
        if n_devices is None and str(config.device).startswith("cuda"):
            visible_devices = [
                value
                for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
                if value.strip()
            ]
            if len(visible_devices) > 1:
                n_devices = len(visible_devices)
        use_device_map = bool(n_devices and n_devices > 1)
        load_device = None if use_device_map else config.device
        self.bridge = TransformerBridge.boot_transformers(
            config.name,
            device=load_device,
            dtype=dtype,
            trust_remote_code=config.trust_remote_code,
            device_map="balanced" if use_device_map else None,
        )
        self.bridge.eval()
        self.tokenizer = self.bridge.tokenizer
        self.cfg = self.bridge.cfg
        self._hook_names = set(self.bridge.hook_dict.keys())

    # -- hook-point resolution -------------------------------------------------

    def resid_hook_name(self, layer: int) -> str:
        """Canonical residual-stream write point at the *output* of ``layer``.

        This is the point injection.py writes to and extraction.py reads
        from — both must agree on it so an extracted vector and an injected
        vector reference the same residual point.
        """
        name = f"blocks.{layer}.hook_resid_post"
        self._check_hook_exists(name)
        return name

    def resid_pre_hook_name(self, layer: int) -> str:
        """Residual-stream write point immediately before ``layer`` runs."""
        name = f"blocks.{layer}.hook_resid_pre"
        self._check_hook_exists(name)
        return name

    def attn_hook_name(self, layer: int, kind: AttnHookKind) -> str:
        """Attention-internals hook point at ``layer`` for the given ``kind``.

        kind:
            "z"       — per-head output, pre-o_proj. Shape
                        [batch, pos, n_heads, d_head]. GQA-expanded: TL
                        exposes this already broadcast to n_heads even
                        though the underlying model has fewer KV heads.
            "pattern" — post-softmax attention weights. Shape
                        [batch, n_heads, query_pos, key_pos].
            "qk_scores" — masked pre-softmax QK attention scores. Shape
                          [batch, n_heads, query_pos, key_pos].
            "v"       — per-head value vectors (pre attention-weighting).
                        Shape [batch, pos, n_kv_heads_or_n_heads, d_head].
            "q" / "k" — per-head query / key vectors.
        """
        if kind not in _ATTN_HOOK_SUFFIX:
            raise ValueError(f"Unknown attn hook kind: {kind!r}; expected one of {sorted(_ATTN_HOOK_SUFFIX)}")
        name = f"blocks.{layer}.{_ATTN_HOOK_SUFFIX[kind]}"
        self._check_hook_exists(name)
        return name

    def sublayer_output_hook_names(self, layer: int) -> tuple[str, str]:
        """Return attention- and MLP-output hook points for one layer."""
        names = (
            f"blocks.{layer}.hook_attn_out",
            f"blocks.{layer}.hook_mlp_out",
        )
        for name in names:
            self._check_hook_exists(name)
        return names

    def _check_hook_exists(self, name: str) -> None:
        if name not in self._hook_names:
            raise KeyError(
                f"Hook point {name!r} not found on this model's bridge.hook_dict "
                f"(this model may use a different hook-naming scheme than expected)."
            )

    # -- tokenization -----------------------------------------------------------

    def encode(self, text: str, *, return_offsets: bool = False):
        """Tokenize already-formatted text with the load-bearing convention
        documented at module level: ``add_special_tokens=False`` (chat
        templates already embed BOS)."""
        return self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
            return_offsets_mapping=return_offsets,
        )

    # -- forward passes -----------------------------------------------------------

    def forward_logits(self, tokens: torch.Tensor, fwd_hooks: Iterable = ()) -> torch.Tensor:
        """Teacher-forced forward pass, hooks optionally attached.

        Returns full logits ``[batch, pos, vocab]`` — callers that only need
        the "first generated token" statistic should index ``[:, -1, :]``
        themselves (see scoring.py), keeping this method reusable for
        multi-position analyses too.
        """
        tokens = tokens.to(self.bridge.cfg.device)
        with torch.inference_mode():
            if fwd_hooks:
                return self.bridge.run_with_hooks(tokens, fwd_hooks=list(fwd_hooks))
            return self.bridge(tokens)

    def run_with_cache(self, tokens: torch.Tensor, names=None, **kwargs):
        """Thin pass-through to ``TransformerBridge.run_with_cache``.

        ``names`` is a predicate ``str -> bool`` (TL's ``names_filter``) or
        ``None`` to cache everything (expensive; prefer a filter).
        """
        tokens = tokens.to(self.bridge.cfg.device)
        with torch.inference_mode():
            return self.bridge.run_with_cache(
                tokens,
                names_filter=names,
                **kwargs,
            )

    @staticmethod
    def clone_kv_cache(cache):
        """Clone an HF KV cache so repeated one-token forwards cannot mutate it.

        Hugging Face cache objects are commonly updated in place.  A head sweep
        must therefore give every patched forward an independent copy of the
        same prefix state; otherwise later heads silently receive a longer
        context than earlier heads.  Legacy tuple caches are handled without
        importing ``transformers`` so this remains version-agnostic.
        """

        if isinstance(cache, torch.Tensor):
            return cache.clone()
        if isinstance(cache, tuple):
            return tuple(HookedModel.clone_kv_cache(value) for value in cache)
        if isinstance(cache, list):
            return [HookedModel.clone_kv_cache(value) for value in cache]
        if isinstance(cache, dict):
            return {
                key: HookedModel.clone_kv_cache(value)
                for key, value in cache.items()
            }
        to_legacy = getattr(cache, "to_legacy_cache", None)
        from_legacy = getattr(type(cache), "from_legacy_cache", None)
        if callable(to_legacy) and callable(from_legacy):
            legacy = HookedModel.clone_kv_cache(to_legacy())
            return from_legacy(legacy)
        return copy.deepcopy(cache)

    def build_prefix_kv_cache(
        self,
        tokens: torch.Tensor,
        *,
        fwd_hooks: Iterable = (),
    ):
        """Run ``tokens[:, :-1]`` once and return its reusable HF KV cache."""

        if tokens.dim() != 2 or tokens.shape[1] < 2:
            raise ValueError("tokens must be [batch, seq] with seq >= 2")
        prefix = tokens[:, :-1].to(self.bridge.cfg.device)
        stale_cache = hasattr(self.bridge, "_last_hf_cache")
        if stale_cache:
            del self.bridge._last_hf_cache
        self.bridge._capture_hf_cache = True
        try:
            with torch.inference_mode(), self.bridge.hooks(
                fwd_hooks=list(fwd_hooks)
            ):
                self.bridge(prefix, return_type=None, use_cache=True)
            cache = getattr(self.bridge, "_last_hf_cache", None)
            if cache is None:
                raise RuntimeError(
                    "Model did not return past_key_values; this sweep requires "
                    "a cache-capable decoder-only transformer"
                )
            return self.clone_kv_cache(cache)
        finally:
            self.bridge._capture_hf_cache = False
            if hasattr(self.bridge, "_last_hf_cache"):
                del self.bridge._last_hf_cache

    @torch.inference_mode()
    def attention_output_weights(self, layer: int) -> torch.Tensor:
        """Return verified Linear output weights as [head, d_head, d_model].

        Split input columns of the actual Linear weight. The installed bridge's
        shape heuristic omits its transpose when the weight matrix is square.
        These are pre-normalization weights, including for Gemma's post-attention
        normalization architecture.
        """
        if not 0 <= layer < int(self.cfg.n_layers):
            raise ValueError("invalid attention layer")
        linear = self.bridge.blocks[layer].attn.o.original_component
        if not isinstance(linear, torch.nn.Linear):
            raise TypeError("attention output projection must be torch.nn.Linear")
        heads, head_width, width = int(self.cfg.n_heads), int(self.cfg.d_head), int(self.cfg.d_model)
        if tuple(linear.weight.shape) != (width, heads * head_width):
            raise ValueError("attention output weight dimensions disagree with model")
        return linear.weight.detach().T.reshape(heads, head_width, width)

    @torch.inference_mode()
    def final_head_ov_inputs(
        self, tokens: torch.Tensor, components: Sequence[tuple[int, int]],
        *, fwd_hooks: Iterable = (), residual_output: dict | None = None,
        attention_output: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Capture last-query z and attention-weighted actual V-projection inputs.

        Returns CPU float32 [batch, head, d_head/d_model] tensors. Uses the
        prefix-cache/one-token path of the original STE measurements; the last
        token stops after the last requested layer. No A/V patching is applied.
        If supplied, residual_output receives last-token residuals by layer.
        attention_output optionally receives actual V inputs and last-query
        patterns/scores by layer, as CPU float32 tensors for diagnostics.
        Intervention hooks apply to the prefix only; callers must restrict
        injection positions to strictly before the final input token.
        """
        from transformer_lens.model_bridge.exceptions import StopAtLayerException

        if tokens.ndim != 2 or tokens.shape[0] < 1 or tokens.shape[1] < 2:
            raise ValueError("requires a nonempty batch and a prefix plus final token")
        if not components or len(set(components)) != len(components):
            raise ValueError("components must be nonempty and unique")
        by_layer = {}
        for index, (layer, head) in enumerate(components):
            if not 0 <= layer < int(self.cfg.n_layers):
                raise ValueError("invalid attention layer")
            if not 0 <= head < int(self.cfg.n_heads):
                raise ValueError("invalid head")
            by_layer.setdefault(layer, []).append((index, head))
        batch = len(tokens)
        z = torch.full((batch, len(components), int(self.cfg.d_head)), torch.nan)
        u = torch.full((batch, len(components), int(self.cfg.d_model)), torch.nan)
        handles, inputs = [], {}
        hooks = []
        try:
            for layer, entries in by_layer.items():
                cols, heads = zip(*entries)
                linear = self.bridge.blocks[layer].attn.v.original_component
                if not isinstance(linear, torch.nn.Linear):
                    raise TypeError("V projection must be a Linear")

                def capture_input(module, args, layer=layer):
                    del module
                    x = args[0].detach()
                    inputs[layer] = torch.cat((inputs[layer], x), dim=1) if layer in inputs else x

                handles.append(linear.register_forward_pre_hook(capture_input))

                def capture_pattern(pattern, hook, layer=layer, cols=cols, heads=heads):
                    del hook
                    x = inputs.pop(layer)
                    attention = pattern[:, list(heads), -1, :].float()
                    if attention.shape[-1] != x.shape[1]:
                        raise ValueError("cached attention/input length mismatch")
                    u[:, list(cols)] = torch.bmm(attention, x.float()).cpu()
                    if attention_output is not None:
                        attention_output.setdefault(layer, {}).update(
                            v_inputs=x.float().cpu(), pattern=attention.cpu())
                    return pattern

                def capture_z(value, hook, cols=cols, heads=heads):
                    del hook
                    z[:, list(cols)] = value[:, -1, list(heads)].float().cpu()
                    return value

                hooks.extend([(self.attn_hook_name(layer, "pattern"), capture_pattern),
                              (self.attn_hook_name(layer, "z"), capture_z)])
                if attention_output is not None:
                    def capture_scores(value, hook, layer=layer, heads=heads):
                        del hook
                        attention_output.setdefault(layer, {})['scores'] = (
                            value[:, list(heads), -1, :].detach().float().cpu())
                        return value
                    hooks.append((self.attn_hook_name(layer, "qk_scores"), capture_scores))

            if residual_output is not None:
                for layer in by_layer:
                    def capture_residual(value, hook, layer=layer):
                        del hook
                        residual_output[layer] = value[:, -1].detach().float().cpu()
                        return value
                    hooks.append((self.resid_hook_name(layer), capture_residual))

            def stop(value, hook):
                del hook
                raise StopAtLayerException(value)

            hooks.append((self.resid_hook_name(max(by_layer)), stop))
            # The original STE evaluator observes hook_v on these layers during
            # the prefix pass. Preserve the same bridge execution path.
            prefix_hooks = list(fwd_hooks)
            prefix_hooks.extend((self.attn_hook_name(layer, "v"), lambda value, hook: value)
                                for layer in by_layer)
            cache = self.build_prefix_kv_cache(tokens, fwd_hooks=prefix_hooks)
            device = self.bridge.cfg.device
            prefix_length = tokens.shape[1] - 1
            with self.bridge.hooks(fwd_hooks=hooks):
                self.bridge(tokens[:, -1:].to(device), return_type=None, use_cache=True,
                    past_key_values=cache,
                    attention_mask=torch.ones(tokens.shape, dtype=torch.long, device=device),
                    position_ids=torch.full((batch, 1), prefix_length, dtype=torch.long, device=device))
        finally:
            for handle in handles:
                handle.remove()
        if not torch.isfinite(z).all() or not torch.isfinite(u).all():
            raise RuntimeError("incomplete OV input capture")
        if attention_output is not None:
            missing = [layer for layer in by_layer
                       if {"v_inputs", "pattern", "scores"} - set(attention_output.get(layer, {}))]
            if missing:
                raise RuntimeError(f"incomplete attention diagnostics at layers {missing}")
        return z, u

    @torch.inference_mode()
    def incremental_last_token_candidate_stats(
        self,
        last_tokens: torch.Tensor,
        *,
        prefix_kv_cache,
        prefix_length: int,
        candidate_token_ids: Sequence[int] | torch.Tensor,
        fwd_hooks: Iterable = (),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score one final token from a reusable prefix cache.

        The cache is cloned internally because HF cache implementations may
        append the final token in place.  Hooks therefore see tensors with a
        sequence axis of length one, and a head patch can only affect the final
        prompt token by construction.
        """

        candidate_logits, last_logits = self._incremental_last_token_candidate_logits(
            last_tokens,
            prefix_kv_cache=prefix_kv_cache,
            prefix_length=prefix_length,
            candidate_token_ids=candidate_token_ids,
            fwd_hooks=fwd_hooks,
        )
        log_normalizer = torch.logsumexp(last_logits, dim=-1, keepdim=True)
        candidate_log_probs = candidate_logits - log_normalizer
        return (
            candidate_logits.detach().cpu(),
            candidate_log_probs.detach().cpu(),
        )

    @torch.inference_mode()
    def incremental_last_token_candidate_logits_only(
        self,
        last_tokens: torch.Tensor,
        *,
        prefix_kv_cache,
        prefix_length: int,
        candidate_token_ids: Sequence[int] | torch.Tensor,
        fwd_hooks: Iterable = (),
    ) -> torch.Tensor:
        """Score only candidate labels, avoiding a full-vocabulary projection."""

        candidate_logits, _ = self._incremental_last_token_candidate_logits(
            last_tokens,
            prefix_kv_cache=prefix_kv_cache,
            prefix_length=prefix_length,
            candidate_token_ids=candidate_token_ids,
            fwd_hooks=fwd_hooks,
            candidate_only=True,
        )
        return candidate_logits.detach().cpu()

    def differentiable_incremental_last_token_candidate_logits(
        self,
        last_tokens: torch.Tensor,
        *,
        prefix_kv_cache,
        prefix_length: int,
        candidate_token_ids: Sequence[int] | torch.Tensor,
        fwd_hooks: Iterable = (),
    ) -> torch.Tensor:
        """Score one cached final token while retaining hook gradients.

        The model parameters and prefix cache may be frozen.  A differentiable
        intervention hook, such as a learned attention-head mask, still
        receives gradients from the returned GPU-resident candidate logits.
        The prefix cache is cloned so repeated optimizer steps all start from
        the same immutable prefix state.
        """

        candidate_logits, _ = self._incremental_last_token_candidate_logits(
            last_tokens,
            prefix_kv_cache=prefix_kv_cache,
            prefix_length=prefix_length,
            candidate_token_ids=candidate_token_ids,
            fwd_hooks=fwd_hooks,
        )
        return candidate_logits

    def _incremental_last_token_candidate_logits(
        self,
        last_tokens: torch.Tensor,
        *,
        prefix_kv_cache,
        prefix_length: int,
        candidate_token_ids: Sequence[int] | torch.Tensor,
        fwd_hooks: Iterable,
        candidate_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Shared grad-mode-preserving cached candidate-logit path."""

        from transformer_lens.model_bridge.exceptions import StopAtLayerException

        if last_tokens.dim() != 2 or last_tokens.shape[1] != 1:
            raise ValueError("last_tokens must have shape [batch, 1]")
        if prefix_length <= 0:
            raise ValueError("prefix_length must be positive")
        if isinstance(candidate_token_ids, torch.Tensor):
            if candidate_token_ids.numel() == 0:
                raise ValueError("candidate_token_ids must be non-empty")
        elif not candidate_token_ids:
            raise ValueError("candidate_token_ids must be non-empty")

        tokens = last_tokens.to(self.bridge.cfg.device)
        batch_size = int(tokens.shape[0])
        hooks = list(fwd_hooks)
        final_norm_name = "ln_final.hook_out"
        stop_at_normalized_output = final_norm_name in self._hook_names
        stop_name = (
            final_norm_name
            if stop_at_normalized_output
            else self.resid_hook_name(int(self.cfg.n_layers) - 1)
        )

        def stop_after_final_block(residual, hook):
            del hook
            raise StopAtLayerException(residual)

        hooks.append((stop_name, stop_after_final_block))
        total_length = prefix_length + 1
        attention_mask = torch.ones(
            (batch_size, total_length),
            dtype=torch.long,
            device=tokens.device,
        )
        position_ids = torch.full(
            (batch_size, 1),
            prefix_length,
            dtype=torch.long,
            device=tokens.device,
        )
        cache = self.clone_kv_cache(prefix_kv_cache)
        with self.bridge.hooks(fwd_hooks=hooks):
            final_residual = self.bridge(
                tokens,
                return_type=None,
                use_cache=True,
                past_key_values=cache,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        if not isinstance(final_residual, torch.Tensor):
            raise RuntimeError(
                "TransformerLens did not return the final residual after the stop hook"
            )
        final_position = final_residual[:, -1:, :]
        normalized = (
            final_position
            if stop_at_normalized_output
            else self.bridge.ln_final(final_position)
        )
        if candidate_only:
            return self._candidate_unembed_logits(normalized, candidate_token_ids), None
        last_logits = self.bridge.unembed(normalized)[:, 0, :]
        last_logits = self._apply_final_logit_transforms(last_logits).float()
        token_ids = torch.as_tensor(
            candidate_token_ids, dtype=torch.long, device=last_logits.device
        )
        if token_ids.dim() == 1:
            candidate_logits = last_logits.index_select(-1, token_ids)
        elif token_ids.dim() == 2:
            if token_ids.shape[0] != batch_size:
                raise ValueError(
                    "row-specific candidate_token_ids batch does not match "
                    f"tokens: {tuple(token_ids.shape)} vs batch={batch_size}"
                )
            candidate_logits = last_logits.gather(-1, token_ids)
        else:
            raise ValueError(
                "candidate_token_ids must be one- or two-dimensional"
            )
        return candidate_logits, last_logits

    def _candidate_unembed_logits(
        self,
        normalized: torch.Tensor,
        candidate_token_ids: Sequence[int] | torch.Tensor,
    ) -> torch.Tensor:
        """Apply the real unembedding Linear to only shared candidate rows."""

        token_ids = torch.as_tensor(
            candidate_token_ids, dtype=torch.long, device=normalized.device
        )
        if token_ids.dim() != 1 or token_ids.numel() == 0:
            raise ValueError("candidate-only scoring requires shared nonempty token ids")
        linear = self.bridge.unembed.original_component
        if not isinstance(linear, torch.nn.Linear):
            raise TypeError("candidate-only scoring requires a Linear unembedding")
        weight = linear.weight.index_select(0, token_ids.to(linear.weight.device))
        bias = (
            linear.bias.index_select(0, token_ids.to(linear.bias.device))
            if linear.bias is not None
            else None
        )
        selected = F.linear(normalized.to(linear.weight.dtype), weight, bias)[:, 0, :]
        return self._apply_final_logit_transforms(selected).float()

    def _apply_final_logit_transforms(
        self, logits: torch.Tensor
    ) -> torch.Tensor:
        """Match model-specific transforms applied after unembedding."""
        soft_cap = float(
            getattr(self.cfg, "output_logits_soft_cap", 0.0) or 0.0
        )
        if soft_cap > 0.0:
            logits = soft_cap * torch.tanh(logits / soft_cap)
        return logits

    def last_token_candidate_stats(
        self,
        tokens: torch.Tensor,
        *,
        candidate_token_ids: Sequence[int] | torch.Tensor,
        fwd_hooks: Iterable = (),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return restricted logits and full-vocabulary log probabilities.

        This avoids materializing ``[batch, sequence, vocabulary]`` logits
        during large localization sweeps. The bridge is stopped after its
        final normalization (or final residual as a fallback), only the last
        sequence position is unembedded, and candidate columns are returned.

        Returns:
            ``(candidate_logits, candidate_log_probs)`` as float32 CPU
            tensors of shape ``[batch, n_candidates]``. The logits are used
            for the closed-set argmax; the log probabilities retain the
            full-vocabulary normalization needed for ``mean_correct_prob``.
        """
        with torch.inference_mode():
            candidate_logits, last_logits = self._last_token_candidate_logits(
                tokens,
                candidate_token_ids=candidate_token_ids,
                fwd_hooks=fwd_hooks,
            )
            log_normalizer = torch.logsumexp(last_logits, dim=-1, keepdim=True)
            candidate_log_probs = candidate_logits - log_normalizer
        return (
            candidate_logits.detach().cpu(),
            candidate_log_probs.detach().cpu(),
        )

    def differentiable_last_token_candidate_logits(
        self,
        tokens: torch.Tensor,
        *,
        candidate_token_ids: Sequence[int] | torch.Tensor,
        fwd_hooks: Iterable = (),
    ) -> torch.Tensor:
        """Return GPU candidate logits with gradients through forward hooks.

        Model parameters can remain frozen while a differentiable intervention
        hook is optimized.  This shares the same final-normalization,
        unembedding, soft-cap, and candidate-indexing path as
        :meth:`last_token_candidate_stats`.
        """

        candidate_logits, _last_logits = self._last_token_candidate_logits(
            tokens,
            candidate_token_ids=candidate_token_ids,
            fwd_hooks=fwd_hooks,
        )
        return candidate_logits

    def _last_token_candidate_logits(
        self,
        tokens: torch.Tensor,
        *,
        candidate_token_ids: Sequence[int] | torch.Tensor,
        fwd_hooks: Iterable,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared grad-mode-preserving candidate-logit implementation."""

        from transformer_lens.model_bridge.exceptions import StopAtLayerException

        if isinstance(candidate_token_ids, torch.Tensor):
            if candidate_token_ids.numel() == 0:
                raise ValueError("candidate_token_ids must be non-empty")
        elif not candidate_token_ids:
            raise ValueError("candidate_token_ids must be non-empty")

        tokens = tokens.to(self.bridge.cfg.device)
        hooks = list(fwd_hooks)
        final_norm_name = "ln_final.hook_out"
        stop_at_normalized_output = final_norm_name in self._hook_names
        stop_name = (
            final_norm_name
            if stop_at_normalized_output
            else self.resid_hook_name(int(self.cfg.n_layers) - 1)
        )

        def stop_after_final_block(residual, hook):
            del hook
            raise StopAtLayerException(residual)

        hooks.append((stop_name, stop_after_final_block))
        with self.bridge.hooks(fwd_hooks=hooks):
            final_residual = self.bridge(tokens, return_type=None)
        if not isinstance(final_residual, torch.Tensor):
            raise RuntimeError(
                "TransformerLens did not return the final residual after the stop hook"
            )

        final_position = final_residual[:, -1:, :]
        normalized = (
            final_position
            if stop_at_normalized_output
            else self.bridge.ln_final(final_position)
        )
        last_logits = self.bridge.unembed(normalized)[:, 0, :]
        last_logits = self._apply_final_logit_transforms(last_logits).float()
        token_ids = torch.as_tensor(
            candidate_token_ids,
            dtype=torch.long,
            device=last_logits.device,
        )
        if token_ids.dim() == 1:
            candidate_logits = last_logits.index_select(-1, token_ids)
        elif token_ids.dim() == 2:
            if token_ids.shape[0] != last_logits.shape[0]:
                raise ValueError(
                    "row-specific candidate_token_ids batch does not "
                    f"match logits: {tuple(token_ids.shape)} vs "
                    f"{tuple(last_logits.shape)}"
                )
            candidate_logits = last_logits.gather(-1, token_ids)
        else:
            raise ValueError(
                "candidate_token_ids must be one- or two-dimensional"
            )
        return candidate_logits, last_logits

    def attention_sums_and_candidate_logits(
        self,
        tokens: torch.Tensor,
        *,
        layers: Sequence[int],
        candidate_token_ids: Sequence[int],
        localization_token_indices: Sequence[int] | None = None,
        fwd_hooks: Iterable = (),
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """Collect batch-summed attention without materializing full logits.

        This is the memory-bounded primitive used by averaged attention
        visualizations.  A normal ``run_with_cache`` retains every requested
        ``[batch, head, query, key]`` pattern and the bridge's full
        ``[batch, position, vocabulary]`` logits.  Large concept batches make
        both tensors needlessly expensive.

        Here, each attention hook immediately sums over the batch and moves
        that sum to CPU.  A hook at the final-normalization output then stops
        the TransformerLens bridge before unembedding.  Only the final
        sequence position is unembedded, so scoring still has model-native
        candidate logits without constructing logits for all prompt positions.
        Bridges without a final-normalization hook fall back to stopping at
        the final residual and applying ``ln_final`` once.

        When ``localization_token_indices`` is provided, each attention hook
        also retains the final-query argmax among those tokens for every
        batch row.  This is only ``batch * layers * heads`` integers and lets
        callers compute trial-level localization accuracy without retaining
        full per-example attention tensors.

        Returns:
            ``(attention_sum, candidate_logits)`` where attention_sum is
            float32 CPU ``[len(layers), n_heads, seq, seq]`` and
            candidate_logits is float32 CPU ``[batch, n_candidates]``.  If
            localization tokens are requested, a third tensor with shape
            ``[batch, len(layers), n_heads]`` contains the winning token
            offset for each trial, layer, and head.
        """
        from transformer_lens.model_bridge.exceptions import StopAtLayerException

        if not layers:
            raise ValueError("layers must be non-empty")
        if not candidate_token_ids:
            raise ValueError("candidate_token_ids must be non-empty")
        if (
            localization_token_indices is not None
            and not localization_token_indices
        ):
            raise ValueError("localization_token_indices must be non-empty")

        tokens = tokens.to(self.bridge.cfg.device)
        sums: dict[int, torch.Tensor] = {}
        localization_predictions: dict[int, torch.Tensor] = {}
        hooks = list(fwd_hooks)

        for layer in layers:
            hook_name = self.attn_hook_name(int(layer), "pattern")

            def sum_pattern(pattern, hook, *, layer_index=int(layer)):
                del hook
                sums[layer_index] = pattern.detach().float().sum(dim=0).cpu()
                if localization_token_indices is not None:
                    token_index = torch.as_tensor(
                        localization_token_indices,
                        dtype=torch.long,
                        device=pattern.device,
                    )
                    localization_predictions[layer_index] = (
                        pattern[:, :, -1, :]
                        .index_select(-1, token_index)
                        .argmax(dim=-1)
                        .detach()
                        .cpu()
                    )
                return pattern

            hooks.append((hook_name, sum_pattern))

        final_norm_name = "ln_final.hook_out"
        stop_at_normalized_output = final_norm_name in self._hook_names
        stop_name = (
            final_norm_name
            if stop_at_normalized_output
            else self.resid_hook_name(int(self.cfg.n_layers) - 1)
        )

        def stop_after_final_block(residual, hook):
            del hook
            raise StopAtLayerException(residual)

        hooks.append((stop_name, stop_after_final_block))

        with torch.inference_mode(), self.bridge.hooks(fwd_hooks=hooks):
            final_residual = self.bridge(tokens, return_type=None)

        missing = [layer for layer in layers if int(layer) not in sums]
        if missing:
            raise RuntimeError(f"Attention hooks did not fire for layers: {missing}")
        if not isinstance(final_residual, torch.Tensor):
            raise RuntimeError(
                "TransformerLens did not return the final residual after the stop hook"
            )

        with torch.inference_mode():
            final_position = final_residual[:, -1:, :]
            normalized = (
                final_position
                if stop_at_normalized_output
                else self.bridge.ln_final(final_position)
            )
            last_logits = self.bridge.unembed(normalized)[:, 0, :]
            last_logits = self._apply_final_logit_transforms(last_logits)
            token_ids = torch.tensor(
                list(candidate_token_ids),
                dtype=torch.long,
                device=last_logits.device,
            )
            candidate_logits = (
                last_logits.index_select(-1, token_ids).detach().float().cpu()
            )
        attention_sum = torch.stack([sums[int(layer)] for layer in layers], dim=0)
        if localization_token_indices is None:
            return attention_sum.contiguous(), candidate_logits
        missing_predictions = [
            layer for layer in layers if int(layer) not in localization_predictions
        ]
        if missing_predictions:
            raise RuntimeError(
                "Attention localization hooks did not fire for layers: "
                f"{missing_predictions}"
            )
        predictions = torch.stack(
            [localization_predictions[int(layer)] for layer in layers],
            dim=1,
        )
        return attention_sum.contiguous(), candidate_logits, predictions
