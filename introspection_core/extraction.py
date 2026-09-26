"""Contrastive concept-vector extraction (Equation 1 of the paper).

For a concept c, the vector is the final-token residual of "Tell me about c."
(rendered with the chat template) minus the mean residual of the same prompt
over the baseline vocabulary, normalized to unit length.

Layer convention: layer L is ``blocks.{L}.hook_resid_post``, the residual
stream after block L, i.e. ``HookedModel.resid_hook_name(L)``. Extraction and
injection read and write the same point, so layer numbers mean the same thing
in extraction.py, injection.py and patching.py.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F


def _last_non_padding_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    """Return each row's final non-padding position for either padding side."""
    if attention_mask.dim() != 2:
        raise ValueError(
            "attention_mask must be 2D, got "
            f"shape {tuple(attention_mask.shape)}"
        )
    positions = torch.arange(
        attention_mask.shape[1], device=attention_mask.device
    ).expand_as(attention_mask)
    last_indices = (
        positions.masked_fill(attention_mask == 0, -1).max(dim=1).values
    )
    if bool((last_indices < 0).any()):
        raise ValueError("attention_mask contains an all-padding row")
    return last_indices


def format_concept_prompt(tokenizer, word: str, template: str = "Tell me about {word}.") -> str:
    """Build the contrastive-extraction prompt for one concept word.

    Uses the chat template with ``add_generation_prompt=True`` when
    available (the assistant turn is left open, so the last token of the
    rendered text is the last token of the *user* turn — matching the
    ``format_prompt`` convention). Falls back to a plain-text "User: ...
    \\n\\nAssistant:" framing for tokenizers with no chat template.
    """
    content = template.format(word=word)
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": content}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"User: {content}\n\nAssistant:"


def load_concepts_from_json(
    path: Path,
    *,
    concept_key: str = "concept_vector_words",
    baseline_key: str = "baseline_words",
    concept_csv: Path | None = None,
    concept_column: str = "concept",
    max_concepts: int | None = None,
) -> tuple[list[str], list[str]]:
    """Load concept + baseline word lists from a small JSON manifest.

    Expects a JSON object with ``{concept_key: [...], baseline_key: [...]}``
    (see
    ``data/dataset/<model>/concepts/train.json`` for
    the shape this package ships). Optionally filters the concept list down to those also
    present in a CSV column (e.g. a quality-filtered subset of a larger
    concept pool) and/or caps the concept count -- both no-ops if omitted.

    Args:
        path: path to the JSON manifest.
        concept_key: JSON key holding the concept word list.
        baseline_key: JSON key holding the baseline word list.
        concept_csv: optional CSV whose ``concept_column`` restricts which
            concepts from ``path`` are kept, preserving ``path``'s order.
        concept_column: column name to read from ``concept_csv``.
        max_concepts: if given, keep only the first ``max_concepts``
            concepts after any CSV filtering (useful for smoke runs).

    Returns:
        ``(concepts, baseline_words)``.
    """
    data = json.loads(Path(path).read_text())
    baseline_words = [str(word) for word in data[baseline_key]]
    concepts = [str(word) for word in data[concept_key]]

    if concept_csv is not None:
        allowed = set(pd.read_csv(concept_csv)[concept_column].astype(str))
        concepts = [concept for concept in concepts if concept in allowed]
    if max_concepts is not None:
        concepts = concepts[:max_concepts]
    if not concepts:
        raise ValueError(f"No concepts selected from {path}")
    return concepts, baseline_words


def load_concept_vector_payload(
    path: Path,
    *,
    concepts: Sequence[str],
    layer: int,
) -> torch.Tensor:
    """Select cached concept vectors by name, returned unit-normalized.

    Payloads are produced by ``scripts/extract_concept_vector_payload.py``.
    Rows are matched on the concept name and returned in ``concepts`` order, so
    one payload extracted over the whole concept population can serve any split
    without re-running extraction. The residual-stream layer is checked so
    cached vectors cannot silently come from the wrong injection site.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    raw_vectors = payload.get("vectors", payload.get("unit_vectors"))
    if not isinstance(raw_vectors, torch.Tensor):
        raise ValueError(f"invalid concept-vector payload: {path}")
    if raw_vectors.dim() != 2:
        raise ValueError(
            "concept-vector payload vectors must be two-dimensional, got "
            f"{tuple(raw_vectors.shape)}"
        )
    rows = payload.get("rows")
    if isinstance(rows, list):
        vector_names = [str(row.get("word", row.get("concept"))) for row in rows]
    elif isinstance(payload.get("concepts"), list):
        vector_names = [str(name) for name in payload["concepts"]]
    else:
        raise ValueError(f"invalid concept-vector payload: {path}")
    if len(vector_names) != raw_vectors.shape[0]:
        raise ValueError(
            f"concept-vector payload names ({len(vector_names)}) and vectors "
            f"({raw_vectors.shape[0]}) disagree: {path}"
        )
    if int(payload.get("layer", layer)) != int(layer):
        raise ValueError("concept-vector payload uses a different layer")

    index: dict[str, int] = {}
    for position, name in enumerate(vector_names):
        # A duplicated name makes selection ambiguous, so refuse rather than
        # silently taking one of them.
        if name in index:
            raise ValueError(f"duplicate concept {name!r} in payload: {path}")
        index[name] = position
    concept_list = [str(concept) for concept in concepts]
    missing = [name for name in concept_list if name not in index]
    if missing:
        raise ValueError(
            f"concept-vector payload {path} is missing {len(missing)} of "
            f"{len(concept_list)} requested concepts, first: {missing[:5]}"
        )
    selection = torch.tensor([index[name] for name in concept_list], dtype=torch.long)
    return F.normalize(raw_vectors.index_select(0, selection).float(), dim=-1)


@torch.inference_mode()
def extract_last_token_residuals(
    model,
    words: list[str],
    *,
    layer: int,
    template: str = "Tell me about {word}.",
    batch_size: int = 16,
    stop_after_layer: bool = False,
) -> torch.Tensor:
    """Run the contrastive-extraction prompt for each word and return the
    last-token residual-stream activation at ``layer`` for each.

    Args:
        model: a HookedModel.
        words: concept or baseline words to prompt with.
        layer: which layer's residual-post point to read
            (``model.resid_hook_name(layer)``).
        template: prompt template, ``{word}`` is substituted.
        batch_size: prompts are batched (left/right padding handled via
            attention_mask; the last *real* token's position is derived
            per-row from the mask, not a fixed ``-1`` index, so padding
            side doesn't matter).
        stop_after_layer: skip later layers when the model backend supports
            ``stop_at_layer``. The requested residual-post hook still runs.

    Returns:
        float32 CPU tensor [len(words), d_model].
    """
    tokenizer = model.tokenizer
    hook_name = model.resid_hook_name(layer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = model.bridge.cfg.device
    chunks: list[torch.Tensor] = []
    for start in range(0, len(words), batch_size):
        batch_words = words[start : start + batch_size]
        prompts = [format_concept_prompt(tokenizer, w, template) for w in batch_words]
        encoded = tokenizer(
            prompts, return_tensors="pt", add_special_tokens=False, padding=True
        ).to(device)

        _logits, cache = model.run_with_cache(
            encoded["input_ids"],
            names=lambda n: n == hook_name,
            attention_mask=encoded["attention_mask"],
            **({"stop_at_layer": layer + 1} if stop_after_layer else {}),
        )
        resid = cache[hook_name]  # [batch, seq, d_model]

        last_indices = _last_non_padding_indices(encoded["attention_mask"])
        batch_indices = torch.arange(resid.shape[0], device=resid.device)
        batch_acts = (
            resid[batch_indices, last_indices.to(resid.device), :]
            .detach()
            .cpu()
            .to(torch.float32)
        )
        chunks.append(batch_acts)

    return torch.cat(chunks, dim=0)


@torch.inference_mode()
def extract_last_token_residuals_by_layer(
    model,
    words: list[str],
    *,
    layers: list[int],
    template: str = "Tell me about {word}.",
    batch_size: int = 16,
) -> dict[int, torch.Tensor]:
    """Extract last-token residuals for several layers in shared forwards."""
    if not layers:
        raise ValueError("layers must be non-empty")
    ordered_layers = list(dict.fromkeys(int(layer) for layer in layers))
    tokenizer = model.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    hook_names = {
        layer: model.resid_hook_name(layer) for layer in ordered_layers
    }
    selected_names = set(hook_names.values())
    device = model.bridge.cfg.device
    chunks: dict[int, list[torch.Tensor]] = {
        layer: [] for layer in ordered_layers
    }

    for start in range(0, len(words), batch_size):
        batch_words = words[start : start + batch_size]
        prompts = [
            format_concept_prompt(tokenizer, word, template)
            for word in batch_words
        ]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        ).to(device)
        _logits, cache = model.run_with_cache(
            encoded["input_ids"],
            names=lambda name: name in selected_names,
            attention_mask=encoded["attention_mask"],
        )
        last_indices = _last_non_padding_indices(encoded["attention_mask"])
        for layer in ordered_layers:
            residual = cache[hook_names[layer]]
            batch_indices = torch.arange(
                residual.shape[0], device=residual.device
            )
            chunks[layer].append(
                residual[
                    batch_indices,
                    last_indices.to(residual.device),
                    :,
                ]
                .detach()
                .cpu()
                .to(torch.float32)
            )

    return {
        layer: torch.cat(layer_chunks, dim=0)
        for layer, layer_chunks in chunks.items()
    }


@torch.inference_mode()
def extract_mean_last_token_residuals_by_layer(
    model,
    words: list[str],
    *,
    layers: list[int],
    template: str = "Tell me about {word}.",
    batch_size: int = 16,
) -> dict[int, torch.Tensor]:
    """Return the streaming mean last-token residual for each layer.

    Unlike :func:`extract_last_token_residuals_by_layer`, this helper keeps
    only one running ``d_model`` sum per layer.  It is therefore suitable for
    baselines defined over an entire tokenizer vocabulary.
    """
    if not words:
        raise ValueError("words must be non-empty")
    if not layers:
        raise ValueError("layers must be non-empty")
    ordered_layers = list(dict.fromkeys(int(layer) for layer in layers))
    tokenizer = model.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    hook_names = {
        layer: model.resid_hook_name(layer) for layer in ordered_layers
    }
    selected_names = set(hook_names.values())
    device = model.bridge.cfg.device
    sums: dict[int, torch.Tensor | None] = {
        layer: None for layer in ordered_layers
    }
    count = 0

    for start in range(0, len(words), batch_size):
        batch_words = words[start : start + batch_size]
        prompts = [
            format_concept_prompt(tokenizer, word, template)
            for word in batch_words
        ]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        ).to(device)
        _logits, cache = model.run_with_cache(
            encoded["input_ids"],
            names=lambda name: name in selected_names,
            attention_mask=encoded["attention_mask"],
        )
        last_indices = _last_non_padding_indices(encoded["attention_mask"])
        batch_indices = torch.arange(
            encoded["input_ids"].shape[0], device=last_indices.device
        )
        for layer in ordered_layers:
            residual = cache[hook_names[layer]]
            batch_sum = (
                residual[
                    batch_indices.to(residual.device),
                    last_indices.to(residual.device),
                    :,
                ]
                .to(torch.float32)
                .sum(dim=0)
                .cpu()
            )
            previous = sums[layer]
            sums[layer] = batch_sum if previous is None else previous + batch_sum
        count += len(batch_words)

    return {
        layer: layer_sum / count
        for layer, layer_sum in sums.items()
        if layer_sum is not None
    }


def extract_concept_vector_matrices(
    model,
    words: list[str],
    baseline_words: list[str],
    *,
    layers: list[int],
    template: str = "Tell me about {word}.",
    batch_size: int = 16,
) -> dict[int, torch.Tensor]:
    """Return one raw ``[concept, d_model]`` vector matrix per layer."""
    if not words:
        raise ValueError("words must be non-empty")
    if not baseline_words:
        raise ValueError("baseline_words must be non-empty")
    baseline_mean_by_layer = extract_mean_last_token_residuals_by_layer(
        model,
        baseline_words,
        layers=layers,
        template=template,
        batch_size=batch_size,
    )
    concept_by_layer = extract_last_token_residuals_by_layer(
        model,
        words,
        layers=layers,
        template=template,
        batch_size=batch_size,
    )
    return {
        layer: concept_by_layer[layer]
        - baseline_mean_by_layer[layer]
        for layer in layers
    }


@dataclass
class ConceptVector:
    """One extracted contrastive concept vector.

    Attributes:
        concept: the concept word.
        layer: the layer it was extracted at (resid_post convention).
        vector: float32 CPU tensor [d_model], NOT unit-normalized —
            callers that want a unit vector (e.g. to hand to
            injection.load_unit_vector's convention) should normalize
            explicitly; this class preserves the raw difference-of-means
            magnitude for anyone who wants it (e.g. for norm-based
            concept-strength comparisons across words).
        baseline_count: how many baseline words the mean was computed over.
        model_name: model identifier the vector was extracted from
            (bookkeeping).
    """

    concept: str
    layer: int
    vector: torch.Tensor
    baseline_count: int
    model_name: str


def extract_concept_vectors(
    model,
    words: list[str],
    baseline_words: list[str],
    *,
    layer: int,
    template: str = "Tell me about {word}.",
    batch_size: int = 16,
) -> list[ConceptVector]:
    """Extract one contrastive concept vector per word in ``words``.

    ``vector = last_token_residual(concept_prompt) -
    mean(last_token_residual(baseline_prompt) for baseline in baseline_words)``,
    all read at ``model.resid_hook_name(layer)`` — the same residual point
    injection.py writes to.

    Args:
        model: a HookedModel.
        words: concept words to extract a vector for (one output per word).
        baseline_words: pooled to a single mean baseline activation,
            shared across every concept in ``words`` (matches the
            "one baseline pool for the whole batch" behavior).
        layer: layer index (resid_post convention).
        template: prompt template, ``{word}`` substituted.
        batch_size: forward-pass batch size.

    Returns:
        list[ConceptVector], one per word in ``words``, same order.
    """
    if not words:
        raise ValueError("words must be non-empty")
    if not baseline_words:
        raise ValueError("baseline_words must be non-empty")

    baseline_acts = extract_last_token_residuals(
        model, baseline_words, layer=layer, template=template, batch_size=batch_size
    )
    concept_acts = extract_last_token_residuals(
        model, words, layer=layer, template=template, batch_size=batch_size
    )
    baseline_mean = baseline_acts.mean(dim=0)

    return [
        ConceptVector(
            concept=word,
            layer=layer,
            vector=concept_acts[idx] - baseline_mean,
            baseline_count=len(baseline_words),
            model_name=model.config.name,
        )
        for idx, word in enumerate(words)
    ]
