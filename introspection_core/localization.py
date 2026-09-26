"""Text -> token-span resolution.

Maps character positions in already-rendered prompt text to token indices with
the fast tokenizer's offset mapping. It has no knowledge of chat templates or
experiments; prompts.py locates the character span and calls into this module.
Any fast HF tokenizer that supports ``return_offsets_mapping=True`` works.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Span:
    """A closed-open token range within one rendered prompt.

    Attributes:
        start: first token index (inclusive).
        end: last token index (exclusive) — so ``end - start`` is the
            number of tokens covered.
        text: the decoded text of tokens[start:end] (for logging/debugging
            — not necessarily identical to the original substring, since
            tokenization can be lossy at the edges).
        token_ids: the actual token ids covered, length ``end - start``.
    """

    start: int
    end: int
    text: str
    token_ids: list[int]


@dataclass
class RenderedPrompt:
    """The output of rendering one prompt template over a batch of items.

    Attributes:
        input_ids: LongTensor [1, seq_len] (or [batch, seq_len] if the
            caller repeats it) ready to feed to ``HookedModel``.
        spans: one Span per item, in the same order as the items passed
            to the renderer (or a single Span for the fixed injection
            target, for templates that use ``injection_marker`` instead of
            a per-item list — see ``PromptTemplate``).
        answer_token_by_choice: maps each candidate answer label (e.g.
            ``"3"`` for "which token") to the token id used to score it.
            Empty dict for templates with no scoring candidates.
        records: one plain dict per item with human-readable bookkeeping
            (choice index, text, token_index, token_id, token_text, ...) —
            convenient for writing directly to a results CSV.
        text: the fully rendered prompt string (for logging).
        clean_target_label: candidate label expected from a clean forward
            pass, or None when clean correctness is undefined.
    """

    input_ids: "object"
    spans: list[Span]
    answer_token_by_choice: dict[str, int]
    records: list[dict]
    text: str
    clean_target_label: str | None = None


def char_pos_to_token(offset_mapping, char_pos: int) -> int:
    """Map a single character position to the token index covering it.

    Falls back to the first token that starts at or after ``char_pos`` if
    no token's [start, end) interval contains it exactly (this happens at
    the boundary of adjacent tokens with no gap). Raises if ``char_pos`` is
    past the end of the text.
    """
    for tok_idx, (tok_start, tok_end) in enumerate(offset_mapping):
        if int(tok_start) <= char_pos < int(tok_end):
            return tok_idx
    for tok_idx, (_tok_start, tok_end) in enumerate(offset_mapping):
        if int(tok_end) > char_pos:
            return tok_idx
    raise ValueError(f"Could not map character offset {char_pos} to a token")


def resolve_char_span_to_tokens(
    offset_mapping,
    input_ids,
    tokenizer,
    char_start: int,
    char_end: int,
    *,
    single_token: bool = False,
) -> Span:
    """Resolve a [char_start, char_end) substring to the Span of tokens
    that cover it.

    Args:
        offset_mapping: the tokenizer's per-token (char_start, char_end)
            pairs for the rendered text (as returned by
            ``tokenizer(..., return_offsets_mapping=True)``).
        input_ids: the corresponding 1D sequence of token ids.
        tokenizer: used only to decode the resolved span's text.
        char_start, char_end: half-open character range in the rendered
            text (e.g. the location of one item's text within a larger
            templated prompt).
        single_token: if True, raise unless the char span is covered by
            exactly one token — used for injection targets and answer
            choices where the caller needs an exact single position, not
            a multi-token span.

    Returns:
        Span with token-index bounds and the decoded text/ids.
    """
    covered = [
        tok_idx
        for tok_idx, (tok_start, tok_end) in enumerate(offset_mapping)
        if int(tok_end) > char_start and int(tok_start) < char_end
    ]
    if not covered:
        raise ValueError(f"Character span [{char_start}, {char_end}) does not map to any token")
    if single_token and len(covered) != 1:
        raise ValueError(
            f"Character span [{char_start}, {char_end}) maps to {len(covered)} tokens "
            f"{covered}, expected exactly 1 (single_token=True)"
        )
    start_idx, end_idx = covered[0], covered[-1] + 1
    token_ids = [int(input_ids[i]) for i in range(start_idx, end_idx)]
    text = tokenizer.decode(token_ids)
    return Span(start=start_idx, end=end_idx, text=text, token_ids=token_ids)


def resolve_text_span_to_tokens(
    text: str,
    offset_mapping,
    input_ids,
    tokenizer,
    char_start: int,
    char_end: int,
    *,
    single_token: bool = False,
) -> Span:
    """Resolve a text span, falling back when a tokenizer reports bad offsets.

    A few fast SentencePiece tokenizers return offsets relative to normalized
    text after chat-template special tokens.  Their token IDs are correct, but
    the offsets no longer index the rendered Python string.  Prefix
    tokenization still gives an exact, model-specific boundary in that case.
    """
    source = text[char_start:char_end]
    offset_error: ValueError | None = None
    offset_span: Span | None = None
    try:
        span = resolve_char_span_to_tokens(
            offset_mapping,
            input_ids,
            tokenizer,
            char_start,
            char_end,
            single_token=single_token,
        )
        if span.text.strip() == source.strip():
            offset_span = span
    except ValueError as error:
        offset_error = error

    full_ids = [int(token_id) for token_id in input_ids]

    def encode_prefix(prefix: str) -> list[int]:
        encoded = tokenizer(prefix, add_special_tokens=False)
        ids = encoded["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(token_id) for token_id in ids]

    def common_prefix_length(left: list[int], right: list[int]) -> int:
        length = 0
        for left_id, right_id in zip(left, right):
            if left_id != right_id:
                break
            length += 1
        return length

    start_idx = common_prefix_length(encode_prefix(text[:char_start]), full_ids)
    end_idx = common_prefix_length(encode_prefix(text[:char_end]), full_ids)
    if end_idx <= start_idx:
        if offset_error is not None:
            raise offset_error
        raise ValueError(
            f"Character span [{char_start}, {char_end}) could not be resolved "
            "by tokenizer prefix alignment"
        )
    token_ids = full_ids[start_idx:end_idx]
    if single_token and len(token_ids) != 1:
        raise ValueError(
            f"Character span [{char_start}, {char_end}) maps to "
            f"{len(token_ids)} tokens via prefix alignment, expected exactly 1 "
            "(single_token=True)"
        )
    prefix_span = Span(
        start=start_idx,
        end=end_idx,
        text=tokenizer.decode(token_ids),
        token_ids=token_ids,
    )
    if (
        offset_span is not None
        and offset_span.start == prefix_span.start
        and offset_span.end == prefix_span.end
    ):
        return offset_span
    return prefix_span
