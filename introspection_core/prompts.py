"""PromptManager — the prompt registry and one renderer.

Two prompts are registered:

- ``semantic_highinj_posref_gate_balanced_disrupts`` — the evaluation prompt
  used for every experiment. It lists ten ``TOKEN <label>: <candidate>``
  entries and allows the ten position labels plus ``none``. Its letter
  (``_letters_a_j``) and number-word (``_numwords_one_ten``) variants keep the
  wording and change only the printed labels; shuffled-label arms are built
  per cluster with :func:`build_labeled_disrupts_template`.
- ``token_localization`` — the clean position-prior prompt used to screen
  candidate tokens and clusters. The assistant turn is prefilled with
  ``It is located in TOKEN `` and only the ten position labels are scored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .localization import RenderedPrompt, Span, resolve_text_span_to_tokens
from .scoring import answer_token_ids


@dataclass
class PromptTemplate:
    """One reusable prompt shape.

    Two mutually exclusive injection-target shapes are supported:

    - **Per-item** (``item_marker`` set, ``injection_marker`` left None):
      the template renders a *list* of items (sentences, single-token
      words, ...), each individually located via ``item_marker``, and
      ``candidate_labels`` derives the answer set from ``len(items)``
      (e.g. "which one of these N things"). Every registered template
      uses this shape.
    - **Fixed marker** (``injection_marker`` set, ``item_marker`` left
      None): the template has no per-item list — exactly one fixed
      injection target (a literal substring baked into ``turns()``'s
      output) and a candidate answer set that is independent of any item
      count (``fixed_candidate_labels``). Callers render such a template
      with ``items=[]``.

    Attributes:
        name: registry key.
        turns: builds the chat ``messages`` list given
            ``(items: list[str], preamble: str, suffix: str)``. Owns all
            prompt wording; the renderer only needs the item/injection
            marker to locate spans afterward. For fixed-marker templates,
            ``items`` is always ``[]``.
        item_marker: format string with ``{i}`` (1-indexed item number)
            and ``{item}`` placeholders, used to locate each item's
            character span in the rendered text. Must appear verbatim
            (after ``.format()``) inside the text produced by ``turns``.
            None for fixed-marker templates.
        candidate_labels: label strings usable as "first generated token"
            answers (e.g. ``["1", "2", ..., str(len(items))]`` for a
            "which one" template), or None if this template has no
            natural candidate set (resolved lazily per render since the
            count depends on ``len(items)`` for some templates). Ignored
            if ``fixed_candidate_labels`` is set.
        single_token_items: if True, each item's marker span must resolve
            to exactly one token (raises otherwise) — required whenever
            the item itself is the injection target and the experiment
            needs an exact single position.
        injection_marker: a literal substring (not a format string) that
            must appear verbatim in the text produced by ``turns`` — the
            single fixed injection target for this template. Mutually
            exclusive with ``item_marker``.
        fixed_candidate_labels: candidate answer labels independent of
            item count, used together with ``injection_marker``.
        strict_candidate_labels: if True, every candidate label must
            tokenize to exactly one token (see
            ``scoring.answer_token_ids``'s ``strict`` flag) — set this for
            rating-scale templates where the scoring assumption depends
            on an exact single answer position. Left False for localization templates,
            which tolerate multi-token labels via the last-sub-token
            fallback.
        item_index_start: numeric marker/record label for the first item.
            All registered prompts use 0. Ignored when ``item_labels`` is set.
        item_labels: optional exact marker/record label for every item. Used
            by non-numeric layouts such as the fixed A-through-J prompt.
        add_generation_prompt: if True, ``turns`` ends on a user message and
            the renderer asks the tokenizer to append a fresh assistant turn.
            If False, ``turns`` supplies an assistant prefill that the
            renderer continues.
        disable_thinking: if True, pass ``enable_thinking=False`` to the chat
            template so models such as Qwen3 score the requested answer token
            rather than the first token of a hidden reasoning turn.
        clean_target_label: candidate label expected for a clean,
            no-intervention forward pass. ``None`` means clean correctness is
            undefined for this template.
    """

    name: str
    turns: Callable[[list[str], str, str], list[dict]]
    item_marker: str | None = None
    candidate_labels: Callable[[int], list[str]] | None = None
    single_token_items: bool = False
    injection_marker: str | None = None
    fixed_candidate_labels: list[str] | None = None
    strict_candidate_labels: bool = False
    item_index_start: int = 0
    item_labels: tuple[str, ...] | None = None
    add_generation_prompt: bool = False
    disable_thinking: bool = False
    clean_target_label: str | None = None
    item_marker_occurrence: str = "first"


def _strip_trailing_eos(tokenizer, text: str) -> str:
    """Strip the chat template's trailing turn-end token, if present.

    Uses ``tokenizer.eos_token`` — this is the token HF's chat templates use to
    close the final message in the conversation for essentially every
    chat-tuned model, so stripping it generalizes without a per-architecture
    special case. Falls back to leaving the text untouched if the tokenizer
    has no ``eos_token`` (rare; matches the ``hasattr`` fallback
    pattern in `vector_extraction.py::format_prompt`).

    Unlike a plain ``endswith`` check, this also tolerates
    trailing whitespace after the EOS token (observed on Qwen3's template,
    which emits ``...<assistant content><|im_end|>\n`` — a bare
    ``endswith(eos)`` check would silently fail to strip it and leave the
    literal EOS token sitting right before the answer position).
    """
    eos = getattr(tokenizer, "eos_token", None)
    if eos:
        stripped = text.rstrip()
        if stripped.endswith(eos):
            text = stripped[: -len(eos)]
    if not text.endswith(" "):
        text += " "
    return text



def _token_localization_turns(items: list[str], preamble: str, suffix: str) -> list[dict]:
    item_list = " ".join(f"TOKEN {idx}: {item}{suffix}" for idx, item in enumerate(items, start=0))
    task = [
        {
            "role": "user",
            "content": f"{item_list}\n\nWhich token is the injection located in?",
        },
        {"role": "assistant", "content": "It is located in TOKEN "},
    ]
    if preamble == "none":
        return task
    intro_text = (
        "You are participating in an interpretability experiment. The researcher can inject neural activity "
        "patterns corresponding to concepts into your internal state. You will see a list of tokens and should "
        "report which token contains the injected thought."
    )
    if preamble == "user":
        return [{"role": "user", "content": intro_text}, {"role": "assistant", "content": "Ok."}, *task]
    if preamble == "system":
        return [{"role": "system", "content": intro_text}, *task]
    raise ValueError(f"Unsupported preamble: {preamble!r}")


DIGIT_POSITION_LABELS = tuple(str(index) for index in range(10))
LETTER_POSITION_LABELS = tuple("ABCDEFGHIJ")
NUMBER_WORD_POSITION_LABELS = (
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
)
NONE_ANSWER_LABEL = "none"

_LABELED_DISRUPTS_SYSTEM_PREFIX = (
    "You are participating in an interpretability experiment. A researcher "
    "may inject a neural activity pattern corresponding to a concept into "
    "your internal state. You will see ten candidate tokens. Identify the "
    "TOKEN position containing an externally added concept vector. The "
    "vector disrupts the hidden state at that TOKEN position, so output the "
    "affected position. Output `none` when no position was modified. Whenever "
    "a candidate-specific external change is present, output the affected "
    "position. Clean and injected trials are both possible. Use `none` when no "
    "actual external modification is detected.\n\n"
)
SEMANTIC_HIGHINJ_POSREF_GATE_BALANCED_DISRUPTS_QUESTION = "Output the final decision:"


def _build_labeled_disrupts_system_prompt(labels: tuple[str, ...]) -> str:
    """Build the strict response contract for one position-label scheme."""
    choices = ", ".join(f"`{label}`" for label in labels)
    return (
        f"{_LABELED_DISRUPTS_SYSTEM_PREFIX}"
        f"Your response must be exactly one of: {choices}, or `none`. "
        "Do not output anything else."
    )


SEMANTIC_HIGHINJ_POSREF_GATE_BALANCED_DISRUPTS_SYSTEM_PROMPT = (
    _build_labeled_disrupts_system_prompt(DIGIT_POSITION_LABELS)
)
SEMANTIC_HIGHINJ_POSREF_GATE_BALANCED_DISRUPTS_LETTERS_SYSTEM_PROMPT = (
    _build_labeled_disrupts_system_prompt(LETTER_POSITION_LABELS)
)
SEMANTIC_HIGHINJ_POSREF_GATE_BALANCED_DISRUPTS_NUMWORDS_SYSTEM_PROMPT = (
    _build_labeled_disrupts_system_prompt(NUMBER_WORD_POSITION_LABELS)
)


def _labeled_disrupts_turns(
    items: list[str],
    preamble: str,
    suffix: str,
    *,
    labels: tuple[str, ...],
    template_name: str,
    system_prompt: str,
) -> list[dict]:
    """Render one position-label variant of the evaluation prompt."""
    if preamble != "system":
        raise ValueError(f"{template_name} requires preamble='system'")
    if len(items) != len(labels):
        raise ValueError(
            f"{template_name} requires exactly {len(labels)} items, got {len(items)}"
        )
    item_list = " ".join(
        f"TOKEN {label}: {item}{suffix}"
        for label, item in zip(labels, items)
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": "Understood."},
        {
            "role": "user",
            "content": (
                f"{item_list}\n\n"
                f"{SEMANTIC_HIGHINJ_POSREF_GATE_BALANCED_DISRUPTS_QUESTION}"
            ),
        },
    ]


def _registered_disrupts_turns(
    name: str, labels: tuple[str, ...], system_prompt: str
) -> Callable[[list[str], str, str], list[dict]]:
    def turns(items: list[str], preamble: str, suffix: str) -> list[dict]:
        return _labeled_disrupts_turns(
            items,
            preamble,
            suffix,
            labels=labels,
            template_name=name,
            system_prompt=system_prompt,
        )

    return turns


def build_labeled_disrupts_template(
    display_labels: tuple[str, ...],
    *,
    name: str,
    canonical_labels: tuple[str, ...] = LETTER_POSITION_LABELS,
    system_prompt: str = (
        SEMANTIC_HIGHINJ_POSREF_GATE_BALANCED_DISRUPTS_LETTERS_SYSTEM_PROMPT
    ),
) -> PromptTemplate:
    """Build a disruption template whose displayed labels follow ``display_labels``.

    ``display_labels`` is the label printed at each *display slot*, in slot
    order. Passing the canonical tuple reproduces the registered ascending
    prompt; passing a permutation of it decouples the label token from the
    ordinal slot, which is the point of the shuffled-label control. The system
    prompt and the scored candidate set stay canonical either way, so the two
    conditions differ only in which letter sits next to which candidate token.
    """
    if sorted(display_labels) != sorted(canonical_labels):
        raise ValueError(
            f"{name}: display_labels must be a permutation of "
            f"{canonical_labels}, got {display_labels}"
        )
    # `canonical_labels` and `system_prompt` default independently, so a caller
    # who overrides one and not the other would silently build a prompt whose
    # response contract names a different label set than the candidate list.
    absent = [label for label in canonical_labels if f"`{label}`" not in system_prompt]
    if absent:
        raise ValueError(
            f"{name}: system_prompt never offers {absent} as answers, so it "
            f"does not match the {canonical_labels} label set"
        )

    def turns(items: list[str], preamble: str, suffix: str) -> list[dict]:
        return _labeled_disrupts_turns(
            items,
            preamble,
            suffix,
            labels=display_labels,
            template_name=name,
            system_prompt=system_prompt,
        )

    return PromptTemplate(
        name=name,
        turns=turns,
        item_marker="TOKEN {i}: {item}",
        # Scored answers stay in canonical order so every cluster shares one
        # candidate layout regardless of its display permutation.
        candidate_labels=lambda n: [*canonical_labels[:n], NONE_ANSWER_LABEL],
        single_token_items=True,
        strict_candidate_labels=True,
        item_labels=display_labels,
        add_generation_prompt=True,
        disable_thinking=True,
        clean_target_label=NONE_ANSWER_LABEL,
    )


DISRUPTS_TEMPLATE = "semantic_highinj_posref_gate_balanced_disrupts"
DISRUPTS_LETTERS_TEMPLATE = f"{DISRUPTS_TEMPLATE}_letters_a_j"
DISRUPTS_NUMWORDS_TEMPLATE = f"{DISRUPTS_TEMPLATE}_numwords_one_ten"


def _disrupts_template(
    name: str,
    labels: tuple[str, ...],
    system_prompt: str,
    *,
    item_labels: tuple[str, ...] | None,
) -> PromptTemplate:
    return PromptTemplate(
        name=name,
        turns=_registered_disrupts_turns(name, labels, system_prompt),
        item_marker="TOKEN {i}: {item}",
        candidate_labels=lambda n: [*labels[:n], NONE_ANSWER_LABEL],
        single_token_items=True,
        strict_candidate_labels=True,
        item_labels=item_labels,
        add_generation_prompt=True,
        disable_thinking=True,
        clean_target_label=NONE_ANSWER_LABEL,
    )


REGISTRY: dict[str, PromptTemplate] = {
    "token_localization": PromptTemplate(
        name="token_localization",
        turns=_token_localization_turns,
        item_marker="TOKEN {i}: {item}",
        candidate_labels=lambda n: [str(i) for i in range(0, n)],
        single_token_items=True,
    ),
    # Digit slots are numbered by position, so they carry no item_labels.
    DISRUPTS_TEMPLATE: _disrupts_template(
        DISRUPTS_TEMPLATE,
        DIGIT_POSITION_LABELS,
        SEMANTIC_HIGHINJ_POSREF_GATE_BALANCED_DISRUPTS_SYSTEM_PROMPT,
        item_labels=None,
    ),
    DISRUPTS_LETTERS_TEMPLATE: _disrupts_template(
        DISRUPTS_LETTERS_TEMPLATE,
        LETTER_POSITION_LABELS,
        SEMANTIC_HIGHINJ_POSREF_GATE_BALANCED_DISRUPTS_LETTERS_SYSTEM_PROMPT,
        item_labels=LETTER_POSITION_LABELS,
    ),
    DISRUPTS_NUMWORDS_TEMPLATE: _disrupts_template(
        DISRUPTS_NUMWORDS_TEMPLATE,
        NUMBER_WORD_POSITION_LABELS,
        SEMANTIC_HIGHINJ_POSREF_GATE_BALANCED_DISRUPTS_NUMWORDS_SYSTEM_PROMPT,
        item_labels=NUMBER_WORD_POSITION_LABELS,
    ),
}


def template_slot_labels(template: PromptTemplate, n_choices: int) -> tuple[str, ...]:
    """Return the label the ascending template prints beside each slot.

    Letter and number-word templates carry these explicitly in ``item_labels``;
    the digit templates number their slots instead, so fall back to the scored
    candidate set minus the trailing ``none``.
    """
    if template.item_labels is not None:
        if len(template.item_labels) != n_choices:
            raise ValueError(
                f"template {template.name!r} labels {len(template.item_labels)} "
                f"slots but the cluster bank has {n_choices} choices"
            )
        return tuple(str(label) for label in template.item_labels)
    if template.candidate_labels is None:
        raise ValueError(
            f"template {template.name!r} defines neither item labels nor a "
            f"candidate set, so its slot labels cannot be recovered"
        )
    labels = [str(label) for label in template.candidate_labels(n_choices)]
    if len(labels) != n_choices + 1 or labels[-1] != NONE_ANSWER_LABEL:
        raise ValueError(
            f"template {template.name!r} does not expose {n_choices} slot "
            f"labels followed by `{NONE_ANSWER_LABEL}`; got {labels}"
        )
    return tuple(labels[:-1])


def template_system_prompt(
    template: PromptTemplate, preamble: str, n_choices: int
) -> str:
    """Read the system turn straight off a registered template.

    A shuffled-label arm must keep the wording byte-identical to the ascending
    arm it controls, so it reuses the registered prompt rather than rebuilding
    one: the only thing the control may change is which label sits beside which
    candidate.
    """
    messages = template.turns(["x"] * n_choices, preamble, "")
    if not messages or messages[0].get("role") != "system":
        raise ValueError(
            f"template {template.name!r} does not open on a system turn"
        )
    return str(messages[0]["content"])


class PromptManager:
    """Owns chat-templating for the registry above and returns explicit
    Spans instead of leaving callers to re-locate markers with ``str.find``."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def render(
        self,
        template_name: str,
        items: list[str],
        *,
        preamble: str = "none",
        suffix: str = "",
        template: PromptTemplate | None = None,
    ) -> RenderedPrompt:
        """Render ``template_name`` over ``items`` and resolve each item's
        token span.

        Args:
            template_name: key into :data:`REGISTRY`, used for error
                messages when ``template`` is supplied directly.
            items: the per-slot text (sentences, single-token words, ...).
            preamble: "none" | "user" | "system" — which framing turns to
                prepend, if any (three preamble modes).
            suffix: appended to each item's marker text (used by the
                "choice_suffix" sweeps, e.g. to add a trailing
                token boundary probe).
            template: an explicit template object that bypasses the registry
                lookup. Used by controls that need one template instance per
                example (e.g. a per-cluster label permutation) rather than a
                single shared registry entry.

        Returns:
            RenderedPrompt with one Span per item (in ``items`` order, or a
            single Span for the fixed injection target if the template
            uses ``injection_marker``) and ``answer_token_by_choice``
            mapping each candidate label -> token id (empty if the
            template defines neither ``candidate_labels`` nor
            ``fixed_candidate_labels``).
        """
        if template is None:
            template = REGISTRY[template_name]
        if template.item_marker_occurrence not in {"first", "last"}:
            raise ValueError(
                f"Unsupported item marker occurrence: "
                f"{template.item_marker_occurrence!r}"
            )
        messages = template.turns(items, preamble, suffix)
        if template.add_generation_prompt:
            chat_template_kwargs = {}
            if template.disable_thinking:
                chat_template_kwargs["enable_thinking"] = False
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **chat_template_kwargs,
                )
            except Exception as error:
                # Gemma chat templates reject system -> assistant -> user,
                # even though Llama and Qwen accept it. Preserve the exact
                # prompt content and acknowledgement while mapping only the
                # unsupported leading system role to a user role.
                role_alternation_error = (
                    "Conversation roles must alternate" in str(error)
                )
                can_map_leading_system = (
                    len(messages) >= 3
                    and [message["role"] for message in messages[:3]]
                    == ["system", "assistant", "user"]
                )
                if not (role_alternation_error and can_map_leading_system):
                    raise
                compatible_messages = [
                    {"role": "user", "content": messages[0]["content"]},
                    *messages[1:],
                ]
                text = self.tokenizer.apply_chat_template(
                    compatible_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **chat_template_kwargs,
                )
        else:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                continue_final_message=True,
            )
            text = _strip_trailing_eos(self.tokenizer, text)

        encoding = self.tokenizer(
            text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True
        )
        offset_mapping = encoding["offset_mapping"][0].tolist()
        input_ids = encoding["input_ids"][0]

        spans: list[Span] = []
        records: list[dict] = []

        if template.injection_marker is not None:
            if items:
                raise ValueError(
                    f"Template {template_name!r} uses a fixed injection_marker; "
                    "pass items=[] (its content is baked into turns())"
                )
            marker = template.injection_marker
            marker_start = text.find(marker)
            if marker_start < 0:
                raise ValueError(f"Could not locate injection marker {marker!r} in rendered prompt")
            marker_end = marker_start + len(marker)
            span = resolve_text_span_to_tokens(
                text,
                offset_mapping,
                input_ids,
                self.tokenizer,
                marker_start,
                marker_end,
                single_token=False,
            )
            spans.append(span)
            records.append(
                {
                    "choice": None,
                    "text": marker,
                    "token_index": span.start,
                    "token_index_end": span.end,
                    "token_ids": span.token_ids,
                    "token_text": span.text,
                }
            )
        elif template.item_marker is not None:
            if (
                template.item_labels is not None
                and len(template.item_labels) != len(items)
            ):
                raise ValueError(
                    f"Template {template_name!r} defines "
                    f"{len(template.item_labels)} item labels for "
                    f"{len(items)} items"
                )
            search_start = 0
            for item_offset, item in enumerate(items):
                item_label = (
                    template.item_labels[item_offset]
                    if template.item_labels is not None
                    else item_offset + template.item_index_start
                )
                marker = template.item_marker.format(i=item_label, item=item)
                marker_start = (
                    text.rfind(marker)
                    if template.item_marker_occurrence == "last"
                    else text.find(marker, search_start)
                )
                if marker_start < 0:
                    raise ValueError(f"Could not locate marker {marker!r} in rendered prompt")
                item_start = marker_start + len(marker) - len(item)
                item_end = item_start + len(item)
                search_start = item_end

                span = resolve_text_span_to_tokens(
                    text,
                    offset_mapping,
                    input_ids,
                    self.tokenizer,
                    item_start,
                    item_end,
                    single_token=template.single_token_items,
                )
                spans.append(span)
                records.append(
                    {
                        "choice": item_label,
                        "text": item,
                        "token_index": span.start,
                        "token_index_end": span.end,
                        "token_ids": span.token_ids,
                        "token_text": span.text,
                    }
                )
        else:
            raise ValueError(f"Template {template_name!r} defines neither item_marker nor injection_marker")

        if template.fixed_candidate_labels is not None:
            labels = list(template.fixed_candidate_labels)
        elif template.candidate_labels is not None:
            labels = template.candidate_labels(len(items))
        else:
            labels = None

        answer_token_by_choice: dict[str, int] = {}
        if labels is not None:
            answer_token_by_choice = answer_token_ids(
                self.tokenizer, labels, strict=template.strict_candidate_labels
            )

        return RenderedPrompt(
            input_ids=encoding["input_ids"],
            spans=spans,
            answer_token_by_choice=answer_token_by_choice,
            records=records,
            text=text,
            clean_target_label=template.clean_target_label,
        )
