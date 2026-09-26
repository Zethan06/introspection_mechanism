"""Task adapters for position-averaged attention visualization.

The visualization backend consumes :class:`AttentionExample` objects and
does not know how a dataset is stored or how a task prompt is worded.  New
tasks should implement ``AttentionTaskAdapter.build_examples`` and keep their
CSV/JSON parsing and prompt-specific span resolution here (or in a sibling
module), while reusing the aggregation and browser modules unchanged.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch

from .localization import Span
from .prompts import PromptManager, build_labeled_disrupts_template


@dataclass(frozen=True)
class AttentionExample:
    """One rendered example accepted by the generic attention aggregator.

    ``positions`` are user-facing labels (normally 0..9), and
    ``injection_spans`` maps each label to its token span.  Elementwise
    attention averaging additionally requires all examples in one run to
    share the same sequence length and span layout; ``validate_aligned_layout``
    checks that invariant before any GPU work starts.
    """

    key: str
    prompt: str
    input_ids: torch.Tensor
    positions: tuple[int, ...]
    injection_spans: dict[int, Span]
    candidate_token_ids: dict[str, int]
    expected_candidate_by_position: dict[int, str]
    item_labels: dict[int, str]
    item_token_indices: tuple[int, ...]
    records: tuple[dict, ...]
    clean_target_label: str | None = None


class AttentionTaskAdapter(Protocol):
    """Minimal extension point for a new task/data-processing pipeline."""

    name: str

    def build_examples(self, prompt_manager: PromptManager) -> list[AttentionExample]:
        """Load task data, render prompts, and return aligned examples."""


def cluster_choice_count(
    cluster_csv: Path, *, choices_column: str = "choices"
) -> int:
    """Read how many candidates each row of a cluster bank carries.

    The shuffled-label control needs the arity before any prompt is rendered,
    because it has to know how many labels the permutation is drawn over.
    """
    with Path(cluster_csv).open(newline="", encoding="utf-8") as handle:
        first = next(csv.DictReader(handle), None)
    if first is None:
        raise ValueError(f"{cluster_csv} has no rows")
    raw = first.get(choices_column)
    if raw is None:
        raise ValueError(f"Missing {choices_column!r} in the first row of {cluster_csv}")
    return len(json.loads(raw))


@dataclass
class TokenLocalizationCsvTask:
    """Read one list of single-token choices per CSV row."""

    path: Path
    max_examples: int | None = None
    choices_column: str = "choices"
    key_column: str = "cluster_key"
    preamble: str = "system"
    choice_suffix: str = ""
    position_index_start: int = 0
    name: str = "token_localization"
    template_name: str | None = None

    def build_examples(self, prompt_manager: PromptManager) -> list[AttentionExample]:
        if self.position_index_start != 0:
            raise ValueError(
                "position_index_start must be 0, got "
                f"{self.position_index_start}"
            )
        with self.path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if self.max_examples is not None:
            rows = rows[: self.max_examples]

        examples: list[AttentionExample] = []
        for row_index, row in enumerate(rows):
            raw_choices = row.get(self.choices_column)
            if raw_choices is None:
                raise ValueError(
                    f"Missing {self.choices_column!r} in row {row_index} of {self.path}"
                )
            choices = json.loads(raw_choices)
            if not isinstance(choices, list) or not choices:
                raise ValueError(
                    f"Invalid choices in row {row_index} of {self.path}: {raw_choices!r}"
                )
            choices = [str(choice) for choice in choices]
            template_name = self.template_name or "token_localization"
            rendered = prompt_manager.render(
                template_name,
                choices,
                preamble=self.preamble,
                suffix=self.choice_suffix,
            )
            positions = tuple(
                range(
                    self.position_index_start,
                    self.position_index_start + len(choices),
                )
            )
            answer_ids = {
                str(label): int(token_id)
                for label, token_id in rendered.answer_token_by_choice.items()
            }
            position_labels = tuple(answer_ids)[: len(positions)]
            spans = {
                position: rendered.spans[item_offset]
                for item_offset, position in enumerate(positions)
            }
            examples.append(
                AttentionExample(
                    key=str(row.get(self.key_column) or row_index),
                    prompt=rendered.text,
                    input_ids=rendered.input_ids.detach().cpu(),
                    positions=positions,
                    injection_spans=spans,
                    candidate_token_ids=answer_ids,
                    expected_candidate_by_position={
                        position: position_labels[item_offset]
                        for item_offset, position in enumerate(positions)
                    },
                    item_labels={
                        position: choices[item_offset]
                        for item_offset, position in enumerate(positions)
                    },
                    item_token_indices=tuple(
                        int(span.start) for span in rendered.spans
                    ),
                    records=tuple(dict(record) for record in rendered.records),
                    clean_target_label=rendered.clean_target_label,
                )
            )
        if not examples:
            raise ValueError(f"No task examples loaded from {self.path}")
        validate_aligned_layout(examples)
        return examples


@dataclass
class ShuffledLabelTokenLocalizationCsvTask:
    """Token-localization task whose position labels are permuted per cluster.

    The registered label prompts print the first label at slot 0, the second at
    slot 1, and so on, whichever label set they use, so the label token is
    perfectly collinear with the ordinal slot: a model that merely counts list
    entries scores exactly like a model that reads the label next to the
    disrupted candidate. This task draws one deterministic derangement per
    cluster, so slot ``p`` carries an arbitrary label and the two strategies
    come apart on every trial.

    Items keep their CSV order and their slots — only the label printed beside
    each item moves. The scored candidate set stays canonical (the ascending
    labels, then ``none``), so every example shares one candidate layout and
    stays comparable with the ascending run trial for trial.

    ``canonical_labels`` and ``system_prompt`` must come from the registered
    ascending template being controlled, so that the arms differ in nothing but
    the permutation. Slots are always numbered from zero, matching the
    disruption prompts this control is built for.
    """

    path: Path
    canonical_labels: tuple[str, ...]
    system_prompt: str
    template_name: str
    max_examples: int | None = None
    choices_column: str = "choices"
    key_column: str = "cluster_key"
    preamble: str = "system"
    choice_suffix: str = ""
    seed: int = 42
    name: str = "shuffled_label_token_localization"

    def build_examples(self, prompt_manager: PromptManager) -> list[AttentionExample]:
        with self.path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if self.max_examples is not None:
            rows = rows[: self.max_examples]

        # Seeded locally so the permutation bank depends only on (seed, row
        # index) and not on whatever global RNG state the caller left behind.
        rng = random.Random(self.seed)
        examples: list[AttentionExample] = []
        for row_index, row in enumerate(rows):
            raw_choices = row.get(self.choices_column)
            if raw_choices is None:
                raise ValueError(
                    f"Missing {self.choices_column!r} in row {row_index} of {self.path}"
                )
            choices = [str(choice) for choice in json.loads(raw_choices)]
            if len(choices) != len(self.canonical_labels):
                raise ValueError(
                    f"Row {row_index} of {self.path} has {len(choices)} choices "
                    f"but {len(self.canonical_labels)} labels are defined"
                )
            display_labels = _derangement(
                self.canonical_labels, rng, key=f"{self.path}:{row_index}"
            )

            template = build_labeled_disrupts_template(
                display_labels,
                name=f"{self.template_name}[{row_index}]",
                canonical_labels=self.canonical_labels,
                system_prompt=self.system_prompt,
            )
            rendered = prompt_manager.render(
                self.template_name,
                choices,
                preamble=self.preamble,
                suffix=self.choice_suffix,
                template=template,
            )
            positions = tuple(range(len(choices)))
            answer_ids = {
                str(label): int(token_id)
                for label, token_id in rendered.answer_token_by_choice.items()
            }
            examples.append(
                AttentionExample(
                    key=str(row.get(self.key_column) or row_index),
                    prompt=rendered.text,
                    input_ids=rendered.input_ids.detach().cpu(),
                    positions=positions,
                    injection_spans={
                        position: rendered.spans[position] for position in positions
                    },
                    candidate_token_ids=answer_ids,
                    # The whole point of the control: slot p is answered by
                    # display_labels[p], not by canonical_labels[p].
                    expected_candidate_by_position={
                        position: display_labels[position] for position in positions
                    },
                    item_labels={
                        position: choices[position] for position in positions
                    },
                    item_token_indices=tuple(
                        int(span.start) for span in rendered.spans
                    ),
                    records=tuple(dict(record) for record in rendered.records),
                    clean_target_label=rendered.clean_target_label,
                )
            )
        if not examples:
            raise ValueError(f"No task examples loaded from {self.path}")
        validate_aligned_layout(examples, shared_answer_key=False)
        return examples


def _derangement(
    labels: tuple[str, ...], rng: random.Random, *, key: str
) -> tuple[str, ...]:
    """Permute ``labels`` so no label keeps its canonical slot.

    A plain shuffle leaves roughly one slot per cluster at its ascending label
    by chance, and those slots are exactly the ones a counting strategy still
    gets right. Rejecting fixed points removes that residual so every scored
    trial discriminates between reading and counting.
    """
    if len(labels) < 2:
        raise ValueError(f"{key}: need at least two labels to derange")
    for _ in range(1000):
        candidate = list(labels)
        rng.shuffle(candidate)
        if all(new != old for new, old in zip(candidate, labels, strict=True)):
            return tuple(candidate)
    raise RuntimeError(f"{key}: could not draw a derangement")


def validate_aligned_layout(
    examples: list[AttentionExample], *, shared_answer_key: bool = True
) -> None:
    """Require the positional alignment needed for elementwise attention means.

    ``shared_answer_key=False`` exempts the position -> expected-label map,
    which the shuffled-label control varies per cluster on purpose. Everything
    that elementwise averaging actually depends on — sequence length, span
    offsets, item token indices, the candidate set — is still required to match.
    """
    if not examples:
        raise ValueError("examples must be non-empty")

    def signature(example: AttentionExample):
        return (
            int(example.input_ids.shape[-1]),
            example.positions,
            tuple(
                (
                    position,
                    int(example.injection_spans[position].start),
                    int(example.injection_spans[position].end),
                )
                for position in example.positions
            ),
            example.item_token_indices,
            tuple(example.candidate_token_ids),
            tuple(
                example.expected_candidate_by_position[position]
                for position in example.positions
            )
            if shared_answer_key
            else None,
            example.clean_target_label,
        )

    expected = signature(examples[0])
    for example in examples[1:]:
        actual = signature(example)
        if actual != expected:
            raise ValueError(
                "Attention examples are not position-aligned. Elementwise "
                "attention averaging requires identical sequence length, "
                "injection-span positions, item-token positions, and candidate "
                f"labels. expected={expected}, key={example.key!r}, got={actual}"
            )


def build_token_rows(tokenizer, example: AttentionExample) -> list[dict]:
    """Build the browser's token metadata from a representative example."""
    encoding = tokenizer(
        example.prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoding["offset_mapping"]
    input_ids = example.input_ids[0].tolist()
    choice_by_token = {
        int(record["token_index"]): record for record in example.records
    }
    rows = []
    for index, (token_id, (char_start, char_end)) in enumerate(
        zip(input_ids, offsets)
    ):
        record = choice_by_token.get(index)
        text = tokenizer.decode(
            [int(token_id)], clean_up_tokenization_spaces=False
        )
        rows.append(
            {
                "idx": int(index),
                "id": int(token_id),
                "text": text,
                "repr": repr(text),
                "slice": example.prompt[int(char_start) : int(char_end)],
                "char_start": int(char_start),
                "char_end": int(char_end),
                "choice": None if record is None else int(record["choice"]),
                "choice_text": "" if record is None else str(record["text"]),
            }
        )
    return rows
