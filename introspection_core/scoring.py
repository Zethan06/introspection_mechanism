"""First-output-token scoring.

Every response is read from ``logits[:, -1, :]`` of a single forward pass,
never from ``.generate()``: the highest-scoring answer among the allowed labels
is the model's response. The functions here are pure tensor math on that
last-position logit vector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def answer_token_ids(tokenizer, labels: list[str], *, strict: bool = False) -> dict[str, int]:
    """Map each answer label (e.g. digits "1".."5", or rating words) to the
    single token id used to score it as a "first generated token" answer.

    Uses the last sub-token id if a label tokenizes to more than one token
    (matches the source convention: ``tokenizer.encode(label,
    add_special_tokens=False)[-1]``) — this is deliberate, since the
    assistant-prefill answer format places these labels immediately after
    a prefix like "It is located in TOKEN ", so what matters is the token
    that would be *generated first* in that specific context, which is the
    label's final sub-token when the prefix has already consumed everything
    up to it.

    Args:
        strict: if True, raise ``ValueError`` unless every label tokenizes
            to exactly one token, instead of silently falling back to the
            last sub-token. Required whenever the experiment's scoring
            assumption depends on an exact single-position answer token
            (e.g. a fixed rating scale like thought-strength detection's
            "1".."5") — a multi-token label there would mean the "first
            generated token" isn't actually the full answer.
    """
    ids: dict[str, int] = {}
    bad: list[tuple[str, list[str]]] = []
    for label in labels:
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        if strict and len(token_ids) != 1:
            bad.append((label, [tokenizer.decode([token_id]) for token_id in token_ids]))
            continue
        ids[label] = token_ids[-1]
    if bad:
        details = "; ".join(f"{label!r}->{decoded}" for label, decoded in bad)
        raise ValueError(f"Labels must each be exactly one token (strict=True): {details}")
    return ids


@dataclass
class FirstTokenScore:
    """Score for one example's first-generated-token distribution.

    Attributes:
        argmax_choice: the candidate label with the highest logit among
            the restricted candidate set (not full vocab).
        candidate_probs: dict label -> probability, softmax renormalized
            over *only* the candidate set (sums to 1 across labels).
        full_vocab_probs: dict label -> probability, taken from a
            full-vocabulary softmax (does not sum to 1 across labels,
            since other vocab tokens absorb probability mass — this is
            the number to report when you care about how much the model
            "wanted" that token relative to everything it could have said).
        correct_prob: full_vocab_probs[expected] if ``expected`` was given,
            else None.
        entropy_bits: Shannon entropy, in bits, of candidate_probs — a
            measure of how concentrated vs. diffuse the answer is.
    """

    argmax_choice: str
    candidate_probs: dict[str, float]
    full_vocab_probs: dict[str, float]
    correct_prob: float | None
    entropy_bits: float


def score_first_token(
    logits_last: torch.Tensor,
    candidate_ids: dict[str, int],
    expected: str | None = None,
) -> FirstTokenScore:
    """Score one example's last-position logits against a closed candidate set.

    Args:
        logits_last: 1D tensor [vocab] — ``logits[i, -1, :]`` for one
            example (call once per row if scoring a batch; this function
            is intentionally single-example so callers can attach
            per-example metadata without unzipping a batched result).
        candidate_ids: label -> token id, as produced by
            :func:`answer_token_ids`.
        expected: the label considered "correct" for this example, if
            applicable (omit for scoring runs with no ground truth, e.g.
            pure distribution/entropy analyses).

    Returns:
        FirstTokenScore.
    """
    if logits_last.dim() != 1:
        raise ValueError(f"logits_last must be 1D [vocab], got shape {tuple(logits_last.shape)}")
    labels = list(candidate_ids.keys())
    ids = torch.tensor([candidate_ids[label] for label in labels], device=logits_last.device)

    full_log_probs = torch.log_softmax(logits_last.float(), dim=-1)
    full_probs_by_label = {
        label: float(torch.exp(full_log_probs[token_id])) for label, token_id in candidate_ids.items()
    }

    candidate_logits = logits_last.float().index_select(0, ids)
    candidate_probs_tensor = torch.softmax(candidate_logits, dim=-1)
    candidate_probs_by_label = {label: float(p) for label, p in zip(labels, candidate_probs_tensor)}

    argmax_idx = int(torch.argmax(candidate_logits).item())
    argmax_choice = labels[argmax_idx]

    correct_prob = full_probs_by_label[expected] if expected is not None else None

    nonzero = candidate_probs_tensor[candidate_probs_tensor > 0]
    entropy_nats = float(-(nonzero * torch.log(nonzero)).sum())
    entropy_bits = entropy_nats / math.log(2)

    return FirstTokenScore(
        argmax_choice=argmax_choice,
        candidate_probs=candidate_probs_by_label,
        full_vocab_probs=full_probs_by_label,
        correct_prob=correct_prob,
        entropy_bits=entropy_bits,
    )


def aggregate(scores: list[FirstTokenScore], expected: list[str] | None = None) -> dict:
    """Aggregate a batch of FirstTokenScore into summary statistics.

    Args:
        scores: one FirstTokenScore per example.
        expected: one expected label per example, parallel to ``scores``;
            if omitted, accuracy-related keys are left out of the result.

    Returns:
        dict with keys: ``n``, ``mean_entropy_bits``, and, if ``expected``
        is given, ``argmax_accuracy`` (fraction where argmax_choice ==
        expected) and ``mean_correct_prob`` (mean of correct_prob across
        examples).
    """
    n = len(scores)
    result: dict = {
        "n": n,
        "mean_entropy_bits": sum(s.entropy_bits for s in scores) / n if n else float("nan"),
    }
    if expected is not None:
        if len(expected) != n:
            raise ValueError(f"expected has {len(expected)} entries, scores has {n}")
        correct = sum(1 for s, e in zip(scores, expected) if s.argmax_choice == e)
        result["argmax_accuracy"] = correct / n if n else float("nan")
        correct_probs = [s.full_vocab_probs[e] for s, e in zip(scores, expected)]
        result["mean_correct_prob"] = sum(correct_probs) / n if n else float("nan")
    return result
