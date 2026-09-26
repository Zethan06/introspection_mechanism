#!/usr/bin/env python3
"""Lexical replacement control: localization without any activation injection.

Each candidate word is replaced in turn by the matching concept word or by a
seeded random word from the model's English-token vocabulary. The prompt is
re-rendered and re-tokenized, and the constrained argmax over the canonical
labels plus ``none`` is compared with the label displayed at the replaced slot.
No hidden state is modified and no head is patched.

Three subcommands:

  plan       Fix the trial list for one model: validation concepts x test
             clusters x ten slots x {concept_word, random_word}. The random word
             is drawn uniformly from the unique vocabulary words, excluding the
             cluster's choices and the concept; a concept already equal to the
             original word is skipped together with its paired random trial.
  run        Score one label arm (digits/letters/words x identity/shuffled) of
             a plan. Shuffled arms draw one derangement per cluster (seed 42),
             exactly as the shuffled-label accuracy runs do. Split a plan over
             workers with --worker/--workers.
  summarize  Pool every arm under a results root into summary.csv/.json.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import replace
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from introspection_core.attention_inputs import _derangement
from introspection_core.model import HookedModel, ModelConfig
from introspection_core.prompts import (
    REGISTRY,
    PromptManager,
    build_labeled_disrupts_template,
    template_slot_labels,
    template_system_prompt,
)

TEMPLATES = {
    "digits": "semantic_highinj_posref_gate_balanced_disrupts",
    "letters": "semantic_highinj_posref_gate_balanced_disrupts_letters_a_j",
    "words": "semantic_highinj_posref_gate_balanced_disrupts_numwords_one_ten",
}
ARMS = tuple(
    f"{label_set}_{permutation}"
    for permutation in ("identity", "shuffled")
    for label_set in TEMPLATES
)
CONDITIONS = ("concept_word", "random_word")
PERMUTATION_SEED = 42


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_plan(args: argparse.Namespace) -> None:
    concepts = json.loads((args.dataset_dir / "concepts/validation.json").read_text())[
        "concept_vector_words"
    ]
    with (args.dataset_dir / "clusters/test.csv").open(newline="") as handle:
        clusters = list(csv.DictReader(handle))
    with (args.dataset_dir / "vocabulary/english.csv").open(newline="") as handle:
        words = sorted({row["word"].strip() for row in csv.DictReader(handle) if row["word"].strip()})
    rng = random.Random(args.seed)
    trials, skipped = [], []
    for concept_id, concept in enumerate(concepts):
        for cluster_id, cluster in enumerate(clusters):
            choices = json.loads(cluster["choices"])
            for position in range(len(choices)):
                if concept == choices[position]:
                    skipped.append([concept_id, cluster_id, position])
                    continue
                word = rng.choice(words)
                while word in choices or word == concept:
                    word = rng.choice(words)
                for condition, replacement in (("concept_word", concept), ("random_word", word)):
                    trials.append(
                        dict(
                            trial_id=len(trials),
                            condition=condition,
                            concept_id=concept_id,
                            concept_word=concept,
                            cluster_id=cluster_id,
                            cluster_key=cluster["cluster_key"],
                            position=position,
                            choices=choices,
                            original_word=choices[position],
                            replacement_word=replacement,
                        )
                    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "plan.json").write_text(json.dumps(trials))
    inputs = [
        args.dataset_dir / "concepts/validation.json",
        args.dataset_dir / "clusters/test.csv",
        args.dataset_dir / "vocabulary/english.csv",
    ]
    (args.output_dir / "protocol.json").write_text(
        json.dumps(
            dict(
                concepts=concepts,
                n_clusters=len(clusters),
                seed=args.seed,
                random_pool="uniform unique words from vocabulary/english.csv; "
                "exclude all current choices and the matched concept",
                random_pool_size=len(words),
                activation_injection=False,
                head_patch=False,
                trials=len(trials),
                skipped_unchanged=skipped,
                hashes={path.name: _sha256(path) for path in inputs},
            ),
            indent=2,
        )
    )
    print(f"{args.output_dir / 'plan.json'}: {len(trials)} trials, {len(skipped)} skipped")


@lru_cache(None)
def arm_layout(arm: str, cluster_id: int):
    """Return the arm's template and the label displayed at each slot."""
    label_set, permutation = arm.split("_")
    name = TEMPLATES[label_set]
    base = REGISTRY[name]
    canonical = template_slot_labels(base, 10)
    display = canonical
    if permutation == "shuffled":
        rng = random.Random(PERMUTATION_SEED)
        for index in range(cluster_id + 1):
            display = _derangement(canonical, rng, key=str(index))
    template = build_labeled_disrupts_template(
        display,
        name=name,
        canonical_labels=canonical,
        system_prompt=template_system_prompt(base, "system", 10),
    )
    # A replacement word may span several tokens.
    return replace(template, single_token_items=False), display


def render_trial(manager: PromptManager, row: dict):
    choices = list(row["choices"])
    position = row["position"]
    word = row["replacement_word"].strip()
    if not word or word == choices[position]:
        raise ValueError(f"trial {row['trial_id']} does not change the prompt")
    choices[position] = word
    template, display = arm_layout(row["arm"], row["cluster_id"])
    rendered = manager.render(template.name, choices, preamble="system", template=template)
    # Exactly one space follows the label colon, and the full word survives.
    marker = f"TOKEN {display[position]}: "
    start = rendered.text.index(marker) + len(marker)
    if rendered.text[start : start + len(word)] != word:
        raise ValueError(f"trial {row['trial_id']}: replacement not rendered verbatim")
    if rendered.spans[position].end <= rendered.spans[position].start:
        raise ValueError(f"trial {row['trial_id']}: empty replacement span")
    return rendered


@torch.inference_mode()
def run_arm(args: argparse.Namespace) -> None:
    rows = json.loads(args.plan.read_text())[args.worker :: args.workers]
    if not rows:
        raise ValueError("this worker has no trials")
    for row in rows:
        row["arm"] = args.arm
        _, display = arm_layout(args.arm, row["cluster_id"])
        row["display_labels"] = list(display)
        row["expected_label"] = display[row["position"]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = HookedModel(ModelConfig(name=args.model, device=args.device, dtype=args.dtype))
    manager = PromptManager(model.tokenizer)

    # The candidate-only readout must match an ordinary full forward. On CUDA the
    # two agree exactly; bfloat16 on CPU can differ by one ulp (2**-6 relative).
    example = render_trial(manager, rows[0])
    candidate_ids = list(example.answer_token_by_choice.values())
    full = model.forward_logits(example.input_ids)[:, -1, :].float().cpu()
    fast, _ = model.last_token_candidate_stats(example.input_ids, candidate_token_ids=candidate_ids)
    torch.testing.assert_close(fast, full[:, candidate_ids], rtol=2**-5, atol=0)

    buckets = defaultdict(list)
    for row in rows:
        rendered = render_trial(manager, row)
        buckets[rendered.input_ids.shape[-1]].append((row, rendered))
    output = args.output_dir / f"worker_{args.worker}.jsonl"
    completed = 0
    with output.open("x") as handle:
        for group in buckets.values():
            for start in range(0, len(group), args.batch_size):
                batch = group[start : start + args.batch_size]
                layout = list(batch[0][1].answer_token_by_choice.items())
                if any(list(rendered.answer_token_by_choice.items()) != layout for _, rendered in batch):
                    raise ValueError("candidate layout differs within a batch")
                labels = [label for label, _ in layout]
                tokens = torch.cat([rendered.input_ids for _, rendered in batch], dim=0)
                logits, log_probs = model.last_token_candidate_stats(
                    tokens, candidate_token_ids=[token for _, token in layout]
                )
                predictions = logits.argmax(-1).tolist()
                for offset, (row, rendered) in enumerate(batch):
                    prediction = labels[predictions[offset]]
                    span = rendered.spans[row["position"]]
                    handle.write(
                        json.dumps(
                            dict(
                                row,
                                prediction=prediction,
                                correct=prediction == row["expected_label"],
                                correct_prob=float(
                                    log_probs[offset, labels.index(row["expected_label"])].exp()
                                ),
                                replacement_token_ids=span.token_ids,
                                replacement_token_count=span.end - span.start,
                            )
                        )
                        + "\n"
                    )
                completed += len(batch)
                print(f"{completed}/{len(rows)}", flush=True)
    (args.output_dir / f"worker_{args.worker}.complete.json").write_text(
        json.dumps(dict(trials=completed, workers=args.workers, arm=args.arm))
    )


def summarize(args: argparse.Namespace) -> None:
    """Pool <root>/<model>/<arm>/worker_*.jsonl against <root>/<model>/plan.json."""
    summary = []
    for plan_path in sorted(args.results_root.glob("*/plan.json")):
        expected = len(json.loads(plan_path.read_text()))
        for arm_dir in sorted(p for p in plan_path.parent.iterdir() if p.name in ARMS):
            counts = defaultdict(lambda: [0, 0, 0])
            seen = set()
            shards = sorted(arm_dir.glob("worker_*.jsonl"))
            for path in shards:
                with path.open() as handle:
                    for line in handle:
                        row = json.loads(line)
                        if row["trial_id"] in seen:
                            raise ValueError(f"{arm_dir}: duplicate trial {row['trial_id']}")
                        seen.add(row["trial_id"])
                        count = counts[row["condition"]]
                        count[0] += 1
                        count[1] += bool(row["correct"])
                        count[2] += row["prediction"] == "none"
            finished = all(
                path.with_name(path.name.replace(".jsonl", ".complete.json")).is_file()
                for path in shards
            )
            complete = bool(shards) and finished and len(seen) == expected
            for condition in CONDITIONS:
                n, correct, none = counts[condition]
                summary.append(
                    dict(
                        model=plan_path.parent.name,
                        arm=arm_dir.name,
                        condition=condition,
                        trials=n,
                        correct=correct,
                        accuracy=correct / n if n else None,
                        none_rate=none / n if n else None,
                        complete=complete,
                    )
                )
    if not summary:
        raise ValueError(f"no <model>/<arm> results under {args.results_root}")
    (args.results_root / "summary.json").write_text(json.dumps(summary, indent=2))
    with (args.results_root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(args.results_root / "summary.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="fix the trial list for one model")
    plan.add_argument("--dataset_dir", type=Path, required=True, help="data/dataset/<model>")
    plan.add_argument("--output_dir", type=Path, required=True, help="<root>/<model>")
    plan.add_argument("--seed", type=int, default=42)

    run = commands.add_parser("run", help="score one label arm of a plan")
    run.add_argument("--arm", choices=ARMS, required=True)
    run.add_argument("--model", required=True, help="HF id or local checkpoint")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--output_dir", type=Path, required=True, help="<root>/<model>/<arm>")
    run.add_argument("--worker", type=int, default=0)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--batch_size", type=int, default=32)
    run.add_argument("--device", default="cuda")
    run.add_argument("--dtype", default="bfloat16")

    summary = commands.add_parser("summarize", help="pool every model and arm")
    summary.add_argument("--results_root", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "run" and not 0 <= args.worker < args.workers:
        parser.error("need 0 <= --worker < --workers")
    {"plan": build_plan, "run": run_arm, "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    main()
