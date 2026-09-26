"""Generate clean token-localization priors for balanced cluster search.

Each eligible single-token word is evaluated once in every answer position.
The resulting ``summary_by_token.csv`` is the model-specific candidate input
consumed by :mod:`introspection_core.balanced_cluster_search`.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Sequence

import torch

from .data_parallel import (
    WorkerSpec,
    build_worker_env,
    current_python,
    parse_gpu_groups,
    run_worker_pool,
    shard_items,
)
from .model import HookedModel, ModelConfig
from .prompts import PromptManager


REPO_ROOT = Path(__file__).resolve().parents[1]
WORD_RE = re.compile(r"^[A-Za-z]+$")
WORD_START_MARKERS = ("▁", "Ġ")

SCORE_FIELDS = [
    "prompt_id",
    "group_idx",
    "rotation",
    "position",
    "token_id",
    "token_text",
    "word",
    "word_lower",
    "word_len",
    "number_logit",
    "number_softmax",
    "full_vocab_prob",
    "clean_prediction",
    "is_argmax",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score clean token-localization priors and write "
            "summary_by_token.csv."
        )
    )
    parser.add_argument("--role", choices=["runner", "worker"], default="runner")
    parser.add_argument("--model", required=True, help="HF model ID or checkpoint")
    parser.add_argument(
        "--results_dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--work_dir", type=Path)
    parser.add_argument("--logs_dir", type=Path)
    parser.add_argument(
        "--gpus",
        default="0",
        help=(
            "Comma-separated one-GPU workers, or semicolon-separated "
            "multi-GPU workers such as '2,3;5,6'."
        ),
    )
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--num_choices", type=int, default=10)
    parser.add_argument(
        "--position_index_start",
        type=int,
        choices=[0],
        default=0,
        help="First TOKEN label and answer number.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--min_word_len", type=int, default=1)
    parser.add_argument("--max_word_len", type=int, default=32)
    parser.add_argument("--max_tokens", type=int)
    parser.add_argument("--word_list", type=Path)
    parser.add_argument("--exclude_word_list", type=Path)
    parser.add_argument(
        "--case_filter",
        choices=["all", "lower", "upper_initial"],
        default="all",
    )
    parser.add_argument(
        "--prompt_preamble",
        choices=["none", "user", "system"],
        default="system",
    )
    parser.add_argument("--progress_every", type=int, default=100)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args(argv)
    validate_args(args, parser)
    return args


def validate_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser | None = None,
) -> None:
    def reject(message: str) -> None:
        if parser is not None:
            parser.error(message)
        raise ValueError(message)

    for name in ("num_workers", "num_choices", "batch_size"):
        if int(getattr(args, name)) < 1:
            reject(f"--{name} must be positive")
    if args.worker_id < 0 or args.worker_id >= args.num_workers:
        reject("--worker_id must be in [0, num_workers)")
    if args.min_word_len < 1:
        reject("--min_word_len must be positive")
    if args.max_word_len < args.min_word_len:
        reject("--max_word_len must be at least --min_word_len")
    if args.max_tokens is not None and args.max_tokens < args.num_choices:
        reject("--max_tokens must be at least --num_choices")
    if not hasattr(torch, args.dtype):
        reject(f"Unknown torch dtype: {args.dtype!r}")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_gpus(gpus: str) -> list[str]:
    if gpus == "auto":
        return [str(index) for index in range(torch.cuda.device_count())] or ["cpu"]
    return parse_gpu_groups(gpus)


def workflow_work_dir(args: argparse.Namespace) -> Path:
    if args.work_dir is not None:
        return resolve_path(args.work_dir)
    results_dir = resolve_path(args.results_dir)
    return REPO_ROOT / "tmp" / "auto_workflow" / results_dir.parent.name / "token_prior"


def workflow_log_dir(args: argparse.Namespace) -> Path:
    if args.logs_dir is not None:
        return resolve_path(args.logs_dir)
    results_dir = resolve_path(args.results_dir)
    return REPO_ROOT / "logs" / "auto_workflow" / results_dir.parent.name / "token_prior"


def _common_worker_args(args: argparse.Namespace, results_dir: Path) -> list[str]:
    values = [
        "--model",
        args.model,
        "--results_dir",
        str(results_dir),
        "--work_dir",
        str(workflow_work_dir(args)),
        "--logs_dir",
        str(workflow_log_dir(args)),
        "--num_choices",
        str(args.num_choices),
        "--position_index_start",
        str(args.position_index_start),
        "--seed",
        str(args.seed),
        "--batch_size",
        str(args.batch_size),
        "--min_word_len",
        str(args.min_word_len),
        "--max_word_len",
        str(args.max_word_len),
        "--case_filter",
        args.case_filter,
        "--prompt_preamble",
        args.prompt_preamble,
        "--progress_every",
        str(args.progress_every),
        "--dtype",
        args.dtype,
    ]
    if args.max_tokens is not None:
        values.extend(["--max_tokens", str(args.max_tokens)])
    if args.word_list is not None:
        values.extend(["--word_list", str(resolve_path(args.word_list))])
    if args.exclude_word_list is not None:
        values.extend(
            ["--exclude_word_list", str(resolve_path(args.exclude_word_list))]
        )
    if args.trust_remote_code:
        values.append("--trust_remote_code")
    return values


def _reset_generated_outputs(
    results_dir: Path, args: argparse.Namespace
) -> tuple[Path, Path]:
    worker_dir = workflow_work_dir(args) / "workers"
    log_dir = workflow_log_dir(args)
    for path in (worker_dir, log_dir):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
    return worker_dir, log_dir


def run_runner(args: argparse.Namespace) -> None:
    results_dir = resolve_path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    existing_outputs = [path for path in results_dir.iterdir() if path.is_file()]
    if existing_outputs:
        raise FileExistsError(
            "Token-prior output already exists; use a new model run root: "
            + ", ".join(str(path) for path in existing_outputs[:5])
        )
    worker_dir, log_dir = _reset_generated_outputs(results_dir, args)
    gpu_groups = resolve_gpus(args.gpus)
    script_path = REPO_ROOT / "scripts/run_vocab_token_clean_prior.py"
    worker_args = _common_worker_args(args, results_dir)
    specs = []
    for worker_id, gpu_group in enumerate(gpu_groups):
        specs.append(
            WorkerSpec(
                name=f"worker_{worker_id}_gpu_{gpu_group}",
                command=[
                    current_python(),
                    str(script_path),
                    "--role",
                    "worker",
                    "--worker_id",
                    str(worker_id),
                    "--num_workers",
                    str(len(gpu_groups)),
                    *worker_args,
                ],
                env=build_worker_env(gpu_group),
            )
        )

    metadata = {
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "gpu_groups": gpu_groups,
        "num_workers": len(gpu_groups),
        "condition": "clean_no_injection",
        "token_filter": (
            "tokenizer entries decoding to one leading-space ASCII "
            "alphabetic word"
        ),
        "position_design": (
            "tokens are grouped and cyclically rotated so every primary "
            "token appears once at every answer position"
        ),
        "score_definition": (
            "softmax over candidate answer-number logits and probability "
            "under the full output vocabulary"
        ),
    }
    metadata_path = results_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"[runner] results_dir={results_dir}", flush=True)
    print(
        f"[runner] launching {len(specs)} workers on {gpu_groups}",
        flush=True,
    )
    run_worker_pool(specs, log_dir)
    summary = combine_worker_outputs(
        results_dir,
        num_workers=len(gpu_groups),
        num_choices=args.num_choices,
        position_index_start=args.position_index_start,
        worker_dir=worker_dir,
    )
    metadata.update(
        {
            "token_count": summary["token_count"],
            "token_position_rows": summary["token_position_rows"],
            "position_summary": summary["position_summary"],
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    shutil.rmtree(worker_dir)
    print(
        f"[runner] wrote {results_dir / 'summary_by_token.csv'} "
        f"for {summary['token_count']} tokens",
        flush=True,
    )


def token_matches_case(word: str, case_filter: str) -> bool:
    if case_filter == "lower":
        return word.islower()
    if case_filter == "upper_initial":
        return bool(word) and word[0].isupper() and word[1:].islower()
    return True


def load_word_set(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    resolved = resolve_path(path)
    words = {
        word.lower()
        for line in resolved.open(encoding="utf-8", errors="ignore")
        if (word := line.strip()) and WORD_RE.fullmatch(word)
    }
    if not words:
        raise ValueError(f"No alphabetic words found in {resolved}")
    return words


def decode_word_start_token(tokenizer, token_id: int) -> tuple[str, str] | None:
    """Return the display text and word for a standalone word-start token."""
    token_text = tokenizer.decode(
        [token_id],
        clean_up_tokenization_spaces=False,
    )
    if token_text.startswith(" "):
        word = token_text[1:]
    else:
        raw_token = tokenizer.convert_ids_to_tokens(token_id)
        if isinstance(raw_token, (list, tuple)):
            raw_token = raw_token[0] if len(raw_token) == 1 else None
        if isinstance(raw_token, str) and raw_token.startswith(WORD_START_MARKERS):
            word = raw_token[1:]
        else:
            # Some tokenizers expose no word-start marker at all. The
            # round-trip check in ``extract_single_word_tokens`` determines
            # whether this is actually a standalone token rather than a
            # continuation piece.
            word = token_text
    if not WORD_RE.fullmatch(word):
        return None
    return token_text, word


def extract_single_word_tokens(
    tokenizer,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    allowed_words = load_word_set(args.word_list)
    excluded_words = load_word_set(args.exclude_word_list)
    tokens = []
    for token_id in range(len(tokenizer)):
        if token_id in special_ids:
            continue
        decoded = decode_word_start_token(tokenizer, token_id)
        if decoded is None:
            continue
        token_text, word = decoded
        if not args.min_word_len <= len(word) <= args.max_word_len:
            continue
        if not token_matches_case(word, args.case_filter):
            continue
        if allowed_words is not None and word.lower() not in allowed_words:
            continue
        if excluded_words is not None and word.lower() in excluded_words:
            continue
        if tokenizer.encode(" " + word, add_special_tokens=False) != [token_id]:
            continue
        tokens.append(
            {
                "token_id": int(token_id),
                "token_text": token_text,
                "word": word,
                "word_lower": word.lower(),
                "word_len": len(word),
            }
        )
    tokens.sort(key=lambda row: (row["word_lower"], row["word"], row["token_id"]))
    if args.max_tokens is not None:
        tokens = tokens[: args.max_tokens]
    random.Random(args.seed).shuffle(tokens)
    return tokens


def make_groups(
    tokens: list[dict[str, Any]],
    num_choices: int,
) -> list[dict[str, Any]]:
    if len(tokens) < num_choices:
        raise ValueError(
            f"Need at least {num_choices} single-word tokens; got {len(tokens)}"
        )
    groups = []
    filler_cursor = 0
    for group_idx, start in enumerate(range(0, len(tokens), num_choices)):
        entries = [
            {"is_primary": True, **token}
            for token in tokens[start : start + num_choices]
        ]
        while len(entries) < num_choices:
            filler = tokens[filler_cursor % len(tokens)]
            filler_cursor += 1
            if any(entry["token_id"] == filler["token_id"] for entry in entries):
                continue
            entries.append({"is_primary": False, **filler})
        groups.append({"group_idx": group_idx, "entries": entries})
    return groups


def rotate(values: list[Any], offset: int) -> list[Any]:
    offset %= len(values)
    return values[offset:] + values[:offset]


def build_task(
    prompt_manager: PromptManager,
    group: dict[str, Any],
    rotation: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    entries = rotate(group["entries"], rotation)
    rendered = prompt_manager.render(
        "token_localization",
        [entry["word"] for entry in entries],
        preamble=args.prompt_preamble,
        suffix="\n",
    )
    for entry, record in zip(entries, rendered.records):
        actual_ids = [int(token_id) for token_id in record["token_ids"]]
        if actual_ids != [int(entry["token_id"])]:
            raise ValueError(
                f"Prompt token mismatch for {entry['word']!r}: "
                f"expected {[entry['token_id']]}, got {actual_ids}"
            )
    return {
        "prompt_id": int(group["group_idx"] * args.num_choices + rotation),
        "group_idx": int(group["group_idx"]),
        "rotation": int(rotation),
        "entries": entries,
        "input_ids": rendered.input_ids,
        "number_tokens": {
            int(label): int(token_id)
            for label, token_id in rendered.answer_token_by_choice.items()
        },
        "position_index_start": args.position_index_start,
    }


def group_tasks_by_length(
    tasks: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Bucket prompts so TransformerLens always scores a real final token."""
    grouped: dict[int, list[dict[str, Any]]] = {}
    for task in tasks:
        length = int(task["input_ids"].shape[1])
        grouped.setdefault(length, []).append(task)
    return list(grouped.values())


def score_candidate_numbers(
    candidate_logits: torch.Tensor,
    candidate_log_probs: torch.Tensor,
    ordered_numbers: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Convert the model's restricted candidate statistics into scores."""
    if candidate_logits.shape != candidate_log_probs.shape:
        raise ValueError("Candidate logits and log probabilities must align")
    if candidate_logits.shape[1] != len(ordered_numbers):
        raise ValueError("Candidate columns and answer numbers must align")
    candidate_probs = torch.softmax(candidate_logits.float(), dim=-1)
    full_probs = candidate_log_probs.float().exp()
    predictions = [
        ordered_numbers[index]
        for index in candidate_logits.argmax(dim=-1).cpu().tolist()
    ]
    return (
        candidate_logits.cpu(),
        candidate_probs.cpu(),
        full_probs.cpu(),
        predictions,
    )


def write_score_rows(
    handle,
    writer: csv.DictWriter,
    tasks: list[dict[str, Any]],
    candidate_logits: torch.Tensor,
    candidate_log_probs: torch.Tensor,
) -> int:
    if not tasks:
        return 0
    expected_numbers = tasks[0]["number_tokens"]
    if any(task["number_tokens"] != expected_numbers for task in tasks[1:]):
        raise ValueError("Answer-number token IDs changed within a batch")
    ordered_numbers = sorted(expected_numbers)
    number_logits, number_probs, full_probs, predictions = (
        score_candidate_numbers(
            candidate_logits,
            candidate_log_probs,
            ordered_numbers,
        )
    )
    rows_written = 0
    for row_index, task in enumerate(tasks):
        prediction = int(predictions[row_index])
        for choice_offset, entry in enumerate(task["entries"]):
            if not entry["is_primary"]:
                continue
            position = task["position_index_start"] + choice_offset
            writer.writerow(
                {
                    "prompt_id": task["prompt_id"],
                    "group_idx": task["group_idx"],
                    "rotation": task["rotation"],
                    "position": position,
                    "token_id": entry["token_id"],
                    "token_text": entry["token_text"],
                    "word": entry["word"],
                    "word_lower": entry["word_lower"],
                    "word_len": entry["word_len"],
                    "number_logit": float(number_logits[row_index, choice_offset]),
                    "number_softmax": float(number_probs[row_index, choice_offset]),
                    "full_vocab_prob": float(full_probs[row_index, choice_offset]),
                    "clean_prediction": prediction,
                    "is_argmax": int(position == prediction),
                }
            )
            rows_written += 1
    handle.flush()
    return rows_written


def run_worker(args: argparse.Namespace) -> None:
    started = time.monotonic()
    results_dir = resolve_path(args.results_dir)
    worker_dir = workflow_work_dir(args) / "workers"
    worker_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    visible_devices = torch.cuda.device_count() if device == "cuda" else 0
    print(
        f"loading TransformerLens bridge for {args.model} dtype={args.dtype} "
        f"visible_devices={visible_devices}",
        flush=True,
    )
    model = HookedModel(
        ModelConfig(
            name=args.model,
            device=device,
            dtype=args.dtype if device == "cuda" else "float32",
            trust_remote_code=args.trust_remote_code,
        )
    )
    tokenizer = model.tokenizer
    tokens = extract_single_word_tokens(tokenizer, args)
    groups = make_groups(tokens, args.num_choices)
    shard_groups = shard_items(groups, args.worker_id, args.num_workers)
    print(
        f"worker_id={args.worker_id} single_word_tokens={len(tokens)} "
        f"groups={len(shard_groups)}/{len(groups)} "
        f"choices={args.num_choices}",
        flush=True,
    )
    prompt_manager = PromptManager(tokenizer)
    output_path = (
        worker_dir / f"worker_{args.worker_id}_token_position_scores.csv"
    )
    rows_written = 0
    prompts_done = 0
    pending: list[dict[str, Any]] = []

    def flush_pending() -> None:
        nonlocal rows_written, prompts_done, pending
        if not pending:
            return
        for tasks in group_tasks_by_length(pending):
            expected_numbers = tasks[0]["number_tokens"]
            if any(
                task["number_tokens"] != expected_numbers
                for task in tasks[1:]
            ):
                raise ValueError("Answer-number token IDs changed within a batch")
            ordered_numbers = sorted(expected_numbers)
            input_ids = torch.cat(
                [task["input_ids"] for task in tasks],
                dim=0,
            )
            candidate_logits, candidate_log_probs = (
                model.last_token_candidate_stats(
                    input_ids,
                    candidate_token_ids=[
                        expected_numbers[number] for number in ordered_numbers
                    ],
                )
            )
            rows_written += write_score_rows(
                handle,
                writer,
                tasks,
                candidate_logits,
                candidate_log_probs,
            )
            prompts_done += len(tasks)
        pending = []

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCORE_FIELDS)
        writer.writeheader()
        for local_group_index, group in enumerate(shard_groups, start=1):
            for rotation in range(args.num_choices):
                pending.append(build_task(prompt_manager, group, rotation, args))
                if len(pending) >= args.batch_size:
                    flush_pending()
            if (
                args.progress_every
                and local_group_index % args.progress_every == 0
            ):
                print(
                    f"groups_done={local_group_index}/{len(shard_groups)} "
                    f"prompts_done={prompts_done} rows={rows_written} "
                    f"elapsed={time.monotonic() - started:.1f}",
                    flush=True,
                )
        flush_pending()

    metadata = {
        "worker_id": args.worker_id,
        "num_workers": args.num_workers,
        "visible_device_count": visible_devices,
        "model_backend": "TransformerLens TransformerBridge",
        "model_device": str(model.bridge.cfg.device),
        "model_dtype": str(model.bridge.cfg.dtype),
        "single_word_tokens_total": len(tokens),
        "groups_total": len(groups),
        "groups_shard": len(shard_groups),
        "prompts_done": prompts_done,
        "rows_written": rows_written,
    }
    (worker_dir / f"worker_{args.worker_id}_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(
        f"saved worker rows={rows_written} prompts={prompts_done} "
        f"to {output_path}",
        flush=True,
    )


def _csv_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _csv_int(row: dict[str, str], key: str) -> int:
    return int(row[key])


def aggregate_scores(
    score_paths: list[Path],
    combined_path: Path,
    num_choices: int,
    position_index_start: int,
) -> dict[str, Any]:
    positions = list(
        range(position_index_start, position_index_start + num_choices)
    )
    token_values: dict[int, dict[str, Any]] = {}
    token_position_rows: list[dict[str, str]] = []
    position_values = {
        position: {"scores": [], "full_probs": [], "argmax": 0}
        for position in positions
    }
    row_count = 0
    with combined_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=SCORE_FIELDS)
        writer.writeheader()
        for path in score_paths:
            with path.open(newline="") as input_handle:
                reader = csv.DictReader(input_handle)
                missing = set(SCORE_FIELDS).difference(reader.fieldnames or [])
                if missing:
                    raise ValueError(f"{path} missing columns: {sorted(missing)}")
                for row in reader:
                    writer.writerow({key: row[key] for key in SCORE_FIELDS})
                    row_count += 1
                    token_id = _csv_int(row, "token_id")
                    position = _csv_int(row, "position")
                    if position not in position_values:
                        raise ValueError(
                            f"Unexpected position {position} in {path}"
                        )
                    score = _csv_float(row, "number_softmax")
                    full_prob = _csv_float(row, "full_vocab_prob")
                    is_argmax = _csv_int(row, "is_argmax")
                    record = token_values.setdefault(
                        token_id,
                        {
                            "token_id": token_id,
                            "token_text": row["token_text"],
                            "word": row["word"],
                            "word_lower": row["word_lower"],
                            "word_len": _csv_int(row, "word_len"),
                            "scores": [],
                            "full_probs": [],
                            "argmax": 0,
                            "by_position": defaultdict(list),
                            "argmax_by_position": defaultdict(int),
                        },
                    )
                    record["scores"].append(score)
                    record["full_probs"].append(full_prob)
                    record["argmax"] += is_argmax
                    record["by_position"][position].append(score)
                    record["argmax_by_position"][position] += is_argmax
                    position_values[position]["scores"].append(score)
                    position_values[position]["full_probs"].append(full_prob)
                    position_values[position]["argmax"] += is_argmax
                    token_position_rows.append(row)
    if not token_values:
        raise ValueError("Worker score files contained no rows")

    token_summary = {}
    for token_id, record in token_values.items():
        missing_positions = [
            position
            for position in positions
            if not record["by_position"][position]
        ]
        if missing_positions:
            raise ValueError(
                f"Token {token_id} missing positions {missing_positions}"
            )
        position_means = {
            position: mean(record["by_position"][position])
            for position in positions
        }
        scores = record["scores"]
        token_summary[token_id] = {
            "token_id": token_id,
            "token_text": record["token_text"],
            "word": record["word"],
            "word_lower": record["word_lower"],
            "word_len": record["word_len"],
            "n": len(scores),
            "mean_number_softmax": mean(scores),
            "std_number_softmax": pstdev(scores) if len(scores) > 1 else 0.0,
            "min_number_softmax": min(scores),
            "max_number_softmax": max(scores),
            "position_score_range": (
                max(position_means.values()) - min(position_means.values())
            ),
            "mean_full_vocab_prob": mean(record["full_probs"]),
            "argmax_count": int(record["argmax"]),
            "argmax_rate": record["argmax"] / len(scores),
            **{
                f"pos_{position:02d}_number_softmax": position_means[position]
                for position in positions
            },
            **{
                f"pos_{position:02d}_argmax": int(
                    record["argmax_by_position"][position]
                )
                for position in positions
            },
        }
    return {
        "row_count": row_count,
        "token_summary": token_summary,
        "token_position_rows": token_position_rows,
        "position_values": position_values,
    }


def write_dict_rows(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames or list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def combine_worker_outputs(
    results_dir: Path,
    num_workers: int,
    num_choices: int,
    position_index_start: int,
    worker_dir: Path | None = None,
) -> dict[str, Any]:
    if worker_dir is None:
        raise ValueError(
            "worker_dir is required so temporary shards stay outside results"
        )
    score_paths = [
        worker_dir / f"worker_{worker_id}_token_position_scores.csv"
        for worker_id in range(num_workers)
    ]
    missing = [path for path in score_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing worker output(s): {missing}")
    combined = aggregate_scores(
        score_paths,
        results_dir / "token_position_scores.csv",
        num_choices,
        position_index_start,
    )
    token_rows = sorted(
        combined["token_summary"].values(),
        key=lambda row: (
            -float(row["mean_number_softmax"]),
            str(row["word_lower"]),
            int(row["token_id"]),
        ),
    )
    write_dict_rows(results_dir / "summary_by_token.csv", token_rows)

    positions = list(
        range(position_index_start, position_index_start + num_choices)
    )
    token_position_rows = [
        {
            "token_id": _csv_int(row, "token_id"),
            "token_text": row["token_text"],
            "word": row["word"],
            "word_lower": row["word_lower"],
            "word_len": _csv_int(row, "word_len"),
            "position": _csv_int(row, "position"),
            "number_softmax": _csv_float(row, "number_softmax"),
            "full_vocab_prob": _csv_float(row, "full_vocab_prob"),
            "is_argmax": _csv_int(row, "is_argmax"),
            "group_idx": _csv_int(row, "group_idx"),
            "rotation": _csv_int(row, "rotation"),
            "prompt_id": _csv_int(row, "prompt_id"),
            "clean_prediction": _csv_int(row, "clean_prediction"),
        }
        for row in sorted(
            combined["token_position_rows"],
            key=lambda item: (
                _csv_int(item, "token_id"),
                _csv_int(item, "position"),
            ),
        )
    ]
    write_dict_rows(
        results_dir / "summary_by_token_position.csv",
        token_position_rows,
    )

    position_rows = []
    for position in positions:
        values = combined["position_values"][position]
        scores = values["scores"]
        position_rows.append(
            {
                "position": position,
                "n": len(scores),
                "mean_number_softmax": mean(scores),
                "std_number_softmax": (
                    pstdev(scores) if len(scores) > 1 else 0.0
                ),
                "min_number_softmax": min(scores),
                "max_number_softmax": max(scores),
                "mean_full_vocab_prob": mean(values["full_probs"]),
                "argmax_count": int(values["argmax"]),
                "argmax_rate": values["argmax"] / len(scores),
            }
        )
    write_dict_rows(results_dir / "summary_by_position.csv", position_rows)
    return {
        "token_count": len(token_rows),
        "token_position_rows": combined["row_count"],
        "position_summary": position_rows,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.role == "runner":
        run_runner(args)
    else:
        run_worker(args)
