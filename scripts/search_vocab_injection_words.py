#!/usr/bin/env python3
"""Rank English tokenizer words by optional-none injection localization.

The runner first extracts layer activations for every eligible ASCII-English
word-start token. By default it centers them with the mean over that same
complete word set. It can exclude a calibration panel from the evaluation
population and optionally evaluate an ordered shortlist without changing the
full-vocabulary baseline. It writes one vector shard per GPU, then evaluates
each shard on every cluster/position combination for each prompt template.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

for _thread_env in (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_env] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from introspection_core.data_parallel import (
    WorkerSpec,
    build_worker_env,
    current_python,
    parse_gpu_groups,
    run_worker_pool,
    shard_items,
)
from introspection_core.cluster_split import require_valid_cluster_split
from introspection_core.cluster_split import file_sha256
from introspection_core.extraction import extract_last_token_residuals
from introspection_core.localization_evaluation import (
    concept_metric_rows,
    evaluate_cluster_localization,
    load_cluster_prompts,
    unit_vector_matrix,
)
from introspection_core.model import HookedModel, ModelConfig
from introspection_core.prompts import PromptManager
from introspection_core.results import write_metadata, write_table
from introspection_core.vocab_token_clean_prior import extract_single_word_tokens


DEFAULT_PROMPT_TEMPLATES = ["semantic_highinj_posref_gate_balanced_disrupts"]

BASELINE_FULL_ENGLISH = "full_english"
BASELINE_MANIFEST = "manifest"
BASELINE_MODES = [BASELINE_FULL_ENGLISH, BASELINE_MANIFEST]
SUPPORTED_PROMPT_TEMPLATES = [*DEFAULT_PROMPT_TEMPLATES]
CONCEPT_RANK_COLUMNS = [
    "injected_argmax_accuracy",
    "injected_mean_correct_prob",
    "accuracy_gain_over_clean",
    "concept",
]
CONCEPT_RANK_ASCENDING = [False, False, False, True]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--role", choices=["runner", "prepare", "worker", "merge"], default="runner"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--cluster_csv", type=Path, required=True)
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help="Canonical data/dataset/<model> directory",
    )
    parser.add_argument(
        "--screening_stage",
        choices=["coarse", "fine"],
        required=True,
    )
    parser.add_argument(
        "--work_dir",
        type=Path,
        help="Temporary workspace; defaults outside results and dataset roots",
    )
    parser.add_argument(
        "--logs_dir",
        type=Path,
        help="Worker-log directory; defaults under logs/auto_workflow/<model>",
    )
    parser.add_argument(
        "--baseline_mode",
        choices=BASELINE_MODES,
        help=(
            "concept-vector baseline: full_english centers every eligible "
            "English vocabulary word on that same complete vocabulary; "
            "manifest uses --baseline_words_json. If omitted, the mode is "
            "inferred for backward compatibility"
        ),
    )
    parser.add_argument(
        "--baseline_words_json",
        type=Path,
        help=(
            "JSON manifest containing a non-empty baseline_words list; only "
            "valid with --baseline_mode manifest"
        ),
    )
    parser.add_argument(
        "--calibration_concepts_json",
        type=Path,
        help=(
            "JSON manifest containing concept_vector_words to exclude "
            "case-insensitively from the screening vocabulary"
        ),
    )
    parser.add_argument(
        "--candidate_words_csv",
        type=Path,
        help=(
            "evaluation-only shortlist drawn from the screening vocabulary; "
            "does not change full-vocabulary baseline extraction"
        ),
    )
    parser.add_argument(
        "--candidate_column",
        default="concept",
        help="Column containing ordered candidate words",
    )
    parser.add_argument(
        "--expected_candidate_count",
        type=int,
        default=3000,
        help=(
            "Required shortlist size (default: formal-protocol top 3000; "
            "override only for an explicitly labeled smoke run)"
        ),
    )
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--strength", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--extraction_batch_size", type=int, default=32)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--expected_clusters", type=int, default=30)
    parser.add_argument(
        "--max_clusters",
        type=int,
        help="Evaluate only the first N rows after validating the cluster CSV.",
    )
    parser.add_argument("--min_word_len", type=int, default=1)
    parser.add_argument("--max_word_len", type=int, default=32)
    parser.add_argument(
        "--case_filter", choices=["all", "lower", "upper_initial"], default="all"
    )
    parser.add_argument("--max_tokens", type=int)
    parser.add_argument(
        "--prompt_templates",
        nargs="+",
        default=DEFAULT_PROMPT_TEMPLATES,
        choices=SUPPORTED_PROMPT_TEMPLATES,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.extraction_batch_size <= 0:
        parser.error("batch sizes must be positive")
    if args.num_workers <= 0:
        parser.error("--num_workers must be positive")
    if not 0 <= args.worker_id < args.num_workers:
        parser.error("--worker_id must be in [0, num_workers)")
    if args.expected_clusters <= 0:
        parser.error("--expected_clusters must be positive")
    if args.max_clusters is not None and not 0 < args.max_clusters <= args.expected_clusters:
        parser.error("--max_clusters must be in [1, expected_clusters]")
    if args.min_word_len <= 0 or args.max_word_len < args.min_word_len:
        parser.error("invalid English word length range")
    if args.max_tokens is not None and args.max_tokens <= 0:
        parser.error("--max_tokens must be positive")
    if args.expected_candidate_count <= 0:
        parser.error("--expected_candidate_count must be positive")
    if not args.candidate_column.strip():
        parser.error("--candidate_column must be non-empty")
    if (
        args.candidate_words_csv is not None
        and args.calibration_concepts_json is None
    ):
        parser.error(
            "--calibration_concepts_json is required with "
            "--candidate_words_csv so shortlist membership can be validated "
            "against V_screen"
        )
    if args.max_tokens is not None and args.max_tokens < args.num_workers:
        parser.error("--max_tokens must be at least --num_workers")
    if (
        args.role == "runner"
        and args.max_tokens is not None
        and args.max_tokens < len(parse_gpu_groups(args.gpus))
    ):
        parser.error("--max_tokens must be at least the number of --gpus")
    if not hasattr(torch, args.dtype):
        parser.error(f"unknown torch dtype: {args.dtype}")
    if len(args.prompt_templates) != 1:
        parser.error(
            "the canonical workflow requires exactly one --prompt_templates value"
        )
    if args.screening_stage == "coarse" and args.candidate_words_csv is not None:
        parser.error("coarse screening cannot use --candidate_words_csv")
    if args.screening_stage == "fine" and args.candidate_words_csv is None:
        parser.error("fine screening requires --candidate_words_csv")
    if args.baseline_mode is None:
        args.baseline_mode = (
            BASELINE_MANIFEST
            if args.baseline_words_json is not None
            else BASELINE_FULL_ENGLISH
        )
    elif (
        args.baseline_mode == BASELINE_FULL_ENGLISH
        and args.baseline_words_json is not None
    ):
        parser.error(
            "--baseline_words_json cannot be used with "
            "--baseline_mode full_english"
        )
    if (
        args.baseline_mode == BASELINE_MANIFEST
        and args.baseline_words_json is None
    ):
        parser.error(
            "--baseline_words_json is required with --baseline_mode manifest"
        )
    return args


def _resolved_dataset_dir(args: argparse.Namespace) -> Path:
    path = args.dataset_dir
    repo_root = Path(__file__).resolve().parents[1]
    return path if path.is_absolute() else repo_root / path


def _work_dir(args: argparse.Namespace) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    if args.work_dir is not None:
        return args.work_dir if args.work_dir.is_absolute() else repo_root / args.work_dir
    dataset_dir = _resolved_dataset_dir(args)
    return (
        repo_root
        / "tmp"
        / "auto_workflow"
        / dataset_dir.name
        / "screening"
        / args.screening_stage
    )


def _logs_dir(args: argparse.Namespace) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    if args.logs_dir is not None:
        return args.logs_dir if args.logs_dir.is_absolute() else repo_root / args.logs_dir
    dataset_dir = _resolved_dataset_dir(args)
    return (
        repo_root
        / "logs"
        / "auto_workflow"
        / dataset_dir.name
        / "screening"
        / args.screening_stage
    )


def _model(args: argparse.Namespace) -> HookedModel:
    return HookedModel(
        ModelConfig(
            name=args.model,
            device="cuda",
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
    )


def _english_vocab(tokenizer, args: argparse.Namespace) -> list[dict]:
    filter_args = SimpleNamespace(
        word_list=None,
        exclude_word_list=None,
        min_word_len=args.min_word_len,
        max_word_len=args.max_word_len,
        case_filter=args.case_filter,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    rows = extract_single_word_tokens(tokenizer, filter_args)
    rows.sort(key=lambda row: int(row["token_id"]))
    if not rows:
        raise ValueError("tokenizer has no eligible ASCII-English word-start tokens")
    return rows


def _load_calibration_words(path: Path) -> list[str]:
    """Load the ordered calibration panel used for lexical exclusion."""
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"calibration manifest must be a JSON object: {path}")
    words = payload.get("concept_vector_words")
    if not isinstance(words, list) or not words:
        raise ValueError(
            "calibration manifest must contain a non-empty "
            f"concept_vector_words list: {path}"
        )
    if any(not isinstance(word, str) or not word.strip() for word in words):
        raise ValueError(
            f"concept_vector_words must contain non-empty strings: {path}"
        )
    return list(words)


def _screening_vocabulary(
    vocab_rows: list[dict],
    calibration_words: list[str],
) -> tuple[list[dict], int]:
    """Return V_screen and the number of case-insensitive lexical exclusions."""
    excluded_lower = {word.lower() for word in calibration_words}
    rows = [
        row
        for row in vocab_rows
        if str(row["word"]).lower() not in excluded_lower
    ]
    return rows, len(vocab_rows) - len(rows)


def _load_candidate_words(
    path: Path,
    *,
    column: str,
    expected_count: int,
) -> list[str]:
    """Load an ordered evaluation shortlist with deterministic cardinality."""
    frame = pd.read_csv(path, keep_default_na=False)
    if column not in frame.columns:
        raise ValueError(f"candidate CSV missing column {column!r}: {path}")
    words = [str(value) for value in frame[column].tolist()]
    if len(words) != expected_count:
        raise ValueError(
            f"candidate CSV has {len(words)} rows; expected exactly "
            f"{expected_count}: {path}"
        )
    if any(not word.strip() for word in words):
        raise ValueError(f"candidate column contains an empty word: {path}")
    seen: set[str] = set()
    duplicates: set[str] = set()
    for word in words:
        if word in seen:
            duplicates.add(word)
        seen.add(word)
    if duplicates:
        raise ValueError(
            "candidate words must be unique; duplicates include "
            f"{sorted(duplicates)[:5]}"
        )
    return words


def _select_evaluation_rows(
    screening_rows: list[dict],
    candidate_words: list[str] | None,
) -> list[dict]:
    """Filter evaluated rows while preserving the frozen shortlist order."""
    if candidate_words is None:
        return list(screening_rows)
    rows_by_word: dict[str, list[dict]] = {}
    for row in screening_rows:
        rows_by_word.setdefault(str(row["word"]), []).append(row)
    selected = []
    for word in candidate_words:
        matches = rows_by_word.get(word, [])
        if len(matches) != 1:
            raise ValueError(
                f"candidate {word!r} occurs {len(matches)} times in V_screen; "
                "every candidate must occur exactly once"
            )
        selected.append(matches[0])
    return selected


def _prepare_manifest_path(work_dir: Path) -> Path:
    return work_dir / "prepared_vectors.json"


def _validate_worker_capacity(word_count: int, num_workers: int) -> None:
    """Reject configurations that would create empty worker shards."""
    if word_count < num_workers:
        raise ValueError(
            f"evaluation vocabulary has {word_count} words, but {num_workers} "
            "workers were requested; reduce --gpus/--num_workers or increase "
            "--max_tokens"
        )


def _load_baseline_words(path: Path) -> list[str]:
    """Load an explicit, ordered baseline-word list from a JSON manifest."""
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"baseline manifest must be a JSON object: {path}")
    words = payload.get("baseline_words")
    if not isinstance(words, list) or not words:
        raise ValueError(
            f"baseline manifest must contain a non-empty baseline_words list: {path}"
        )
    if any(not isinstance(word, str) or not word.strip() for word in words):
        raise ValueError(f"baseline_words must contain non-empty strings: {path}")
    return list(words)


def _baseline_words_sha256(words: list[str]) -> str:
    encoded = json.dumps(
        words,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stored_baseline_mode(manifest: dict) -> str | None:
    """Read new manifests and recognize pre-baseline_mode full-vocab runs."""
    mode = manifest.get("baseline_mode")
    if mode in BASELINE_MODES:
        return str(mode)
    baseline = str(manifest.get("baseline", "")).lower()
    source_path = manifest.get("baseline_source_path")
    if source_path:
        return BASELINE_MANIFEST
    if "complete eligible english vocabulary" in baseline:
        return BASELINE_FULL_ENGLISH
    return None


def _resume_manifest_mismatches(
    manifest: dict,
    args: argparse.Namespace,
) -> list[str]:
    """Return preparation fields that would make cached vectors unsafe."""
    requested = {
        "model": str(args.model),
        "layer": int(args.layer),
        "num_workers": int(args.num_workers),
        "case_filter": str(args.case_filter),
        "min_word_len": int(args.min_word_len),
        "max_word_len": int(args.max_word_len),
    }
    mismatches = []
    for field, value in requested.items():
        if field in manifest and manifest[field] != value:
            mismatches.append(
                f"{field}: cached={manifest[field]!r}, requested={value!r}"
            )

    stored_mode = _stored_baseline_mode(manifest)
    if stored_mode != args.baseline_mode:
        mismatches.append(
            "baseline_mode: "
            f"cached={stored_mode!r}, requested={args.baseline_mode!r}"
        )
    elif args.baseline_mode == BASELINE_FULL_ENGLISH:
        cached_baseline_count = manifest.get("baseline_word_count")
        cached_english_count = manifest.get("english_word_count")
        if (
            cached_baseline_count is not None
            and cached_english_count is not None
            and cached_baseline_count != cached_english_count
        ):
            mismatches.append(
                "full-English baseline count differs from English vocabulary "
                f"count: {cached_baseline_count!r} != {cached_english_count!r}"
            )
        if manifest.get("baseline_source_path") is not None:
            mismatches.append(
                "baseline_source_path must be null for full_english: "
                f"cached={manifest['baseline_source_path']!r}"
            )
    elif args.baseline_mode == BASELINE_MANIFEST:
        assert args.baseline_words_json is not None
        words = _load_baseline_words(args.baseline_words_json)
        expected_hash = _baseline_words_sha256(words)
        cached_hash = manifest.get("baseline_words_sha256")
        if cached_hash != expected_hash:
            mismatches.append(
                "baseline_words_sha256: "
                f"cached={cached_hash!r}, requested={expected_hash!r}"
            )
        cached_count = manifest.get("baseline_word_count")
        if cached_count != len(words):
            mismatches.append(
                "baseline_word_count: "
                f"cached={cached_count!r}, requested={len(words)!r}"
            )

    if "max_tokens" not in manifest:
        mismatches.append(
            "max_tokens: cached manifest predates truncation tracking and "
            "cannot prove complete-vocabulary membership"
        )
    elif manifest["max_tokens"] != args.max_tokens:
        mismatches.append(
            "max_tokens: "
            f"cached={manifest['max_tokens']!r}, requested={args.max_tokens!r}"
        )
    cached_dtype = manifest.get("dtype")
    if cached_dtype is not None and cached_dtype != args.dtype:
        mismatches.append(
            f"dtype: cached={cached_dtype!r}, requested={args.dtype!r}"
        )

    calibration_words = (
        _load_calibration_words(args.calibration_concepts_json)
        if args.calibration_concepts_json is not None
        else None
    )
    requested_calibration_hash = (
        _baseline_words_sha256(calibration_words)
        if calibration_words is not None
        else None
    )
    cached_calibration_hash = manifest.get("calibration_words_sha256")
    if cached_calibration_hash != requested_calibration_hash:
        mismatches.append(
            "calibration_words_sha256: "
            f"cached={cached_calibration_hash!r}, "
            f"requested={requested_calibration_hash!r}"
        )

    candidate_words = (
        _load_candidate_words(
            args.candidate_words_csv,
            column=args.candidate_column,
            expected_count=args.expected_candidate_count,
        )
        if args.candidate_words_csv is not None
        else None
    )
    requested_candidate_hash = (
        _baseline_words_sha256(candidate_words)
        if candidate_words is not None
        else None
    )
    cached_candidate_hash = manifest.get("candidate_words_sha256")
    if cached_candidate_hash != requested_candidate_hash:
        mismatches.append(
            "candidate_words_sha256: "
            f"cached={cached_candidate_hash!r}, "
            f"requested={requested_candidate_hash!r}"
        )
    cached_candidate_column = manifest.get("candidate_column")
    requested_candidate_column = (
        args.candidate_column if candidate_words is not None else None
    )
    if cached_candidate_column != requested_candidate_column:
        mismatches.append(
            "candidate_column: "
            f"cached={cached_candidate_column!r}, "
            f"requested={requested_candidate_column!r}"
        )
    return mismatches


def _validate_resume_manifest(
    manifest: dict,
    args: argparse.Namespace,
) -> None:
    mismatches = _resume_manifest_mismatches(manifest, args)
    if mismatches:
        detail = "; ".join(mismatches)
        raise ValueError(
            "Cached vocabulary vectors do not match this run ("
            f"{detail}). Use matching preparation settings or a new "
            "--results_dir; never reuse vectors across baseline definitions "
            "or evaluation populations."
        )


def prepare(args: argparse.Namespace) -> None:
    work_dir = _work_dir(args)
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _prepare_manifest_path(work_dir)
    shard_paths = [
        work_dir / "vector_shards" / f"worker_{worker_id}.pt"
        for worker_id in range(args.num_workers)
    ]
    if (
        args.resume
        and manifest_path.exists()
        and all(path.exists() for path in shard_paths)
    ):
        manifest = json.loads(manifest_path.read_text())
        _validate_resume_manifest(manifest, args)
        _validate_worker_capacity(
            int(
                manifest.get(
                    "evaluation_word_count",
                    manifest["english_word_count"],
                )
            ),
            args.num_workers,
        )
        print(f"[prepare] reusing {manifest_path}", flush=True)
        return

    print(f"[prepare] loading model {args.model}", flush=True)
    model = _model(args)
    vocab_rows = _english_vocab(model.tokenizer, args)
    words = [str(row["word"]) for row in vocab_rows]
    calibration_words = (
        _load_calibration_words(args.calibration_concepts_json)
        if args.calibration_concepts_json is not None
        else []
    )
    screening_rows, excluded_count = _screening_vocabulary(
        vocab_rows,
        calibration_words,
    )
    candidate_words = (
        _load_candidate_words(
            args.candidate_words_csv,
            column=args.candidate_column,
            expected_count=args.expected_candidate_count,
        )
        if args.candidate_words_csv is not None
        else None
    )
    evaluation_rows = _select_evaluation_rows(screening_rows, candidate_words)
    _validate_worker_capacity(len(evaluation_rows), args.num_workers)
    print(
        f"[prepare] extracting L{args.layer} activations for "
        f"{len(words)} full-vocabulary words; evaluating "
        f"{len(evaluation_rows)} V_screen rows",
        flush=True,
    )
    activations = extract_last_token_residuals(
        model,
        words,
        layer=args.layer,
        batch_size=args.extraction_batch_size,
    )
    if args.baseline_mode == BASELINE_FULL_ENGLISH:
        baseline_words = words
        baseline_activations = activations
        baseline_source = "complete eligible English vocabulary"
        baseline_source_path = None
    else:
        assert args.baseline_words_json is not None
        baseline_words = _load_baseline_words(args.baseline_words_json)
        print(
            f"[prepare] extracting {len(baseline_words)} specified baseline "
            f"words from {args.baseline_words_json}",
            flush=True,
        )
        baseline_activations = extract_last_token_residuals(
            model,
            baseline_words,
            layer=args.layer,
            batch_size=args.extraction_batch_size,
        )
        baseline_source = "specified baseline_words manifest"
        baseline_source_path = str(args.baseline_words_json.resolve())
    baseline_mean = baseline_activations.mean(dim=0)
    vectors = unit_vector_matrix(activations - baseline_mean)
    activation_index_by_token_id = {
        int(row["token_id"]): index for index, row in enumerate(vocab_rows)
    }
    evaluation_indices = torch.tensor(
        [
            activation_index_by_token_id[int(row["token_id"])]
            for row in evaluation_rows
        ],
        dtype=torch.long,
    )
    evaluation_vectors = vectors.index_select(0, evaluation_indices)

    vocabulary_dir = _resolved_dataset_dir(args) / "vocabulary"
    write_table(vocabulary_dir, "english.csv", vocab_rows)
    write_table(vocabulary_dir, "screening.csv", screening_rows)
    write_table(work_dir, "evaluation.csv", evaluation_rows)
    torch.save(baseline_mean, work_dir / "baseline_mean.pt")
    (work_dir / "vector_shards").mkdir(parents=True, exist_ok=True)
    for worker_id, shard_path in enumerate(shard_paths):
        shard_indices = list(
            range(worker_id, len(evaluation_rows), args.num_workers)
        )
        index_tensor = torch.tensor(shard_indices, dtype=torch.long)
        payload = {
            "rows": [evaluation_rows[index] for index in shard_indices],
            "vectors": evaluation_vectors.index_select(0, index_tensor),
        }
        torch.save(payload, shard_path)
        print(
            f"[prepare] worker {worker_id}: {len(shard_indices)} words -> {shard_path}",
            flush=True,
        )

    manifest = {
        "model": args.model,
        "layer": args.layer,
        "dtype": args.dtype,
        "baseline_mode": args.baseline_mode,
        "baseline": f"mean activation over {baseline_source}",
        "baseline_source_path": baseline_source_path,
        "baseline_word_count": len(baseline_words),
        "baseline_words_sha256": _baseline_words_sha256(baseline_words),
        "english_filter": "ASCII letters only; standalone word-start token round trip",
        "case_filter": args.case_filter,
        "min_word_len": args.min_word_len,
        "max_word_len": args.max_word_len,
        "max_tokens": args.max_tokens,
        "vocab_size": len(model.tokenizer),
        "english_word_count": len(vocab_rows),
        "english_words_sha256": _baseline_words_sha256(words),
        "screening_word_count": len(screening_rows),
        "screening_words_sha256": _baseline_words_sha256(
            [str(row["word"]) for row in screening_rows]
        ),
        "calibration_source_path": (
            str(args.calibration_concepts_json.resolve())
            if args.calibration_concepts_json is not None
            else None
        ),
        "calibration_word_count": len(calibration_words),
        "calibration_words_sha256": (
            _baseline_words_sha256(calibration_words)
            if calibration_words
            else None
        ),
        "calibration_lexical_exclusion_count": excluded_count,
        "evaluation_word_count": len(evaluation_rows),
        "candidate_source_path": (
            str(args.candidate_words_csv.resolve())
            if args.candidate_words_csv is not None
            else None
        ),
        "candidate_column": (
            args.candidate_column if candidate_words is not None else None
        ),
        "candidate_words_sha256": (
            _baseline_words_sha256(candidate_words)
            if candidate_words is not None
            else None
        ),
        "num_workers": args.num_workers,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[prepare] wrote {manifest_path}", flush=True)


def _clean_stats(model, examples, *, include_exact: bool) -> dict:
    tokens = torch.cat([example.input_ids for example in examples], dim=0)
    candidate_ids = examples[0].candidate_token_ids
    candidate_logits, candidate_log_probs = model.last_token_candidate_stats(
        tokens, candidate_token_ids=candidate_ids
    )
    labels = examples[0].candidate_labels
    none_index = labels.index("none")
    restricted_none_prob = torch.softmax(candidate_logits, dim=-1)[:, none_index]
    full_none_prob = candidate_log_probs[:, none_index].exp()
    exact_none_count = None
    exact_none_rate = None
    if include_exact:
        exact_none_count = 0
        for index in range(tokens.shape[0]):
            greedy_id = int(
                model.forward_logits(tokens[index : index + 1])[:, -1, :]
                .argmax(dim=-1)
                .item()
            )
            exact_none_count += int(greedy_id == candidate_ids[none_index])
        exact_none_rate = exact_none_count / len(examples)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "clean_prompt_count": len(examples),
        "clean_exact_none_count": exact_none_count,
        "clean_exact_none_rate": exact_none_rate,
        "clean_restricted_none_count": int(
            candidate_logits.argmax(dim=-1).eq(none_index).sum()
        ),
        "clean_mean_full_vocab_p_none": float(full_none_prob.mean()),
        "clean_mean_restricted_p_none": float(restricted_none_prob.mean()),
    }


def _load_summary_rows(path: Path) -> dict[str, dict[str, str]]:
    """Load resumable summaries keyed by prompt template."""
    if not path.exists() or path.stat().st_size == 0:
        return {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "prompt_template" not in reader.fieldnames:
            return {}
        return {
            row["prompt_template"]: row
            for row in reader
            if row.get("prompt_template")
        }


def worker(args: argparse.Namespace) -> None:
    work_dir = _work_dir(args)
    shard_path = work_dir / "vector_shards" / f"worker_{args.worker_id}.pt"
    if not shard_path.exists():
        raise FileNotFoundError(f"missing prepared vector shard: {shard_path}")
    payload = torch.load(shard_path, map_location="cpu", weights_only=False)
    vocab_rows = list(payload["rows"])
    vectors = payload["vectors"].float()
    if not vocab_rows:
        raise ValueError(
            f"prepared vector shard is empty: {shard_path}; reduce the worker count"
        )
    concept_names = [str(row["word"]) for row in vocab_rows]
    print(
        f"[worker {args.worker_id}] loading model for {len(concept_names)} words",
        flush=True,
    )
    model = _model(args)
    prompt_manager = PromptManager(model.tokenizer)
    summary_path = (
        work_dir / "workers" / f"worker_{args.worker_id}_summary.csv"
    )
    existing_summaries = (
        _load_summary_rows(summary_path) if args.resume else {}
    )
    summary_rows = []
    for template_index, template_name in enumerate(args.prompt_templates, start=1):
        output_path = (
            work_dir
            / "workers"
            / f"worker_{args.worker_id}_{template_name}.csv"
        )
        if args.resume and output_path.exists():
            with output_path.open(newline="") as handle:
                completed = sum(1 for _ in csv.DictReader(handle))
            if completed == len(concept_names):
                existing_summary = existing_summaries.get(template_name)
                if existing_summary is not None:
                    summary_rows.append(existing_summary)
                    print(
                        f"[worker {args.worker_id}] {template_name} already complete",
                        flush=True,
                    )
                    continue
                print(
                    f"[worker {args.worker_id}] {template_name} result is complete "
                    "but its summary is missing; recomputing",
                    flush=True,
                )
        print(
            f"[worker {args.worker_id}] template {template_index}/"
            f"{len(args.prompt_templates)}: {template_name}",
            flush=True,
        )
        examples, n_choices = load_cluster_prompts(
            args.cluster_csv,
            prompt_manager,
            preamble="system",
            template_name=template_name,
        )
        if len(examples) != args.expected_clusters:
            raise ValueError(
                f"expected {args.expected_clusters} clusters, got {len(examples)}"
            )
        if args.max_clusters is not None:
            examples = examples[: args.max_clusters]
        clean = _clean_stats(
            model,
            examples,
            include_exact=args.worker_id == 0,
        )
        evaluation = evaluate_cluster_localization(
            model,
            examples=examples,
            concept_names=concept_names,
            unit_vectors=vectors,
            injection_layer=args.layer,
            strength=args.strength,
            scale_mode="relative_hidden_norm",
            batch_size=args.batch_size,
            include_rows=False,
            progress_prefix=f"[worker {args.worker_id} {template_name}] ",
            progress_every=100,
            trial_mode="cross_product",
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        metric_rows = concept_metric_rows(evaluation, concept_names)
        for vocab_row, metric_row in zip(vocab_rows, metric_rows):
            metric_row.update(
                {
                    "token_id": int(vocab_row["token_id"]),
                    "token_text": str(vocab_row["token_text"]),
                    "word_lower": str(vocab_row["word_lower"]),
                    "word_len": int(vocab_row["word_len"]),
                    "prompt_template": template_name,
                    "worker_id": args.worker_id,
                }
            )
        metric_rows.sort(key=_concept_rank_key)
        write_table(output_path.parent, output_path.name, metric_rows)
        trials = int(evaluation.n_trials.sum())
        correct = int(evaluation.injected_correct.sum())
        summary = {
            "worker_id": args.worker_id,
            "prompt_template": template_name,
            "n_words": len(concept_names),
            "n_clusters": len(examples),
            "n_choices": n_choices,
            "n_trials": trials,
            "n_correct": correct,
            "injected_accuracy": correct / trials,
            "injected_mean_correct_prob": (
                float(evaluation.injected_correct_prob_sum.sum()) / trials
            ),
            **clean,
        }
        summary_rows.append(summary)
        print(json.dumps(summary), flush=True)
    write_table(
        summary_path.parent,
        summary_path.name,
        summary_rows,
    )


def _read_csv_preserving_strings(path: Path) -> pd.DataFrame:
    """Read generated CSV without converting valid token strings to NA."""
    return pd.read_csv(path, keep_default_na=False)


def _concept_rank_key(row: dict) -> tuple[float, float, float, str]:
    """Return the registered deterministic concept-ranking key."""
    return (
        -float(row["injected_argmax_accuracy"]),
        -float(row["injected_mean_correct_prob"]),
        -float(row["accuracy_gain_over_clean"]),
        str(row["concept"]),
    )


def _sort_concept_metric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the same registered ordering used inside each worker."""
    return frame.sort_values(
        CONCEPT_RANK_COLUMNS,
        ascending=CONCEPT_RANK_ASCENDING,
        kind="stable",
    )


def _write_screening_selection(
    args: argparse.Namespace,
    metrics: pd.DataFrame,
    preparation: dict,
) -> Path:
    """Publish the fixed shortlist or selected concept population."""
    dataset_dir = _resolved_dataset_dir(args)
    concepts_dir = dataset_dir / "concepts"
    concepts_dir.mkdir(parents=True, exist_ok=True)
    count = 3000 if args.screening_stage == "coarse" else 300
    if len(metrics) < count:
        raise ValueError(
            f"{args.screening_stage} screening produced {len(metrics)} concepts; "
            f"need at least {count}"
        )
    selected = metrics.iloc[:count].copy()
    stem = "shortlist" if args.screening_stage == "coarse" else "selected"
    csv_path = concepts_dir / f"{stem}.csv"
    selected.to_csv(csv_path, index=False)

    if args.screening_stage == "fine":
        vocabulary = pd.read_csv(
            dataset_dir / "vocabulary" / "english.csv",
            keep_default_na=False,
        )
        words = [str(word) for word in vocabulary["word"].tolist()]
        (concepts_dir / "selected.json").write_text(
            json.dumps(
                {
                    "concept_vector_words": selected["concept"].astype(str).tolist(),
                    "baseline_words": words,
                    "baseline_mode": BASELINE_FULL_ENGLISH,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        shuffled = selected["concept"].astype(str).tolist()
        random.Random(args.seed).shuffle(shuffled)
        split_names = {
            "train": shuffled[:100],
            "validation": shuffled[100:200],
            "test": shuffled[200:300],
        }
        selected_by_concept = selected.set_index("concept", drop=False)
        split_manifest = {
            "seed": args.seed,
            "source_file": str(csv_path.resolve()),
            "source_file_sha256": file_sha256(csv_path),
            "baseline_mode": BASELINE_FULL_ENGLISH,
            "baseline_word_count": len(words),
            "baseline_words_sha256": _baseline_words_sha256(words),
            "splits": {},
        }
        for split_name, split_words in split_names.items():
            split_frame = selected_by_concept.loc[split_words].reset_index(drop=True)
            split_csv = concepts_dir / f"{split_name}.csv"
            split_json = concepts_dir / f"{split_name}.json"
            split_frame.to_csv(split_csv, index=False)
            split_json.write_text(
                json.dumps(
                    {
                        "concept_vector_words": split_words,
                        "baseline_words": words,
                        "baseline_mode": BASELINE_FULL_ENGLISH,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            split_manifest["splits"][split_name] = {
                "count": len(split_words),
                "concepts": split_words,
                "csv_sha256": file_sha256(split_csv),
                "json_sha256": file_sha256(split_json),
            }
        concept_manifest_path = (
            dataset_dir / "manifests" / "concept_split.json"
        )
        concept_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        concept_manifest_path.write_text(
            json.dumps(split_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    manifest_path = dataset_dir / "manifests" / "vocabulary_screening.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    )
    payload.update(
        {
            "model": args.model,
            "baseline_mode": preparation["baseline_mode"],
            "baseline_word_count": preparation["baseline_word_count"],
            "baseline_words_sha256": preparation["baseline_words_sha256"],
            "english_word_count": preparation["english_word_count"],
            "english_words_sha256": preparation["english_words_sha256"],
            "screening_word_count": preparation["screening_word_count"],
            "screening_words_sha256": preparation["screening_words_sha256"],
            "calibration_words_sha256": preparation["calibration_words_sha256"],
            "calibration_lexical_exclusion_count": preparation[
                "calibration_lexical_exclusion_count"
            ],
        }
    )
    payload[args.screening_stage] = {
        "cluster_file": str(args.cluster_csv.resolve()),
        "cluster_file_sha256": file_sha256(args.cluster_csv.resolve()),
        "metrics_file": str((args.results_dir / "metrics.csv").resolve()),
        "metrics_file_sha256": file_sha256(args.results_dir / "metrics.csv"),
        "selection_file": str(csv_path.resolve()),
        "selection_file_sha256": file_sha256(csv_path),
        "selection_count": count,
        "ranking_columns": CONCEPT_RANK_COLUMNS,
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return csv_path


def merge(args: argparse.Namespace) -> None:
    work_dir = _work_dir(args)
    manifest = json.loads(_prepare_manifest_path(work_dir).read_text())
    args.results_dir.mkdir(parents=True, exist_ok=True)
    merged_summaries = []
    merged_metrics: pd.DataFrame | None = None
    for template_name in args.prompt_templates:
        paths = [
            work_dir
            / "workers"
            / f"worker_{worker_id}_{template_name}.csv"
            for worker_id in range(args.num_workers)
        ]
        missing = [path for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"missing worker results: {missing}")
        # Tokens such as ``NA``, ``None``, ``null``, and ``nan`` are valid
        # standalone English vocabulary entries.  Pandas treats these strings
        # as missing values by default, which corrupts the concept name during
        # worker-result merging.
        frames = [_read_csv_preserving_strings(path) for path in paths]
        frame = pd.concat(frames, ignore_index=True)
        frame = _sort_concept_metric_frame(frame)
        merged_metrics = frame

        summary_frames = []
        for worker_id in range(args.num_workers):
            summary_path = (
                work_dir / "workers" / f"worker_{worker_id}_summary.csv"
            )
            summary_frame = _read_csv_preserving_strings(summary_path)
            summary_frames.append(
                summary_frame[summary_frame["prompt_template"] == template_name]
            )
        summary_frame = pd.concat(summary_frames, ignore_index=True)
        trials = int(summary_frame["n_trials"].sum())
        correct = int(summary_frame["n_correct"].sum())
        merged_summaries.append(
            {
                "prompt_template": template_name,
                "n_words": int(frame.shape[0]),
                "n_trials": trials,
                "n_correct": correct,
                "injected_accuracy": correct / trials,
                "injected_mean_correct_prob": float(
                    (
                        summary_frame["injected_mean_correct_prob"]
                        * summary_frame["n_trials"]
                    ).sum()
                    / trials
                ),
                "clean_mean_full_vocab_p_none": float(
                    summary_frame["clean_mean_full_vocab_p_none"].mean()
                ),
                "clean_exact_none_count": int(
                    summary_frame["clean_exact_none_count"].dropna().iloc[0]
                ),
                "clean_prompt_count": int(summary_frame["clean_prompt_count"].iloc[0]),
            }
        )
    assert merged_metrics is not None
    cluster_count = args.max_clusters or args.expected_clusters
    trials_per_word = (
        int(merged_summaries[0]["n_trials"])
        // int(merged_summaries[0]["n_words"])
    )
    position_count = trials_per_word // cluster_count
    position_labels = list(range(position_count))
    metrics_path = args.results_dir / "metrics.csv"
    merged_metrics.to_csv(metrics_path, index=False)
    write_metadata(
        args.results_dir,
        model_name=args.model,
        args=vars(args),
        seed=args.seed,
        date=datetime.now().strftime("%Y-%m-%d"),
        gpus=parse_gpu_groups(args.gpus),
        extra={
            **manifest,
            "protocol": (
                f"raw closed-set argmax over {position_labels[0]}-"
                f"{position_labels[-1]} and none"
            ),
            "position_labels": position_labels,
            "trials_per_word_per_prompt": trials_per_word,
            "summary": merged_summaries[0],
        },
    )
    selection_path = _write_screening_selection(args, merged_metrics, manifest)
    print(
        f"[merge] wrote {metrics_path} and {selection_path}",
        flush=True,
    )


def _forwarded_args(args: argparse.Namespace) -> list[str]:
    values = [
        "--model", args.model,
        "--cluster_csv", str(args.cluster_csv),
        "--results_dir", str(args.results_dir),
        "--dataset_dir", str(args.dataset_dir),
        "--screening_stage", args.screening_stage,
        "--work_dir", str(_work_dir(args)),
        "--logs_dir", str(_logs_dir(args)),
        "--baseline_mode", args.baseline_mode,
        *(
            ["--baseline_words_json", str(args.baseline_words_json)]
            if args.baseline_words_json is not None
            else []
        ),
        *(
            [
                "--calibration_concepts_json",
                str(args.calibration_concepts_json),
            ]
            if args.calibration_concepts_json is not None
            else []
        ),
        *(
            ["--candidate_words_csv", str(args.candidate_words_csv)]
            if args.candidate_words_csv is not None
            else []
        ),
        "--candidate_column", args.candidate_column,
        "--expected_candidate_count", str(args.expected_candidate_count),
        "--gpus", args.gpus,
        "--num_workers", str(args.num_workers),
        "--layer", str(args.layer),
        "--strength", str(args.strength),
        "--batch_size", str(args.batch_size),
        "--extraction_batch_size", str(args.extraction_batch_size),
        "--dtype", args.dtype,
        "--expected_clusters", str(args.expected_clusters),
        "--min_word_len", str(args.min_word_len),
        "--max_word_len", str(args.max_word_len),
        "--case_filter", args.case_filter,
        "--seed", str(args.seed),
        "--prompt_templates", *args.prompt_templates,
    ]
    if args.max_clusters is not None:
        values.extend(["--max_clusters", str(args.max_clusters)])
    if args.max_tokens is not None:
        values.extend(["--max_tokens", str(args.max_tokens)])
    if args.trust_remote_code:
        values.append("--trust_remote_code")
    if args.resume:
        values.append("--resume")
    return values


def _freeze_coarse_panel(args: argparse.Namespace) -> None:
    """Create the deterministic three-cluster calibration panel once."""
    source = args.cluster_csv.resolve()
    frame = pd.read_csv(source, keep_default_na=False)
    if len(frame) != 30:
        raise ValueError(
            f"coarse source has {len(frame)} clusters; "
            f"expected the frozen 30-cluster calibration bank: {source}"
        )
    indices = sorted(random.Random(args.seed).sample(range(len(frame)), 3))
    panel = frame.iloc[indices].copy().reset_index(drop=True)
    dataset_dir = _resolved_dataset_dir(args)
    panel_path = dataset_dir / "vocabulary" / "coarse_panel.csv"
    manifest_path = dataset_dir / "manifests" / "coarse_panel.json"
    if panel_path.exists() or manifest_path.exists():
        if not panel_path.exists() or not manifest_path.exists():
            raise FileExistsError("Coarse-panel artifacts are only partially present")
        existing = pd.read_csv(panel_path, keep_default_na=False)
        stored = json.loads(manifest_path.read_text())
        expected_keys = panel["cluster_key"].astype(str).tolist()
        if (
            not existing.equals(panel)
            or stored.get("seed") != args.seed
            or stored.get("source_file_sha256") != file_sha256(source)
            or stored.get("selected_row_indices") != indices
            or stored.get("selected_cluster_keys") != expected_keys
            or stored.get("panel_file_sha256") != file_sha256(panel_path)
        ):
            raise ValueError("Existing coarse-panel artifacts do not match this run")
        args.cluster_csv = panel_path
        args.expected_clusters = 3
        return
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(panel_path, index=False)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "source_file": str(source),
                "source_file_sha256": file_sha256(source),
                "selected_row_indices": indices,
                "selected_cluster_keys": panel["cluster_key"].astype(str).tolist(),
                "panel_file": str(panel_path.resolve()),
                "panel_file_sha256": file_sha256(panel_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    args.cluster_csv = panel_path
    args.expected_clusters = 3


def runner(args: argparse.Namespace) -> None:
    results_dir = args.results_dir.resolve()
    if (results_dir / "metrics.csv").exists() or (
        results_dir.exists()
        and not args.resume
        and any(path.is_file() for path in results_dir.rglob("*"))
    ):
        raise FileExistsError(
            f"Screening output already exists: {results_dir}; "
            "use a new model run root"
        )
    concepts_dir = _resolved_dataset_dir(args) / "concepts"
    published_selection = concepts_dir / (
        "shortlist.csv" if args.screening_stage == "coarse" else "selected.csv"
    )
    if published_selection.exists():
        raise FileExistsError(
            f"Screening selection already exists: {published_selection}"
        )
    if args.screening_stage == "coarse":
        _freeze_coarse_panel(args)
    gpus = parse_gpu_groups(args.gpus)
    if args.max_tokens is not None:
        _validate_worker_capacity(args.max_tokens, len(gpus))
    args.num_workers = len(gpus)
    script = str(Path(__file__).resolve())
    forwarded = _forwarded_args(args)
    prepare_env = build_worker_env(gpus[0])
    prepare_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    prepare_spec = WorkerSpec(
        name=f"prepare_gpu{gpus[0]}",
        command=[current_python(), script, "--role", "prepare", *forwarded],
        env=prepare_env,
    )
    run_worker_pool([prepare_spec], log_dir=_logs_dir(args))

    worker_specs = []
    for worker_id, gpu in enumerate(gpus):
        worker_env = build_worker_env(gpu)
        worker_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        worker_specs.append(
            WorkerSpec(
                name=f"gpu{gpu}_worker{worker_id}",
                command=[
                    current_python(), script, "--role", "worker", *forwarded,
                    "--worker_id", str(worker_id),
                ],
                env=worker_env,
            )
        )
    run_worker_pool(worker_specs, log_dir=_logs_dir(args))
    merge_args = parse_args(["--role", "merge", *forwarded])
    merge(merge_args)
    shutil.rmtree(_work_dir(args), ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    require_valid_cluster_split(args.cluster_csv)
    if args.role == "runner":
        runner(args)
    elif args.role == "prepare":
        prepare(args)
    elif args.role == "worker":
        worker(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
