"""Balanced similar-token cluster search.

Candidate tokens come from the clean position prior; each seed token is grown
into ten-token groups of embedding neighbours, and every group is scored with
the ``token_localization`` prompt across several orderings. Groups whose tokens
receive similar average selection probabilities are kept, and the result is
written to data/dataset/<model_slug>/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F

from .data_parallel import (
    WorkerSpec,
    build_worker_env,
    current_python,
    parse_gpu_groups,
    run_worker_pool,
)
from .cluster_split import (
    CANDIDATE_BANKS,
    file_sha256,
    write_cluster_split_manifest,
)
from .model import HookedModel, ModelConfig
from .prompts import PromptManager
from .vocab_token_clean_prior import decode_word_start_token


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_BASE = Path("data/dataset")
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Search for balanced similar-token clusters and write a per-model dataset."
    )
    p.add_argument("--role", choices=["runner", "worker"], default="runner")
    p.add_argument("--model", required=True, help="HF id or local checkpoint path")
    p.add_argument("--prior_summary", type=Path, required=True,
                   help="CSV produced by vocab_token_clean_prior (summary_by_token.csv)")
    p.add_argument("--dataset_base", type=Path, default=DEFAULT_DATASET_BASE,
                   help="Root dataset directory; model slug is appended as a sub-dir")
    p.add_argument("--dataset_name", default=None,
                   help="Override the model-slug sub-dir name (default: basename of --model)")
    p.add_argument(
        "--gpus",
        default="0,1,2,3,4",
        help=(
            "comma-separated GPU ids, or semicolon-separated multi-GPU "
            "worker groups such as 2,3;5,6"
        ),
    )
    p.add_argument("--worker_id", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--num_choices", type=int, default=10)
    p.add_argument(
        "--position_index_start",
        choices=[0],
        type=int,
        default=0,
        help="First user-facing TOKEN label (default: 0).",
    )
    p.add_argument("--seed", type=int, default=20260623)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--candidate_min_len", type=int, default=3)
    p.add_argument("--candidate_max_len", type=int, default=12)
    p.add_argument("--candidate_min_mean_prior", type=float, default=0.0)
    p.add_argument("--candidate_max_mean_prior", type=float, default=0.25)
    p.add_argument("--candidate_max_argmax_rate", type=float, default=0.35)
    p.add_argument("--candidate_max_position_range", type=float, default=999.0)
    p.add_argument(
        "--candidate_bank",
        choices=CANDIDATE_BANKS,
        required=True,
        help=(
            "Select one deterministic disjoint candidate bank. The first "
            "--candidate_token_count ranked tokens are shuffled in blocks of "
            "four and assigned to calibration, train, validation, and test."
        ),
    )
    p.add_argument(
        "--candidate_token_count",
        type=int,
        default=4000,
        help="Number of ranked tokens divided among the four candidate banks",
    )
    p.add_argument(
        "--work_root",
        type=Path,
        default=Path("tmp/auto_workflow"),
        help="Root for resumable cluster-search intermediates",
    )
    p.add_argument(
        "--logs_root",
        type=Path,
        default=Path("logs/auto_workflow"),
        help="Root for cluster-search worker logs",
    )
    p.add_argument("--max_seed_tokens", type=int, default=800)
    p.add_argument("--neighbor_pool", type=int, default=72)
    p.add_argument("--variants_per_seed", type=int, default=4)
    p.add_argument("--variant_stride", type=int, default=5)
    p.add_argument("--permutations_per_cluster", type=int, default=30)
    p.add_argument("--keep_top_clusters", type=int, default=100)
    p.add_argument("--dataset_clusters", type=int, default=30)
    p.add_argument("--max_abs_deviation", type=float, default=0.035)
    p.add_argument("--max_prob_range", type=float, default=0.08)
    p.add_argument("--prompt_preamble", choices=["none", "user", "system"], default="system")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--progress_every", type=int, default=50)
    args = p.parse_args(argv)
    bank_count = len(CANDIDATE_BANKS)
    if args.candidate_bank and (
        args.candidate_token_count < bank_count
        or args.candidate_token_count % bank_count
    ):
        p.error("--candidate_token_count must be a positive multiple of four")
    return args


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_gpus(gpus: str) -> list[str]:
    if gpus == "auto":
        count = torch.cuda.device_count()
        return [str(i) for i in range(count)] or ["cpu"]
    return parse_gpu_groups(gpus)


def dataset_dir(args) -> Path:
    slug = args.dataset_name or Path(args.model).name.lower()
    base = resolve_path(args.dataset_base)
    return base / slug


def model_slug(args) -> str:
    """Return the one stable model slug used by every workflow root."""
    return str(args.dataset_name or Path(args.model).name.lower())


def bank_work_dir(out_dir: Path, args) -> Path:
    """Return the temporary artifact directory for one cluster bank."""
    del out_dir
    return (
        resolve_path(args.work_root)
        / model_slug(args)
        / "cluster_search"
        / args.candidate_bank
    )


def worker_dir(out_dir: Path, args) -> Path:
    """Return an isolated worker directory outside the dataset tree."""
    return bank_work_dir(out_dir, args) / "workers"


def worker_log_dir(out_dir: Path, args) -> Path:
    """Return the canonical per-model cluster-search log directory."""
    del out_dir
    return (
        resolve_path(args.logs_root)
        / model_slug(args)
        / "cluster_search"
        / args.candidate_bank
    )


def output_filenames(args) -> dict[str, str]:
    """Return stable, semantic paths for one registered cluster bank."""
    bank = args.candidate_bank
    if bank not in CANDIDATE_BANKS:
        raise ValueError(f"candidate_bank must be one of {CANDIDATE_BANKS}")
    return {
        "candidate_tokens": f"candidates/{bank}.csv",
        "cluster_pool": "pool.csv",
        "clusters": f"clusters/{bank}.csv",
        "choice_banks": "choice_banks.csv",
        "target_tokens": "target_tokens.csv",
        "summary": f"manifests/cluster_search/{bank}.json",
    }


def output_paths(out_dir: Path, args) -> dict[str, Path]:
    """Resolve published outputs and non-canonical working artifacts."""
    names = output_filenames(args)
    work_dir = bank_work_dir(out_dir, args)
    paths = {
        "candidate_tokens": out_dir / names["candidate_tokens"],
        "clusters": out_dir / names["clusters"],
        "summary": out_dir / names["summary"],
        "cluster_pool": work_dir / names["cluster_pool"],
        "choice_banks": work_dir / names["choice_banks"],
        "target_tokens": work_dir / names["target_tokens"],
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    return paths


def cluster_split_settings(args) -> dict[str, Any]:
    """Return the shared settings fingerprint recorded by every bank."""
    return {
        "model": str(args.model),
        "source_prior_sha256": file_sha256(resolve_path(args.prior_summary)),
        "seed": int(args.seed),
        "candidate_token_count": int(args.candidate_token_count),
        "candidate_ranking": [
            "absolute distance from uniform prior ascending",
            "position score range ascending",
            "lowercase word ascending",
            "token ID ascending",
        ],
        "candidate_filters": {
            "min_len": int(args.candidate_min_len),
            "max_len": int(args.candidate_max_len),
            "min_mean_prior": float(args.candidate_min_mean_prior),
            "max_mean_prior": float(args.candidate_max_mean_prior),
            "max_argmax_rate": float(args.candidate_max_argmax_rate),
            "max_position_range": float(args.candidate_max_position_range),
        },
        "num_choices": int(args.num_choices),
        "position_index_start": int(getattr(args, "position_index_start", 0)),
        "dataset_clusters": int(args.dataset_clusters),
        "cluster_search": {
            "max_seed_tokens": int(args.max_seed_tokens),
            "neighbor_pool": int(args.neighbor_pool),
            "variants_per_seed": int(args.variants_per_seed),
            "variant_stride": int(args.variant_stride),
            "permutations_per_cluster": int(args.permutations_per_cluster),
            "keep_top_clusters": int(args.keep_top_clusters),
            "max_abs_deviation": float(args.max_abs_deviation),
            "max_prob_range": float(args.max_prob_range),
            "prompt_preamble": str(args.prompt_preamble),
        },
    }


# ---------------------------------------------------------------------------
# Candidate token loading (pure CSV — model-agnostic)
# ---------------------------------------------------------------------------

def _safe_float(row: Any, key: str, default: float = 0.0) -> float:
    v = getattr(row, key, default)
    return default if pd.isna(v) else float(v)


def _safe_int(row: Any, key: str, default: int = 0) -> int:
    v = getattr(row, key, default)
    return default if pd.isna(v) else int(v)


def split_candidate_banks(
    tokens: list[dict[str, Any]],
    *,
    candidate_token_count: int,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Split ranked candidates into deterministic, disjoint four-token banks."""
    bank_count = len(CANDIDATE_BANKS)
    if candidate_token_count < bank_count or candidate_token_count % bank_count:
        raise ValueError("candidate_token_count must be a positive multiple of four")
    if len(tokens) < candidate_token_count:
        raise ValueError(
            f"Only {len(tokens)} eligible candidate tokens found; "
            f"need {candidate_token_count} for four disjoint banks"
        )

    rng = random.Random(seed)
    banks: dict[str, list[dict[str, Any]]] = {
        name: [] for name in CANDIDATE_BANKS
    }
    for block_index, start in enumerate(
        range(0, candidate_token_count, bank_count)
    ):
        block = [
            {
                **tokens[index],
                "candidate_rank": index + 1,
                "candidate_block": block_index,
            }
            for index in range(start, start + bank_count)
        ]
        rng.shuffle(block)
        for bank, token in zip(CANDIDATE_BANKS, block):
            token["candidate_bank"] = bank
            token["candidate_bank_index"] = len(banks[bank])
            banks[bank].append(token)
    return banks


def load_candidate_tokens(args, tokenizer) -> list[dict[str, Any]]:
    prior = pd.read_csv(resolve_path(args.prior_summary))
    required = {"word", "word_lower", "word_len", "mean_number_softmax"}
    missing = sorted(required.difference(prior.columns))
    if missing:
        raise ValueError(f"Prior summary missing columns: {missing}")

    tokens: list[dict] = []
    seen: set[str] = set()
    seen_token_ids: set[int] = set()
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])

    for row in prior.itertuples(index=False):
        word = str(getattr(row, "word"))
        lower = str(getattr(row, "word_lower"))
        if lower in seen or word != lower or not word.isalpha():
            continue
        word_len = _safe_int(row, "word_len", len(word))
        if not (args.candidate_min_len <= word_len <= args.candidate_max_len):
            continue
        mean_prior = _safe_float(row, "mean_number_softmax")
        if not (args.candidate_min_mean_prior <= mean_prior <= args.candidate_max_mean_prior):
            continue
        if _safe_float(row, "position_score_range") > args.candidate_max_position_range:
            continue
        n = max(_safe_int(row, "n", args.num_choices), 1)
        argmax_count = _safe_int(row, "argmax_count", 0)
        argmax_rate = _safe_float(row, "argmax_rate", argmax_count / n)
        if argmax_rate > args.candidate_max_argmax_rate:
            continue
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(ids) != 1:
            continue
        tid = int(ids[0])
        if tid in special_ids or tid in seen_token_ids:
            continue
        decoded = decode_word_start_token(tokenizer, tid)
        if decoded is None or decoded[1] != word:
            continue
        token_text, _ = decoded
        tokens.append({
            "word": word, "word_lower": lower, "word_len": word_len,
            "token_id": tid,
            "token_text": token_text,
            "mean_clean_prior": mean_prior,
            "argmax_count": argmax_count, "argmax_rate": argmax_rate,
            "position_score_range": _safe_float(row, "position_score_range"),
        })
        seen.add(lower)
        seen_token_ids.add(tid)

    if len(tokens) < args.num_choices:
        raise ValueError(f"Only {len(tokens)} candidate tokens found (need {args.num_choices})")
    target_prob = 1.0 / args.num_choices
    tokens.sort(key=lambda t: (
        abs(t["mean_clean_prior"] - target_prob),
        t["position_score_range"],
        t["word_lower"], t["token_id"],
    ))
    candidate_bank = getattr(args, "candidate_bank", None)
    if candidate_bank is None:
        return tokens
    banks = split_candidate_banks(
        tokens,
        candidate_token_count=args.candidate_token_count,
        seed=args.seed,
    )
    return banks[candidate_bank]


# ---------------------------------------------------------------------------
# Pure-math helpers (unchanged from source)
# ---------------------------------------------------------------------------

def cluster_key(words: list[str]) -> str:
    payload = "|".join(sorted(w.lower() for w in words))
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def entropy_normalized(probs: list[float]) -> float:
    total = sum(probs)
    if total <= 0:
        return float("nan")
    norm = [max(p / total, 1e-300) for p in probs]
    return -sum(p * math.log2(p) for p in norm) / math.log2(len(probs))


def mean_pairwise_similarity(
    unit_embeddings: torch.Tensor, indices: list[int]
) -> tuple[float, float]:
    if len(indices) < 2:
        return float("nan"), float("nan")
    sel = unit_embeddings.index_select(
        0, torch.tensor(indices, device=unit_embeddings.device, dtype=torch.long)
    )
    sim = sel @ sel.T
    upper = sim[torch.triu(torch.ones_like(sim, dtype=torch.bool), diagonal=1)]
    return float(upper.mean().cpu()), float(upper.min().cpu())


# ---------------------------------------------------------------------------
# Cluster construction (unchanged from source)
# ---------------------------------------------------------------------------

def rotate(values: list[Any], offset: int) -> list[Any]:
    offset %= len(values)
    return values[offset:] + values[:offset]


def nearest_neighbors_for_seed(
    seed_idx: int,
    candidates: list[dict[str, Any]],
    unit_embeddings: torch.Tensor,
    args,
) -> list[dict[str, Any]]:
    sims = unit_embeddings @ unit_embeddings[seed_idx]
    top_k = min(len(candidates), args.neighbor_pool + 1)
    values, indices = torch.topk(sims, k=top_k)
    rows: list[dict] = []
    used = {candidates[seed_idx]["word_lower"]}
    for val, idx in zip(values.cpu().tolist(), indices.cpu().tolist()):
        cand = candidates[int(idx)]
        lower = cand["word_lower"]
        if lower in used:
            continue
        rows.append({**cand, "similarity_to_seed": float(val)})
        used.add(lower)
        if len(rows) >= args.neighbor_pool:
            break
    return rows


def build_cluster_variants(
    seed: dict[str, Any],
    seed_idx: int,
    neighbor_rows: list[dict[str, Any]],
    candidate_index_by_lower: dict[str, int],
    unit_embeddings: torch.Tensor,
    args,
) -> list[dict[str, Any]]:
    variants: list[dict] = []
    seen_keys: set[str] = set()
    for variant_idx in range(args.variants_per_seed):
        offset = variant_idx * args.variant_stride
        neighbors = neighbor_rows[offset: offset + args.num_choices - 1]
        if len(neighbors) != args.num_choices - 1:
            break
        token_rows = [seed, *neighbors]
        words = [r["word"] for r in token_rows]
        key = cluster_key(words)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        indices = [candidate_index_by_lower[r["word_lower"]] for r in token_rows]
        mean_sim, min_sim = mean_pairwise_similarity(unit_embeddings, indices)
        variants.append({
            "cluster_key": key,
            "seed_word": seed["word"],
            "seed_word_lower": seed["word_lower"],
            "seed_token_id": int(seed["token_id"]),
            "seed_candidate_idx": int(seed_idx),
            "variant_idx": int(variant_idx),
            "neighbor_offset": int(offset),
            "choices": words,
            "token_rows": token_rows,
            "candidate_indices": indices,
            "mean_pairwise_similarity": mean_sim,
            "min_pairwise_similarity": min_sim,
            "mean_seed_neighbor_similarity": float(
                sum(r["similarity_to_seed"] for r in neighbors) / len(neighbors)
            ),
            "min_seed_neighbor_similarity": float(
                min(r["similarity_to_seed"] for r in neighbors)
            ),
        })
    return variants


def cluster_permutations(choices: list[str], count: int, seed: int) -> list[list[str]]:
    count = max(count, len(choices))
    rng = random.Random(seed)
    perms: list[list[str]] = []
    seen: set[tuple] = set()
    for offset in range(len(choices)):
        perm = tuple(rotate(choices, offset))
        perms.append(list(perm))
        seen.add(perm)
        if len(perms) >= count:
            return perms
    while len(perms) < count:
        perm = tuple(rng.sample(choices, len(choices)))
        if perm not in seen:
            seen.add(perm)
            perms.append(list(perm))
    return perms


# ---------------------------------------------------------------------------
# Prompt building — reuses PromptManager localization templates
# ---------------------------------------------------------------------------

def build_task(
    prompt_manager: PromptManager,
    cluster: dict[str, Any],
    choices: list[str],
    permutation_idx: int,
    args,
) -> dict[str, Any]:
    """Render one permutation of a cluster into a scorable task dict.

    Uses the zero-based ``token_localization`` clean position-prior prompt. Asserts len(choices)==num_choices to catch
    mis-configured clusters early.
    """
    assert len(choices) == args.num_choices, (
        f"Cluster {cluster['cluster_key']!r} has {len(choices)} choices, "
        f"expected {args.num_choices}"
    )

    if int(getattr(args, "position_index_start", 0)) != 0:
        raise ValueError("token_localization labels its slots TOKEN 0..N-1")
    rendered = prompt_manager.render(
        "token_localization",
        choices,
        preamble=args.prompt_preamble,
    )

    # number_tokens: {1: token_id_of_"1", 2: token_id_of_"2", ...}
    # PromptManager returns string keys; convert to int.
    number_tokens = {int(k): v for k, v in rendered.answer_token_by_choice.items()}

    # Verify that each choice's token id matches what was recorded during
    # candidate selection (catches tokenizer / context mismatches).
    token_by_lower = {r["word_lower"]: int(r["token_id"]) for r in cluster["token_rows"]}
    for record in rendered.records:
        word = str(record["text"])
        expected_id = token_by_lower[word.lower()]
        # RenderedPrompt.records have "token_ids": [int] (single-token items)
        actual_id = record["token_ids"][0]
        if actual_id != expected_id:
            raise ValueError(
                f"Token id mismatch for {word!r}: expected {expected_id}, got {actual_id}"
            )

    return {
        "cluster_key": cluster["cluster_key"],
        "choices": choices,
        "permutation_idx": int(permutation_idx),
        "input_ids": rendered.input_ids,   # [1, seq_len]
        "number_tokens": number_tokens,
    }


# ---------------------------------------------------------------------------
# Batched scoring — uses model.forward_logits (no attention_mask needed)
# ---------------------------------------------------------------------------

def _padded_forward(
    model: HookedModel,
    tasks: list[dict[str, Any]],
) -> torch.Tensor:
    """Right-pad a heterogeneous batch, run forward, return last-real-token logits.

    Right-padding is safe with causal attention: the last real token at
    position len-1 only attends to positions 0..len-1, none of which are pad
    tokens, so the logit there is identical to a single-sequence forward pass.
    """
    device = torch.device(model.bridge.cfg.device)
    pad_id = int(model.tokenizer.pad_token_id or model.tokenizer.eos_token_id)
    lengths = [int(t["input_ids"].shape[1]) for t in tasks]
    max_len = max(lengths)
    input_ids = torch.full((len(tasks), max_len), fill_value=pad_id, dtype=torch.long, device=device)
    for row, (task, length) in enumerate(zip(tasks, lengths)):
        input_ids[row, :length] = task["input_ids"][0].to(device)

    logits = model.forward_logits(input_ids)            # [batch, max_len, vocab]
    last_idx = torch.tensor([l - 1 for l in lengths], device=device, dtype=torch.long)
    rows = torch.arange(len(tasks), device=device)
    return logits[rows, last_idx].detach()              # [batch, vocab]


# ---------------------------------------------------------------------------
# Accumulator logic (unchanged from source)
# ---------------------------------------------------------------------------

def init_accumulator(cluster: dict[str, Any], args) -> dict[str, Any]:
    lowers = [w.lower() for w in cluster["choices"]]
    start = int(getattr(args, "position_index_start", 0))
    labels = range(start, start + args.num_choices)
    return {
        "cluster": cluster,
        "prompt_count": 0,
        "token_sums":     {l: 0.0 for l in lowers},
        "token_counts":   {l: 0   for l in lowers},
        "selected_counts":{l: 0   for l in lowers},
        "position_sums":  {i: 0.0 for i in labels},
        "position_counts":{i: 0   for i in labels},
        "prompt_entropy_sum": 0.0,
        "best_prompt": None,
    }


def consume_scores(
    pending: list[dict[str, Any]],
    logits: torch.Tensor,
    accumulators: dict[str, dict[str, Any]],
    args,
) -> None:
    position_index_start = int(getattr(args, "position_index_start", 0))
    ordered = list(
        range(position_index_start, position_index_start + args.num_choices)
    )
    answer_ids = torch.tensor(
        [pending[0]["number_tokens"][i] for i in ordered],
        device=logits.device, dtype=torch.long,
    )
    cand_logits = logits.float().index_select(dim=1, index=answer_ids)
    cand_probs  = torch.softmax(cand_logits, dim=-1).cpu()
    predictions = torch.argmax(cand_logits, dim=-1).cpu().tolist()
    target_prob = 1.0 / args.num_choices

    for row, task in enumerate(pending):
        probs   = [float(v) for v in cand_probs[row].tolist()]
        choices = task["choices"]
        acc     = accumulators[task["cluster_key"]]
        pred_offset = int(predictions[row])
        pred_pos = ordered[pred_offset]
        entropy = entropy_normalized(probs)
        max_abs = max(abs(p - target_prob) for p in probs)
        prob_range = max(probs) - min(probs)
        record = {
            "permutation_idx": int(task["permutation_idx"]),
            "choices": choices,
            "candidate_probs": probs,
            "max_abs_deviation": max_abs,
            "prob_range": prob_range,
            "entropy_normalized": entropy,
            "argmax_position": pred_pos,
        }
        best = acc["best_prompt"]
        if best is None or (max_abs, prob_range, -entropy) < (
            best["max_abs_deviation"], best["prob_range"], -best["entropy_normalized"]
        ):
            acc["best_prompt"] = record

        acc["prompt_count"] += 1
        acc["prompt_entropy_sum"] += entropy
        for pos, (word, prob) in enumerate(
            zip(choices, probs),
            start=position_index_start,
        ):
            lower = word.lower()
            acc["token_sums"][lower]   += prob
            acc["token_counts"][lower] += 1
            acc["position_sums"][pos]  += prob
            acc["position_counts"][pos]+= 1
            if pos == pred_pos:
                acc["selected_counts"][lower] += 1


def score_pending(
    pending: list[dict[str, Any]],
    model: HookedModel,
    accumulators: dict[str, dict[str, Any]],
    args,
) -> None:
    if not pending:
        return
    logits = _padded_forward(model, pending)
    consume_scores(pending, logits, accumulators, args)
    pending.clear()


# ---------------------------------------------------------------------------
# Accumulator finalization (unchanged from source)
# ---------------------------------------------------------------------------

def finalize_accumulator(
    acc: dict[str, Any], args
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cluster = acc["cluster"]
    choices = cluster["choices"]
    target_prob = 1.0 / args.num_choices
    token_means = {
        lower: acc["token_sums"][lower] / max(acc["token_counts"][lower], 1)
        for lower in acc["token_sums"]
    }
    token_mean_vals = [token_means[w.lower()] for w in choices]
    token_abs = [abs(v - target_prob) for v in token_mean_vals]
    position_index_start = int(getattr(args, "position_index_start", 0))
    position_means = {
        str(pos): acc["position_sums"][pos] / max(acc["position_counts"][pos], 1)
        for pos in range(
            position_index_start,
            position_index_start + args.num_choices,
        )
    }
    best = acc["best_prompt"]
    if best is None:
        raise ValueError(f"Cluster {cluster['cluster_key']} has no scored prompts")

    row = {
        "cluster_key":  cluster["cluster_key"],
        "seed_word":    cluster["seed_word"],
        "seed_word_lower": cluster["seed_word_lower"],
        "seed_token_id": int(cluster["seed_token_id"]),
        "variant_idx":  int(cluster["variant_idx"]),
        "neighbor_offset": int(cluster["neighbor_offset"]),
        "num_choices":  args.num_choices,
        "choices":      json.dumps(choices),
        "best_choices": json.dumps(best["choices"]),
        "best_candidate_probs": json.dumps(best["candidate_probs"]),
        "best_argmax_position": int(best["argmax_position"]),
        "best_prompt_max_abs_deviation": float(best["max_abs_deviation"]),
        "best_prompt_prob_range": float(best["prob_range"]),
        "best_prompt_entropy_normalized": float(best["entropy_normalized"]),
        "token_mean_candidate_probs": json.dumps(token_means),
        "token_selected_counts": json.dumps(acc["selected_counts"]),
        "position_mean_candidate_probs": json.dumps(position_means),
        "prompt_count": int(acc["prompt_count"]),
        "avg_prompt_entropy_normalized": float(
            acc["prompt_entropy_sum"] / acc["prompt_count"]
        ),
        "token_mean_max_abs_deviation": float(max(token_abs)),
        "token_mean_prob_range": float(max(token_mean_vals) - min(token_mean_vals)),
        "token_mean_l1_deviation": float(sum(token_abs)),
        "token_mean_l2_deviation": float(
            math.sqrt(sum((v - target_prob) ** 2 for v in token_mean_vals))
        ),
        "token_mean_entropy_normalized": float(entropy_normalized(token_mean_vals)),
        "mean_pairwise_similarity": float(cluster["mean_pairwise_similarity"]),
        "min_pairwise_similarity":  float(cluster["min_pairwise_similarity"]),
        "mean_seed_neighbor_similarity": float(cluster["mean_seed_neighbor_similarity"]),
        "min_seed_neighbor_similarity":  float(cluster["min_seed_neighbor_similarity"]),
    }

    best_prob_by_lower = {
        w.lower(): p for w, p in zip(best["choices"], best["candidate_probs"])
    }
    token_rows: list[dict] = []
    token_by_lower = {r["word_lower"]: r for r in cluster["token_rows"]}
    for token_order, word in enumerate(choices):
        lower = word.lower()
        token = token_by_lower[lower]
        token_rows.append({
            "cluster_key":  cluster["cluster_key"],
            "token_order":  int(token_order),
            "word":         word,
            "word_lower":   lower,
            "token_id":     int(token["token_id"]),
            "token_text":   token["token_text"],
            "word_len":     int(token["word_len"]),
            "mean_clean_prior":  float(token["mean_clean_prior"]),
            "argmax_rate":       float(token["argmax_rate"]),
            "position_score_range": float(token["position_score_range"]),
            "token_mean_candidate_prob": float(token_means[lower]),
            "best_prompt_candidate_prob": float(best_prob_by_lower[lower]),
            "selected_count": int(acc["selected_counts"][lower]),
            "prompt_count":   int(acc["prompt_count"]),
            "similarity_to_seed": float(token.get("similarity_to_seed", 1.0)),
        })
    return row, token_rows


# ---------------------------------------------------------------------------
# Worker: loads model + embeddings, scores all clusters for its shard
# ---------------------------------------------------------------------------

def run_worker(args) -> None:
    t0 = time.monotonic()
    out_dir = dataset_dir(args)
    output_worker_dir = worker_dir(out_dir, args)
    output_worker_dir.mkdir(parents=True, exist_ok=True)

    model = HookedModel(ModelConfig(
        name=args.model, device=args.device, dtype=args.dtype
    ))
    prompt_manager = PromptManager(model.tokenizer)

    # Candidate tokens use the model's tokenizer; loaded identically to source.
    candidates = load_candidate_tokens(args, model.tokenizer)
    if args.worker_id == 0:
        candidate_path = output_paths(out_dir, args)["candidate_tokens"]
        pd.DataFrame(candidates).to_csv(candidate_path, index=False)
    seed_candidates = candidates[: args.max_seed_tokens]
    shard_indices = [i for i in range(len(seed_candidates)) if i % args.num_workers == args.worker_id]
    print(
        f"worker_id={args.worker_id}  candidates={len(candidates)} "
        f"seed_shard={len(shard_indices)}/{len(seed_candidates)} "
        f"choices={args.num_choices}  permutations={args.permutations_per_cluster}",
        flush=True,
    )

    # Input embedding matrix — TransformerLens exposes this as bridge.embed.W_E
    # with shape [vocab, d_model].
    with torch.inference_mode():
        W_E = model.bridge.embed.W_E.detach().float()
        candidate_ids = torch.tensor(
            [r["token_id"] for r in candidates],
            device=W_E.device, dtype=torch.long,
        )
        unit_embeddings = F.normalize(W_E.index_select(0, candidate_ids), dim=-1)

    candidate_index_by_lower = {r["word_lower"]: i for i, r in enumerate(candidates)}

    accumulators: dict[str, dict] = {}
    pending: list[dict] = []
    local_seen: set[str] = set()
    proposed = 0
    scored_prompts = 0

    for local_count, seed_idx in enumerate(shard_indices, start=1):
        seed = seed_candidates[seed_idx]
        global_idx = candidate_index_by_lower[seed["word_lower"]]
        neighbors = nearest_neighbors_for_seed(global_idx, candidates, unit_embeddings, args)
        variants  = build_cluster_variants(
            seed, global_idx, neighbors, candidate_index_by_lower, unit_embeddings, args
        )
        for cluster in variants:
            if cluster["cluster_key"] in local_seen:
                continue
            local_seen.add(cluster["cluster_key"])
            accumulators[cluster["cluster_key"]] = init_accumulator(cluster, args)
            perm_seed = args.seed + int(cluster["seed_token_id"]) * 1009 + int(cluster["variant_idx"])
            for perm_idx, choices in enumerate(
                cluster_permutations(cluster["choices"], args.permutations_per_cluster, perm_seed)
            ):
                pending.append(build_task(prompt_manager, cluster, choices, perm_idx, args))
                scored_prompts += 1
                if len(pending) >= args.batch_size:
                    score_pending(pending, model, accumulators, args)
            proposed += 1
        if args.progress_every and local_count % args.progress_every == 0:
            score_pending(pending, model, accumulators, args)
            print(
                f"seeds={local_count}/{len(shard_indices)}  clusters={proposed}  "
                f"prompts={scored_prompts}  elapsed={time.monotonic()-t0:.1f}s",
                flush=True,
            )

    score_pending(pending, model, accumulators, args)

    cluster_rows: list[dict] = []
    token_rows_all: list[dict] = []
    for acc in accumulators.values():
        c_row, t_rows = finalize_accumulator(acc, args)
        cluster_rows.append(c_row)
        token_rows_all.extend(t_rows)

    pd.DataFrame(cluster_rows).to_csv(
        output_worker_dir / f"worker_{args.worker_id}_scored_clusters.csv", index=False
    )
    pd.DataFrame(token_rows_all).to_csv(
        output_worker_dir / f"worker_{args.worker_id}_cluster_tokens.csv", index=False
    )
    print(
        f"worker {args.worker_id} done: {len(cluster_rows)} clusters, "
        f"{scored_prompts} prompts, elapsed={time.monotonic()-t0:.1f}s",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Combine worker outputs → dataset files
# ---------------------------------------------------------------------------

def _read_csvs(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths if p.exists() and p.stat().st_size]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


class InsufficientDisjointClustersError(ValueError):
    """Raised when a ranked pool cannot supply the requested disjoint set."""


def select_choice_token_disjoint_clusters(
    clusters: pd.DataFrame,
    token_rows: pd.DataFrame,
    *,
    count: int,
    num_choices: int,
) -> pd.DataFrame:
    """Greedily select ranked clusters without reusing a choice token ID."""
    if count <= 0:
        raise ValueError("count must be positive")
    required_token_columns = {"cluster_key", "token_id"}
    missing = sorted(required_token_columns.difference(token_rows.columns))
    if missing:
        raise ValueError(f"Cluster-token rows missing columns: {missing}")

    token_ids_by_cluster = {
        str(cluster_key): {int(token_id) for token_id in group["token_id"]}
        for cluster_key, group in token_rows.groupby("cluster_key", sort=False)
    }
    accepted_indices: list[Any] = []
    used_token_ids: set[int] = set()
    for index, row in clusters.iterrows():
        key = str(row["cluster_key"])
        token_ids = token_ids_by_cluster.get(key, set())
        if len(token_ids) != num_choices:
            raise ValueError(
                f"Cluster {key!r} has {len(token_ids)} unique choice token IDs; "
                f"expected {num_choices}"
            )
        if used_token_ids.isdisjoint(token_ids):
            accepted_indices.append(index)
            used_token_ids.update(token_ids)
            if len(accepted_indices) == count:
                break

    if len(accepted_indices) != count:
        raise InsufficientDisjointClustersError(
            f"Only {len(accepted_indices)} choice-token-disjoint clusters are "
            f"available; need {count}. Expand the cluster search pool and rerun."
        )
    return clusters.loc[accepted_indices].copy().reset_index(drop=True)


def select_disjoint_clusters_with_full_pool_fallback(
    clusters: pd.DataFrame,
    qualified: pd.DataFrame,
    token_rows: pd.DataFrame,
    *,
    count: int,
    num_choices: int,
) -> pd.DataFrame:
    """Prefer quality-qualified rows, then retry the complete ranking."""
    if len(qualified) < count:
        source = clusters
    else:
        source = qualified
    try:
        return select_choice_token_disjoint_clusters(
            source,
            token_rows,
            count=count,
            num_choices=num_choices,
        )
    except InsufficientDisjointClustersError:
        if source is clusters:
            raise
        return select_choice_token_disjoint_clusters(
            clusters,
            token_rows,
            count=count,
            num_choices=num_choices,
        )


def write_choice_banks(out_dir: Path, selected: pd.DataFrame, token_rows: pd.DataFrame, args) -> None:
    banks: list[dict] = []
    targets: list[dict] = []
    bank_idx = 0
    position_index_start = int(getattr(args, "position_index_start", 0))
    for cluster_rank, row in enumerate(selected.itertuples(index=False), start=1):
        base_choices = json.loads(row.choices)
        assert len(base_choices) == args.num_choices, (
            f"Cluster {row.cluster_key!r} stored {len(base_choices)} choices, "
            f"expected {args.num_choices}"
        )
        token_mean_probs = json.loads(row.token_mean_candidate_probs)
        cluster_tokens = token_rows[token_rows["cluster_key"] == row.cluster_key]
        token_by_lower = {
            str(t.word_lower): t for t in cluster_tokens.itertuples(index=False)
        }
        for rotation in range(args.num_choices):
            choices = rotate(base_choices, rotation)
            for target_pos, target_tok in enumerate(
                choices,
                start=position_index_start,
            ):
                lower = target_tok.lower()
                tr = token_by_lower.get(lower)
                mean_prior = float(getattr(tr, "mean_clean_prior", 0.0)) if tr else 0.0
                banks.append({
                    "bank_idx":        bank_idx,
                    "bank_mode":       "balanced_similar_cluster",
                    "token_category":  "balanced_similar",
                    "cluster_rank":    cluster_rank,
                    "cluster_rotation": rotation,
                    "cluster_key":     row.cluster_key,
                    "num_choices":     args.num_choices,
                    "target_token":    target_tok,
                    "target_token_lower": lower,
                    "target_mean_clean_prior": mean_prior,
                    "target_position": target_pos,
                    "choices":         json.dumps(choices),
                    "matched_tokens":  json.dumps([
                        {
                            "choice_position": pos,
                            "match_token": c,
                            "match_token_lower": c.lower(),
                            "token_mean_candidate_prob": token_mean_probs[c.lower()],
                        }
                        for pos, c in enumerate(
                            choices,
                            start=position_index_start,
                        )
                        if pos != target_pos
                    ]),
                    "token_mean_candidate_probs": json.dumps(token_mean_probs),
                    "best_choices":    row.best_choices,
                    "best_candidate_probs": row.best_candidate_probs,
                    "best_prompt_max_abs_deviation": float(row.best_prompt_max_abs_deviation),
                    "best_prompt_prob_range": float(row.best_prompt_prob_range),
                    "token_mean_max_abs_deviation": float(row.token_mean_max_abs_deviation),
                    "token_mean_prob_range": float(row.token_mean_prob_range),
                    "mean_pairwise_similarity": float(row.mean_pairwise_similarity),
                })
                bank_idx += 1
                targets.append({
                    "cluster_rank": cluster_rank,
                    "cluster_key":  row.cluster_key,
                    "word":         target_tok,
                    "word_lower":   lower,
                    "mean_clean_prior": mean_prior,
                    "token_category": "balanced_similar",
                })
    paths = output_paths(out_dir, args)
    pd.DataFrame(banks).to_csv(paths["choice_banks"], index=False)
    pd.DataFrame(targets).drop_duplicates(["cluster_key", "word_lower"]).to_csv(
        paths["target_tokens"], index=False
    )


def combine_worker_outputs(out_dir: Path, num_workers: int, args) -> None:
    output_worker_dir = worker_dir(out_dir, args)
    paths = output_paths(out_dir, args)
    cluster_paths = [output_worker_dir / f"worker_{i}_scored_clusters.csv" for i in range(num_workers)]
    token_paths   = [output_worker_dir / f"worker_{i}_cluster_tokens.csv"  for i in range(num_workers)]
    missing = [p for p in [*cluster_paths, *token_paths] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing worker output(s): {missing}")

    clusters = _read_csvs(cluster_paths)
    if clusters.empty:
        raise ValueError("No clusters were scored")

    sort_cols = [
        "token_mean_max_abs_deviation",
        "token_mean_prob_range",
        "best_prompt_max_abs_deviation",
        "best_prompt_prob_range",
        "mean_pairwise_similarity",
    ]
    clusters = (
        clusters
        .sort_values(sort_cols, ascending=[True, True, True, True, False])
        .drop_duplicates("cluster_key", keep="first")
        .reset_index(drop=True)
    )
    clusters.insert(0, "rank", range(1, len(clusters) + 1))

    # Full scored pool — used for inspection and re-selection.
    clusters.to_csv(paths["cluster_pool"], index=False)

    token_rows = _read_csvs(token_paths)
    token_rows = token_rows[token_rows["cluster_key"].isin(set(clusters["cluster_key"]))]
    token_rows = token_rows.drop_duplicates(["cluster_key", "word_lower"])

    qualified = clusters[
        (clusters["token_mean_max_abs_deviation"] <= args.max_abs_deviation)
        & (clusters["token_mean_prob_range"]       <= args.max_prob_range)
    ]
    selected = select_disjoint_clusters_with_full_pool_fallback(
        clusters,
        qualified,
        token_rows,
        count=args.dataset_clusters,
        num_choices=args.num_choices,
    )
    selected.insert(0, "dataset_rank", range(1, len(selected) + 1))

    # Verify option count on every selected cluster.
    for _, row in selected.iterrows():
        choices = json.loads(row["choices"])
        assert len(choices) == args.num_choices, (
            f"Selected cluster {row['cluster_key']!r} has {len(choices)} choices, "
            f"expected {args.num_choices}"
        )

    # Top-30 (or dataset_clusters) file — consumed by localization experiments.
    selected.to_csv(paths["clusters"], index=False)

    selected_tokens = (
        token_rows[token_rows["cluster_key"].isin(set(selected["cluster_key"]))]
        .merge(selected[["cluster_key", "dataset_rank"]], on="cluster_key", how="left")
        .sort_values(["dataset_rank", "token_order"])
    )
    write_choice_banks(out_dir, selected, selected_tokens, args)

    candidate_bank = getattr(args, "candidate_bank", None)
    summary = {
        "candidate_bank": candidate_bank,
        "candidate_token_count": (
            args.candidate_token_count if candidate_bank is not None else None
        ),
        "candidate_bank_size": (
            args.candidate_token_count // len(CANDIDATE_BANKS)
            if candidate_bank is not None else None
        ),
        "candidate_split_seed": args.seed if candidate_bank is not None else None,
        "prior_summary": str(resolve_path(args.prior_summary)),
        "split_settings": (
            cluster_split_settings(args) if candidate_bank is not None else None
        ),
        "num_choices":        args.num_choices,
        "position_index_start": int(getattr(args, "position_index_start", 0)),
        "clusters_scored":    int(len(clusters)),
        "qualified_clusters": int(len(qualified)),
        "dataset_clusters":   int(len(selected)),
        "selected_unique_choice_token_ids": int(
            selected_tokens["token_id"].nunique()
        ),
        "choice_banks":       int(len(selected) * args.num_choices * args.num_choices),
        "thresholds": {
            "token_mean_max_abs_deviation": float(args.max_abs_deviation),
            "token_mean_prob_range":        float(args.max_prob_range),
        },
        "top_clusters": selected.head(10).to_dict(orient="records"),
    }
    paths["summary"].write_text(json.dumps(summary, indent=2) + "\n")
    if candidate_bank == CANDIDATE_BANKS[-1]:
        manifest_path = write_cluster_split_manifest(out_dir)
        print(f"[runner] wrote validated split manifest: {manifest_path}", flush=True)
    print("[runner] done:", json.dumps({k: v for k, v in summary.items() if k != "top_clusters"}, indent=2))


# ---------------------------------------------------------------------------
# Runner: spawns one worker per GPU via data_parallel.run_worker_pool
# ---------------------------------------------------------------------------

def run_runner(args) -> None:
    out_dir = dataset_dir(args)
    prior_summary = resolve_path(args.prior_summary)
    out_dir.mkdir(parents=True, exist_ok=True)
    published = output_paths(out_dir, args)
    existing_outputs = [
        published[name]
        for name in ("candidate_tokens", "clusters", "summary")
        if published[name].exists()
    ]
    split_manifest = out_dir / "manifests" / "cluster_split.json"
    if existing_outputs or split_manifest.exists():
        raise FileExistsError(
            "Cluster-bank output already exists; use a new model run root: "
            + ", ".join(
                str(path) for path in [*existing_outputs, split_manifest]
                if path.exists()
            )
        )

    output_worker_dir = worker_dir(out_dir, args)
    log_dir = worker_log_dir(out_dir, args)
    for d in (output_worker_dir, log_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    gpus = resolve_gpus(args.gpus)
    script_path = REPO_ROOT / "scripts/search_balanced_token_clusters.py"

    # Propagate every worker-relevant arg; only omit runner-only flags.
    base_args = [
        "--model",           args.model,
        "--prior_summary",   str(prior_summary),
        "--dataset_base",    str(resolve_path(args.dataset_base)),
        "--work_root",       str(resolve_path(args.work_root)),
        "--logs_root",       str(resolve_path(args.logs_root)),
        "--num_choices",     str(args.num_choices),
        "--position_index_start", str(args.position_index_start),
        "--seed",            str(args.seed),
        "--batch_size",      str(args.batch_size),
        "--candidate_min_len",            str(args.candidate_min_len),
        "--candidate_max_len",            str(args.candidate_max_len),
        "--candidate_min_mean_prior",     str(args.candidate_min_mean_prior),
        "--candidate_max_mean_prior",     str(args.candidate_max_mean_prior),
        "--candidate_max_argmax_rate",    str(args.candidate_max_argmax_rate),
        "--candidate_max_position_range", str(args.candidate_max_position_range),
        "--candidate_token_count", str(args.candidate_token_count),
        "--max_seed_tokens",        str(args.max_seed_tokens),
        "--neighbor_pool",          str(args.neighbor_pool),
        "--variants_per_seed",      str(args.variants_per_seed),
        "--variant_stride",         str(args.variant_stride),
        "--permutations_per_cluster", str(args.permutations_per_cluster),
        "--keep_top_clusters",      str(args.keep_top_clusters),
        "--dataset_clusters",       str(args.dataset_clusters),
        "--max_abs_deviation",      str(args.max_abs_deviation),
        "--max_prob_range",         str(args.max_prob_range),
        "--prompt_preamble",        args.prompt_preamble,
        "--dtype",                  args.dtype,
        "--device",                 args.device,
        "--progress_every",         str(args.progress_every),
    ]
    if args.dataset_name:
        base_args += ["--dataset_name", args.dataset_name]
    if args.candidate_bank:
        base_args += ["--candidate_bank", args.candidate_bank]

    specs = [
        WorkerSpec(
            name=f"worker_{wid}_gpu_{gpu}",
            command=[
                current_python(), str(script_path),
                "--role", "worker",
                "--worker_id", str(wid),
                "--num_workers", str(len(gpus)),
                *base_args,
            ],
            env=build_worker_env(gpu),
        )
        for wid, gpu in enumerate(gpus)
    ]

    print(f"[runner] out_dir={out_dir}", flush=True)
    print(f"[runner] launching {len(specs)} workers on GPUs {','.join(gpus)}", flush=True)
    run_worker_pool(specs, log_dir)
    combine_worker_outputs(out_dir, len(gpus), args)
    shutil.rmtree(output_worker_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    args = parse_args(argv)
    if args.role == "runner":
        run_runner(args)
    else:
        run_worker(args)
