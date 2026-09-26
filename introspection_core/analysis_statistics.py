"""Small, deterministic statistical helpers for offline experiment analyses."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from scipy.stats import wilcoxon

from .boundary_direction_auc import roc_auc


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    """Write a non-empty row table, creating its parent directory."""

    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    for row in rows[1:]:
        if list(row) != fieldnames:
            raise ValueError("CSV rows must have identical ordered fields")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    """Write formatted UTF-8 JSON, creating its parent directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def primitive_outcome(
    predictions: torch.Tensor,
    target_positions: torch.Tensor,
    number_candidate_count: int,
) -> torch.Tensor:
    """Map closed-set predictions to none=0, exact-number=1, wrong-number=2."""

    predictions = predictions.detach().cpu().long().reshape(-1)
    target_positions = target_positions.detach().cpu().long().reshape(-1)
    if predictions.shape != target_positions.shape:
        raise ValueError("predictions and target positions must have equal shape")
    if number_candidate_count <= 0:
        raise ValueError("number_candidate_count must be positive")
    if bool(predictions.lt(0).any()) or bool(
        predictions.gt(number_candidate_count).any()
    ):
        raise ValueError("prediction lies outside number-plus-none candidates")
    result = torch.full_like(predictions, 2)
    result[predictions.eq(number_candidate_count)] = 0
    result[predictions.eq(target_positions)] = 1
    return result


def bh_qvalues(pvalues: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg adjusted q-values, preserving non-finite entries."""

    values = np.asarray(pvalues, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if finite_indices.size == 0:
        return result.tolist()
    finite = values[finite_indices]
    if bool(((finite < 0) | (finite > 1)).any()):
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(finite)
    ranked = finite[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0, 1)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    result[finite_indices] = restored
    return result.tolist()


def paired_test(
    left: np.ndarray,
    right: np.ndarray,
    *,
    rng: np.random.Generator,
    permutations: int,
    bootstrap_samples: int,
) -> dict[str, float | int]:
    """Return paired descriptive, sign-permutation, bootstrap, and Wilcoxon stats."""

    left = np.asarray(left, dtype=float).reshape(-1)
    right = np.asarray(right, dtype=float).reshape(-1)
    if left.shape != right.shape:
        raise ValueError("paired samples must have equal shape")
    if permutations <= 0 or bootstrap_samples <= 0:
        raise ValueError("permutation and bootstrap counts must be positive")
    finite = np.isfinite(left) & np.isfinite(right)
    left = left[finite]
    right = right[finite]
    count = int(left.size)
    if count == 0:
        return {
            "paired_concept_count": 0,
            "number_mean": math.nan,
            "none_mean": math.nan,
            "mean_difference_number_minus_none": math.nan,
            "relative_difference": math.nan,
            "cohen_dz": math.nan,
            "ci95_low": math.nan,
            "ci95_high": math.nan,
            "permutation_p": math.nan,
            "wilcoxon_p": math.nan,
        }

    differences = left - right
    difference_mean = float(differences.mean())
    difference_std = float(differences.std(ddof=1)) if count > 1 else 0.0
    cohen_dz = (
        difference_mean / difference_std
        if difference_std > 1e-12
        else math.nan
    )
    right_mean = float(right.mean())
    relative = (
        difference_mean / abs(right_mean) if abs(right_mean) > 1e-12 else math.nan
    )

    exceedances = 0
    remaining = permutations
    while remaining:
        chunk = min(remaining, 4096)
        signs = rng.integers(0, 2, size=(chunk, count), dtype=np.int8) * 2 - 1
        permuted = (signs * differences).mean(axis=1)
        exceedances += int(np.count_nonzero(np.abs(permuted) >= abs(difference_mean)))
        remaining -= chunk
    permutation_p = (exceedances + 1) / (permutations + 1)

    bootstrap_means = np.empty(bootstrap_samples, dtype=float)
    remaining = bootstrap_samples
    offset = 0
    while remaining:
        chunk = min(remaining, 4096)
        indices = rng.integers(0, count, size=(chunk, count))
        bootstrap_means[offset : offset + chunk] = differences[indices].mean(axis=1)
        offset += chunk
        remaining -= chunk
    ci_low, ci_high = np.quantile(bootstrap_means, (0.025, 0.975))

    if bool(np.allclose(differences, 0.0)):
        wilcoxon_p = 1.0
    else:
        wilcoxon_p = float(wilcoxon(differences).pvalue)
    return {
        "paired_concept_count": count,
        "number_mean": float(left.mean()),
        "none_mean": right_mean,
        "mean_difference_number_minus_none": difference_mean,
        "relative_difference": relative,
        "cohen_dz": cohen_dz,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "permutation_p": permutation_p,
        "wilcoxon_p": wilcoxon_p,
    }


def _cluster_pair_wins(
    labels: np.ndarray,
    scores: np.ndarray,
    cluster_indices: np.ndarray,
    cluster_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate Mann-Whitney wins for every positive/negative cluster pair."""

    positive_scores = scores[labels]
    negative_scores = scores[~labels]
    positive_clusters = cluster_indices[labels]
    negative_clusters = cluster_indices[~labels]
    positive_counts = np.bincount(positive_clusters, minlength=cluster_count)
    negative_counts = np.bincount(negative_clusters, minlength=cluster_count)
    pair_wins = np.zeros((cluster_count, cluster_count), dtype=np.float64)
    for negative_cluster in range(cluster_count):
        comparison = np.sort(negative_scores[negative_clusters == negative_cluster])
        if len(comparison) == 0:
            continue
        lower = np.searchsorted(comparison, positive_scores, side="left")
        upper = np.searchsorted(comparison, positive_scores, side="right")
        wins = lower + 0.5 * (upper - lower)
        pair_wins[:, negative_cluster] = np.bincount(
            positive_clusters, weights=wins, minlength=cluster_count
        )
    return pair_wins, positive_counts, negative_counts


def cluster_bootstrap_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    cluster_indices: np.ndarray,
    *,
    cluster_count: int,
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> np.ndarray:
    """Bootstrap pooled ROC-AUC while resampling independent clusters.

    Every sampled cluster retains all of its observations. Repeated clusters
    receive multiplicity weights, so within-cluster dependence is preserved.
    Tied positive/negative score pairs contribute half a win, matching
    :func:`roc_auc`.
    """

    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    cluster_indices = np.asarray(cluster_indices, dtype=np.int64)
    if labels.ndim != 1 or scores.ndim != 1 or cluster_indices.ndim != 1:
        raise ValueError("labels, scores, and cluster indices must be one-dimensional")
    if labels.shape != scores.shape or labels.shape != cluster_indices.shape:
        raise ValueError("labels, scores, and cluster indices must have equal shape")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    if cluster_count <= 0 or bootstrap_samples <= 0:
        raise ValueError("cluster count and bootstrap samples must be positive")
    if len(cluster_indices) == 0:
        raise ValueError("cluster bootstrap input must not be empty")
    if cluster_indices.min() < 0 or cluster_indices.max() >= cluster_count:
        raise ValueError("cluster index lies outside the declared cluster count")

    pair_wins, positive_counts, negative_counts = _cluster_pair_wins(
        labels, scores, cluster_indices, cluster_count
    )
    bootstrap_counts = rng.multinomial(
        cluster_count,
        np.full(cluster_count, 1.0 / cluster_count),
        size=bootstrap_samples,
    ).astype(np.float64)
    numerator = np.einsum(
        "bi,ij,bj->b",
        bootstrap_counts,
        pair_wins,
        bootstrap_counts,
        optimize=True,
    )
    denominator = (
        bootstrap_counts @ positive_counts
    ) * (bootstrap_counts @ negative_counts)
    return np.divide(
        numerator,
        denominator,
        out=np.full(bootstrap_samples, np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default)."""

    if total <= 0:
        raise ValueError("total must be positive")
    if not 0 <= successes <= total:
        raise ValueError("successes must lie in [0, total]")
    rate = successes / total
    scale = 1.0 + z * z / total
    center = (rate + z * z / (2 * total)) / scale
    half = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / scale
    return max(0.0, center - half), min(1.0, center + half)


def cluster_bootstrap_rate(
    successes: np.ndarray,
    totals: np.ndarray,
    *,
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> np.ndarray:
    """Bootstrap pooled rates while resampling whole clusters.

    ``successes`` and ``totals`` share shape ``[cluster, ...]``; trailing axes
    (for example, separate arms evaluated on the same clusters) are resampled
    jointly, so every draw uses one cluster multiset for all of them. Returns
    pooled rates of shape ``[bootstrap_samples, ...]``.
    """

    successes = np.asarray(successes, dtype=np.float64)
    totals = np.asarray(totals, dtype=np.float64)
    if successes.shape != totals.shape or successes.ndim == 0:
        raise ValueError("successes and totals must share a [cluster, ...] shape")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if np.any(totals <= 0) or np.any(successes < 0) or np.any(successes > totals):
        raise ValueError("each cluster needs 0 <= successes <= totals and totals > 0")
    cluster_count = successes.shape[0]
    weights = rng.multinomial(
        cluster_count,
        np.full(cluster_count, 1.0 / cluster_count),
        size=bootstrap_samples,
    ).astype(np.float64)
    flat_successes = successes.reshape(cluster_count, -1)
    flat_totals = totals.reshape(cluster_count, -1)
    rates = (weights @ flat_successes) / (weights @ flat_totals)
    return rates.reshape((bootstrap_samples,) + successes.shape[1:])
