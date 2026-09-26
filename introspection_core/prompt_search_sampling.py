"""Deterministic nested sampling for vocabulary-wide prompt search."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np


def contiguous_shard_bounds(
    total: int, shard_id: int, num_shards: int
) -> tuple[int, int]:
    """Return balanced contiguous bounds for one zero-based shard."""
    if total < 0:
        raise ValueError("total must be non-negative")
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_id < num_shards:
        raise ValueError("shard_id must be in [0, num_shards)")
    return (
        total * shard_id // num_shards,
        total * (shard_id + 1) // num_shards,
    )


def _rank_bins(values: Sequence[tuple[float, int]], n_bins: int) -> list[int]:
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    order = sorted(range(len(values)), key=lambda index: values[index])
    bins = [0] * len(values)
    for rank, index in enumerate(order):
        bins[index] = min(n_bins - 1, rank * n_bins // len(values))
    return bins


def _length_bucket(length: int) -> int:
    if length <= 3:
        return 0
    if length <= 6:
        return 1
    if length <= 10:
        return 2
    return 3


def _case_bucket(word: str) -> str:
    if word.islower():
        return "lower"
    if word.istitle():
        return "title"
    if word.isupper():
        return "upper"
    return "mixed"


def nested_stratified_order(
    rows: Sequence[dict[str, Any]],
    *,
    difficulty_key: str = "baseline_difficulty",
    n_difficulty_bins: int = 10,
    n_token_bins: int = 10,
    seed: int = 42,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Return one nested, approximately proportional stratified ordering.

    Prefixes of the returned order form successively larger panels. Strata
    jointly cover baseline difficulty rank, tokenizer-ID rank, word length,
    and casing. Items within a stratum use a stable seeded hash instead of
    input order.
    """
    if not rows:
        raise ValueError("rows must be non-empty")
    token_ids = [int(row["token_id"]) for row in rows]
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("token_id values must be unique")

    difficulty_values = [
        (float(row[difficulty_key]), token_ids[index])
        for index, row in enumerate(rows)
    ]
    token_values = [
        (float(token_id), token_id) for token_id in token_ids
    ]
    difficulty_bins = _rank_bins(difficulty_values, n_difficulty_bins)
    token_bins = _rank_bins(token_values, n_token_bins)

    strata: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    annotations: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        word = str(row["word"])
        key = (
            difficulty_bins[index],
            token_bins[index],
            _length_bucket(int(row.get("word_len", len(word)))),
            _case_bucket(word),
        )
        strata[key].append(index)
        annotations.append(
            {
                "difficulty_bin": key[0],
                "token_id_bin": key[1],
                "length_bin": key[2],
                "case_bin": key[3],
            }
        )

    for key, indices in strata.items():
        indices.sort(
            key=lambda index: hashlib.blake2b(
                f"{seed}:{token_ids[index]}".encode(), digest_size=8
            ).digest()
        )

    total = len(rows)
    selected_counts: Counter[tuple[Any, ...]] = Counter()
    cursors: Counter[tuple[Any, ...]] = Counter()
    ordered_indices: list[int] = []
    ordered_keys = sorted(strata, key=repr)
    for step in range(total):
        available = [
            key for key in ordered_keys if cursors[key] < len(strata[key])
        ]
        key = max(
            available,
            key=lambda candidate: (
                (step + 1) * len(strata[candidate]) / total
                - selected_counts[candidate],
                repr(candidate),
            ),
        )
        ordered_indices.append(strata[key][cursors[key]])
        cursors[key] += 1
        selected_counts[key] += 1
    return ordered_indices, annotations


def geometric_coreset_order(
    vectors: np.ndarray,
    *,
    max_size: int,
    projection_dim: int = 64,
    seed: int = 42,
    batch_size: int = 4096,
) -> dict[str, np.ndarray]:
    """Build a nested coreset in normalized vector-direction space.

    The full vocabulary is compressed into ``max_size`` fine clusters after
    a deterministic Gaussian random projection. A farthest-first walk over
    the centroids then orders actual vocabulary medoids so that every prefix
    is a geometry-covering prompt-search panel.
    """
    from sklearn.cluster import MiniBatchKMeans

    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("vectors must be a non-empty two-dimensional array")
    if not 0 < max_size <= values.shape[0]:
        raise ValueError("max_size must be in [1, number of vectors]")
    if projection_dim <= 0:
        raise ValueError("projection_dim must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = values / np.maximum(norms, np.finfo(np.float32).eps)
    target_dim = min(projection_dim, values.shape[1])
    generator = np.random.default_rng(seed)
    projection = generator.standard_normal(
        (values.shape[1], target_dim), dtype=np.float32
    ) / np.sqrt(target_dim)
    projected = normalized @ projection
    projected_norms = np.linalg.norm(projected, axis=1, keepdims=True)
    projected /= np.maximum(projected_norms, np.finfo(np.float32).eps)

    clustering = MiniBatchKMeans(
        n_clusters=max_size,
        random_state=seed,
        batch_size=max(batch_size, 3 * max_size),
        n_init=3,
        max_iter=100,
        reassignment_ratio=0.0,
    )
    labels = clustering.fit_predict(projected)
    centers = np.asarray(clustering.cluster_centers_, dtype=np.float32)
    counts = np.bincount(labels, minlength=max_size).astype(np.int64)
    if np.any(counts == 0):
        raise RuntimeError("geometric clustering produced an empty cluster")

    center_distances = np.sum((projected - centers[labels]) ** 2, axis=1)
    medoid_indices = np.full(max_size, -1, dtype=np.int64)
    for row_index, (label, distance) in enumerate(
        zip(labels, center_distances, strict=True)
    ):
        current = medoid_indices[label]
        if current < 0 or distance < center_distances[current]:
            medoid_indices[label] = row_index

    # K-means already allocates more centroids to dense regions. The
    # farthest-first traversal therefore adds geometric coverage without
    # letting the largest cluster consume every slot in a small panel.
    first = int(np.argmax(counts))
    order = np.empty(max_size, dtype=np.int64)
    order[0] = first
    selected = np.zeros(max_size, dtype=bool)
    selected[first] = True
    min_distances = np.sum((centers - centers[first]) ** 2, axis=1)
    for position in range(1, max_size):
        scores = min_distances.copy()
        scores[selected] = -1.0
        next_index = int(np.argmax(scores))
        order[position] = next_index
        selected[next_index] = True
        distances = np.sum((centers - centers[next_index]) ** 2, axis=1)
        np.minimum(min_distances, distances, out=min_distances)

    return {
        "medoid_indices": medoid_indices,
        "center_order": order,
        "fine_centers": centers,
        "fine_counts": counts,
        "point_center_squared_distances": center_distances,
        "labels": labels.astype(np.int64, copy=False),
    }


def geometric_panel_weights(
    fine_centers: np.ndarray,
    fine_counts: np.ndarray,
    selected_center_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign full-vocabulary mass and coverage distances to panel medoids."""
    centers = np.asarray(fine_centers, dtype=np.float32)
    counts = np.asarray(fine_counts, dtype=np.int64)
    selected = np.asarray(selected_center_indices, dtype=np.int64)
    if centers.ndim != 2 or counts.shape != (centers.shape[0],):
        raise ValueError("fine center/count shapes do not match")
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError("selected_center_indices must be non-empty")
    if np.any(selected < 0) or np.any(selected >= centers.shape[0]):
        raise ValueError("selected center index is out of range")
    if len(set(selected.tolist())) != selected.size:
        raise ValueError("selected center indices must be unique")

    selected_centers = centers[selected]
    squared_distances = (
        np.sum(centers**2, axis=1, keepdims=True)
        + np.sum(selected_centers**2, axis=1)[None, :]
        - 2.0 * centers @ selected_centers.T
    )
    assignments = np.argmin(squared_distances, axis=1)
    weights = np.bincount(
        assignments, weights=counts, minlength=selected.size
    ).astype(np.int64)
    coverage = np.sqrt(
        np.maximum(
            squared_distances[np.arange(centers.shape[0]), assignments], 0
        )
    )
    return weights, coverage
