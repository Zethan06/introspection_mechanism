"""Ranked cluster loading and aligned full-vector averaging."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class RankedCluster:
    """One ranked token-choice bank."""

    rank: int | None
    cluster_key: str | None
    seed_word: str | None
    choices: tuple[str, ...]
    dataset_rank: int | None = None

    def metadata(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "dataset_rank": self.dataset_rank,
            "cluster_key": self.cluster_key,
            "seed_word": self.seed_word,
            "choices": list(self.choices),
        }


def load_ranked_clusters(
    path: Path,
    *,
    start_rank: int = 1,
    count: int = 1,
    cluster_key: str | None = None,
) -> list[RankedCluster]:
    """Load one keyed cluster or consecutive ranked clusters.

    Selected cluster-bank CSVs retain the global search-pool ``rank`` while
    assigning a consecutive ``dataset_rank``.  Preserve the legacy global-rank
    lookup when it covers the requested interval, then fall back to the
    dataset-local rank for multi-cluster bank aggregation.
    """
    if start_rank < 1:
        raise ValueError(f"cluster_rank must be positive; got {start_rank}")
    if count < 1:
        raise ValueError(f"cluster_count must be positive; got {count}")
    if cluster_key is not None and count != 1:
        raise ValueError("--cluster_key requires --cluster_count 1")

    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty cluster CSV: {path}")

    if cluster_key is not None:
        selected = [row for row in rows if row.get("cluster_key") == cluster_key]
        if not selected:
            raise ValueError(f"cluster_key {cluster_key!r} not found in {path}")
        selected = selected[:1]
    else:
        by_rank: dict[int, dict[str, str]] = {}
        for row in rows:
            rank_text = row.get("rank")
            if not rank_text:
                continue
            rank = int(rank_text)
            if rank in by_rank:
                raise ValueError(f"duplicate cluster rank {rank} in {path}")
            by_rank[rank] = row
        requested = list(range(start_rank, start_rank + count))
        missing = [rank for rank in requested if rank not in by_rank]
        if not missing:
            selected = [by_rank[rank] for rank in requested]
        else:
            by_dataset_rank: dict[int, dict[str, str]] = {}
            for row in rows:
                rank_text = row.get("dataset_rank")
                if not rank_text:
                    continue
                dataset_rank = int(rank_text)
                if dataset_rank in by_dataset_rank:
                    raise ValueError(
                        f"duplicate dataset rank {dataset_rank} in {path}"
                    )
                by_dataset_rank[dataset_rank] = row
            missing_dataset_ranks = [
                rank for rank in requested if rank not in by_dataset_rank
            ]
            if missing_dataset_ranks:
                raise ValueError(
                    f"cluster ranks missing from {path}: {missing}; "
                    "dataset ranks also missing: "
                    f"{missing_dataset_ranks}"
                )
            selected = [by_dataset_rank[rank] for rank in requested]

    clusters: list[RankedCluster] = []
    expected_positions: int | None = None
    for row in selected:
        choices = tuple(str(choice) for choice in json.loads(row["choices"]))
        if not choices:
            raise ValueError(f"cluster {row.get('cluster_key')} has no choices")
        if expected_positions is None:
            expected_positions = len(choices)
        elif len(choices) != expected_positions:
            raise ValueError(
                f"cluster rank {row.get('rank')} has {len(choices)} choices; "
                f"expected {expected_positions}"
            )
        clusters.append(
            RankedCluster(
                rank=int(row["rank"]) if row.get("rank") else None,
                dataset_rank=(
                    int(row["dataset_rank"])
                    if row.get("dataset_rank")
                    else None
                ),
                cluster_key=row.get("cluster_key"),
                seed_word=row.get("seed_word"),
                choices=choices,
            )
        )
    return clusters


def mean_full_vectors(
    vector_sums: torch.Tensor,
    cluster_count: int,
) -> torch.Tensor:
    """Average aligned latent vectors without reducing their hidden dimension."""
    if cluster_count < 1:
        raise ValueError("cluster_count must be positive")
    return vector_sums / float(cluster_count)
