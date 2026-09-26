"""Validation for the frozen four-bank cluster split."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


CANDIDATE_BANKS = ("calibration", "train", "validation", "test")
CLUSTER_SPLIT_MANIFEST = Path("manifests/cluster_split.json")
_BANK_CLUSTER_NAME = re.compile(rf"^({'|'.join(CANDIDATE_BANKS)})\.csv$")


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_summary(dataset_dir: Path, bank: str) -> dict[str, Any]:
    path = dataset_dir / "manifests" / "cluster_search" / f"{bank}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing cluster-bank summary: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Cluster-bank summary must be a JSON object: {path}")
    if payload.get("candidate_bank") != bank:
        raise ValueError(
            f"Cluster-bank summary {path} identifies bank "
            f"{payload.get('candidate_bank')!r}, expected {bank!r}"
        )
    if not isinstance(payload.get("split_settings"), dict):
        raise ValueError(
            f"Cluster-bank summary lacks reusable split_settings: {path}; "
            "rerun that bank with the current search implementation"
        )
    return payload


def _required_columns(frame: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Artifact {path} is missing columns: {missing}")


def build_cluster_split_manifest(dataset_dir: Path) -> dict[str, Any]:
    """Load and jointly validate all four candidate and cluster banks."""
    dataset_dir = dataset_dir.resolve()
    summaries = {
        bank: _load_summary(dataset_dir, bank) for bank in CANDIDATE_BANKS
    }
    settings = summaries[CANDIDATE_BANKS[0]]["split_settings"]
    for bank in CANDIDATE_BANKS[1:]:
        if summaries[bank]["split_settings"] != settings:
            raise ValueError(
                f"Cluster bank {bank!r} uses different split/search settings"
            )

    expected_clusters = int(settings["dataset_clusters"])
    num_choices = int(settings["num_choices"])
    candidate_token_count = int(settings["candidate_token_count"])
    expected_candidates = candidate_token_count // len(CANDIDATE_BANKS)
    if candidate_token_count % len(CANDIDATE_BANKS):
        raise ValueError("candidate_token_count must be divisible by four")

    bank_details: dict[str, dict[str, Any]] = {}
    token_ids_by_bank: dict[str, set[int]] = {}
    words_by_bank: dict[str, set[str]] = {}
    candidate_ranks: set[int] = set()
    candidate_hashes: dict[str, str] = {}
    cluster_hashes: dict[str, str] = {}

    for bank in CANDIDATE_BANKS:
        candidate_path = dataset_dir / "candidates" / f"{bank}.csv"
        cluster_path = dataset_dir / "clusters" / f"{bank}.csv"
        for path in (candidate_path, cluster_path):
            if not path.exists():
                raise FileNotFoundError(f"Missing cluster-split artifact: {path}")

        candidates = pd.read_csv(candidate_path, keep_default_na=False)
        clusters = pd.read_csv(cluster_path, keep_default_na=False)
        _required_columns(
            candidates,
            {
                "candidate_bank",
                "candidate_rank",
                "token_id",
                "word_lower",
            },
            candidate_path,
        )
        _required_columns(clusters, {"cluster_key", "choices"}, cluster_path)

        if len(candidates) != expected_candidates:
            raise ValueError(
                f"Bank {bank!r} has {len(candidates)} candidate rows; "
                f"expected {expected_candidates}"
            )
        if set(candidates["candidate_bank"].astype(str)) != {bank}:
            raise ValueError(f"Candidate file is mislabeled or copied: {candidate_path}")
        if candidates["token_id"].nunique() != expected_candidates:
            raise ValueError(f"Candidate token IDs are not unique in {candidate_path}")
        if candidates["word_lower"].astype(str).nunique() != expected_candidates:
            raise ValueError(f"Candidate lowercase words are not unique in {candidate_path}")

        ranks = {int(value) for value in candidates["candidate_rank"]}
        if candidate_ranks.intersection(ranks):
            raise ValueError(f"Candidate ranks overlap across banks at {candidate_path}")
        candidate_ranks.update(ranks)

        if len(clusters) != expected_clusters:
            raise ValueError(
                f"Bank {bank!r} has {len(clusters)} clusters; "
                f"expected {expected_clusters}"
            )
        if clusters["cluster_key"].astype(str).nunique() != expected_clusters:
            raise ValueError(f"Cluster keys are not unique in {cluster_path}")

        candidate_by_word = {
            str(row.word_lower): int(row.token_id)
            for row in candidates.itertuples(index=False)
        }
        choice_words: set[str] = set()
        choice_token_ids: set[int] = set()
        for row in clusters.itertuples(index=False):
            choices = json.loads(row.choices)
            if not isinstance(choices, list) or len(choices) != num_choices:
                raise ValueError(
                    f"Cluster {row.cluster_key!r} in {cluster_path} must contain "
                    f"exactly {num_choices} choices"
                )
            lower_choices = [str(choice).lower() for choice in choices]
            if len(set(lower_choices)) != num_choices:
                raise ValueError(
                    f"Cluster {row.cluster_key!r} contains duplicate choice words"
                )
            missing_words = [
                word for word in lower_choices if word not in candidate_by_word
            ]
            if missing_words:
                raise ValueError(
                    f"Cluster {row.cluster_key!r} contains words outside its "
                    f"candidate bank: {missing_words[:5]}"
                )
            choice_words.update(lower_choices)
            choice_token_ids.update(candidate_by_word[word] for word in lower_choices)

        expected_choice_count = expected_clusters * num_choices
        if len(choice_words) != expected_choice_count:
            raise ValueError(
                f"Bank {bank!r} has {len(choice_words)} unique choice words; "
                f"expected {expected_choice_count}"
            )
        if len(choice_token_ids) != expected_choice_count:
            raise ValueError(
                f"Bank {bank!r} has {len(choice_token_ids)} unique choice token IDs; "
                f"expected {expected_choice_count}"
            )

        candidate_hashes[bank] = file_sha256(candidate_path)
        cluster_hashes[bank] = file_sha256(cluster_path)
        token_ids_by_bank[bank] = choice_token_ids
        words_by_bank[bank] = choice_words
        bank_details[bank] = {
            "candidate_file": str(candidate_path.relative_to(dataset_dir)),
            "cluster_file": str(cluster_path.relative_to(dataset_dir)),
            "candidate_rows": len(candidates),
            "unique_cluster_keys": int(clusters["cluster_key"].nunique()),
            "unique_choice_token_ids": len(choice_token_ids),
            "unique_choice_words": len(choice_words),
        }

    expected_ranks = set(range(1, candidate_token_count + 1))
    if candidate_ranks != expected_ranks:
        raise ValueError(
            "The four candidate banks do not cover each registered candidate "
            "rank exactly once"
        )

    token_checks: dict[str, bool] = {}
    word_checks: dict[str, bool] = {}
    for index, left in enumerate(CANDIDATE_BANKS):
        for right in CANDIDATE_BANKS[index + 1 :]:
            pair = f"{left}__{right}"
            token_checks[pair] = token_ids_by_bank[left].isdisjoint(
                token_ids_by_bank[right]
            )
            word_checks[pair] = words_by_bank[left].isdisjoint(words_by_bank[right])
    if not all(token_checks.values()):
        raise ValueError("Choice token IDs overlap across cluster banks")
    if not all(word_checks.values()):
        raise ValueError("Lowercase choice words overlap across cluster banks")

    return {
        "bank_names": list(CANDIDATE_BANKS),
        "split_settings": settings,
        "source_prior_sha256": settings["source_prior_sha256"],
        "candidate_file_sha256": candidate_hashes,
        "cluster_file_sha256": cluster_hashes,
        "banks": bank_details,
        "pairwise_disjoint_choice_token_ids": token_checks,
        "pairwise_disjoint_lowercase_choice_words": word_checks,
        "valid": True,
    }


def write_cluster_split_manifest(dataset_dir: Path) -> Path:
    """Validate all banks and atomically publish their shared manifest."""
    payload = build_cluster_split_manifest(dataset_dir)
    path = dataset_dir.resolve() / CLUSTER_SPLIT_MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)
    return path


def require_valid_cluster_split(cluster_csv: Path) -> None:
    """Gate standard bank artifacts on a current four-bank manifest."""
    cluster_csv = cluster_csv.resolve()
    if (
        _BANK_CLUSTER_NAME.fullmatch(cluster_csv.name) is None
        or cluster_csv.parent.name != "clusters"
    ):
        return
    dataset_dir = cluster_csv.parent.parent
    manifest_path = dataset_dir / CLUSTER_SPLIT_MANIFEST
    if not manifest_path.exists():
        raise ValueError(
            f"Missing {CLUSTER_SPLIT_MANIFEST} beside {cluster_csv}; "
            "validate all four cluster banks before using them"
        )
    stored = json.loads(manifest_path.read_text())
    current = build_cluster_split_manifest(dataset_dir)
    if stored != current:
        raise ValueError(
            f"{CLUSTER_SPLIT_MANIFEST} does not match the current bank artifacts; "
            "rerun cluster-split validation"
        )
