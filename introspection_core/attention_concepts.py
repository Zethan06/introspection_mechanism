"""Reusable concept-bank loading for averaged intervention experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .extraction import (
    extract_concept_vectors,
    load_concept_vector_payload,
    load_concepts_from_json,
)
from .injection import load_unit_vector, normalize_unit_vector


@dataclass
class ConceptBank:
    names: list[str]
    vectors: torch.Tensor
    source_paths: list[str]
    baseline_words: list[str]


def load_concept_bank(
    model,
    *,
    concepts_json: Path,
    layer: int,
    average_all_concepts: bool,
    concept: str,
    concept_csv: Path | None = None,
    concept_column: str = "concept",
    max_concepts: int | None = None,
    max_baselines: int | None = None,
    vector_source: str = "on_the_fly",
    vectors_dir: Path | None = None,
    state_vectors: Path | None = None,
    vec_type: str = "last",
    extraction_batch_size: int = 16,
) -> ConceptBank:
    """Select concepts and return one independently normalized vector each.

    ``vector_source='payload'`` slices ``state_vectors`` by concept name, which
    reuses one population-wide extraction instead of recomputing the contrastive
    baseline for every split.
    """
    concepts, baseline_words = load_concepts_from_json(
        concepts_json,
        concept_csv=concept_csv,
        concept_column=concept_column,
        max_concepts=max_concepts,
    )
    if max_baselines is not None:
        baseline_words = baseline_words[:max_baselines]
    if not baseline_words:
        raise ValueError("At least one baseline word is required")
    if average_all_concepts:
        selected = concepts
    else:
        if concept not in concepts:
            raise ValueError(
                f"Concept {concept!r} is not in the selected manifest "
                f"({len(concepts)} concepts)"
            )
        selected = [concept]

    if vector_source == "on_the_fly":
        extracted = extract_concept_vectors(
            model,
            words=selected,
            baseline_words=baseline_words,
            layer=layer,
            batch_size=extraction_batch_size,
        )
        vectors = torch.stack(
            [normalize_unit_vector(item.vector) for item in extracted], dim=0
        )
        return ConceptBank(
            names=selected,
            vectors=vectors,
            source_paths=[],
            baseline_words=baseline_words,
        )

    if vector_source == "payload":
        if state_vectors is None:
            raise ValueError("state_vectors is required when vector_source='payload'")
        vectors = load_concept_vector_payload(
            state_vectors, concepts=selected, layer=layer
        )
        return ConceptBank(
            names=selected,
            vectors=vectors,
            source_paths=[str(state_vectors)],
            baseline_words=baseline_words,
        )

    if vector_source != "saved":
        raise ValueError(
            f"Unknown vector_source={vector_source!r}; expected payload, saved, "
            "or on_the_fly"
        )
    if vectors_dir is None:
        raise ValueError("vectors_dir is required when vector_source='saved'")

    manifest_path = vectors_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "Saved concept vectors require vectors_dir/manifest.json; "
            "concept, layer, and vector type must be recorded in the manifest "
            "instead of encoded in filenames"
        )
    manifest = json.loads(manifest_path.read_text())
    entries = manifest.get("vectors")
    if not isinstance(entries, list):
        raise ValueError(f"{manifest_path} must contain a 'vectors' list")
    index: dict[tuple[str, int, str], Path] = {}
    for entry in entries:
        key = (
            str(entry["concept"]),
            int(entry["layer"]),
            str(entry["vec_type"]),
        )
        if key in index:
            raise ValueError(f"Duplicate saved-vector manifest entry: {key}")
        relative_path = Path(str(entry["path"]))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"Saved-vector paths must stay relative to {vectors_dir}: "
                f"{relative_path}"
            )
        index[key] = vectors_dir / relative_path
    paths = []
    missing_keys = []
    for name in selected:
        key = (name, layer, vec_type)
        if key not in index:
            missing_keys.append(key)
        else:
            paths.append(index[key])
    if missing_keys:
        raise FileNotFoundError(
            "Missing saved-vector manifest entries: "
            + ", ".join(repr(key) for key in missing_keys[:10])
        )
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing saved concept vectors: "
            + ", ".join(str(path) for path in missing[:10])
        )
    vectors = torch.stack(
        [
            load_unit_vector(path, device="cpu", dtype=torch.float32)
            for path in paths
        ],
        dim=0,
    )
    return ConceptBank(
        names=selected,
        vectors=vectors,
        source_paths=[str(path) for path in paths],
        baseline_words=baseline_words,
    )
