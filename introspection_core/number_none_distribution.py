"""Helpers for visualizing held-out number-vs-none direction scores."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Mapping

import numpy as np


def load_ste_plot_layers(path: Path, layers: np.ndarray) -> tuple[list[int], list[int]]:
    """Read STE_LAYERS=l-r without executing shell code; pad by two layers."""
    assignments = re.findall(
        r"^\s*(?:export\s+)?STE_LAYERS\s*=\s*(.*?)\s*$",
        Path(path).read_text(encoding="utf-8"), re.MULTILINE,
    )
    if not assignments:
        raise ValueError(f"STE_LAYERS is missing from {path}")
    value = assignments[-1].split("#", 1)[0].strip().strip("\"'")
    match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", value)
    if match is None:
        raise ValueError("STE_LAYERS must have the form l-r")
    left, right = map(int, match.groups())
    available = set(map(int, layers))
    if left > right or not set(range(left, right + 1)) <= available:
        raise ValueError("STE_LAYERS must be increasing and present in the score artifact")
    selected = list(range(max(min(available), left - 2),
                          min(max(available), right + 2) + 1))
    if not set(selected) <= available:
        raise ValueError("score artifact is missing layers inside the padded STE range")
    return [left, right], selected


def load_number_none_test_scores(
    path: Path,
    *,
    capture_position: str = "final_token",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load the all-boundary number-vs-none projections from a test artifact."""

    score_path = Path(path).resolve()
    if not score_path.is_file():
        raise FileNotFoundError(score_path)
    with np.load(score_path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata"]))
        if metadata.get("protocol") != "train_direction_then_test_once":
            raise ValueError("scores were not produced by a frozen train direction")
        if metadata.get("capture_position", "routing_boundary") != capture_position:
            raise ValueError(
                "the score artifact does not capture the "
                + capture_position.replace("_", " ")
            )

        contrasts = list(metadata["contrasts"])
        panels = list(metadata["panels"])
        outcomes = list(metadata["primitive_outcomes"])
        try:
            contrast_index = contrasts.index("any_number_vs_none")
            panel_index = panels.index("all_boundaries")
            none_code = outcomes.index("none")
        except ValueError as error:
            raise ValueError("score artifact lacks the number-vs-none contrast") from error

        scores = np.asarray(
            payload["unit_contrast_projection"][
                :, contrast_index, panel_index, :
            ],
            dtype=np.float64,
        )
        primitive_outcome = np.asarray(payload["primitive_outcome"])
        if not np.issubdtype(primitive_outcome.dtype, np.integer):
            raise ValueError("primitive outcomes must use integer codes")
        if primitive_outcome.size and (
            primitive_outcome.min() < 0
            or primitive_outcome.max() >= len(outcomes)
        ):
            raise ValueError("primitive outcome code is outside the metadata range")
        number_labels = np.asarray(primitive_outcome != none_code, dtype=bool)
        layers = np.asarray(metadata["layers"], dtype=np.int64)

    if layers.ndim != 1 or not len(layers):
        raise ValueError("layers must be a non-empty one-dimensional array")
    if number_labels.ndim != 1:
        raise ValueError("primitive outcomes must be one-dimensional")
    if scores.shape != (len(number_labels), len(layers)):
        raise ValueError("scores do not align with trials and layers")
    if not number_labels.any() or number_labels.all():
        raise ValueError("test scores must contain both number and none outcomes")
    return layers, scores, number_labels, metadata


def normalize_density_ridge_scores(
    scores: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Put all ridge-plot scores on one shared finite [0, 1] coordinate."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("scores must have shape [trial, layer]")
    normalized = np.full(values.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(values)
    if not valid.any():
        raise ValueError("scores contain no finite values")
    minimum = float(values[valid].min())
    maximum = float(values[valid].max())
    if maximum <= minimum:
        raise ValueError("finite scores have zero range")
    normalized[valid] = (values[valid] - minimum) / (maximum - minimum)
    parameters: dict[str, object] = {
        "mode": "shared_minmax",
        "minimum": minimum,
        "maximum": maximum,
    }

    finite = normalized[np.isfinite(normalized)]
    if finite.size and (finite.min() < 0.0 or finite.max() > 1.0):
        raise AssertionError("normalization produced values outside [0, 1]")
    return normalized, parameters


def normalization_metadata(
    source_metadata: Mapping,
    parameters: Mapping[str, object],
) -> dict[str, object]:
    """Build compact provenance for a normalized visualization."""

    return {
        "schema_version": 1,
        "model": source_metadata.get("model"),
        "capture_position": source_metadata.get("capture_position"),
        "protocol": source_metadata.get("protocol"),
        "direction_definition": source_metadata.get("direction_definition"),
        "contrast": "any_number_vs_none",
        "panel": "all_boundaries",
        "normalization": dict(parameters),
        "normalization_scope": "both test classes share the same scale",
    }
