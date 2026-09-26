#!/usr/bin/env python3
"""Draw Figure 2 (layerwise report outcome and injection position).

(a) Held-out injected validation trials scored on the position--none direction
    (Stage 02b), grouped by response, one density ridge per layer.
(b) K-means position-clustering accuracy by layer (Stage 01). The star marks
    each model's transition layer: the layer with the largest accuracy increase
    over the layer below it.

Figure 3 is drawn by plot_fig3_label_average.py.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from introspection_core.manuscript_figures import (
    configure_manuscript_style, draw_layerwise_panels,
)
from introspection_core.number_none_distribution import (
    load_number_none_test_scores, normalize_density_ridge_scores,
)


MODELS = [
    ("qwen3-4b-instruct-2507", "Qwen3-4B-IT"),
    ("llama3.1-8b-instruct", "LLaMA-3.1-8B-IT"),
    ("gemma3-12b-it", "Gemma-3-12B-IT"),
]

# Layers shown as density ridges in Figure 2a (display window only).
RIDGE_LAYERS = {
    "qwen3-4b-instruct-2507": list(range(17, 26)),
    "llama3.1-8b-instruct": list(range(12, 21)),
    "gemma3-12b-it": list(range(22, 31)),
}


def transition_layer(layers: np.ndarray, accuracy: np.ndarray) -> int:
    """Return the layer whose clustering accuracy rises most over the one below."""
    order = np.argsort(layers)
    layers, accuracy = layers[order], accuracy[order]
    if len(layers) < 2 or np.any(np.diff(layers) != 1):
        raise ValueError("clustering metrics must cover consecutive layers")
    return int(layers[1:][np.argmax(np.diff(accuracy))])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    configure_manuscript_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panels, provenance = [], []
    for model, title in MODELS:
        root = args.results_root / model
        score_path = root / "position_none_direction/test_scores.npz"
        metric_path = root / "validation_diagnostics/position_kmeans/layer_metrics.csv"
        layers, raw, labels, _ = load_number_none_test_scores(score_path)
        scores, normalization = normalize_density_ridge_scores(raw)
        with metric_path.open() as handle:
            rows = list(csv.DictReader(handle))
        accuracy_layers = np.array([int(row["layer"]) for row in rows])
        accuracy = np.array([100 * float(row["test_accuracy"]) for row in rows])
        marked = transition_layer(accuracy_layers, accuracy)
        panels.append(dict(title=title, layers=layers, scores=scores, labels=labels,
                           ridge_layers=RIDGE_LAYERS[model], marked_layer=marked,
                           accuracy_layers=accuracy_layers, accuracy=accuracy))
        provenance.append(dict(
            model=model,
            sources=[dict(path=str(path.relative_to(args.results_root)),
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                     for path in (score_path, metric_path)],
            normalization=normalization,
            ridge_layers=RIDGE_LAYERS[model],
            transition_layer=marked,
            groups={"none_response": int((~labels).sum()),
                    "position_response": int(labels.sum())},
        ))

    figure, details = draw_layerwise_panels(panels)
    for extension in ("pdf", "svg", "png"):
        figure.savefig(args.output_dir / f"fig2_layerwise.{extension}", dpi=250)
    plt.close(figure)
    metadata = dict(panels=provenance, kde=details,
                    density_grouping="injected trials grouped by emitted response")
    (args.output_dir / "provenance_fig2.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(args.output_dir)


if __name__ == "__main__":
    main()
