#!/usr/bin/env python3
"""Cluster full layerwise latents on train concepts and evaluate on test concepts.

For every observed layer, K-means is fit without position labels on the full
train latent vectors.  A train-only Hungarian assignment names the ten cluster
IDs as token positions, and frozen centroids plus that assignment are evaluated
once on the disjoint test split.  No projection or cross-validation is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from introspection_core import (  # noqa: E402
    HookedModel,
    ModelConfig,
    PromptManager,
    collect_averaged_injection_projection,
    load_concepts_from_json,
    load_ranked_clusters,
)
from introspection_core.cluster_split import file_sha256  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-concepts-json", type=Path, required=True)
    parser.add_argument("--train-cluster-csv", type=Path, required=True)
    parser.add_argument("--test-concepts-json", type=Path, required=True)
    parser.add_argument("--test-cluster-csv", type=Path, required=True)
    parser.add_argument("--state-vectors", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--injection-layer", type=int, required=True)
    parser.add_argument("--strength", type=float, required=True)
    parser.add_argument("--start-layer", type=int)
    parser.add_argument("--end-layer", type=int)
    parser.add_argument("--cluster-count", type=int, default=30)
    parser.add_argument("--concept-batch-size", type=int, default=32)
    parser.add_argument("--n-init", type=int, default=50)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument(
        "--scale-mode",
        choices=("unit", "relative_hidden_norm"),
        default="relative_hidden_norm",
    )
    parser.add_argument(
        "--prompt-template",
        default="semantic_highinj_posref_gate_balanced_disrupts",
    )
    parser.add_argument("--prompt-preamble", default="system")
    parser.add_argument("--capture-token-index", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "train_concepts_json",
        "train_cluster_csv",
        "test_concepts_json",
        "test_cluster_csv",
        "state_vectors",
    ):
        path = getattr(args, name).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        setattr(args, name, path)
    if args.train_concepts_json == args.test_concepts_json:
        raise ValueError("train and test concept manifests must differ")
    if args.train_cluster_csv == args.test_cluster_csv:
        raise ValueError("train and test cluster CSVs must differ")
    if args.injection_layer < 0:
        raise ValueError("injection-layer must be non-negative")
    if args.cluster_count < 1 or args.concept_batch_size < 1:
        raise ValueError("cluster-count and concept-batch-size must be positive")
    if args.n_init < 1 or args.max_iter < 1:
        raise ValueError("n-init and max-iter must be positive")
    args.output_dir = args.output_dir.resolve()


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fit_position_mapping(
    cluster_ids: np.ndarray,
    positions: np.ndarray,
    *,
    position_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cluster-to-position mapping and its train contingency table."""
    contingency = np.zeros((position_count, position_count), dtype=np.int64)
    np.add.at(contingency, (cluster_ids, positions), 1)
    cluster_order, position_order = linear_sum_assignment(-contingency)
    mapping = np.full(position_count, -1, dtype=np.int64)
    mapping[cluster_order] = position_order
    if np.any(mapping < 0):
        raise ValueError("Hungarian assignment did not map every cluster")
    return mapping, contingency


def _injected_labels(rows: Sequence[Mapping[str, object]]) -> tuple[np.ndarray, list[str]]:
    injected = [row for row in rows if row["condition"] == "injected"]
    positions = np.asarray(
        [int(row["target_position"]) for row in injected], dtype=np.int64
    )
    concepts = [str(row["concept"]) for row in injected]
    return positions, concepts


def _collect_split(
    model: HookedModel,
    prompt_manager: PromptManager,
    *,
    concepts_path: Path,
    clusters_path: Path,
    args: argparse.Namespace,
    layers: Sequence[int],
):
    concepts, _baselines = load_concepts_from_json(concepts_path)
    clusters = load_ranked_clusters(
        clusters_path, start_rank=1, count=args.cluster_count
    )
    payload = torch.load(args.state_vectors, map_location="cpu", weights_only=False)
    if int(payload["layer"]) != args.injection_layer:
        raise ValueError("cached state-vector layer does not match injection-layer")
    cached_names = [str(value) for value in payload["concepts"]]
    if len(cached_names) != len(set(cached_names)):
        raise ValueError("cached state vectors contain duplicate concept names")
    cached_index = {name: index for index, name in enumerate(cached_names)}
    missing = [name for name in concepts if name not in cached_index]
    if missing:
        raise ValueError(f"concept missing from cached state vectors: {missing[0]!r}")
    unit_vectors = payload["unit_vectors"]
    selected = torch.tensor([cached_index[name] for name in concepts])
    unit_matrix = unit_vectors.index_select(0, selected).to(
        device=model.bridge.cfg.device,
        dtype=model.bridge.cfg.dtype,
    )
    names = list(concepts)
    collected = collect_averaged_injection_projection(
        model,
        prompt_manager,
        clusters=clusters,
        concept_names=names,
        concept_vectors=unit_matrix,
        injection_layer=args.injection_layer,
        coeffs=[args.strength],
        scale_mode=args.scale_mode,
        prompt_preamble=args.prompt_preamble,
        observe_layers=layers,
        concept_batch_size=args.concept_batch_size,
        prompt_template=args.prompt_template,
        capture_token_index=args.capture_token_index,
    )
    return names, clusters, collected


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = HookedModel(
        ModelConfig(
            name=args.model,
            device=args.device,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
    )
    model_layer_count = int(model.cfg.n_layers)
    start_layer = (
        args.start_layer
        if args.start_layer is not None
        else 0
    )
    end_layer = args.end_layer if args.end_layer is not None else model_layer_count - 1
    if not (0 <= args.injection_layer < model_layer_count):
        raise ValueError(
            "injection-layer must be within the model layer range "
            f"0..{model_layer_count - 1}"
        )
    if not (
        0 <= start_layer <= args.injection_layer <= end_layer < model_layer_count
    ):
        raise ValueError(
            "layers must satisfy 0 <= start-layer <= injection-layer "
            "<= end-layer < model layer count"
        )
    layers = list(range(start_layer, end_layer + 1))
    prompt_manager = PromptManager(model.tokenizer)

    print("[train] collecting full latents", flush=True)
    train_names, train_clusters, train = _collect_split(
        model,
        prompt_manager,
        concepts_path=args.train_concepts_json,
        clusters_path=args.train_cluster_csv,
        args=args,
        layers=layers,
    )
    train_y, train_row_concepts = _injected_labels(train.rows)
    if set(train_row_concepts) != set(train_names):
        raise ValueError("train latent rows do not match train concepts")
    position_values = np.unique(train_y)
    if not np.array_equal(position_values, np.arange(len(position_values))):
        raise ValueError(f"positions must be contiguous from zero: {position_values}")
    position_count = len(position_values)

    fitted: dict[int, tuple[KMeans, np.ndarray]] = {}
    train_cluster_ids: dict[int, np.ndarray] = {}
    train_contingencies: dict[int, np.ndarray] = {}
    for layer in layers:
        train_x = train.residuals[layer][1:].numpy()
        estimator = KMeans(
            n_clusters=position_count,
            n_init=args.n_init,
            max_iter=args.max_iter,
            random_state=args.seed,
            algorithm="lloyd",
        )
        cluster_ids = estimator.fit_predict(train_x)
        mapping, contingency = _fit_position_mapping(
            cluster_ids, train_y, position_count=position_count
        )
        fitted[layer] = (estimator, mapping)
        train_cluster_ids[layer] = cluster_ids
        train_contingencies[layer] = contingency
        train_accuracy = accuracy_score(train_y, mapping[cluster_ids])
        print(
            f"[train] layer={layer} inertia={estimator.inertia_:.6g} "
            f"mapped_accuracy={train_accuracy:.3%}",
            flush=True,
        )
    del train

    print("[test] collecting full latents", flush=True)
    test_names, test_clusters, test = _collect_split(
        model,
        prompt_manager,
        concepts_path=args.test_concepts_json,
        clusters_path=args.test_cluster_csv,
        args=args,
        layers=layers,
    )
    overlap = sorted(set(train_names).intersection(test_names))
    if overlap:
        raise ValueError(f"train/test concept leakage: {overlap[0]!r}")
    test_y, test_row_concepts = _injected_labels(test.rows)
    if set(test_row_concepts) != set(test_names):
        raise ValueError("test latent rows do not match test concepts")
    if not np.array_equal(np.unique(test_y), position_values):
        raise ValueError("train and test positions differ")

    metric_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    test_cluster_matrix: list[np.ndarray] = []
    test_prediction_matrix: list[np.ndarray] = []
    centers: list[torch.Tensor] = []
    mappings: list[torch.Tensor] = []
    for layer in layers:
        estimator, mapping = fitted[layer]
        test_x = test.residuals[layer][1:].numpy()
        test_cluster = estimator.predict(test_x)
        test_prediction = mapping[test_cluster]
        train_cluster = train_cluster_ids[layer]
        train_prediction = mapping[train_cluster]
        metric_rows.append(
            {
                "model": args.model,
                "layer": layer,
                "train_points": len(train_y),
                "test_points": len(test_y),
                "train_concepts": len(train_names),
                "test_concepts": len(test_names),
                "positions": position_count,
                "chance_accuracy": 1.0 / position_count,
                "train_mapped_accuracy": accuracy_score(train_y, train_prediction),
                "train_ari": adjusted_rand_score(train_y, train_cluster),
                "train_ami": adjusted_mutual_info_score(train_y, train_cluster),
                "test_accuracy": accuracy_score(test_y, test_prediction),
                "test_macro_f1": f1_score(
                    test_y, test_prediction, average="macro", zero_division=0
                ),
                "test_ari": adjusted_rand_score(test_y, test_cluster),
                "test_ami": adjusted_mutual_info_score(test_y, test_cluster),
                "train_inertia": estimator.inertia_,
                "kmeans_iterations": estimator.n_iter_,
            }
        )
        matrix = confusion_matrix(
            test_y, test_prediction, labels=np.arange(position_count)
        )
        for true_position in range(position_count):
            for predicted_position in range(position_count):
                confusion_rows.append(
                    {
                        "layer": layer,
                        "true_position": true_position,
                        "predicted_position": predicted_position,
                        "count": int(matrix[true_position, predicted_position]),
                    }
                )
        test_cluster_matrix.append(test_cluster)
        test_prediction_matrix.append(test_prediction)
        centers.append(torch.from_numpy(estimator.cluster_centers_))
        mappings.append(torch.from_numpy(mapping))
        print(
            f"[test] layer={layer} accuracy={metric_rows[-1]['test_accuracy']:.3%} "
            f"ARI={metric_rows[-1]['test_ari']:.6f} "
            f"AMI={metric_rows[-1]['test_ami']:.6f}",
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "layer_metrics.csv"
    confusion_path = args.output_dir / "test_confusion.csv"
    scores_path = args.output_dir / "test_assignments.npz"
    centers_path = args.output_dir / "kmeans_centers.pt"
    metadata_path = args.output_dir / "metadata.json"
    _write_csv(metrics_path, metric_rows)
    _write_csv(confusion_path, confusion_rows)
    np.savez_compressed(
        scores_path,
        layers=np.asarray(layers, dtype=np.int64),
        true_position=test_y,
        concept=np.asarray(test_row_concepts),
        cluster_id=np.stack(test_cluster_matrix),
        predicted_position=np.stack(test_prediction_matrix),
    )
    torch.save(
        {
            "schema_version": 1,
            "layers": layers,
            "centers": torch.stack(centers),
            "cluster_to_position": torch.stack(mappings),
            "train_contingency": torch.from_numpy(
                np.stack([train_contingencies[layer] for layer in layers])
            ),
        },
        centers_path,
    )
    metadata = {
        "schema_version": 1,
        "protocol": "train_kmeans_hungarian_then_test_once",
        "representation": "full_final_token_resid_post_without_projection",
        "distance": "euclidean",
        "position_labels_used_by_kmeans": False,
        "hungarian_mapping_source": "train_only",
        "test_usage": "single_final_evaluation",
        "model": args.model,
        "model_layer_count": model_layer_count,
        "model_width": int(model.cfg.d_model),
        "layers": layers,
        "injection_layer": args.injection_layer,
        "strength": args.strength,
        "scale_mode": args.scale_mode,
        "prompt_template": args.prompt_template,
        "prompt_preamble": args.prompt_preamble,
        "capture_token_index": args.capture_token_index,
        "train_concepts_json": str(args.train_concepts_json),
        "train_cluster_csv": str(args.train_cluster_csv),
        "test_concepts_json": str(args.test_concepts_json),
        "test_cluster_csv": str(args.test_cluster_csv),
        "state_vectors": str(args.state_vectors),
        "input_sha256": {
            name: file_sha256(getattr(args, name))
            for name in (
                "train_concepts_json",
                "train_cluster_csv",
                "test_concepts_json",
                "test_cluster_csv",
                "state_vectors",
            )
        },
        "train_concepts": train_names,
        "test_concepts": test_names,
        "train_cluster_keys": [item.cluster_key for item in train_clusters],
        "test_cluster_keys": [item.cluster_key for item in test_clusters],
        "cluster_reduction": "full_vector_mean",
        "cluster_count_per_split": args.cluster_count,
        "kmeans_n_clusters": position_count,
        "kmeans_n_init": args.n_init,
        "kmeans_max_iter": args.max_iter,
        "seed": args.seed,
        "metrics_file": str(metrics_path),
        "confusion_file": str(confusion_path),
        "assignments_file": str(scores_path),
        "centers_file": str(centers_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"[done] wrote results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
