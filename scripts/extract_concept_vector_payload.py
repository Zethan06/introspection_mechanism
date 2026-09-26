#!/usr/bin/env python3
"""Extract contrastive concept vectors into evaluator-compatible shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from introspection_core.extraction import (  # noqa: E402
    extract_concept_vectors,
    extract_last_token_residuals,
    load_concepts_from_json,
)
from introspection_core.model import HookedModel, ModelConfig  # noqa: E402
from introspection_core.prompt_search_sampling import (  # noqa: E402
    contiguous_shard_bounds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--concepts_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument(
        "--baseline_mean_file",
        type=Path,
        help=(
            "Reuse a locked full-baseline mean and extract only concept "
            "activations instead of recomputing every baseline word"
        ),
    )
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0 or args.batch_size <= 0:
        raise ValueError("shard count and batch size must be positive")
    concepts, baseline_words = load_concepts_from_json(args.concepts_json)
    model = HookedModel(
        ModelConfig(
            name=args.model,
            device=args.device,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
    )
    if args.baseline_mean_file is None:
        extracted = extract_concept_vectors(
            model,
            words=concepts,
            baseline_words=baseline_words,
            layer=args.layer,
            batch_size=args.batch_size,
        )
        vectors = F.normalize(
            torch.stack([item.vector for item in extracted]).float(), dim=-1
        )
        baseline_source = "on_the_fly"
        baseline_count = len(baseline_words)
    else:
        payload = torch.load(
            args.baseline_mean_file, map_location="cpu", weights_only=False
        )
        if int(payload.get("layer", -1)) != args.layer:
            raise ValueError("baseline mean uses a different extraction layer")
        baseline_mean = payload.get("baseline_mean")
        if not isinstance(baseline_mean, torch.Tensor) or baseline_mean.dim() != 1:
            raise ValueError("baseline mean payload has no valid baseline_mean")
        activations = extract_last_token_residuals(
            model,
            concepts,
            layer=args.layer,
            batch_size=args.batch_size,
        )
        if activations.shape[1] != baseline_mean.shape[0]:
            raise ValueError("baseline mean width does not match model residual width")
        vectors = F.normalize(
            activations - baseline_mean.float()[None, :], dim=-1
        )
        baseline_source = str(args.baseline_mean_file.resolve())
        baseline_count = int(payload.get("baseline_word_count", 0))
    rows = [
        {"word": concept, "source_index": index}
        for index, concept in enumerate(concepts)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = []
    for shard_id in range(args.num_shards):
        start, end = contiguous_shard_bounds(
            len(rows), shard_id, args.num_shards
        )
        path = args.output_dir / f"worker_{shard_id}.pt"
        torch.save(
            {
                "rows": rows[start:end],
                "vectors": vectors[start:end],
                "source": str(args.concepts_json),
                "layer": args.layer,
                "baseline_count": baseline_count,
                "baseline_source": baseline_source,
                "shard_id": shard_id,
                "num_shards": args.num_shards,
            },
            path,
        )
        shard_paths.append(str(path))
    canonical_path = args.output_dir / "state_vectors.pt"
    torch.save(
        {
            "schema_version": 1,
            "concepts": list(concepts),
            "unit_vectors": vectors,
            "layer": args.layer,
            "baseline_count": baseline_count,
            "baseline_source": baseline_source,
            "source": str(args.concepts_json),
        },
        canonical_path,
    )
    metadata = {
        "model": args.model,
        "concepts_json": str(args.concepts_json),
        "layer": args.layer,
        "concept_count": len(concepts),
        "baseline_count": baseline_count,
        "baseline_source": baseline_source,
        "num_shards": args.num_shards,
        "shards": shard_paths,
        "state_vectors": str(canonical_path),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
