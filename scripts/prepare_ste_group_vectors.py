#!/usr/bin/env python3
"""Recreate validation100 and sampled-bottom100 with one frozen baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from introspection_core.analysis_statistics import write_json
from introspection_core.cluster_split import file_sha256
from introspection_core.extraction import extract_last_token_residuals, load_concept_vector_payload
from introspection_core.model import HookedModel, ModelConfig
from introspection_core.concept_groups import load_populations


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--reference_vectors", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--results_dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    directory = args.dataset_dir / "concepts"
    names, groups = load_populations(directory / "validation.csv", directory / "bottom100_seed42.csv", 100)
    high_meta = json.loads((directory / "validation.json").read_text())
    if high_meta["concept_vector_words"] != names[:100]:
        raise ValueError("CSV/JSON population identity mismatch")
    output = args.results_dir
    if (output / "concept_vectors.pt").exists():
        raise FileExistsError("vector bank already exists")
    output.mkdir(parents=True, exist_ok=True)
    model = HookedModel(ModelConfig(name=args.model, dtype="bfloat16", device="cuda"))
    print(f"loaded {args.model}; verifying early-stop extraction", flush=True)
    check_words = names[:args.batch_size]
    full = extract_last_token_residuals(model, check_words, layer=args.layer, batch_size=args.batch_size)
    early = extract_last_token_residuals(model, check_words, layer=args.layer, batch_size=args.batch_size, stop_after_layer=True)
    error = float((full - early).abs().max())
    if error != 0:
        raise ValueError(f"early-stop extraction differs from full forward: {error}")
    baseline_words = high_meta["baseline_words"]
    acts = []
    start_time = time.monotonic()
    chunk_size = args.batch_size * 100
    for start in range(0, len(baseline_words), chunk_size):
        acts.append(extract_last_token_residuals(model, baseline_words[start:start + chunk_size],
            layer=args.layer, batch_size=args.batch_size, stop_after_layer=True))
        print(f"baseline={min(start + chunk_size, len(baseline_words))}/{len(baseline_words)} elapsed_s={time.monotonic()-start_time:.1f}", flush=True)
    mean = torch.cat(acts).mean(0)
    del acts
    torch.save({"layer": args.layer, "baseline_mean": mean, "baseline_word_count": len(baseline_words),
                "model": args.model}, output / "baseline_mean.pt")
    high = extract_last_token_residuals(model, names[:100], layer=args.layer, batch_size=args.batch_size, stop_after_layer=True)
    low = extract_last_token_residuals(model, names[100:], layer=args.layer, batch_size=args.batch_size, stop_after_layer=True)
    vectors = F.normalize(torch.cat([high, low]) - mean[None], dim=-1)
    reference = load_concept_vector_payload(args.reference_vectors, concepts=names[:100], layer=args.layer)
    cosine = F.cosine_similarity(reference, vectors[:100])
    drift = {"max_abs_difference": float((reference - vectors[:100]).abs().max()),
             "min_cosine": float(cosine.min()), "mean_cosine": float(cosine.mean())}
    metadata = {"model": args.model, "layer": args.layer, "batch_size": args.batch_size,
                "group_counts": {"validation100": 100, "bottom100": 100},
                "baseline_count": len(baseline_words), "early_stop_full_forward_max_abs_difference": error,
                "selected_reference_comparison": drift,
                "inputs": {str(p.resolve()): file_sha256(p) for p in
                    (directory / "validation.json", directory / "validation.csv", directory / "bottom100_seed42.csv", args.reference_vectors)},
                "elapsed_seconds": time.monotonic() - start_time}
    write_json(output / "metadata.json", metadata)
    if drift["min_cosine"] < 0.9999:
        raise ValueError(f"regenerated selected vectors do not match current bank closely enough: {drift}; inspect before evaluating")
    torch.save({"concepts": names, "unit_vectors": vectors, "layer": args.layer, "groups": groups,
                "baseline_source": str((output / "baseline_mean.pt").resolve())}, output / "concept_vectors.pt")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
