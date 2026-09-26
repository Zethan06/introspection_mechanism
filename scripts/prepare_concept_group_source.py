#!/usr/bin/env python3
"""Write the frozen C_intro / C_nonintro source directory for the QK and OV analyses.

Every Section-4 capture (``capture_ste_context_modes``, ``capture_qk_attention_kl``,
``run_ste_qk_score_ablation``, ``capture_ov_output_svd``, ``run_ov_causal_ablation``)
reads ``<source>/sources.json``. Its ``metadata`` block fixes the model, the 200
concepts in order (validation100, then the seeded bottom100 sample), the frozen
prompt and injection settings, the evaluation cluster bank (test), the
concept-vector bank, and the candidate heads of the STE search window.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from introspection_core.cluster_split import file_sha256
from introspection_core.head_output_patch import GATE_MASK_DIR, load_head_selection, parse_head_group_spec
from introspection_core.concept_groups import GROUPS, load_populations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_results", type=Path, required=True, help="results/<model>")
    parser.add_argument("--dataset_dir", type=Path, required=True, help="data/dataset/<model>")
    parser.add_argument(
        "--concept_vectors",
        type=Path,
        required=True,
        help="concept-vector bank covering both groups (prepare_ste_group_vectors.py)",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    concepts_dir = args.dataset_dir / "concepts"
    clusters_dir = args.dataset_dir / "clusters"
    gate_dir = args.model_results / GATE_MASK_DIR
    names, groups = load_populations(
        concepts_dir / "validation.csv", concepts_dir / "bottom100_seed42.csv", 100
    )
    bank = torch.load(args.concept_vectors, map_location="cpu", weights_only=False)
    # Consumers slice the bank by name, so it may hold more than these 200.
    missing = sorted(set(names) - set(map(str, bank["concepts"])))
    if missing:
        raise ValueError(f"concept-vector bank lacks {len(missing)} concepts, e.g. {missing[:5]}")

    selections, layers, config, model, n_heads = {}, set(), None, None, None
    for direction in ("on", "off"):
        selection = load_head_selection(gate_dir / f"train_{direction}/selected_heads.json")
        if selection["selection_direction"] != direction:
            raise ValueError(f"train_{direction} holds a {selection['selection_direction']} mask")
        mask = torch.load(
            gate_dir / f"train_{direction}/head_mask.pt", map_location="cpu", weights_only=False
        )
        current = json.loads(
            (gate_dir / f"train_{direction}/configuration.json").read_text()
        )["args"]
        if config is not None and any(
            current[key] != config[key]
            for key in ("injection_layer", "strength", "prompt_template", "prompt_preamble", "scale_mode")
        ):
            raise ValueError("gate-on and gate-off masks were trained under different settings")
        if model is not None and selection["model"] != model:
            raise ValueError("gate-on and gate-off masks name different models")
        config, model, n_heads = current, selection["model"], int(mask["n_heads"])
        selections[direction] = parse_head_group_spec(selection["gate_group"])[1]
        layers.update(int(layer) for layer in selection["layers_searched"])
    if int(bank["layer"]) != int(config["injection_layer"]):
        raise ValueError("concept vectors were extracted at a different layer than the masks")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    cluster_csv = clusters_dir / "test.csv"
    inputs = [
        concepts_dir / "validation.csv",
        concepts_dir / "bottom100_seed42.csv",
        cluster_csv,
        args.concept_vectors,
        *[gate_dir / f"train_{d}/{name}" for d in ("on", "off") for name in ("selected_heads.json", "configuration.json")],
    ]
    metadata = {
        "schema_version": 1,
        "status": "complete",
        "model": model,
        "population_definition": "fixed validation100 versus frozen random bottom100; no outcome ranking",
        "groups": list(GROUPS),
        "concepts": names,
        "concept_groups": groups,
        "inputs": {str(path): file_sha256(path) for path in inputs},
        "args": {
            "model": model,
            "model_results": str(args.model_results),
            "validation_concepts_csv": str(concepts_dir / "validation.csv"),
            "bottom_sample_csv": str(concepts_dir / "bottom100_seed42.csv"),
            "cluster_csv": str(cluster_csv),
            "concept_vectors": str(args.concept_vectors),
            "injection_layer": str(config["injection_layer"]),
            "strength": str(config["strength"]),
        },
        "frozen_prompt_config": {
            key: config[key] for key in ("prompt_template", "prompt_preamble", "scale_mode")
        },
        "components": [(layer, head) for layer in sorted(layers) for head in range(n_heads)],
        "selections": selections,
        "note": "both 100-concept groups are frozen before evaluation; all prompt clusters come from the test bank",
    }
    (output / "sources.json").write_text(json.dumps({"metadata": metadata}, indent=2) + "\n")

    shutil.copyfile(gate_dir / "train_on/configuration.json", output / "training_configuration.json")
    print(output / "sources.json")


if __name__ == "__main__":
    main()
