#!/usr/bin/env python3
"""Ablate final-token STE-head query/key score terms on frozen concept trials."""

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from introspection_core.qk_score_ablation import (
    rotated_capture, score_term, subtract_score_term, validate_frozen_prompt,
)


CONDITIONS = ("native", "without_query", "without_key")


def ablation_hooks(model, selected, clean, injected_prefix, mode):
    """Use current-run final Q/K at each layer, including upstream ablations."""
    hooks = []
    for layer, heads in selected.items():
        current = {}
        for name in ("q", "k"):
            def save(value, hook, name=name, current=current):
                del hook
                current[name] = value.detach()
                return value
            hooks.append((f"blocks.{layer}.attn.hook_rot_{name}", save))

        def patch(scores, hook, layer=layer, heads=heads, current=current):
            del hook
            if set(current) != {"q", "k"} or current["q"].shape[1] != 1 or current["k"].shape[1] != 1:
                raise RuntimeError(f"missing final-token Q/K at layer {layer}")
            k1 = torch.cat((injected_prefix[layer].to(scores.device), current["k"]), dim=1)
            k0 = clean[layer]["k"].to(scores.device)
            q0 = clean[layer]["q"].to(scores.device)
            q1 = current["q"][:, -1]
            term = score_term(k0, k1, q0, q1, heads, mode)
            if term.shape[-1] != scores.shape[-1]:
                raise RuntimeError("QK score and key context lengths disagree")
            current.clear()
            return subtract_score_term(scores, term, heads)
        hooks.append((model.attn_hook_name(layer, "qk_scores"), patch))
    return hooks


def summarize(predictions, positions, groups):
    rows = []
    for group in ("validation100", "bottom100"):
        mask = torch.tensor([g == group for g in groups]).repeat_interleave(10)[:len(positions)]
        if not bool(mask.any()):
            continue
        expected = positions[mask]
        native = predictions["native"][mask]
        for condition in CONDITIONS:
            current = predictions[condition][mask]
            rows.append({
                "group": group, "condition": condition, "n_trials": len(current),
                "number_count": int((current < 10).sum()),
                "number_rate": float((current < 10).float().mean()),
                "exact_count": int((current == expected).sum()),
                "exact_rate": float((current == expected).float().mean()),
                "none_count": int((current == 10).sum()),
                "native_number_to_none": int(((native < 10) & (current == 10)).sum()),
                "native_none_to_number": int(((native == 10) & (current < 10)).sum()),
            })
    return rows


@torch.inference_mode()
def run_cluster(model, example, vectors, source, selected, cluster, args):
    from introspection_core.injected_trials import InjectedTrial, build_injected_batch
    from introspection_core.label_accuracy import resolve_label_candidate_layout

    meta = source["metadata"]
    layout = resolve_label_candidate_layout([example])
    if layout.position_count != 10 or layout.labels[-1] != "none":
        raise ValueError("expected ten number labels followed by none")
    tokens_one = example.input_ids
    positions_one = torch.tensor([[example.injection_spans[p].start for p in range(10)]])
    if not (positions_one.min() > 0 and positions_one.max() < tokens_one.shape[1] - 1):
        raise ValueError("injection must be in the prefix")
    layers = sorted(selected)
    clean_prefix = {}
    cache = model.build_prefix_kv_cache(tokens_one, fwd_hooks=rotated_capture(layers, clean_prefix, "k"))
    clean_q, clean_last_k = {}, {}
    clean_logits, _ = model.incremental_last_token_candidate_stats(
        tokens_one[:, -1:], prefix_kv_cache=cache, prefix_length=tokens_one.shape[1] - 1,
        candidate_token_ids=layout.token_ids,
        fwd_hooks=[*rotated_capture(layers, clean_q, "q"),
                   *rotated_capture(layers, clean_last_k, "k")],
    )
    if set(clean_prefix) != set(layers) or set(clean_q) != set(layers) or set(clean_last_k) != set(layers):
        raise RuntimeError("clean Q/K capture incomplete")
    clean = {layer: {"k": torch.cat((clean_prefix[layer], clean_last_k[layer]), 1),
                     "q": clean_q[layer][:, -1]} for layer in layers}
    count = len(meta["concepts"]) * 10
    if args.trial_end is not None:
        count = min(count, args.trial_end)
    predictions = {name: torch.empty(count, dtype=torch.long) for name in CONDITIONS}
    position_indices = torch.arange(10).repeat(len(meta["concepts"]))[:count]
    start_time = time.monotonic()
    for start in range(0, count, args.batch_size):
        stop = min(start + args.batch_size, count)
        trials = [InjectedTrial(i // 10, 0, i % 10, False, False) for i in range(start, stop)]
        tokens, injection_hook = build_injected_batch(
            trials, base_tokens=tokens_one, injection_token_positions=positions_one,
            concept_vectors=vectors, model=model,
            injection_layer=int(meta["args"]["injection_layer"]),
            strength=float(meta["args"]["strength"]),
            scale_mode=meta["frozen_prompt_config"]["scale_mode"],
        )
        injected_prefix = {}
        injected_cache = model.build_prefix_kv_cache(
            tokens, fwd_hooks=[injection_hook, *rotated_capture(layers, injected_prefix, "k")])
        if set(injected_prefix) != set(layers):
            raise RuntimeError("injected key capture incomplete")
        common = dict(prefix_kv_cache=injected_cache, prefix_length=tokens.shape[1] - 1,
                      candidate_token_ids=layout.token_ids)
        native, _ = model.incremental_last_token_candidate_stats(tokens[:, -1:], **common)
        predictions["native"][start:stop] = native.argmax(-1)
        for condition, mode in (("without_query", "query"), ("without_key", "key")):
            logits, _ = model.incremental_last_token_candidate_stats(
                tokens[:, -1:], **common,
                fwd_hooks=ablation_hooks(model, selected, clean, injected_prefix, mode))
            predictions[condition][start:stop] = logits.argmax(-1)
        if start == 0 or stop == count or stop % (args.batch_size * 25) == 0:
            print(f"cluster={cluster} trials={stop}/{count} elapsed_s={time.monotonic()-start_time:.1f}", flush=True)
    rows = summarize(predictions, position_indices, meta["concept_groups"])
    return predictions, rows, int(clean_logits.argmax(-1)[0]), time.monotonic() - start_time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--results_dir", type=Path, required=True)
    cluster_group = parser.add_mutually_exclusive_group(required=True)
    cluster_group.add_argument("--cluster", type=int)
    cluster_group.add_argument("--all-clusters", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--trial-end", type=int, help="Preflight: exclusive trial limit")
    parser.add_argument("--resume", action="store_true", help="Skip completed, matching clusters")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.trial_end is not None and args.trial_end < 1):
        parser.error("batch_size and trial-end must be positive")
    if args.all_clusters and args.trial_end is not None:
        parser.error("--all-clusters requires full trials")
    source = json.loads((args.source / "sources.json").read_text())
    meta = source["metadata"]
    if len(meta["concepts"]) != 200 or meta["concept_groups"].count("validation100") != 100 or meta["concept_groups"].count("bottom100") != 100:
        raise ValueError("expected frozen valid100 and bottom100 populations")
    from introspection_core.attention_inputs import TokenLocalizationCsvTask
    from introspection_core.extraction import load_concept_vector_payload
    from introspection_core.injected_trials import validate_token0_9_candidate_layout
    from introspection_core.head_output_patch import GATE_MASK_DIR, load_head_selection, parse_head_group_spec
    from introspection_core.model import HookedModel, ModelConfig
    from introspection_core.prompts import PromptManager

    selection_path = Path(meta["args"]["model_results"]) / GATE_MASK_DIR / "train_on/selected_heads.json"
    selection = load_head_selection(selection_path)
    if selection["selection_direction"] != "on" or Path(selection["model"]).resolve() != Path(meta["model"]).resolve():
        raise ValueError("STE on-head selection/checkpoint mismatch")
    selected_heads = parse_head_group_spec(selection["gate_group"])[1]
    if len(selected_heads) != 32:
        raise ValueError("expected exactly 32 STE heads")
    selected = {}
    for layer, head in selected_heads:
        selected.setdefault(layer, []).append(head)
    torch.set_num_threads(4)
    model = HookedModel(ModelConfig(name=meta["model"], device=args.device, dtype="bfloat16"))
    config = json.loads(selection_path.with_name("configuration.json").read_text())
    frozen = meta["frozen_prompt_config"]
    examples = TokenLocalizationCsvTask(
        path=Path(meta["args"]["cluster_csv"]), preamble=frozen["prompt_preamble"],
        choice_suffix=config.get("choice_suffix", ""), position_index_start=0,
        template_name=frozen["prompt_template"], name=frozen["prompt_template"],
    ).build_examples(PromptManager(model.tokenizer))
    validate_token0_9_candidate_layout(examples)
    if len(examples) != 30 or (args.cluster is not None and not 0 <= args.cluster < len(examples)):
        raise ValueError("expected 30 frozen clusters and a valid cluster index")
    vectors = load_concept_vector_payload(
        Path(meta["args"]["concept_vectors"]), concepts=meta["concepts"],
        layer=int(meta["args"]["injection_layer"]))
    args.results_dir.mkdir(parents=True, exist_ok=True)
    clusters = range(30) if args.all_clusters else (args.cluster,)
    for cluster in clusters:
        validate_frozen_prompt(
            examples[cluster].input_ids, [examples[cluster].injection_spans[p].start for p in range(10)],
            args.source.parent / f"cluster_{cluster:02d}" / "complete.json")
        out = args.results_dir / f"cluster_{cluster:02d}"
        if out.exists():
            if not args.resume:
                raise FileExistsError(f"refusing overwrite: {out}")
            marker = out / "manifest.json"
            if not marker.is_file() or not (out / "predictions.pt").is_file() or not (out / "summary.csv").is_file():
                raise RuntimeError(f"incomplete cluster requires inspection: {out}")
            previous = json.loads(marker.read_text())
            expected = {"source": str(args.source), "selection": str(selection_path),
                        "model": meta["model"], "cluster": cluster,
                        "trial_end": args.trial_end or 2000, "batch_size": args.batch_size,
                        "selected_heads": [list(head) for head in selected_heads],
                        "conditions": list(CONDITIONS)}
            if any(previous.get(key) != value for key, value in expected.items()):
                raise ValueError(f"incompatible completed cluster: {out}")
            print(f"SKIP cluster={cluster}", flush=True)
            continue
        predictions, rows, clean_prediction, elapsed = run_cluster(
            model, examples[cluster], vectors, source, selected, cluster, args)
        out.mkdir()
        torch.save({"predictions": predictions, "trial_positions": torch.arange(10).repeat(200)[:len(predictions["native"])]}, out / "predictions.pt")
        for row in rows:
            row["cluster"] = cluster
            row["model"] = str(meta["model"])
            row["clean_prediction"] = clean_prediction
        with (out / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        (out / "manifest.json").write_text(json.dumps({
            "source": str(args.source), "selection": str(selection_path),
            "model": meta["model"], "cluster": cluster,
            "trial_end": len(predictions["native"]), "batch_size": args.batch_size,
            "clean_prediction": clean_prediction, "elapsed_seconds": elapsed,
            "selected_heads": selected_heads, "conditions": CONDITIONS,
        }, indent=2))
        print(f"COMPLETE cluster={cluster} elapsed_s={elapsed:.1f}", flush=True)
    print("RUN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
