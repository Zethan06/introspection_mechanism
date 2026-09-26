#!/usr/bin/env python3
"""Train one formal STE Top-k number-output head mask on the train split."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from introspection_core.attention_routing import (  # noqa: E402
    candidate_routing_key_positions,
)
from introspection_core.cluster_split import file_sha256  # noqa: E402
from introspection_core.injected_trials import initialize_distributed  # noqa: E402
from introspection_core.head_mask_gate import (  # noqa: E402
    FORMAL_TRAINING_PROTOCOL,
    FORMAL_TOP_K,
    HEAD_MASK_SCHEMA_VERSION,
    TopKHeadMask,
    directional_head_gate_bce,
    formal_train_pair_filter,
    formal_training_pair_mask,
    hard_topk_mask,
    incremental_one_hot_router_hook,
    load_gate_on_candidate_trials,
    load_head_gate_dataset,
    masked_final_token_head_hooks,
    prepare_cached_head_gate_batch,
)
from introspection_core.head_output_patch import (  # noqa: E402
    head_group_spec,
    parse_layer_spec,
)
from introspection_core.model import HookedModel, ModelConfig  # noqa: E402
from introspection_core.results import write_table  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--direction", choices=("on", "off"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train_cluster_csv", type=Path, required=True)
    parser.add_argument("--train_concept_vectors", type=Path, required=True)
    parser.add_argument(
        "--train_outcomes_csv",
        type=Path,
        help="Locked injected outcomes; required only for gate-on candidates.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--layers", required=True, help="e.g. 17-23 or 14-16")
    parser.add_argument(
        "--top_k",
        type=int,
        default=FORMAL_TOP_K,
        help="Exact number of attention heads selected by the STE mask.",
    )
    parser.add_argument("--injection_layer", type=int, required=True)
    parser.add_argument("--strength", type=float, required=True)
    parser.add_argument(
        "--forced_router_layer",
        type=int,
        help="Required for gate-on; forbidden for gate-off.",
    )
    parser.add_argument(
        "--forced_router_heads",
        type=int,
        nargs="+",
        help="Required for gate-on; forbidden for gate-off.",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Per-rank batch size when --distributed is enabled.",
    )
    parser.add_argument("--learning_rate", type=float, default=3e-3)
    parser.add_argument("--gate_temperature", type=float, default=0.1)
    parser.add_argument("--ste_temperature_start", type=float, default=1.0)
    parser.add_argument("--ste_temperature_end", type=float, default=0.1)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--initialization_std", type=float, default=1e-3)
    parser.add_argument(
        "--prompt_template",
        default="semantic_highinj_posref_gate_balanced_disrupts",
    )
    parser.add_argument(
        "--prompt_preamble", choices=("none", "user", "system"), default="system"
    )
    parser.add_argument("--choice_suffix", default="")
    parser.add_argument(
        "--scale_mode",
        choices=("unit", "relative_hidden_norm"),
        default="relative_hidden_norm",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress_every", type=int, default=20)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    positive = {
        "strength": args.strength,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "gate_temperature": args.gate_temperature,
        "ste_temperature_start": args.ste_temperature_start,
        "ste_temperature_end": args.ste_temperature_end,
        "gradient_clip_norm": args.gradient_clip_norm,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        parser.error(f"these arguments must be positive: {invalid}")
    if args.initialization_std < 0:
        parser.error("--initialization_std must be non-negative")
    if args.top_k <= 0:
        parser.error("--top_k must be positive")

    router_supplied = (
        args.forced_router_layer is not None
        and args.forced_router_heads is not None
    )
    router_partial = (
        args.forced_router_layer is None
    ) != (args.forced_router_heads is None)
    if router_partial:
        parser.error(
            "--forced_router_layer and --forced_router_heads must be supplied together"
        )
    if args.direction == "on" and not router_supplied:
        parser.error("gate-on training requires the forced router")
    if args.direction == "on" and args.train_outcomes_csv is None:
        parser.error("gate-on training requires --train_outcomes_csv")
    if args.direction == "off" and router_supplied:
        parser.error("gate-off training must use the Native router")
    if args.direction == "off" and args.train_outcomes_csv is not None:
        parser.error("gate-off training uses the full grid and no outcomes CSV")
    if args.forced_router_heads is not None:
        if len(set(args.forced_router_heads)) != len(args.forced_router_heads):
            parser.error("--forced_router_heads must be unique")
    return args


def _temperature(
    step: int,
    *,
    total_steps: int,
    start: float,
    end: float,
) -> float:
    if total_steps <= 1:
        return float(end)
    fraction = min(max(step / (total_steps - 1), 0.0), 1.0)
    return float(start * ((end / start) ** fraction))


def _distributed_epoch_layout(
    candidate_count: int,
    *,
    per_rank_batch_size: int,
    world_size: int,
) -> tuple[int, int]:
    """Return a non-empty-per-rank sample count and global step count.

    Only a tail smaller than ``world_size`` must be dropped.  A partial global
    batch containing at least one example per rank is safe because gradients
    are synchronized with explicit sample-count weighting.
    """

    if candidate_count <= 0 or per_rank_batch_size <= 0 or world_size <= 0:
        raise ValueError("candidate count, batch size, and world size must be positive")
    usable_count = candidate_count - candidate_count % world_size
    if usable_count == 0:
        raise ValueError("Candidate pool is smaller than the distributed world size")
    global_batch_size = per_rank_batch_size * world_size
    batches_per_epoch = (usable_count + global_batch_size - 1) // global_batch_size
    return usable_count, batches_per_epoch


def _json_args(args: argparse.Namespace) -> dict:
    return {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def _synchronize_mask_gradient(
    parameter: torch.nn.Parameter,
    *,
    local_count: int,
) -> int:
    """All-reduce local mean gradients into a sample-weighted global mean."""

    import torch.distributed as dist

    global_count = torch.tensor(
        float(local_count), device=parameter.device, dtype=torch.float32
    )
    dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
    total = int(global_count.item())
    if parameter.grad is None:
        parameter.grad = torch.zeros_like(parameter)
    elif local_count > 0:
        parameter.grad.mul_(local_count)
    else:
        parameter.grad.zero_()
    dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
    if total > 0:
        parameter.grad.div_(global_count)
    return total


def _validate_model_configuration(
    args: argparse.Namespace,
    *,
    model: HookedModel,
    layers: Sequence[int],
) -> None:
    if not 0 <= args.injection_layer < int(model.cfg.n_layers):
        raise ValueError("Injection layer is outside the model")
    invalid_layers = [
        layer for layer in layers if not 0 <= layer < int(model.cfg.n_layers)
    ]
    if invalid_layers:
        raise ValueError(f"layers outside model range: {invalid_layers}")
    if min(layers) <= args.injection_layer:
        raise ValueError("Every searched head layer must follow the injection layer")
    if args.top_k > len(layers) * int(model.cfg.n_heads):
        raise ValueError(
            f"Top{args.top_k} exceeds the searched layer/head cell count"
        )
    if args.direction != "on":
        return
    assert args.forced_router_layer is not None
    assert args.forced_router_heads is not None
    if args.forced_router_layer <= max(layers):
        raise ValueError("Forced router layer must follow every searched head layer")
    if not 0 <= args.forced_router_layer < int(model.cfg.n_layers):
        raise ValueError("Forced router layer is outside the model")
    invalid_heads = [
        head
        for head in args.forced_router_heads
        if not 0 <= head < int(model.cfg.n_heads)
    ]
    if invalid_heads:
        raise ValueError(f"Forced router heads outside model: {invalid_heads}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    context = initialize_distributed(args.distributed)
    layers = parse_layer_spec(args.layers)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"non-empty output directory exists: {output_dir}")
    if context.is_primary:
        output_dir.mkdir(parents=True, exist_ok=True)
    if context.enabled:
        torch.distributed.barrier()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = HookedModel(
        ModelConfig(
            name=args.model,
            device=args.device,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            n_devices=1 if context.enabled else None,
        )
    )
    _validate_model_configuration(args, model=model, layers=layers)
    n_heads = int(model.cfg.n_heads)
    train_pair_filter = formal_train_pair_filter(args.direction)

    train_cluster_csv = args.train_cluster_csv.resolve()
    train_concept_vectors = args.train_concept_vectors.resolve()
    dataset = load_head_gate_dataset(
        model,
        cluster_csv=train_cluster_csv,
        concept_vectors_file=train_concept_vectors,
        injection_layer=args.injection_layer,
        prompt_template=args.prompt_template,
        prompt_preamble=args.prompt_preamble,
        choice_suffix=args.choice_suffix,
    )
    if args.direction == "on":
        assert args.train_outcomes_csv is not None
        candidate_trials = load_gate_on_candidate_trials(
            args.train_outcomes_csv.resolve(), dataset=dataset
        )
        candidate_pool = "locked_native_injected_exact_targets"
    else:
        candidate_trials = dataset.trials
        candidate_pool = "full_coordinate_grid"
    router_keys = None
    if args.direction == "on":
        router_keys = torch.tensor(
            [
                [
                    int(
                        candidate_routing_key_positions(
                            model.tokenizer, example
                        )[position]
                    )
                    for position in range(10)
                ]
                for example in dataset.examples
            ],
            dtype=torch.long,
        )

    for parameter in model.bridge.parameters():
        parameter.requires_grad_(False)
    model.bridge.eval()
    head_mask = TopKHeadMask(
        layers,
        n_heads=n_heads,
        top_k=args.top_k,
        device=model.bridge.cfg.device,
        seed=args.seed,
        initialization_std=args.initialization_std,
    )
    optimizer = torch.optim.Adam(
        [head_mask.scores], lr=args.learning_rate
    )
    global_batch_size = args.batch_size * context.world_size
    usable_count, batches_per_epoch = _distributed_epoch_layout(
        len(candidate_trials),
        per_rank_batch_size=args.batch_size,
        world_size=context.world_size,
    )
    total_steps = args.epochs * batches_per_epoch
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    history: list[dict] = []
    global_step = 0

    if context.is_primary:
        router_label = "forced_target_key" if args.direction == "on" else "native"
        print(
            f"model={args.model} split=train direction={args.direction} "
            f"layers={layers} top_k={args.top_k} "
            f"full_coordinates={len(dataset.trials)} "
            f"candidate_coordinates={len(candidate_trials)} "
            f"candidate_pool={candidate_pool} filter={train_pair_filter} "
            f"router={router_label} loss=bce_only "
            f"world_size={context.world_size} per_rank_batch={args.batch_size} "
            f"global_batch={global_batch_size} final_token_only=true",
            flush=True,
        )

    final_candidate_count = 0
    final_train_count = 0
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(candidate_trials), generator=generator).tolist()
        candidate_count = 0
        valid_count = 0
        loss_sum = 0.0
        ideal_count = 0
        exact_count = 0

        for start in range(0, usable_count, global_batch_size):
            global_indices = order[start : start + global_batch_size]
            local_indices = global_indices[context.rank :: context.world_size]
            if not local_indices:
                raise RuntimeError(
                    f"Distributed rank {context.rank} received an empty batch"
                )
            trials = [candidate_trials[index] for index in local_indices]
            batch = prepare_cached_head_gate_batch(
                model,
                trials=trials,
                dataset=dataset,
                layers=layers,
                injection_layer=args.injection_layer,
                strength=args.strength,
                scale_mode=args.scale_mode,
            )
            valid = formal_training_pair_mask(
                batch.clean_logits,
                batch.injected_logits,
                positions=batch.positions,
                direction=args.direction,
            )
            train_indices = valid.nonzero(as_tuple=False).squeeze(-1)
            count = int(train_indices.numel())
            candidate_count += len(trials)
            valid_count += count
            temperature = _temperature(
                global_step,
                total_steps=total_steps,
                start=args.ste_temperature_start,
                end=args.ste_temperature_end,
            )
            global_step += 1
            if count == 0 and not context.enabled:
                continue
            optimization_indices = (
                train_indices
                if count > 0
                else torch.zeros(1, dtype=torch.long)
            )

            optimizer.zero_grad(set_to_none=True)
            mask, _soft_mask, _hard_mask = head_mask(temperature=temperature)
            if args.direction == "on":
                donor_z = batch.injected_z
                recipient_cache = batch.clean_cache
            else:
                donor_z = batch.clean_z
                recipient_cache = batch.injected_cache
            hooks = masked_final_token_head_hooks(
                model,
                layers=layers,
                donor_z_by_layer=donor_z,
                mask=mask,
            )
            if args.direction == "on":
                assert router_keys is not None
                assert args.forced_router_layer is not None
                assert args.forced_router_heads is not None
                cluster_indices = torch.tensor(
                    [trial.cluster_index for trial in batch.trials],
                    dtype=torch.long,
                )
                forced_keys = router_keys[
                    cluster_indices, batch.positions
                ]
                hooks.append(
                    incremental_one_hot_router_hook(
                        model,
                        layer=args.forced_router_layer,
                        heads=args.forced_router_heads,
                        key_positions=forced_keys,
                    )
                )

            patched_all = model.differentiable_incremental_last_token_candidate_logits(
                batch.last_tokens,
                prefix_kv_cache=recipient_cache,
                prefix_length=batch.prefix_length,
                candidate_token_ids=dataset.candidate_token_ids,
                fwd_hooks=hooks,
            )
            device_indices = optimization_indices.to(patched_all.device)
            patched = patched_all.index_select(0, device_indices)
            loss = directional_head_gate_bce(
                patched,
                target_is_number=args.direction == "on",
                temperature=args.gate_temperature,
            )
            (loss if count > 0 else loss * 0.0).backward()
            if head_mask.scores.grad is None or not bool(
                torch.isfinite(head_mask.scores.grad).all()
            ):
                raise FloatingPointError("Head-mask gradient is missing or non-finite")
            global_valid = count
            if context.enabled:
                global_valid = _synchronize_mask_gradient(
                    head_mask.scores, local_count=count
                )
            if global_valid > 0:
                torch.nn.utils.clip_grad_norm_(
                    [head_mask.scores],
                    max_norm=args.gradient_clip_norm,
                    error_if_nonfinite=True,
                )
                optimizer.step()

            if count > 0:
                predictions = patched.detach().argmax(dim=-1).cpu()
                positions = batch.positions.index_select(
                    0, optimization_indices
                )
                loss_sum += float(loss.detach()) * count
                if args.direction == "on":
                    ideal_count += int(predictions.lt(10).sum())
                    exact_count += int(predictions.eq(positions).sum())
                else:
                    ideal_count += int(predictions.eq(10).sum())

            if (
                context.is_primary
                and args.progress_every > 0
                and global_step % args.progress_every == 0
            ):
                selected = ",".join(
                    f"L{layer}H{head}"
                    for layer, head in head_mask.selected_components()
                )
                print(
                    f"epoch={epoch} step={global_step} valid={valid_count}/"
                    f"{candidate_count} loss={float(loss):.6f} "
                    f"temp={temperature:.4f} selected={selected}",
                    flush=True,
                )

        if context.enabled:
            aggregate = torch.tensor(
                [
                    candidate_count,
                    valid_count,
                    loss_sum,
                    ideal_count,
                    exact_count,
                ],
                dtype=torch.float64,
                device=head_mask.scores.device,
            )
            torch.distributed.all_reduce(
                aggregate, op=torch.distributed.ReduceOp.SUM
            )
            candidate_count = int(aggregate[0].item())
            valid_count = int(aggregate[1].item())
            loss_sum = float(aggregate[2].item())
            ideal_count = int(aggregate[3].item())
            exact_count = int(aggregate[4].item())
        if valid_count == 0:
            raise RuntimeError(
                f"No train coordinates satisfied {train_pair_filter}"
            )
        row = {
            "epoch": epoch,
            "direction": args.direction,
            "global_steps": global_step,
            "candidate_coordinates": candidate_count,
            "trained_coordinates": valid_count,
            "train_pair_filter": train_pair_filter,
            "train_rate": valid_count / candidate_count,
            "mean_bce": loss_sum / valid_count,
            "ideal_output_rate": ideal_count / valid_count,
            "exact_target_rate": (
                exact_count / valid_count if args.direction == "on" else None
            ),
        }
        history.append(row)
        final_candidate_count = candidate_count
        final_train_count = valid_count
        if context.is_primary:
            print(json.dumps(row, indent=2), flush=True)

    selection_rows = head_mask.selection_rows()
    selected_rows = [row for row in selection_rows if row["selected"]]
    selected_rows.sort(key=lambda row: int(row["selection_rank"]))
    selected_components = [
        (int(row["layer"]), int(row["head"])) for row in selected_rows
    ]
    group_name = f"{args.direction}_top{args.top_k}"
    gate_group = head_group_spec(group_name, selected_rows)
    input_hashes = {
        "train_cluster_csv": file_sha256(train_cluster_csv),
        "train_concept_vectors": file_sha256(train_concept_vectors),
    }
    if args.train_outcomes_csv is not None:
        input_hashes["train_outcomes_csv"] = file_sha256(
            args.train_outcomes_csv.resolve()
        )
    router_mode = "forced_target_key" if args.direction == "on" else "native"
    objective = (
        f"clean_with_injected_top{args.top_k}_forced_router_number_bce"
        if args.direction == "on"
        else f"injected_with_clean_top{args.top_k}_native_router_none_bce"
    )
    checkpoint = {
        "schema_version": HEAD_MASK_SCHEMA_VERSION,
        "artifact_type": "ste_topk_head_gate",
        "formal_training_protocol": FORMAL_TRAINING_PROTOCOL,
        "model": args.model,
        "source_split": "train",
        "layers": layers,
        "n_heads": n_heads,
        "top_k": args.top_k,
        "scores": head_mask.scores.detach().float().cpu(),
        "hard_mask": hard_topk_mask(
            head_mask.scores.detach(), top_k=args.top_k
        ).cpu().bool(),
        "selected_components": selected_components,
        "group_name": group_name,
        "gate_group": gate_group,
        "selection_direction": args.direction,
        "objective": objective,
        "loss": "binary_cross_entropy_with_logits",
        "gate_score_definition": "tau_logmeanexp_number_minus_none",
        "gate_temperature": args.gate_temperature,
        "injection_layer": args.injection_layer,
        "strength": args.strength,
        "scale_mode": args.scale_mode,
        "training_router_mode": router_mode,
        "bce_router_mode": router_mode,
        "training_router_layer": args.forced_router_layer,
        "training_router_heads": args.forced_router_heads,
        "train_pair_filter": train_pair_filter,
        "train_candidate_pool": candidate_pool,
        "full_coordinate_count": len(dataset.trials),
        "candidate_coordinate_count": len(candidate_trials),
        "final_epoch_processed_candidate_count": final_candidate_count,
        "final_epoch_train_count": final_train_count,
        "prompt_template": args.prompt_template,
        "prompt_preamble": args.prompt_preamble,
        "choice_suffix": args.choice_suffix,
        "input_sha256": input_hashes,
        "seed": args.seed,
        "distributed_world_size": context.world_size,
        "per_rank_batch_size": args.batch_size,
        "effective_global_batch_size": global_batch_size,
    }
    checkpoint_path = output_dir / "head_mask.pt"
    if context.is_primary:
        torch.save(checkpoint, checkpoint_path)
        write_table(output_dir, "history.csv", history)
        write_table(output_dir, "head_scores.csv", selection_rows)
        manifest = {
            "schema_version": 1,
            "model": args.model,
            "source_split": "train",
            "formal_training_protocol": FORMAL_TRAINING_PROTOCOL,
            "selection_direction": args.direction,
            "objective": objective,
            "training_router_mode": router_mode,
            "group_name": group_name,
            "gate_group": gate_group,
            "top_k": args.top_k,
            "layers_searched": layers,
            "heads": selected_rows,
            "checkpoint": str(checkpoint_path),
        }
        (output_dir / "selected_heads.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        configuration = {
            "model": args.model,
            "dataset_split": "train",
            "formal_training_protocol": FORMAL_TRAINING_PROTOCOL,
            "direction": args.direction,
            "training_population": train_pair_filter,
            "training_candidate_pool": candidate_pool,
            "patch_site": "attention hook_z at final prompt token only",
            "patch_operation": (
                f"clean recipient <- paired injected Top{args.top_k} head outputs"
                if args.direction == "on"
                else f"injected recipient <- paired clean Top{args.top_k} head outputs"
            ),
            "training_router_mode": router_mode,
            "loss": (
                f"BCEWithLogits(d_tau(clean <- injected Top{args.top_k}, forced router), 1)"
                if args.direction == "on"
                else f"BCEWithLogits(d_tau(injected <- clean Top{args.top_k}, native router), 0)"
            ),
            "input_sha256": input_hashes,
            "args": _json_args(args),
            "checkpoint": "head_mask.pt",
            "selected_heads": "selected_heads.json",
        }
        (output_dir / "configuration.json").write_text(
            json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"saved formal {args.direction} Top{args.top_k} mask to "
            f"{checkpoint_path}"
        )
        print(f"selected {gate_group}")
    if context.enabled:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
