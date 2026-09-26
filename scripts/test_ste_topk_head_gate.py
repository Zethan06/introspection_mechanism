#!/usr/bin/env python3
"""Evaluate one frozen formal STE Top-k mask on a held-out data split."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from introspection_core.attention_routing import (  # noqa: E402
    candidate_routing_key_positions,
)
from introspection_core.cluster_split import file_sha256  # noqa: E402
from introspection_core.donor_router_mismatch import (  # noqa: E402
    DonorRouterAccumulator,
    POSITION_COUNT,
    aggregate_position_rows,
    router_source_indices,
    sharded_complete_position_batches,
)
from introspection_core.injected_trials import (  # noqa: E402
    gate_logit,
    initialize_distributed,
)
from introspection_core.head_mask_gate import (  # noqa: E402
    TransitionStats,
    incremental_one_hot_router_hook,
    load_head_gate_dataset,
    load_head_mask_checkpoint,
    masked_final_token_head_hooks,
    prepare_cached_head_gate_batch,
)
from introspection_core.head_output_patch import (  # noqa: E402
    final_token_head_group_patch_hooks,
)
from introspection_core.model import HookedModel, ModelConfig  # noqa: E402
from introspection_core.prompts import REGISTRY  # noqa: E402


@dataclass
class OutputStats:
    n: int = 0
    none: int = 0
    number: int = 0
    exact: int = 0
    gate_sum: float = 0.0

    def update(
        self,
        logits: torch.Tensor,
        *,
        positions: torch.Tensor,
        temperature: float,
    ) -> None:
        selected = logits.detach().float()
        targets = positions.to(selected.device)
        predictions = selected.argmax(dim=-1)
        self.n += int(selected.shape[0])
        self.none += int(predictions.eq(10).sum())
        self.number += int(predictions.lt(10).sum())
        self.exact += int(predictions.eq(targets).sum())
        self.gate_sum += float(
            gate_logit(selected, temperature=temperature).sum()
        )

    def tensor(self, device: torch.device | str) -> torch.Tensor:
        return torch.tensor(
            [self.n, self.none, self.number, self.exact, self.gate_sum],
            dtype=torch.float64,
            device=device,
        )

    @classmethod
    def from_tensor(cls, values: torch.Tensor) -> "OutputStats":
        n, none, number, exact, gate_sum = values.detach().cpu().tolist()
        return cls(
            n=int(n),
            none=int(none),
            number=int(number),
            exact=int(exact),
            gate_sum=float(gate_sum),
        )

    def row(
        self,
        *,
        condition: str,
        gate_state: str,
        router: str,
    ) -> dict[str, object]:
        if self.n <= 0:
            raise RuntimeError("cannot summarize zero examples")
        return {
            "condition": condition,
            "gate_state": gate_state,
            "router": router,
            "n_trials": self.n,
            "none_rate": self.none / self.n,
            "number_rate": self.number / self.n,
            "exact_target_accuracy": self.exact / self.n,
            "mean_gate_score": self.gate_sum / self.n,
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--head_mask", type=Path, required=True)
    parser.add_argument("--test_cluster_csv", type=Path, required=True)
    parser.add_argument("--test_concept_vectors", type=Path, required=True)
    parser.add_argument(
        "--dataset_split",
        choices=("validation", "test"),
        default="test",
        help="Name recorded for the supplied evaluation inputs.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--recipient",
        choices=("selection", "clean", "injected"),
        default="selection",
        help=(
            "Recipient computation for the factorial. 'selection' preserves "
            "the checkpoint-direction default: clean for gate-on masks and "
            "injected for gate-off masks."
        ),
    )
    parser.add_argument(
        "--forced_router_layer",
        type=int,
        help="Optional downstream router for the post-training Forced test.",
    )
    parser.add_argument("--forced_router_heads", type=int, nargs="+")
    parser.add_argument(
        "--router_position_mode",
        choices=("matched", "all"),
        default="matched",
        help=(
            "Use the donor-matched router target, or evaluate the complete "
            "donor-position by router-position matrix."
        ),
    )
    parser.add_argument(
        "--router_intervention",
        choices=("forced_attention", "injected_output_patch"),
        default="forced_attention",
        help=(
            "For the all-position grid, either force router attention one-hot "
            "to j or patch the router head output from the natural injected-j "
            "state."
        ),
    )
    parser.add_argument(
        "--router_donor_state",
        choices=("opposite", "injected"),
        default="opposite",
        help=(
            "All-position output-patch grid only: take the router heads from "
            "the state the recipient is not in (opposite), or always from the "
            "natural injected run at j (injected). Fig 3d's gate-clean x "
            "router-injected cell needs injected on the injected recipient."
        ),
    )
    parser.add_argument(
        "--clean_router_patch",
        action="store_true",
        help=(
            "Matched mode only: replace the forced-router heads' final-token "
            "output with the paired clean run's value instead of forcing their "
            "attention, pinning the router to clean in both factorial cells."
        ),
    )
    parser.add_argument(
        "--prompt_template",
        help=(
            "Evaluate under this registered template instead of the one the "
            "mask was trained under; the mask itself is never re-optimized."
        ),
    )
    parser.add_argument(
        "--label_permutation",
        choices=("identity", "shuffled"),
        default="identity",
        help=(
            "shuffled draws one seeded derangement of the template's labels "
            "per cluster (--seed), as in the router label arms."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=32)
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
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")
    if (args.forced_router_layer is None) != (
        args.forced_router_heads is None
    ):
        parser.error(
            "--forced_router_layer and --forced_router_heads must be supplied together"
        )
    if args.forced_router_heads is not None and len(
        set(args.forced_router_heads)
    ) != len(args.forced_router_heads):
        parser.error("--forced_router_heads must be unique")
    if args.router_position_mode == "all" and args.forced_router_layer is None:
        parser.error(
            "--router_position_mode=all requires a router layer and heads"
        )
    if (
        args.router_position_mode != "all"
        and args.router_intervention != "forced_attention"
    ):
        parser.error("--router_intervention applies only to the all-position grid")
    if args.router_donor_state != "opposite" and (
        args.router_position_mode != "all"
        or args.router_intervention != "injected_output_patch"
    ):
        parser.error(
            "--router_donor_state applies only to the all-position "
            "injected_output_patch grid"
        )
    if args.clean_router_patch and (
        args.forced_router_layer is None or args.router_position_mode != "matched"
    ):
        parser.error(
            "--clean_router_patch requires router heads and --router_position_mode=matched"
        )
    if args.prompt_template is not None and args.prompt_template not in REGISTRY:
        parser.error(f"unknown --prompt_template {args.prompt_template!r}")
    return args


def _assert_test_split(
    checkpoint: dict,
    *,
    cluster_csv: Path,
    concept_vectors_file: Path,
) -> dict[str, str]:
    test_hashes = {
        "test_cluster_csv": file_sha256(cluster_csv),
        "test_concept_vectors": file_sha256(concept_vectors_file),
    }
    train_hashes = checkpoint.get("input_sha256", {})
    comparisons = (
        (
            "cluster",
            train_hashes.get("train_cluster_csv"),
            test_hashes["test_cluster_csv"],
        ),
        (
            "concept_vectors",
            train_hashes.get("train_concept_vectors"),
            test_hashes["test_concept_vectors"],
        ),
    )
    reused = [
        name
        for name, train_hash, test_hash in comparisons
        if train_hash is not None and train_hash == test_hash
    ]
    if reused:
        raise ValueError(
            "Evaluation inputs reuse train artifacts for: " + ", ".join(reused)
        )
    return test_hashes


def sharded_batches(
    items: Sequence,
    *,
    batch_size: int,
    rank: int,
    world_size: int,
) -> list[Sequence]:
    """Shard whole batches so results are stable across worker counts."""

    batches = [
        items[start : start + batch_size]
        for start in range(0, len(items), batch_size)
    ]
    return batches[rank::world_size]


def _condition_labels(
    *, recipient: str, top_k: int
) -> tuple[str, str, str, str]:
    """Return baseline/patch labels for one directional Top-k intervention."""

    if recipient == "clean":
        return (
            "clean_baseline",
            "off",
            f"clean_with_injected_top{top_k}",
            "on",
        )
    if recipient == "injected":
        return (
            "injected_baseline",
            "on",
            f"injected_with_clean_top{top_k}",
            "off",
        )
    raise ValueError(f"unsupported recipient state: {recipient!r}")


def _validate_model_configuration(
    args: argparse.Namespace,
    *,
    checkpoint: dict,
    model: HookedModel,
) -> None:
    layers = [int(layer) for layer in checkpoint["layers"]]
    invalid_layers = [
        layer for layer in layers if not 0 <= layer < int(model.cfg.n_layers)
    ]
    if invalid_layers:
        raise ValueError(f"Selected-head layers outside model: {invalid_layers}")
    injection_layer = int(checkpoint["injection_layer"])
    if not 0 <= injection_layer < int(model.cfg.n_layers):
        raise ValueError("Checkpoint injection layer is outside the model")
    if injection_layer >= min(layers):
        raise ValueError("Selected heads must follow the injection layer")
    if int(checkpoint["n_heads"]) != int(model.cfg.n_heads):
        raise ValueError("Checkpoint head count does not match the model")
    if args.forced_router_layer is None:
        return
    assert args.forced_router_heads is not None
    if args.forced_router_layer <= max(layers):
        raise ValueError("Forced test router must follow every selected-head layer")
    if not 0 <= args.forced_router_layer < int(model.cfg.n_layers):
        raise ValueError("Forced test router layer is outside the model")
    invalid_heads = [
        head
        for head in args.forced_router_heads
        if not 0 <= head < int(model.cfg.n_heads)
    ]
    if invalid_heads:
        raise ValueError(f"Forced test router heads outside model: {invalid_heads}")
    if (
        checkpoint["selection_direction"] == "on"
        and getattr(args, "router_intervention", "forced_attention")
        == "forced_attention"
        and not getattr(args, "clean_router_patch", False)
    ):
        trained_layer = checkpoint.get("training_router_layer")
        trained_heads = checkpoint.get("training_router_heads")
        if trained_layer is not None and int(trained_layer) != args.forced_router_layer:
            raise ValueError("Forced test router layer differs from gate-on training")
        if trained_heads is not None and [int(head) for head in trained_heads] != [
            int(head) for head in args.forced_router_heads
        ]:
            raise ValueError("Forced test router heads differ from gate-on training")


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    context = initialize_distributed(args.distributed)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"non-empty output directory exists: {output_dir}")
    if context.is_primary:
        output_dir.mkdir(parents=True, exist_ok=True)
    if context.enabled:
        torch.distributed.barrier()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    checkpoint = load_head_mask_checkpoint(
        args.head_mask.resolve(), expected_top_k=None
    )
    if str(checkpoint.get("model")) != args.model:
        raise ValueError("head-mask model does not match --model")

    test_cluster_csv = args.test_cluster_csv.resolve()
    test_concept_vectors = args.test_concept_vectors.resolve()
    test_hashes = _assert_test_split(
        checkpoint,
        cluster_csv=test_cluster_csv,
        concept_vectors_file=test_concept_vectors,
    )
    model = HookedModel(
        ModelConfig(
            name=args.model,
            device=args.device,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            n_devices=1 if context.enabled else None,
        )
    )
    _validate_model_configuration(args, checkpoint=checkpoint, model=model)

    direction = str(checkpoint["selection_direction"])
    top_k = int(checkpoint["top_k"])
    recipient = (
        ("clean" if direction == "on" else "injected")
        if args.recipient == "selection"
        else args.recipient
    )
    transition_direction = "on" if recipient == "clean" else "off"
    donor_state = "injected" if recipient == "clean" else "clean"
    router_donor_state = (
        "injected" if args.router_donor_state == "injected" else donor_state
    )
    (
        baseline_condition,
        baseline_gate_state,
        patch_condition,
        patch_gate_state,
    ) = _condition_labels(recipient=recipient, top_k=top_k)
    if args.router_position_mode == "all":
        expected_recipient = "clean" if direction == "on" else "injected"
        if recipient != expected_recipient:
            raise ValueError(
                "the donor-router grid requires the checkpoint-direction "
                f"recipient ({expected_recipient})"
            )
    layers = [int(layer) for layer in checkpoint["layers"]]
    capture_layers = list(layers)
    if args.router_intervention == "injected_output_patch" or args.clean_router_patch:
        assert args.forced_router_layer is not None
        capture_layers = sorted(set([*layers, args.forced_router_layer]))
    hard_mask = checkpoint["hard_mask"].float().to(model.bridge.cfg.device)
    mask_prompt_template = str(checkpoint["prompt_template"])
    prompt_template = args.prompt_template or mask_prompt_template
    dataset = load_head_gate_dataset(
        model,
        cluster_csv=test_cluster_csv,
        concept_vectors_file=test_concept_vectors,
        injection_layer=int(checkpoint["injection_layer"]),
        prompt_template=prompt_template,
        prompt_preamble=str(checkpoint["prompt_preamble"]),
        choice_suffix=str(checkpoint.get("choice_suffix", "")),
        label_permutation=args.label_permutation,
        label_seed=args.seed,
    )
    label_arm = {
        "prompt_template": prompt_template,
        "label_permutation": args.label_permutation,
        "label_seed": args.seed,
        "mask_prompt_template": mask_prompt_template,
        "transferred_mask": (
            prompt_template != mask_prompt_template
            or args.label_permutation != "identity"
        ),
        "slot_labels_cluster0": [
            str(dataset.examples[0].expected_candidate_by_position[position])
            for position in range(10)
        ],
        "scored_candidates": [
            str(label) for label in dataset.examples[0].candidate_token_ids
        ],
    }
    if context.is_primary:
        print(f"label arm: {json.dumps(label_arm)}", flush=True)
    router_keys = None
    if (
        args.forced_router_layer is not None
        and args.router_intervention == "forced_attention"
        and not args.clean_router_patch
    ):
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

    if (
        args.router_position_mode == "all"
        and args.router_intervention == "injected_output_patch"
    ):
        local_batches = sharded_complete_position_batches(
            dataset.trials,
            batch_size=args.batch_size,
            rank=context.rank,
            world_size=context.world_size,
        )
    else:
        local_batches = sharded_batches(
            dataset.trials,
            batch_size=args.batch_size,
            rank=context.rank,
            world_size=context.world_size,
        )
    local_trial_count = sum(len(batch) for batch in local_batches)
    routers = ["native"]
    if router_keys is not None:
        routers.append("forced")
    if args.clean_router_patch:
        routers.append("clean_patch")
    stats = {
        (router, intervention): OutputStats()
        for router in routers
        for intervention in ("baseline", "topk_patch")
    }
    transitions = {router: TransitionStats() for router in routers}
    mismatch_stats = {
        (route_position, donor_position, intervention): DonorRouterAccumulator(
            donor_position=donor_position,
            router_position=route_position,
        )
        for route_position in range(POSITION_COUNT)
        for donor_position in range(POSITION_COUNT)
        for intervention in (baseline_condition, patch_condition)
    }
    if context.is_primary:
        print(
            f"model={args.model} split={args.dataset_split} direction={direction} "
            f"coordinates={len(dataset.trials)} top_k={checkpoint['top_k']} "
            f"routers={routers} world_size={context.world_size}",
            flush=True,
        )

    for batch_index, trials in enumerate(local_batches, start=1):
        batch = prepare_cached_head_gate_batch(
            model,
            trials=trials,
            dataset=dataset,
            layers=capture_layers,
            injection_layer=int(checkpoint["injection_layer"]),
            strength=float(checkpoint["strength"]),
            scale_mode=str(checkpoint["scale_mode"]),
        )

        def slot_order(logits: torch.Tensor) -> torch.Tensor:
            return dataset.slot_order_logits(logits, batch.trials)

        if recipient == "clean":
            baseline_logits = batch.clean_logits
            recipient_cache = batch.clean_cache
            donor_z = batch.injected_z
        else:
            baseline_logits = batch.injected_logits
            recipient_cache = batch.injected_cache
            donor_z = batch.clean_z
        baseline_logits = slot_order(baseline_logits)
        patch_hooks = masked_final_token_head_hooks(
            model,
            layers=layers,
            donor_z_by_layer=donor_z,
            mask=hard_mask,
        )
        if args.router_position_mode == "matched":
            patched_native = slot_order(
                model.incremental_last_token_candidate_stats(
                    batch.last_tokens,
                    prefix_kv_cache=recipient_cache,
                    prefix_length=batch.prefix_length,
                    candidate_token_ids=dataset.candidate_token_ids,
                    fwd_hooks=patch_hooks,
                )[0]
            )
            stats[("native", "baseline")].update(
                baseline_logits,
                positions=batch.positions,
                temperature=float(checkpoint["gate_temperature"]),
            )
            stats[("native", "topk_patch")].update(
                patched_native,
                positions=batch.positions,
                temperature=float(checkpoint["gate_temperature"]),
            )
            transitions["native"].update(
                baseline_logits,
                patched_native,
                direction=transition_direction,
            )

        if router_keys is not None and args.router_position_mode == "matched":
            assert args.forced_router_layer is not None
            assert args.forced_router_heads is not None
            keys = torch.tensor(
                [
                    int(router_keys[trial.cluster_index, trial.position])
                    for trial in batch.trials
                ],
                dtype=torch.long,
            )
            router_hook = incremental_one_hot_router_hook(
                model,
                layer=args.forced_router_layer,
                heads=args.forced_router_heads,
                key_positions=keys,
            )
            forced_baseline = slot_order(
                model.incremental_last_token_candidate_stats(
                    batch.last_tokens,
                    prefix_kv_cache=recipient_cache,
                    prefix_length=batch.prefix_length,
                    candidate_token_ids=dataset.candidate_token_ids,
                    fwd_hooks=[router_hook],
                )[0]
            )
            forced_patched = slot_order(
                model.incremental_last_token_candidate_stats(
                    batch.last_tokens,
                    prefix_kv_cache=recipient_cache,
                    prefix_length=batch.prefix_length,
                    candidate_token_ids=dataset.candidate_token_ids,
                    fwd_hooks=[*patch_hooks, router_hook],
                )[0]
            )
            stats[("forced", "baseline")].update(
                forced_baseline,
                positions=batch.positions,
                temperature=float(checkpoint["gate_temperature"]),
            )
            stats[("forced", "topk_patch")].update(
                forced_patched,
                positions=batch.positions,
                temperature=float(checkpoint["gate_temperature"]),
            )
            transitions["forced"].update(
                forced_baseline,
                forced_patched,
                direction=transition_direction,
            )
        elif args.clean_router_patch:
            assert args.forced_router_layer is not None
            assert args.forced_router_heads is not None
            router_layer = int(args.forced_router_layer)
            # Pin the router to the paired clean run. Leaving it native in an
            # injected recipient would still read injected prefix keys/values.
            clean_router_hooks = final_token_head_group_patch_hooks(
                model,
                components=[
                    (router_layer, int(head)) for head in args.forced_router_heads
                ],
                source_z_by_layer={router_layer: batch.clean_z[router_layer]},
            )
            router_clean_baseline = slot_order(
                model.incremental_last_token_candidate_stats(
                    batch.last_tokens,
                    prefix_kv_cache=recipient_cache,
                    prefix_length=batch.prefix_length,
                    candidate_token_ids=dataset.candidate_token_ids,
                    fwd_hooks=clean_router_hooks,
                )[0]
            )
            router_clean_patched = slot_order(
                model.incremental_last_token_candidate_stats(
                    batch.last_tokens,
                    prefix_kv_cache=recipient_cache,
                    prefix_length=batch.prefix_length,
                    candidate_token_ids=dataset.candidate_token_ids,
                    fwd_hooks=[*patch_hooks, *clean_router_hooks],
                )[0]
            )
            stats[("clean_patch", "baseline")].update(
                router_clean_baseline,
                positions=batch.positions,
                temperature=float(checkpoint["gate_temperature"]),
            )
            stats[("clean_patch", "topk_patch")].update(
                router_clean_patched,
                positions=batch.positions,
                temperature=float(checkpoint["gate_temperature"]),
            )
            transitions["clean_patch"].update(
                router_clean_baseline,
                router_clean_patched,
                direction=transition_direction,
            )
        elif (
            router_keys is not None
            and args.router_intervention == "forced_attention"
        ):
            assert args.forced_router_layer is not None
            assert args.forced_router_heads is not None
            for route_position in range(POSITION_COUNT):
                keys = torch.tensor(
                    [
                        int(router_keys[trial.cluster_index, route_position])
                        for trial in batch.trials
                    ],
                    dtype=torch.long,
                )
                router_hook = incremental_one_hot_router_hook(
                    model,
                    layer=args.forced_router_layer,
                    heads=args.forced_router_heads,
                    key_positions=keys,
                )
                forced_baseline = slot_order(
                    model.incremental_last_token_candidate_stats(
                        batch.last_tokens,
                        prefix_kv_cache=recipient_cache,
                        prefix_length=batch.prefix_length,
                        candidate_token_ids=dataset.candidate_token_ids,
                        fwd_hooks=[router_hook],
                    )[0]
                )
                forced_patched = slot_order(
                    model.incremental_last_token_candidate_stats(
                        batch.last_tokens,
                        prefix_kv_cache=recipient_cache,
                        prefix_length=batch.prefix_length,
                        candidate_token_ids=dataset.candidate_token_ids,
                        fwd_hooks=[*patch_hooks, router_hook],
                    )[0]
                )
                for donor_position in range(POSITION_COUNT):
                    donor_rows = batch.positions.eq(donor_position).to(
                        forced_baseline.device
                    )
                    mismatch_stats[
                        (route_position, donor_position, baseline_condition)
                    ].update(
                        forced_baseline[donor_rows],
                        gate_temperature=float(checkpoint["gate_temperature"]),
                    )
                    mismatch_stats[
                        (
                            route_position,
                            donor_position,
                            patch_condition,
                        )
                    ].update(
                        forced_patched[donor_rows],
                        gate_temperature=float(checkpoint["gate_temperature"]),
                    )
        elif args.router_position_mode == "all":
            assert args.forced_router_layer is not None
            assert args.forced_router_heads is not None
            router_layer = int(args.forced_router_layer)
            # By default the ENV donor mirrors the Top-k donor: the state the
            # recipient is not in. --router_donor_state=injected instead always
            # reads the natural injected run at j, as the 2026-09-12 Fig 3d
            # grid did; on an injected recipient its diagonal is then the
            # native router.
            donor_router_z = (
                batch.injected_z
                if args.router_donor_state == "injected"
                else donor_z
            )[router_layer]
            router_components = [
                (router_layer, int(head))
                for head in args.forced_router_heads
            ]
            for route_position in range(POSITION_COUNT):
                source_indices = router_source_indices(
                    batch.trials,
                    router_position=route_position,
                )
                router_source_z = donor_router_z.index_select(
                    0, source_indices
                )
                router_patch_hooks = final_token_head_group_patch_hooks(
                    model,
                    components=router_components,
                    source_z_by_layer={router_layer: router_source_z},
                )
                patched_baseline = slot_order(
                    model.incremental_last_token_candidate_stats(
                        batch.last_tokens,
                        prefix_kv_cache=recipient_cache,
                        prefix_length=batch.prefix_length,
                        candidate_token_ids=dataset.candidate_token_ids,
                        fwd_hooks=router_patch_hooks,
                    )[0]
                )
                patched_topk = slot_order(
                    model.incremental_last_token_candidate_stats(
                        batch.last_tokens,
                        prefix_kv_cache=recipient_cache,
                        prefix_length=batch.prefix_length,
                        candidate_token_ids=dataset.candidate_token_ids,
                        fwd_hooks=[*patch_hooks, *router_patch_hooks],
                    )[0]
                )
                for donor_position in range(POSITION_COUNT):
                    donor_rows = batch.positions.eq(donor_position).to(
                        patched_baseline.device
                    )
                    mismatch_stats[
                        (route_position, donor_position, baseline_condition)
                    ].update(
                        patched_baseline[donor_rows],
                        gate_temperature=float(checkpoint["gate_temperature"]),
                    )
                    mismatch_stats[
                        (
                            route_position,
                            donor_position,
                            patch_condition,
                        )
                    ].update(
                        patched_topk[donor_rows],
                        gate_temperature=float(checkpoint["gate_temperature"]),
                    )

        if (
            context.is_primary
            and args.progress_every > 0
            and batch_index % args.progress_every == 0
        ):
            if args.router_position_mode == "matched":
                progress = stats[("native", "baseline")].n
            else:
                progress = sum(
                    mismatch_stats[(0, donor, baseline_condition)].n
                    for donor in range(POSITION_COUNT)
                )
            print(
                f"rank0_batch={batch_index} rank0_scored={progress}/"
                f"{local_trial_count}",
                flush=True,
            )

    if args.router_position_mode == "all":
        ordered_mismatch_keys = list(mismatch_stats)
        packed_mismatch = torch.stack(
            [
                mismatch_stats[key].tensor(hard_mask.device)
                for key in ordered_mismatch_keys
            ]
        )
        if context.enabled:
            torch.distributed.all_reduce(
                packed_mismatch, op=torch.distributed.ReduceOp.SUM
            )
        mismatch_stats = {
            key: DonorRouterAccumulator.from_tensor(
                packed_mismatch[index],
                donor_position=key[1],
                router_position=key[0],
            )
            for index, key in enumerate(ordered_mismatch_keys)
        }
        expected_per_cell = len(dataset.trials) // POSITION_COUNT
        incomplete = {
            key: value.n
            for key, value in mismatch_stats.items()
            if value.n != expected_per_cell
        }
        if incomplete:
            raise RuntimeError(
                f"donor-router grid has incomplete cells: {incomplete}"
            )
        if context.is_primary:
            position_rows = [
                mismatch_stats[key].row(intervention=key[2])
                for key in ordered_mismatch_keys
            ]
            aggregate_rows = aggregate_position_rows(position_rows)
            with (output_dir / "position_results.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(position_rows[0])
                )
                writer.writeheader()
                writer.writerows(position_rows)
            with (output_dir / "aggregate_results.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(aggregate_rows[0])
                )
                writer.writeheader()
                writer.writerows(aggregate_rows)
            summary = {
                "model": args.model,
                "dataset_split": args.dataset_split,
                "evaluation_population": "full_concept_cluster_position_grid",
                f"{args.dataset_split}_input_sha256": test_hashes,
                "head_mask": str(args.head_mask.resolve()),
                "selection_direction": direction,
                "top_k": top_k,
                "recipient_state": recipient,
                "selected_components": checkpoint["selected_components"],
                "patch_site": "attention hook_z at final prompt token only",
                "patch_operation": (
                    f"clean recipient <- paired injected Top{top_k} donor outputs "
                    "captured at donor position i"
                    if recipient == "clean"
                    else f"injected recipient <- paired clean Top{top_k} donor "
                    "outputs captured at donor position i"
                ),
                "router_intervention": {
                    "layer": args.forced_router_layer,
                    "heads": args.forced_router_heads,
                    "kind": args.router_intervention,
                    "patch_site": (
                        "attention pattern at final query"
                        if args.router_intervention == "forced_attention"
                        else "attention hook_z at final token"
                    ),
                    "donor_state": (
                        None
                        if args.router_intervention == "forced_attention"
                        else f"natural fully {router_donor_state} state at position j"
                    ),
                },
                "design": (
                    f"complete Top{top_k} donor i x forced ENV position j"
                    if args.router_intervention == "forced_attention"
                    else f"complete Top{top_k} donor i x {router_donor_state} "
                    "ENV-head-output donor j"
                ),
                "accuracy_denominator": "all trials, including none predictions",
                "accuracy_target": (
                    "forced router position j"
                    if args.router_intervention == "forced_attention"
                    else "router-state donor position j"
                ),
                "n_position_cells_per_intervention": POSITION_COUNT ** 2,
                "n_trials_per_position_cell": expected_per_cell,
                "label_arm": label_arm,
                "world_size": context.world_size,
                "per_rank_batch_size": args.batch_size,
                "args": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "aggregate_results": aggregate_rows,
            }
            (output_dir / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(aggregate_rows, indent=2), flush=True)
            print(f"wrote {output_dir}", flush=True)
        if context.enabled:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
        return

    ordered_keys = [
        (router, intervention)
        for router in routers
        for intervention in ("baseline", "topk_patch")
    ]
    packed = torch.stack(
        [stats[key].tensor(hard_mask.device) for key in ordered_keys]
    )
    if context.enabled:
        torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
    stats = {
        key: OutputStats.from_tensor(packed[index])
        for index, key in enumerate(ordered_keys)
    }
    packed_transitions = torch.stack(
        [transitions[router].tensor(hard_mask.device) for router in routers]
    )
    if context.enabled:
        torch.distributed.all_reduce(
            packed_transitions, op=torch.distributed.ReduceOp.SUM
        )
    transitions = {
        router: TransitionStats.from_tensor(packed_transitions[index])
        for index, router in enumerate(routers)
    }
    for key, value in stats.items():
        if value.n != len(dataset.trials):
            raise RuntimeError(
                f"full {args.dataset_split} grid incomplete for {key}: {value.n} != "
                f"{len(dataset.trials)}"
            )

    rows = []
    for router in routers:
        rows.append(
            stats[(router, "baseline")].row(
                condition=baseline_condition,
                gate_state=baseline_gate_state,
                router=router,
            )
        )
        rows.append(
            stats[(router, "topk_patch")].row(
                condition=patch_condition,
                gate_state=patch_gate_state,
                router=router,
            )
        )

    transition_rows = [
        transitions[router].row(direction=transition_direction, router=router)
        for router in routers
    ]
    if context.is_primary:
        with (output_dir / "results.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with (output_dir / "transitions.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(transition_rows[0])
            )
            writer.writeheader()
            writer.writerows(transition_rows)
        summary = {
            "model": args.model,
            "dataset_split": args.dataset_split,
            "evaluation_population": "full_concept_cluster_position_grid",
            f"{args.dataset_split}_input_sha256": test_hashes,
            "head_mask": str(args.head_mask.resolve()),
            "selection_direction": direction,
            "top_k": top_k,
            "recipient_state": recipient,
            "selected_components": checkpoint["selected_components"],
            "patch_site": "attention hook_z at final prompt token only",
            "patch_operation": (
                f"clean recipient <- paired injected Top{top_k} head outputs"
                if recipient == "clean"
                else f"injected recipient <- paired clean Top{top_k} head outputs"
            ),
            "forced_router": (
                None
                if args.forced_router_layer is None or args.clean_router_patch
                else {
                    "layer": args.forced_router_layer,
                    "heads": args.forced_router_heads,
                    "intervention": "final query one-hot to target candidate routing key",
                }
            ),
            "clean_router_patch": (
                None
                if not args.clean_router_patch
                else {
                    "layer": args.forced_router_layer,
                    "heads": args.forced_router_heads,
                    "patch_site": "attention hook_z at final token",
                    "donor_state": "paired clean run",
                    "router_label": "clean_patch",
                }
            ),
            "label_arm": label_arm,
            "world_size": context.world_size,
            "per_rank_batch_size": args.batch_size,
            "results": rows,
            "transitions": transition_rows,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        printable = {"results": rows, "transitions": transition_rows}
        print(json.dumps(printable, indent=2), flush=True)
        print(f"wrote {output_dir}", flush=True)
    if context.enabled:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
