"""Collect cluster-averaged latent points for explicit vector injection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .cluster_aggregation import RankedCluster, mean_full_vectors
from .hooks import write_span
from .layer_cache import run_injected_and_capture


@dataclass
class AveragedInjectionProjection:
    """Aligned points ready for projection and artifact generation."""

    rows: list[dict[str, object]]
    residuals: dict[int, torch.Tensor]
    head_z: torch.Tensor | None
    clean_prediction: str
    cluster_metrics: list[dict[str, object]]


def _candidate_predictions(
    logits: torch.Tensor,
    candidate_ids: dict[str, int],
) -> tuple[list[str], torch.Tensor]:
    labels = list(candidate_ids)
    ids = torch.tensor(
        [candidate_ids[label] for label in labels],
        device=logits.device,
        dtype=torch.long,
    )
    scores = logits.index_select(-1, ids)
    return labels, scores.argmax(dim=-1).detach().cpu()


def _clean_target_label(rendered) -> str | None:
    """Return the template's clean target, with legacy optional-none support."""
    target = getattr(rendered, "clean_target_label", None)
    if target is not None:
        return str(target)
    if "none" in rendered.answer_token_by_choice:
        return "none"
    return None


def _run_injected_candidate_capture(
    model,
    tokens: torch.Tensor,
    *,
    injection_layer: int,
    span: tuple[int, int],
    vectors: torch.Tensor,
    strength: float,
    scale_mode: str,
    candidate_token_ids: Sequence[int],
    capture_last_hooks: dict[str, torch.Tensor],
    capture_token_index: int = -1,
) -> torch.Tensor:
    """Return candidate logits while capturing latents at one token index."""
    injection_hook_name = model.resid_hook_name(injection_layer)

    def injection_hook(activation, hook):
        del hook
        return write_span(
            activation,
            spans=[span] * int(activation.shape[0]),
            value=vectors,
            mode="add",
            strength=strength,
            scale=scale_mode,
        )

    hooks = [(injection_hook_name, injection_hook)]
    for hook_name in list(capture_last_hooks):
        def make_capture(name: str):
            def capture(activation, hook):
                del hook
                capture_last_hooks[name] = (
                    activation[:, capture_token_index].detach().cpu().float()
                )
                return activation

            return capture

        hooks.append((hook_name, make_capture(hook_name)))

    candidate_logits, _ = model.last_token_candidate_stats(
        tokens,
        candidate_token_ids=candidate_token_ids,
        fwd_hooks=hooks,
    )
    return candidate_logits


def collect_averaged_injection_projection(
    model,
    prompt_manager,
    *,
    clusters: Sequence[RankedCluster],
    concept_names: Sequence[str],
    concept_vectors: torch.Tensor,
    injection_layer: int,
    coeffs: Sequence[float],
    scale_mode: str,
    prompt_preamble: str,
    observe_layers: Sequence[int],
    concept_batch_size: int,
    prompt_template: str = "token_localization",
    head_layer: int | None = None,
    positions: Sequence[int] | None = None,
    capture_token_index: int = -1,
) -> AveragedInjectionProjection:
    """Inject every concept into every aligned cluster position and average."""
    if not clusters:
        raise ValueError("clusters must be non-empty")
    if not concept_names:
        raise ValueError("concept_names must be non-empty")
    if concept_batch_size < 1:
        raise ValueError("concept_batch_size must be positive")
    if len(concept_names) != int(concept_vectors.shape[0]):
        raise ValueError("concept names and vectors must have equal lengths")

    n_clusters = len(clusters)
    n_positions = len(clusters[0].choices)
    selected_positions = (
        list(range(n_positions))
        if positions is None
        else [int(position) for position in positions]
    )
    if not selected_positions:
        raise ValueError("positions must be non-empty")
    if len(set(selected_positions)) != len(selected_positions):
        raise ValueError("positions must be unique")
    if any(
        position < 0 or position >= n_positions
        for position in selected_positions
    ):
        raise ValueError(
            f"positions must be in 0..{n_positions - 1}, got "
            f"{selected_positions}"
        )
    n_concepts = len(concept_names)
    n_coeffs = len(coeffs)
    representative = prompt_manager.render(
        prompt_template,
        list(clusters[0].choices),
        preamble=prompt_preamble,
    )
    clean_target_label = _clean_target_label(representative)
    if (
        clean_target_label is not None
        and clean_target_label not in representative.answer_token_by_choice
    ):
        raise ValueError(
            f"clean target {clean_target_label!r} is not a candidate for "
            f"prompt template {prompt_template!r}"
        )
    n_candidates = len(representative.answer_token_by_choice)
    total_points = (
        1 + len(selected_positions) * n_coeffs * n_concepts
    )
    d_model = int(model.cfg.d_model)
    residual_sums = {
        int(layer): torch.zeros(total_points, d_model, dtype=torch.float32)
        for layer in observe_layers
    }
    vote_counts = torch.zeros(
        total_points,
        n_candidates,
        dtype=torch.int32,
    )
    z_sums: torch.Tensor | None = None
    residual_hooks = {
        model.resid_hook_name(int(layer)): int(layer)
        for layer in observe_layers
    }
    z_hook_name = (
        model.attn_hook_name(head_layer, "z")
        if head_layer is not None
        else None
    )
    canonical_labels: list[str] | None = None
    canonical_ids: dict[str, int] | None = None
    cluster_metrics: list[dict[str, object]] = []

    for cluster_offset, cluster in enumerate(clusters, start=1):
        rendered = prompt_manager.render(
            prompt_template,
            list(cluster.choices),
            preamble=prompt_preamble,
        )
        sequence_length = int(rendered.input_ids.shape[1])
        if not -sequence_length <= capture_token_index < sequence_length:
            raise ValueError(
                f"capture_token_index {capture_token_index} is invalid for "
                f"cluster rank {cluster.rank} with sequence length "
                f"{sequence_length}"
            )
        if len(rendered.spans) != n_positions:
            raise ValueError(
                f"cluster rank {cluster.rank} rendered {len(rendered.spans)} "
                f"positions; expected {n_positions}"
            )
        candidate_ids = rendered.answer_token_by_choice
        if _clean_target_label(rendered) != clean_target_label:
            raise ValueError(
                f"cluster rank {cluster.rank} changed the clean target label"
            )
        labels = list(candidate_ids)
        if canonical_labels is None:
            canonical_labels = labels
            canonical_ids = candidate_ids
        elif labels != canonical_labels or candidate_ids != canonical_ids:
            raise ValueError(
                f"cluster rank {cluster.rank} changed answer-token mapping"
            )

        hook_names = set(residual_hooks)
        if z_hook_name is not None:
            hook_names.add(z_hook_name)
        with torch.inference_mode():
            clean_logits, clean_cache = model.run_with_cache(
                rendered.input_ids,
                names=lambda name: name in hook_names,
            )
        _, clean_predictions = _candidate_predictions(
            clean_logits[:, -1],
            candidate_ids,
        )
        vote_counts[0, int(clean_predictions[0])] += 1
        clean_prediction = labels[int(clean_predictions[0])]
        for hook_name, layer in residual_hooks.items():
            residual_sums[layer][0] += (
                clean_cache[hook_name][0, capture_token_index]
                .detach()
                .cpu()
                .float()
            )
        if z_hook_name is not None:
            clean_z = (
                clean_cache[z_hook_name][0, capture_token_index]
                .detach()
                .cpu()
                .float()
            )
            if z_sums is None:
                z_sums = torch.zeros(
                    total_points,
                    clean_z.shape[0],
                    clean_z.shape[1],
                    dtype=torch.float32,
                )
            z_sums[0] += clean_z

        correct = 0
        trials = 0
        for selected_offset, position_index in enumerate(
            selected_positions
        ):
            span = rendered.spans[position_index]
            target = int(span.start)
            for coeff_index, coeff in enumerate(coeffs):
                base_index = (
                    1
                    + (
                        selected_offset * n_coeffs + coeff_index
                    )
                    * n_concepts
                )
                for batch_start in range(0, n_concepts, concept_batch_size):
                    batch_end = min(
                        batch_start + concept_batch_size,
                        n_concepts,
                    )
                    vectors = concept_vectors[batch_start:batch_end]
                    batch_size = int(vectors.shape[0])
                    tokens = rendered.input_ids.expand(
                        batch_size,
                        -1,
                    ).contiguous()
                    captures = {
                        hook_name: torch.empty(0)
                        for hook_name in residual_hooks
                    }
                    if z_hook_name is not None:
                        captures[z_hook_name] = torch.empty(0)
                    if hasattr(model, "last_token_candidate_stats"):
                        candidate_logits = _run_injected_candidate_capture(
                            model,
                            tokens,
                            injection_layer=injection_layer,
                            span=(target, int(span.end)),
                            vectors=vectors,
                            strength=float(coeff),
                            scale_mode=scale_mode,
                            candidate_token_ids=[
                                candidate_ids[label] for label in labels
                            ],
                            capture_last_hooks=captures,
                            capture_token_index=capture_token_index,
                        )
                        predictions = (
                            candidate_logits.argmax(dim=-1).detach().cpu()
                        )
                    else:
                        logits, _ = run_injected_and_capture(
                            model,
                            tokens,
                            injection_layer=injection_layer,
                            spans=[(target, int(span.end))],
                            vector=vectors,
                            strength=float(coeff),
                            scale=scale_mode,
                            capture_layers=[],
                            capture_last_hooks=captures,
                            capture_token_index=capture_token_index,
                        )
                        _, predictions = _candidate_predictions(
                            logits,
                            candidate_ids,
                        )
                    row_start = base_index + batch_start
                    row_end = row_start + batch_size
                    row_indices = torch.arange(row_start, row_end)
                    vote_counts[row_indices, predictions] += 1
                    correct += int(
                        (predictions == position_index).sum().item()
                    )
                    trials += batch_size
                    for hook_name, layer in residual_hooks.items():
                        residual_sums[layer][row_start:row_end] += captures[
                            hook_name
                        ]
                    if z_hook_name is not None and z_sums is not None:
                        z_sums[row_start:row_end] += captures[z_hook_name]

        cluster_metrics.append(
            {
                **cluster.metadata(),
                "clean_prediction": clean_prediction,
                "clean_target": clean_target_label or "",
                "clean_correct": (
                    int(clean_prediction == clean_target_label)
                    if clean_target_label is not None
                    else ""
                ),
                "clean_none_correct": (
                    int(clean_prediction == clean_target_label)
                    if clean_target_label is not None
                    else ""
                ),
                "n_injected_examples": trials,
                "argmax_accuracy": correct / trials,
            }
        )
        print(
            f"cluster {cluster_offset}/{n_clusters} "
            f"(rank={cluster.rank}, seed={cluster.seed_word!r}) done; "
            f"accuracy={correct / trials:.3%}",
            flush=True,
        )

    assert canonical_labels is not None
    majority_indices = vote_counts.argmax(dim=1)
    majority_predictions = [
        canonical_labels[int(index)] for index in majority_indices
    ]
    majority_fractions = (
        vote_counts.max(dim=1).values.float() / float(n_clusters)
    )
    rows: list[dict[str, object]] = [
        {
            "condition": "clean",
            "intervention": "clean",
            "target_position": -1,
            "target_choice": "",
            "target_token": "",
            "inject_position": -1,
            "inject_choice": "",
            "inject_token": "",
            "sample": -1,
            "coeff": 0.0,
            "concept": "",
            "prediction": majority_predictions[0],
            "prediction_vote_fraction": float(majority_fractions[0]),
            "clean_target": clean_target_label or "",
            "correct": (
                int(majority_predictions[0] == clean_target_label)
                if clean_target_label is not None
                else ""
            ),
            "cluster_count": n_clusters,
            "aggregation": "full_vector_mean",
        }
    ]
    for selected_offset, position_index in enumerate(selected_positions):
        choice_label = str(representative.records[position_index]["choice"])
        for coeff_index, coeff in enumerate(coeffs):
            for concept_index, concept in enumerate(concept_names):
                row_index = (
                    1
                    + (
                        selected_offset * n_coeffs + coeff_index
                    )
                    * n_concepts
                    + concept_index
                )
                prediction = majority_predictions[row_index]
                rows.append(
                    {
                        "condition": "injected",
                        "intervention": "inject",
                        "target_position": position_index,
                        "target_choice": choice_label,
                        "target_token": f"mean over {n_clusters} clusters",
                        "inject_position": position_index,
                        "inject_choice": choice_label,
                        "inject_token": f"mean over {n_clusters} clusters",
                        "sample": concept_index,
                        "coeff": float(coeff),
                        "concept": str(concept),
                        "prediction": prediction,
                        "prediction_vote_fraction": float(
                            majority_fractions[row_index]
                        ),
                        "correct": int(prediction == choice_label),
                        "cluster_count": n_clusters,
                        "aggregation": "full_vector_mean",
                    }
                )

    return AveragedInjectionProjection(
        rows=rows,
        residuals={
            layer: mean_full_vectors(values, n_clusters)
            for layer, values in residual_sums.items()
        },
        head_z=(
            mean_full_vectors(z_sums, n_clusters)
            if z_sums is not None
            else None
        ),
        clean_prediction=majority_predictions[0],
        cluster_metrics=cluster_metrics,
    )
