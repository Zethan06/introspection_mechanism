#!/usr/bin/env python3
"""Token-cluster PCA projection gallery (latent and head-output PCA figures).

Injects concept vectors at each candidate position and renders a layerwise
selected-token residual projection gallery.

Pipeline:
  1. Load one cluster (10 single-token choices) from the balanced-cluster CSV.
  2. Extract contrastive concept vectors, or slice them from a saved payload.
  3. Render the matching token-localization prompt over the 10 cluster choices.
  4. One clean forward pass, caching every observe layer's resid-post -> the single
     clean point (selected-token residual per layer).
  5. Inject at each choice position and cache the same layer outputs.
  6. Per (observe layer, method), project clean + intervened selected-token residuals to
     3D and assemble a Plotly gallery, one scene per card.
  7. Write metadata.json + points.csv + projection_variance.csv + the HTML gallery.

Observe layers default to inject_layer+1 .. last.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from introspection_core import (
    HookedModel,
    ModelConfig,
    PromptManager,
    collect_averaged_injection_projection,
    extract_concept_vectors,
    load_concept_vector_payload,
    load_concepts_from_json,
    load_ranked_clusters,
    normalize_unit_vector,
)
from introspection_core import projection as projection_module
from introspection_core.prompts import REGISTRY
from introspection_core.cluster_split import require_valid_cluster_split
from introspection_core.results import write_metadata, write_table
from introspection_core.scatter_html import (
    CLEAN_ORIGIN_STYLE,
    injection_outcome_filter_groups,
    token_position_style,
    write_scatter_html,
)

DEFAULT_METHODS = "pca"


def _format_coeff(coeff: float) -> str:
    return f"{coeff:+g}"


def _capture_token_metadata(
    rendered_prompts,
    clusters,
    tokenizer,
    capture_token_index: int,
) -> dict:
    """Describe the captured token for every prompt contributing to an average."""
    captured_tokens = []
    for rendered, cluster in zip(rendered_prompts, clusters, strict=True):
        sequence_length = int(rendered.input_ids.shape[1])
        if not -sequence_length <= capture_token_index < sequence_length:
            raise ValueError(
                f"--capture_token_index {capture_token_index} is invalid for "
                f"cluster rank {cluster.rank} with sequence length {sequence_length}"
            )
        token_id = int(rendered.input_ids[0, capture_token_index].item())
        captured_tokens.append(
            {
                "cluster_rank": cluster.rank,
                "cluster_key": cluster.cluster_key,
                "sequence_length": sequence_length,
                "token_id": token_id,
                "token_text": tokenizer.decode([token_id]),
            }
        )

    identities = {
        (token["token_id"], token["token_text"])
        for token in captured_tokens
    }
    shared_identity = len(identities) == 1
    return {
        "capture_token_index": capture_token_index,
        "capture_token_identity": (
            "shared" if shared_identity else "varies_by_cluster"
        ),
        "captured_token_id": (
            captured_tokens[0]["token_id"] if shared_identity else None
        ),
        "captured_token_text": (
            captured_tokens[0]["token_text"] if shared_identity else None
        ),
        "captured_tokens": captured_tokens,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model id or local checkpoint path")
    parser.add_argument(
        "--intervention",
        choices=("inject",),
        default="inject",
        help="latent intervention to visualize",
    )

    # Cluster selection.
    parser.add_argument("--cluster_csv", required=True,
                        help="balanced similar-token clusters CSV (has a 'choices' JSON column)")
    parser.add_argument("--cluster_rank", type=int, default=1,
                        help="first 1-indexed cluster rank to include")
    parser.add_argument(
        "--cluster_count",
        type=int,
        default=1,
        help="consecutive clusters to average before projection",
    )
    parser.add_argument("--cluster_key", default=None,
                        help="select the cluster by its 'cluster_key' instead of --cluster_rank")

    # Concepts.
    parser.add_argument("--concepts_json",
                        help="JSON manifest with concept_vector_words + baseline_words")
    parser.add_argument("--max_concepts", type=int, default=None,
                        help="cap the number of concepts (default: all in the manifest)")
    parser.add_argument(
        "--state_vectors",
        type=Path,
        help=(
            "state_vectors.pt covering at least the manifest's concepts; sliced "
            "by name instead of re-extracting the contrastive vectors"
        ),
    )

    # Injection-only options.
    parser.add_argument("--injection_layer", type=int)
    parser.add_argument("--coeffs", type=float, nargs="+",
                        help="signed injection coeff(s); traces group by coeff and marker symbol encodes sign")
    parser.add_argument(
        "--concept_batch_size",
        type=int,
        default=16,
        help="concept-injection batch size",
    )
    parser.add_argument("--scale_mode", choices=["unit", "relative_hidden_norm"],
                        default="relative_hidden_norm")
    parser.add_argument(
        "--prompt_preamble",
        choices=["none", "user", "system"],
        default="system",
        help="injection prompt framing",
    )
    parser.add_argument(
        "--prompt_template",
        choices=sorted(REGISTRY),
        default="token_localization",
        help="injection prompt template",
    )
    parser.add_argument(
        "--positions",
        type=int,
        nargs="+",
        help=(
            "injection positions to include (default: every prompt position)"
        ),
    )


    # Observe layers (mode-specific default).
    parser.add_argument("--start_layer", type=int, default=None,
                        help="first observe layer (default: injection_layer+1)")
    parser.add_argument("--end_layer", type=int, default=None,
                        help="last observe layer, inclusive (default: model's last layer)")
    parser.add_argument("--methods", default=DEFAULT_METHODS, help="projection method (PCA)")
    parser.add_argument(
        "--capture_token_index",
        type=int,
        default=-1,
        help=(
            "sequence index whose latent representation is projected "
            "(default: -1, the final token; use -2 for the penultimate token)"
        ),
    )

    # Per-head OV output visualization (optional).
    parser.add_argument("--head_layer", type=int, default=None,
                        help="if given, also project all-head OV outputs (hook_z @ W_O) at this layer "
                             "and write token_head_ov_projection.html alongside the residual gallery")

    parser.add_argument("--max_points", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--date", default=None)
    args = parser.parse_args(argv)
    missing = [
        name
        for name in ("concepts_json", "injection_layer", "coeffs")
        if getattr(args, name) is None
    ]
    if missing:
        parser.error(
            "--intervention inject requires: "
            + ", ".join(f"--{name}" for name in missing)
        )
    return args


def _resolve_path(raw: str) -> Path:
    """Resolve a possibly-relative data path against CWD then the repo root."""
    p = Path(raw)
    if p.is_absolute() or p.exists():
        return p
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / raw


def load_cluster_choices(cluster_csv: str, *, rank: int, key: str | None):
    """Return (choices, cluster_meta) for one cluster row.

    choices: the 10 single-token words in the row's 'choices' JSON list.
    cluster_meta: {cluster_key, seed_word, rank} for bookkeeping.
    """
    path = _resolve_path(cluster_csv)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty cluster CSV: {path}")

    if key is not None:
        matches = [r for r in rows if r.get("cluster_key") == key]
        if not matches:
            raise ValueError(f"cluster_key {key!r} not found in {path}")
        row = matches[0]
    else:
        by_rank = {int(r["rank"]): r for r in rows if r.get("rank")}
        if rank not in by_rank:
            raise ValueError(f"cluster_rank {rank} not found in {path} (have {min(by_rank)}..{max(by_rank)})")
        row = by_rank[rank]

    choices = json.loads(row["choices"])
    if not choices:
        raise ValueError(f"cluster {row.get('cluster_key')} has no choices")
    meta = {
        "cluster_key": row.get("cluster_key"),
        "seed_word": row.get("seed_word"),
        "rank": int(row["rank"]) if row.get("rank") else None,
    }
    return [str(c) for c in choices], meta


def _aggregate_cluster_metadata(clusters) -> dict[str, object]:
    """Describe aggregation while retaining legacy identity for one cluster."""
    metadata: dict[str, object] = {
        "aggregation": "full_vector_mean",
        "count": len(clusters),
        "ranks": [cluster.rank for cluster in clusters],
        "clusters": [cluster.metadata() for cluster in clusters],
    }
    if len(clusters) == 1:
        metadata.update(clusters[0].metadata())
    return metadata


def main(argv=None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if not methods:
        raise ValueError("--methods must list at least one projection method")

    cluster_path = _resolve_path(args.cluster_csv)
    require_valid_cluster_split(cluster_path)
    clusters = load_ranked_clusters(
        cluster_path,
        start_rank=args.cluster_rank,
        count=args.cluster_count,
        cluster_key=args.cluster_key,
    )
    choices = list(clusters[0].choices)
    cluster_meta = clusters[0].metadata()
    concepts: list[str] = []
    baseline_words: list[str] = []
    concepts, baseline_words = load_concepts_from_json(
        _resolve_path(args.concepts_json), max_concepts=args.max_concepts
    )
    if not concepts:
        raise ValueError("no concepts loaded from --concepts_json")

    model = HookedModel(
        ModelConfig(
            name=args.model,
            device=args.device,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
    )
    prompt_manager = PromptManager(model.tokenizer)

    n_layers = int(model.cfg.n_layers)
    default_start_layer = args.injection_layer + 1
    start_layer = (
        args.start_layer if args.start_layer is not None else default_start_layer
    )
    end_layer = args.end_layer if args.end_layer is not None else n_layers - 1
    if not (0 <= start_layer <= end_layer < n_layers):
        raise ValueError(
            f"observe range [{start_layer}, {end_layer}] out of bounds for {n_layers} layers"
        )
    if start_layer <= args.injection_layer:
        print(
            f"WARNING observe start_layer {start_layer} <= injection_layer "
            f"{args.injection_layer}: the captured residual there is unaffected by "
            f"injection (no signal); the reference observes inject_layer+1..last.",
            flush=True,
        )
    layers = list(range(start_layer, end_layer + 1))
    resid_hook_names = {layer: model.resid_hook_name(layer) for layer in layers}
    hook_name_set = set(resid_hook_names.values())

    # Optional per-head OV visualization at a single specified layer.
    head_layer = args.head_layer
    z_hook_name = None
    z_acts_list: list[torch.Tensor] = []  # one [n_heads, d_head] per row (same order as `rows`)
    if head_layer is not None:
        if not (0 <= head_layer < n_layers):
            raise ValueError(f"--head_layer {head_layer} out of range [0, {n_layers})")
        z_hook_name = model.attn_hook_name(head_layer, "z")
        hook_name_set.add(z_hook_name)

    # Render every contributing prompt so averaged captures have truthful metadata.
    render_template = args.prompt_template
    render_preamble = args.prompt_preamble
    rendered_prompts = [
        prompt_manager.render(
            render_template,
            list(cluster.choices),
            preamble=render_preamble,
        )
        for cluster in clusters
    ]
    rendered = rendered_prompts[0]
    capture_metadata = _capture_token_metadata(
        rendered_prompts,
        clusters,
        model.tokenizer,
        args.capture_token_index,
    )
    captured_token_text = capture_metadata["captured_token_text"]
    captured_token_description = (
        repr(captured_token_text)
        if capture_metadata["capture_token_identity"] == "shared"
        else "identity varies by cluster; see captured_tokens metadata"
    )
    concept_names: list[str] = []
    device = model.bridge.cfg.device
    dtype = model.bridge.cfg.dtype
    if args.state_vectors is not None:
        unit_matrix = load_concept_vector_payload(
            _resolve_path(args.state_vectors),
            concepts=concepts,
            layer=args.injection_layer,
        ).to(device=device, dtype=dtype)
        concept_names = list(concepts)
    else:
        concept_vectors = extract_concept_vectors(
            model,
            words=concepts,
            baseline_words=baseline_words,
            layer=args.injection_layer,
        )
        unit_matrix = torch.stack(
            [
                normalize_unit_vector(cv.vector).to(device=device, dtype=dtype)
                for cv in concept_vectors
            ],
            dim=0,
        )
        concept_names = [cv.concept for cv in concept_vectors]
    averaged = collect_averaged_injection_projection(
        model,
        prompt_manager,
        clusters=clusters,
        concept_names=concept_names,
        concept_vectors=unit_matrix,
        injection_layer=args.injection_layer,
        coeffs=args.coeffs,
        scale_mode=args.scale_mode,
        prompt_preamble=args.prompt_preamble,
        observe_layers=layers,
        concept_batch_size=args.concept_batch_size,
        prompt_template=args.prompt_template,
        head_layer=head_layer,
        positions=args.positions,
        capture_token_index=args.capture_token_index,
    )
    rows = averaged.rows
    acts_by_layer = {
        layer: list(values.unbind(dim=0))
        for layer, values in averaged.residuals.items()
    }
    if averaged.head_z is not None:
        z_acts_list = list(averaged.head_z.unbind(dim=0))
    clean_prediction = averaged.clean_prediction
    cluster_meta = _aggregate_cluster_metadata(clusters)
    cluster_metrics = averaged.cluster_metrics

    n_concepts = len(concept_names)

    # -- project each (observe layer, method) to 3D --------------------------------------
    total_points = len(rows)
    views: list[dict] = []
    for layer in layers:
        stacked = torch.stack(acts_by_layer[layer], dim=0)  # [total_points, d_model]
        if stacked.shape[0] != total_points:
            raise ValueError(f"layer {layer}: {stacked.shape[0]} acts vs {total_points} rows")
        for method in methods:
            result = projection_module.project(stacked, method=method, n_components=3, seed=args.seed)
            views.append(
                {
                    "layer": layer,
                    "method": method,
                    "result": result,
                    "hover_fields": [
                        "intervention",
                        "target_position",
                        "target_token",
                        "sample",
                        "concept",
                        "coeff",
                        "prediction",
                        "prediction_vote_fraction",
                        "correct",
                        "cluster_count",
                        "aggregation",
                    ],
                }
            )

    # -- color by inject-token position; split injected traces by coeff/sign -------------
    n_choices = len(rendered.spans)
    group_legend = [
        {
            "label": "clean (origin)",
            "predicate": lambda row: row["condition"] == "clean",
            **CLEAN_ORIGIN_STYLE,
        }
    ]
    injected_positions = (
        list(range(n_choices))
        if args.positions is None
        else list(args.positions)
    )
    for choice_idx, record in enumerate(rendered.records):
        if choice_idx not in injected_positions:
            continue
        for coeff in args.coeffs:
            label = (
                f"TOKEN {record['choice']}: {record['text']} "
                f"coeff={_format_coeff(coeff)}"
            )
            group_legend.append(
                {
                    "label": label,
                    "predicate": (
                        lambda pos, c: (
                            lambda row: (
                                row["condition"] == "injected"
                                and row["target_position"] == pos
                                and row["coeff"] == c
                            )
                        )
                    )(choice_idx, coeff),
                    **token_position_style(choice_idx),
                }
            )

    point_filter_groups = injection_outcome_filter_groups()

    # -- write results -------------------------------------------------------------------
    run_dir = _resolve_path(args.results_dir) / "visualizations"
    run_dir.mkdir(parents=True, exist_ok=True)
    intervened_rows = [row for row in rows if row["condition"] != "clean"]
    intervention_accuracy = (
        sum(int(row["correct"]) for row in intervened_rows)
        / len(intervened_rows)
    )
    intervention_metadata = (
        {
            "injection_layer": args.injection_layer,
            "coeffs": list(args.coeffs),
            "scale_mode": args.scale_mode,
            "concepts": concept_names,
            "n_concepts": n_concepts,
            "baseline_word_count": len(baseline_words),
            "cluster_count": len(clusters),
            "cluster_ranks": [cluster.rank for cluster in clusters],
            "cluster_keys": [cluster.cluster_key for cluster in clusters],
            "seed_words": [cluster.seed_word for cluster in clusters],
            "cluster_reduction": "full_vector_mean_before_projection",
            **capture_metadata,
            "injection_positions": (
                list(range(len(rendered.spans)))
                if args.positions is None
                else list(args.positions)
            ),
        }
    )
    write_metadata(
        run_dir,
        model_name=args.model,
        args=vars(args),
        seed=args.seed,
        date=args.date,
        filename="validation_latent.metadata.json",
        extra={
            "prompt_text": rendered.text,
            "cluster": cluster_meta,
            "choices": choices,
            "intervention": args.intervention,
            **intervention_metadata,
            "observe_layers": layers,
            "methods": methods,
            "point_count": total_points,
            "clean_prediction": clean_prediction,
            "intervention_accuracy": intervention_accuracy,
        },
    )
    write_table(run_dir, "validation_latent.points.csv", rows)
    if cluster_metrics:
        write_table(run_dir, "validation_latent.cluster_metrics.csv", cluster_metrics)

    variance_rows = [
        {"layer": view["layer"], "method": view["method"], "component": i + 1, "variance_ratio": vr}
        for view in views
        if view["result"].variance_ratio is not None
        for i, vr in enumerate(view["result"].variance_ratio)
    ]
    if variance_rows:
        write_table(run_dir, "validation_latent.variance.csv", variance_rows)

    seed_word = cluster_meta.get("seed_word")
    cluster_label = (
        (
            f"{cluster_meta.get('cluster_key')} "
            f"(seed {seed_word!r})"
        )
        if len(clusters) == 1
        else f"ranks {clusters[0].rank}-{clusters[-1].rank} mean"
    )
    injected_position_count = len(
        intervention_metadata["injection_positions"]
    )
    page_description = (
        f"Inject each of {n_concepts} concept vectors at layer "
        f"{args.injection_layer} into each of "
        f"{injected_position_count} selected aligned TOKEN "
        f"positions across {len(clusters)} clusters, then average full "
        "latent vectors before projection "
        f"(coeffs {list(args.coeffs)}, {args.scale_mode}). Per observe "
        f"layer L{layers[0]}-L{layers[-1]}: 3D projections of the "
        f"token-index {args.capture_token_index} residual "
        f"({captured_token_description}) across {total_points} points. "
        "Color marks "
        "the intervention position; all injected points use the same "
        "circle marker mapping as the subspace-patch gallery "
        "(clean=black)."
    )
    write_scatter_html(
        run_dir / "validation_latent.html",
        title=(f"Token-cluster {args.intervention} projection — {args.model} — cluster "
               f"{cluster_label}"),
        views=views,
        rows=rows,
        group_legend=group_legend,
        max_points=args.max_points,
        seed=args.seed,
        page_description=page_description,
        extra_meta={
            "model": args.model,
            "intervention": args.intervention,
            "cluster_key": cluster_meta.get("cluster_key"),
            "seed_word": seed_word,
            "cluster_count": len(clusters),
            "cluster_ranks": [cluster.rank for cluster in clusters],
            "cluster_reduction": "full_vector_mean_before_projection",
            "observe_layers": layers,
            "methods": methods,
            "intervention_accuracy": intervention_accuracy,
            **intervention_metadata,
        },
        point_filter_groups=point_filter_groups,
    )

    print(f"clean argmax={clean_prediction}", flush=True)
    print(
        f"clusters={len(clusters)} ranks="
        f"{[cluster.rank for cluster in clusters]} choices_per_cluster="
        f"{len(choices)}",
        flush=True,
    )
    print(
        f"intervention={args.intervention} points={total_points} "
        f"accuracy={intervention_accuracy:.3%}",
        flush=True,
    )
    print(f"wrote results to {run_dir}", flush=True)

    # -- per-head OV output projection (only when --head_layer is given) ------------------
    if head_layer is not None and z_acts_list:
        if len(z_acts_list) != total_points:
            raise ValueError(f"z_acts_list length {len(z_acts_list)} != total_points {total_points}")
        # W_O: [n_heads, d_head, d_model] for this layer (TransformerLens convention)
        W_O = model.bridge.W_O[head_layer].detach().to("cpu", dtype=torch.float32)
        n_heads = W_O.shape[0]
        # stacked_z: [total_points, n_heads, d_head]
        stacked_z = torch.stack(z_acts_list, dim=0)

        head_views: list[dict] = []
        for head in range(n_heads):
            # OV output for head: z[:, head, :] @ W_O[head] → [total_points, d_model]
            ov = stacked_z[:, head, :] @ W_O[head]
            for method in methods:
                result = projection_module.project(ov, method=method, n_components=3, seed=args.seed)
                head_views.append({
                    "layer": head,          # "layer" key is structural; displayed via "label"
                    "label": f"Head {head}",
                    "nav_label": f"H{head} {method}",
                    "method": method,
                    "result": result,
                })

        head_variance_rows = [
            {"head": view["layer"], "method": view["method"], "component": i + 1, "variance_ratio": vr}
            for view in head_views
            if view["result"].variance_ratio is not None
            for i, vr in enumerate(view["result"].variance_ratio)
        ]
        if head_variance_rows:
            write_table(run_dir, "validation_head_ov.variance.csv", head_variance_rows)

        write_scatter_html(
            run_dir / "validation_head_ov.html",
            title=(f"Head OV {args.intervention} projection L{head_layer} — "
                   f"{args.model} — cluster "
                   f"{cluster_label}"),
            views=head_views,
            rows=rows,
            group_legend=group_legend,
            max_points=args.max_points,
            seed=args.seed,
            page_description=(
                f"Per-head OV output (hook_z @ W_O) at layer {head_layer}, "
                f"token index {args.capture_token_index} "
                f"({captured_token_description}), projected to 3D under "
                f"{args.intervention}. "
                f"One scene per head (H0-H{n_heads - 1}), colored by "
                "intervention position (clean=black)."
            ),
            extra_meta={
                "model": args.model,
                "intervention": args.intervention,
                "cluster_key": cluster_meta.get("cluster_key"),
                "seed_word": seed_word,
                "head_layer": head_layer,
                "n_heads": n_heads,
                "methods": methods,
                **intervention_metadata,
            },
            point_filter_groups=point_filter_groups,
        )
        print(f"wrote head OV gallery (L{head_layer}, {n_heads} heads) to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
