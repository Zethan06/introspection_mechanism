"""Build the compact attention browser from task examples and concept vectors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from .attention_aggregation import (
    average_attention,
)
from .attention_artifacts import (
    head_summary_rows,
    write_attention_payloads,
    write_csv,
)
from .attention_concepts import ConceptBank
from .attention_inputs import (
    AttentionTaskAdapter,
    build_token_rows,
)
from .layer_cache import ResidualHookPoint
from .prompts import PromptManager
from .token_attention_browser import make_token_attention_browser


def build_position_averaged_attention_visualization(
    model,
    *,
    task: AttentionTaskAdapter,
    concept_bank: ConceptBank,
    output_dir: Path,
    injection_layer: int,
    positions: Sequence[int] | None = None,
    layers: Sequence[int] | None = None,
    strength: float = 3.0,
    scale_mode: str = "relative_hidden_norm",
    concept_batch_size: int = 16,
    provenance: dict | None = None,
    artifact_stem: str = "token_attention_browser",
) -> Path:
    """Run aggregation and write browser, compact tensors, tables, and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_manager = PromptManager(model.tokenizer)
    examples = task.build_examples(prompt_manager)
    aggregate = average_attention(
        model,
        examples=examples,
        concept_vectors=concept_bank.vectors,
        injection_layer=injection_layer,
        positions=positions,
        strength=strength,
        scale_mode=scale_mode,
        concept_batch_size=concept_batch_size,
        layers=layers,
    )

    representative = examples[0]
    token_rows = build_token_rows(model.tokenizer, representative)
    target_indices = {
        str(position): int(representative.injection_spans[position].start)
        for position in aggregate.positions
    }
    target_tokens = {
        str(position): representative.item_labels[position]
        for position in aggregate.positions
    }
    position_summaries = {}
    for position in aggregate.positions:
        summary = aggregate.summaries[position].as_dict()
        summary["representative_target_token"] = target_tokens[str(position)]
        position_summaries[str(position)] = summary

    payload_metadata = write_attention_payloads(
        output_dir,
        clean=aggregate.clean_mean,
        injected_by_position=aggregate.injected_means,
        artifact_stem=artifact_stem,
    )
    concept_label = (
        f"mean of all {len(concept_bank.names)} concepts"
        if len(concept_bank.names) > 1
        else concept_bank.names[0]
    )
    first_position = aggregate.positions[0]
    metadata = {
        "title": "Token attention averaged across concepts and task examples",
        "task": task.name,
        "aggregation": "mean_across_concepts_and_examples_by_injection_position",
        "model": model.config.name,
        "concept": concept_label,
        "concepts": concept_bank.names,
        "num_concepts": len(concept_bank.names),
        "intervention_label": concept_label,
        "intervention_action": "injected",
        "num_interventions": len(concept_bank.names),
        "intervention_count_label": "concept(s)",
        "num_clusters": len(examples),
        "available_target_positions": aggregate.positions,
        "position_summaries": position_summaries,
        "target_position": first_position,
        "target_token": target_tokens[str(first_position)],
        "target_token_index": target_indices[str(first_position)],
        "target_token_indices": target_indices,
        "representative_target_tokens": target_tokens,
        "choices": [
            representative.item_labels[position]
            for position in representative.positions
        ],
        "choice_token_indices": list(representative.item_token_indices),
        "injection_layer": int(injection_layer),
        "strength": float(strength),
        "scale_mode": scale_mode,
        "seq_len": int(aggregate.clean_mean.shape[-1]),
        "num_layers": len(aggregate.layers),
        "num_heads": int(aggregate.clean_mean.shape[1]),
        "layers": aggregate.layers,
        "attention_injection_backend": "transformer_lens",
        "tokens": token_rows,
        "prompt": representative.prompt,
        **payload_metadata,
    }
    if provenance:
        metadata["provenance"] = provenance

    browser_path = output_dir / f"{artifact_stem}.html"
    browser_path.write_text(
        make_token_attention_browser(metadata),
        encoding="utf-8",
    )
    run_metadata = {
        key: value for key, value in metadata.items() if key not in {"tokens", "prompt"}
    }
    (output_dir / f"{artifact_stem}.metadata.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    (output_dir / f"{artifact_stem}.prompt.txt").write_text(
        representative.prompt,
        encoding="utf-8",
    )
    write_csv(output_dir / f"{artifact_stem}.tokens.csv", token_rows)
    write_csv(
        output_dir / f"{artifact_stem}.positions.csv",
        [
            position_summaries[str(position)]
            for position in aggregate.positions
        ],
    )

    summary_rows = []
    localization_trials = len(examples) * len(concept_bank.names)
    for position in aggregate.positions:
        rows = head_summary_rows(
            aggregate.clean_mean,
            aggregate.injected_means[position],
            layers=aggregate.layers,
            target_token_index=target_indices[str(position)],
            item_token_indices=representative.item_token_indices,
        )
        for row in rows:
            layer_offset = aggregate.layers.index(int(row["layer"]))
            head = int(row["head"])
            clean_correct = int(
                aggregate.clean_attention_correct[position][layer_offset, head]
            )
            injected_correct = int(
                aggregate.injected_attention_correct[position][layer_offset, head]
            )
            clean_accuracy = clean_correct / localization_trials
            injected_accuracy = injected_correct / localization_trials
            summary_rows.append(
                {
                    "target_position": position,
                    "clean_attention_localization_correct": clean_correct,
                    "injected_attention_localization_correct": injected_correct,
                    "attention_localization_trials": localization_trials,
                    "clean_attention_localization_accuracy": clean_accuracy,
                    "injected_attention_localization_accuracy": injected_accuracy,
                    "attention_localization_gain": (
                        injected_accuracy - clean_accuracy
                    ),
                    **row,
                }
            )
    write_csv(output_dir / f"{artifact_stem}.heads.csv", summary_rows)
    return browser_path


