"""introspection_core — model-agnostic LLM introspection primitives.

Public API is curated here; internal modules may have more surface than
what's re-exported. Import from this package root in scripts, e.g.:

    from introspection_core import HookedModel, ModelConfig

The package is built exclusively on TransformerLens (``TransformerBridge``)
hooks.
"""

from __future__ import annotations

from .context_key_modes import (
    capture_context_qk,
    context_key_mode_metrics,
    load_context_mode_captures,
    summarize_context_modes,
)

from .attention_aggregation import (
    AveragedAttention,
    AveragedPatchedAttention,
    average_attention,
)
from .attention_concepts import ConceptBank, load_concept_bank
from .attention_inputs import (
    AttentionExample,
    AttentionTaskAdapter,
    ShuffledLabelTokenLocalizationCsvTask,
    TokenLocalizationCsvTask,
    cluster_choice_count,
)
from .attention_routing import (
    candidate_routing_key_positions,
    force_attention_to_keys,
    one_hot_attention_hook,
    trailing_newline_key_positions,
)
from .attention_routing_evaluation import (
    AttentionRoutingEvaluation,
    evaluate_attention_routing,
)
from .cluster_aggregation import (
    RankedCluster,
    load_ranked_clusters,
    mean_full_vectors,
)
from .clean_head_patch import (
    CandidateLayoutExample,
    CleanHeadPatchAccumulator,
    CleanHeadPatchCondition,
    build_patch_conditions,
    concept_bootstrap_interval,
    format_accuracy_change_sentence,
    format_head_components,
    layer_matched_control_groups,
    parse_head_components,
    resolve_candidate_layout,
)
from .donor_router_mismatch import (
    DonorRouterAccumulator,
    aggregate_position_rows,
    router_source_indices,
    sharded_complete_position_batches,
)
from .extraction import (
    ConceptVector,
    extract_concept_vectors,
    format_concept_prompt,
    load_concept_vector_payload,
    load_concepts_from_json,
)
from .injected_trials import InjectedTrial, gate_logit
from .head_mask_gate import (
    FORMAL_TOP_K,
    FORMAL_TRAINING_PROTOCOL,
    TopKHeadMask,
    TransitionStats,
    directional_head_gate_bce,
    formal_train_pair_filter,
    formal_training_pair_mask,
    incremental_one_hot_router_hook,
    load_gate_on_candidate_trials,
    load_head_mask_checkpoint,
    masked_final_token_head_hooks,
    native_number_inducing_mask,
)
from .injection import (
    inject,
    load_unit_vector,
    make_injection_hook,
    normalize_unit_vector,
    run_injected,
)
from .injection_projection import (
    AveragedInjectionProjection,
    collect_averaged_injection_projection,
)
from .label_accuracy import (
    LabelAccuracyCounts,
    LabelAccuracyEvaluation,
    LabelAccuracyRunMetadata,
    LabelCandidateLayout,
    build_label_accuracy_summary,
    evaluate_split_label_accuracy,
    label_accuracy_cluster_rows,
    label_accuracy_concept_rows,
    label_accuracy_outputs_match,
    label_accuracy_position_rows,
    load_label_accuracy_vectors,
    resolve_label_candidate_layout,
    write_label_accuracy_outputs,
)
from .layer_cache import capture_resid_layers, gather_target_tokens, patch_resid_and_forward, run_injected_and_capture
from .localization import (
    RenderedPrompt,
    Span,
    resolve_char_span_to_tokens,
    resolve_text_span_to_tokens,
)
from .model import HookedModel, ModelConfig
from .patching import collect, patch
from .prompts import (
    REGISTRY,
    PromptManager,
    PromptTemplate,
    template_slot_labels,
    template_system_prompt,
)
from .position_averaged_attention import (
    build_position_averaged_attention_visualization,
)
from .scoring import FirstTokenScore, aggregate, answer_token_ids, score_first_token

__all__ = [
    "capture_context_qk",
    "context_key_mode_metrics",
    "load_context_mode_captures",
    "summarize_context_modes",
    # model
    "HookedModel",
    "ModelConfig",
    # validation clean-head patch
    "CandidateLayoutExample",
    "CleanHeadPatchAccumulator",
    "CleanHeadPatchCondition",
    "build_patch_conditions",
    "concept_bootstrap_interval",
    "format_accuracy_change_sentence",
    "format_head_components",
    "layer_matched_control_groups",
    "parse_head_components",
    "resolve_candidate_layout",
    # averaged attention visualization
    "AttentionExample",
    "AttentionTaskAdapter",
    "TokenLocalizationCsvTask",
    "ShuffledLabelTokenLocalizationCsvTask",
    "cluster_choice_count",
    "ConceptBank",
    "load_concept_bank",
    "AveragedAttention",
    "AveragedPatchedAttention",
    "average_attention",
    "build_position_averaged_attention_visualization",
    # one-hot attention routing
    "candidate_routing_key_positions",
    "force_attention_to_keys",
    "one_hot_attention_hook",
    "trailing_newline_key_positions",
    "AttentionRoutingEvaluation",
    "evaluate_attention_routing",
    "load_concept_vector_payload",
    "InjectedTrial",
    "gate_logit",
    "DonorRouterAccumulator",
    "aggregate_position_rows",
    "router_source_indices",
    "sharded_complete_position_batches",
    # formal STE Top-k head selection
    "FORMAL_TOP_K",
    "FORMAL_TRAINING_PROTOCOL",
    "TopKHeadMask",
    "TransitionStats",
    "directional_head_gate_bce",
    "formal_train_pair_filter",
    "formal_training_pair_mask",
    "incremental_one_hot_router_hook",
    "load_gate_on_candidate_trials",
    "load_head_mask_checkpoint",
    "masked_final_token_head_hooks",
    "native_number_inducing_mask",
    # cluster aggregation
    "RankedCluster",
    "load_ranked_clusters",
    "mean_full_vectors",
    # localization
    "Span",
    "RenderedPrompt",
    "resolve_char_span_to_tokens",
    "resolve_text_span_to_tokens",
    # label-agnostic localization accuracy
    "LabelCandidateLayout",
    "LabelAccuracyCounts",
    "LabelAccuracyEvaluation",
    "LabelAccuracyRunMetadata",
    "resolve_label_candidate_layout",
    "load_label_accuracy_vectors",
    "evaluate_split_label_accuracy",
    "build_label_accuracy_summary",
    "label_accuracy_position_rows",
    "label_accuracy_cluster_rows",
    "label_accuracy_concept_rows",
    "label_accuracy_outputs_match",
    "write_label_accuracy_outputs",
    # prompts
    "PromptManager",
    "PromptTemplate",
    "REGISTRY",
    "template_slot_labels",
    "template_system_prompt",
    # injection
    "load_unit_vector",
    "normalize_unit_vector",
    "make_injection_hook",
    "inject",
    "run_injected",
    "AveragedInjectionProjection",
    "collect_averaged_injection_projection",
    # layer cache
    "capture_resid_layers",
    "run_injected_and_capture",
    "patch_resid_and_forward",
    "gather_target_tokens",
    # patching
    "collect",
    "patch",
    # scoring
    "answer_token_ids",
    "score_first_token",
    "FirstTokenScore",
    "aggregate",
    # extraction
    "ConceptVector",
    "extract_concept_vectors",
    "format_concept_prompt",
    "load_concepts_from_json",
]
