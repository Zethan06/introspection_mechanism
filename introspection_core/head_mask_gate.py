"""Sparse Top-k attention-head selection for number-output interventions.

The formal experiment has exactly two independently trained directions:

* ``on``: among native clean-none/injected-exact transitions, patch injected
  final-token head outputs into a clean recipient and optimize number output
  under a forced downstream router.
* ``off``: patch clean final-token head outputs into an injected recipient and
  optimize ``none`` output under the native router for native
  clean-none/injected-number transitions.

Both directions use an exactly-Top-k straight-through mask and BCE only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .attention_inputs import (
    AttentionExample,
    ShuffledLabelTokenLocalizationCsvTask,
    TokenLocalizationCsvTask,
    cluster_choice_count,
)
from .injected_trials import (
    ALL_LABELS,
    InjectedTrial,
    build_injected_batch,
    gate_logit,
    load_injected_trials,
    validate_token0_9_candidate_layout,
)
from .head_output_patch import capture_final_token_heads
from .prompts import (
    NONE_ANSWER_LABEL,
    REGISTRY,
    PromptManager,
    template_slot_labels,
    template_system_prompt,
)


HEAD_MASK_SCHEMA_VERSION = 2
FORMAL_TOP_K = 32
FORMAL_CONCEPT_COUNT = 100
FORMAL_CLUSTER_COUNT = 30
FORMAL_POSITION_COUNT = 10
FORMAL_COORDINATE_COUNT = (
    FORMAL_CONCEPT_COUNT * FORMAL_CLUSTER_COUNT * FORMAL_POSITION_COUNT
)
GATE_ON_TRAIN_PAIR_FILTER = "native_clean_none_injected_exact_target"
GATE_OFF_TRAIN_PAIR_FILTER = "native_clean_none_injected_any_number"
FORMAL_TRAINING_PROTOCOL = (
    "gate_on_forced_locked_exact_gate_off_native_fullgrid_bce_only"
)


@dataclass(frozen=True)
class HeadGateDataset:
    """Full concept-by-cluster-by-position grid for one explicit split."""

    concepts: tuple[str, ...]
    examples: tuple[AttentionExample, ...]
    candidate_token_ids: tuple[int, ...]
    base_tokens: torch.Tensor
    injection_token_positions: torch.Tensor
    concept_vectors: torch.Tensor
    trials: tuple[InjectedTrial, ...]
    # ``[clusters, 11]`` map from slot order (ten positions, then ``none``) to
    # candidate columns. None when every cluster answers slot p with column p;
    # only a shuffled label arm needs it.
    slot_columns: torch.Tensor | None = None

    def slot_order_logits(
        self, logits: torch.Tensor, trials: Sequence[InjectedTrial]
    ) -> torch.Tensor:
        """Reorder candidate logits so column p answers slot p in every row."""

        if self.slot_columns is None:
            return logits
        clusters = torch.tensor(
            [int(trial.cluster_index) for trial in trials], dtype=torch.long
        )
        columns = self.slot_columns.index_select(0, clusters).to(logits.device)
        return logits.gather(-1, columns)


def _ordered_concepts_from_payload(payload: dict) -> tuple[str, ...]:
    """Return ordered concept labels from train or held-out vector artifacts."""

    concepts = payload.get("concepts")
    if concepts is None:
        rows = payload.get("rows")
        if isinstance(rows, (list, tuple)) and rows and all(
            isinstance(row, dict) and isinstance(row.get("word"), str)
            for row in rows
        ):
            # Held-out QK-circuit vector shards store the same ordered labels
            # as row metadata rather than under the train artifact's
            # ``concepts`` key.
            concepts = [row["word"] for row in rows]
    if not isinstance(concepts, (list, tuple)) or not concepts:
        raise ValueError("Concept-vector payload has no ordered concept labels")
    normalized = tuple(str(concept) for concept in concepts)
    if len(set(normalized)) != len(normalized):
        raise ValueError("Concept-vector payload contains duplicate concept labels")
    return normalized


def labeled_candidate_layout(
    examples: Sequence[AttentionExample],
) -> tuple[tuple[int, ...], torch.Tensor | None]:
    """Return shared candidate token IDs and the per-cluster slot-column map.

    Candidates are scored in the first example's label order with ``none``
    last, so column 10 stays the ``none`` column that ``gate_logit`` expects.
    Every example must score the same candidate tokens; only the label
    printed beside each slot may differ between clusters.
    """

    if not examples:
        raise ValueError("no examples to lay out")
    labels = tuple(str(label) for label in examples[0].candidate_token_ids)
    if len(labels) != FORMAL_POSITION_COUNT + 1 or labels[-1] != NONE_ANSWER_LABEL:
        raise ValueError(
            f"expected ten slot labels followed by `{NONE_ANSWER_LABEL}`, got {labels}"
        )
    token_ids = tuple(int(examples[0].candidate_token_ids[label]) for label in labels)
    if len(set(token_ids)) != len(token_ids):
        raise ValueError(f"candidate labels share answer tokens: {token_ids}")
    column_by_label = {label: index for index, label in enumerate(labels)}
    rows: list[list[int]] = []
    for example in examples:
        example_labels = tuple(str(label) for label in example.candidate_token_ids)
        example_ids = tuple(
            int(example.candidate_token_ids[label]) for label in example_labels
        )
        if example_labels != labels or example_ids != token_ids:
            raise ValueError(
                f"example {example.key!r} scores a different candidate set: "
                f"{dict(zip(example_labels, example_ids))}"
            )
        if tuple(example.positions) != tuple(range(FORMAL_POSITION_COUNT)):
            raise ValueError(
                f"example {example.key!r} positions must be 0..9, got {example.positions}"
            )
        rows.append(
            [
                column_by_label[str(example.expected_candidate_by_position[position])]
                for position in range(FORMAL_POSITION_COUNT)
            ]
            + [FORMAL_POSITION_COUNT]
        )
    slot_columns = torch.tensor(rows, dtype=torch.long)
    if bool(slot_columns[:, :FORMAL_POSITION_COUNT].sort(dim=-1).values.ne(
        torch.arange(FORMAL_POSITION_COUNT)
    ).any()):
        raise ValueError("each cluster must answer its ten slots with ten distinct labels")
    if torch.equal(
        slot_columns, torch.arange(FORMAL_POSITION_COUNT + 1).expand_as(slot_columns)
    ):
        return token_ids, None
    return token_ids, slot_columns


def load_head_gate_dataset(
    model,
    *,
    cluster_csv: Path,
    concept_vectors_file: Path,
    injection_layer: int,
    prompt_template: str,
    prompt_preamble: str,
    choice_suffix: str,
    label_permutation: str = "identity",
    label_seed: int = 42,
) -> HeadGateDataset:
    """Load the full split without conditioning on saved outcomes.

    The digit identity arm keeps the strict ``0..9, none`` layout check. Other
    label arms score a candidate set shared by every cluster and record in
    ``slot_columns`` which column answers each slot.
    """

    payload = torch.load(
        concept_vectors_file, map_location="cpu", weights_only=False
    )
    normalized_concepts = _ordered_concepts_from_payload(payload)

    vectors = payload.get("unit_vectors")
    if vectors is None:
        raw_vectors = payload.get("vectors")
        if not isinstance(raw_vectors, torch.Tensor):
            raise ValueError("Concept-vector payload has no vectors")
        vectors = F.normalize(raw_vectors.float(), dim=-1)
    if not isinstance(vectors, torch.Tensor):
        raise ValueError("Concept-vector payload vectors must be a tensor")
    concept_vectors = F.normalize(vectors.float(), dim=-1).cpu()
    if not bool(torch.isfinite(concept_vectors).all()):
        raise ValueError("Concept vectors contain non-finite values")
    if bool(concept_vectors.norm(dim=-1).le(1e-8).any()):
        raise ValueError("Concept-vector payload contains a zero vector")
    if len(normalized_concepts) != int(concept_vectors.shape[0]):
        raise ValueError(
            "Concept labels and vectors have different row counts: "
            f"{len(normalized_concepts)} != {int(concept_vectors.shape[0])}"
        )
    vector_layer = payload.get("layer")
    if vector_layer is not None and int(vector_layer) != int(injection_layer):
        raise ValueError(
            f"Concept vectors use layer {int(vector_layer)}, expected "
            f"injection layer {int(injection_layer)}"
        )
    if concept_vectors.dim() != 2 or concept_vectors.shape[1] != int(
        model.cfg.d_model
    ):
        raise ValueError(
            "Concept-vector shape does not match model hidden width: "
            f"{tuple(concept_vectors.shape)} vs d_model={int(model.cfg.d_model)}"
        )

    if label_permutation == "shuffled":
        # Same construction as the router label arms and Table 1: one seeded
        # derangement of the template's labels per cluster.
        template = REGISTRY[prompt_template]
        n_choices = cluster_choice_count(cluster_csv)
        task = ShuffledLabelTokenLocalizationCsvTask(
            path=cluster_csv,
            canonical_labels=template_slot_labels(template, n_choices),
            system_prompt=template_system_prompt(
                template, prompt_preamble, n_choices
            ),
            template_name=prompt_template,
            preamble=prompt_preamble,
            choice_suffix=choice_suffix,
            seed=label_seed,
            name=prompt_template,
        )
    elif label_permutation == "identity":
        task = TokenLocalizationCsvTask(
            path=cluster_csv,
            preamble=prompt_preamble,
            choice_suffix=choice_suffix,
            position_index_start=0,
            template_name=prompt_template,
            name=prompt_template,
        )
    else:
        raise ValueError(f"unknown label_permutation: {label_permutation!r}")
    examples = task.build_examples(PromptManager(model.tokenizer))
    if not examples:
        raise ValueError("Cluster split produced no prompt examples")
    candidate_labels = tuple(str(label) for label in examples[0].candidate_token_ids)
    if label_permutation == "identity" and candidate_labels == ALL_LABELS:
        candidate_token_ids = tuple(validate_token0_9_candidate_layout(examples))
        for example in examples[1:]:
            example_token_ids = tuple(validate_token0_9_candidate_layout([example]))
            if example_token_ids != candidate_token_ids:
                raise ValueError(
                    "Candidate token IDs differ across cluster prompts: "
                    f"{example_token_ids} != {candidate_token_ids}"
                )
        slot_columns = None
    else:
        candidate_token_ids, slot_columns = labeled_candidate_layout(examples)
    base_tokens = torch.cat([example.input_ids for example in examples], dim=0)
    injection_token_positions = torch.tensor(
        [
            [
                int(example.injection_spans[position].start)
                for position in range(FORMAL_POSITION_COUNT)
            ]
            for example in examples
        ],
        dtype=torch.long,
    )
    trials = tuple(
        InjectedTrial(
            concept_index=concept_index,
            cluster_index=cluster_index,
            position=position,
            correct=False,
            predicted_none=False,
        )
        for concept_index in range(len(concept_vectors))
        for cluster_index in range(len(examples))
        for position in range(FORMAL_POSITION_COUNT)
    )
    if len(normalized_concepts) != FORMAL_CONCEPT_COUNT:
        raise ValueError(
            f"Formal split requires {FORMAL_CONCEPT_COUNT} concepts, got "
            f"{len(normalized_concepts)}"
        )
    if len(examples) != FORMAL_CLUSTER_COUNT:
        raise ValueError(
            f"Formal split requires {FORMAL_CLUSTER_COUNT} prompt clusters, got "
            f"{len(examples)}"
        )
    if len(trials) != FORMAL_COORDINATE_COUNT:
        raise RuntimeError(
            f"Formal grid must contain {FORMAL_COORDINATE_COUNT} coordinates, "
            f"got {len(trials)}"
        )
    return HeadGateDataset(
        concepts=normalized_concepts,
        examples=tuple(examples),
        candidate_token_ids=candidate_token_ids,
        base_tokens=base_tokens,
        injection_token_positions=injection_token_positions,
        concept_vectors=concept_vectors,
        trials=trials,
        slot_columns=slot_columns,
    )


@dataclass(frozen=True)
class CachedHeadGateBatch:
    """Paired clean/injected caches and final-token head outputs."""

    trials: tuple[InjectedTrial, ...]
    last_tokens: torch.Tensor
    prefix_length: int
    clean_cache: object
    injected_cache: object
    clean_z: dict[int, torch.Tensor]
    injected_z: dict[int, torch.Tensor]
    clean_logits: torch.Tensor
    injected_logits: torch.Tensor
    positions: torch.Tensor


def load_gate_on_candidate_trials(
    outcomes_csv: Path,
    *,
    dataset: HeadGateDataset,
) -> tuple[InjectedTrial, ...]:
    """Load the locked injected-exact candidate pool used by gate-on training."""

    candidates = [
        trial for trial in load_injected_trials(outcomes_csv) if trial.correct
    ]
    if not candidates:
        raise ValueError("Gate-on outcomes contain no injected-exact candidates")
    invalid = [
        (trial.concept_index, trial.cluster_index, trial.position)
        for trial in candidates
        if not (
            0 <= trial.concept_index < len(dataset.concepts)
            and 0 <= trial.cluster_index < len(dataset.examples)
            and 0 <= trial.position < FORMAL_POSITION_COUNT
            and not trial.predicted_none
        )
    ]
    if invalid:
        raise ValueError(
            "Gate-on outcomes contain invalid exact coordinates: "
            f"{invalid[:5]}"
        )
    candidates.sort(
        key=lambda trial: (
            trial.concept_index,
            trial.cluster_index,
            trial.position,
        )
    )
    return tuple(candidates)


def prepare_cached_head_gate_batch(
    model,
    *,
    trials: Sequence[InjectedTrial],
    dataset: HeadGateDataset,
    layers: Sequence[int],
    injection_layer: int,
    strength: float,
    scale_mode: str,
) -> CachedHeadGateBatch:
    """Capture paired native clean/injected donors for one coordinate batch."""

    batch = tuple(trials)
    if not batch:
        raise ValueError("trials must be non-empty")
    tokens, injection_hook = build_injected_batch(
        batch,
        base_tokens=dataset.base_tokens,
        injection_token_positions=dataset.injection_token_positions,
        concept_vectors=dataset.concept_vectors,
        model=model,
        injection_layer=injection_layer,
        strength=strength,
        scale_mode=scale_mode,
    )
    prefix_length = int(tokens.shape[1]) - 1
    batch_injection_positions = torch.tensor(
        [
            int(
                dataset.injection_token_positions[
                    trial.cluster_index, trial.position
                ]
            )
            for trial in batch
        ],
        dtype=torch.long,
    )
    if bool(batch_injection_positions.ge(prefix_length).any()):
        raise ValueError(
            "Injection touches the final prompt token; prefix KV reuse requires "
            "every injection position to precede it"
        )

    clean_cache = model.build_prefix_kv_cache(tokens)
    injected_cache = model.build_prefix_kv_cache(
        tokens, fwd_hooks=[injection_hook]
    )
    last_tokens = tokens[:, -1:]
    clean_z, clean_logits = capture_final_token_heads(
        model,
        last_tokens=last_tokens,
        prefix_kv_cache=clean_cache,
        prefix_length=prefix_length,
        candidate_token_ids=dataset.candidate_token_ids,
        layers=layers,
    )
    injected_z, injected_logits = capture_final_token_heads(
        model,
        last_tokens=last_tokens,
        prefix_kv_cache=injected_cache,
        prefix_length=prefix_length,
        candidate_token_ids=dataset.candidate_token_ids,
        layers=layers,
    )
    return CachedHeadGateBatch(
        trials=batch,
        last_tokens=last_tokens,
        prefix_length=prefix_length,
        clean_cache=clean_cache,
        injected_cache=injected_cache,
        clean_z=clean_z,
        injected_z=injected_z,
        clean_logits=clean_logits,
        injected_logits=injected_logits,
        positions=torch.tensor(
            [trial.position for trial in batch], dtype=torch.long
        ),
    )


def native_number_inducing_mask(
    clean_logits: torch.Tensor,
    injected_logits: torch.Tensor,
) -> torch.Tensor:
    """Select native clean=none to injected=any-number transitions."""

    if clean_logits.shape != injected_logits.shape:
        raise ValueError("clean and injected logits must have identical shapes")
    if clean_logits.dim() != 2 or clean_logits.shape[-1] != 11:
        raise ValueError("clean and injected logits must both be [batch, 11]")
    return clean_logits.argmax(dim=-1).eq(10) & injected_logits.argmax(
        dim=-1
    ).ne(10)


def native_exact_inducing_mask(
    clean_logits: torch.Tensor,
    injected_logits: torch.Tensor,
    *,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Select native clean=none to injected=target-number transitions."""

    if clean_logits.shape != injected_logits.shape:
        raise ValueError("clean and injected logits must have identical shapes")
    if clean_logits.dim() != 2 or clean_logits.shape[-1] != 11:
        raise ValueError("clean and injected logits must both be [batch, 11]")
    targets = positions.to(device=injected_logits.device, dtype=torch.long)
    if targets.shape != (clean_logits.shape[0],):
        raise ValueError("positions must contain one target per logit row")
    if bool(((targets < 0) | (targets >= FORMAL_POSITION_COUNT)).any()):
        raise ValueError("positions must be in [0, 9]")
    return clean_logits.argmax(dim=-1).eq(10) & injected_logits.argmax(
        dim=-1
    ).eq(targets)


def formal_training_pair_mask(
    clean_logits: torch.Tensor,
    injected_logits: torch.Tensor,
    *,
    positions: torch.Tensor,
    direction: str,
) -> torch.Tensor:
    """Apply the direction-specific population filter of the formal protocol."""

    if direction == "on":
        return native_exact_inducing_mask(
            clean_logits, injected_logits, positions=positions
        )
    if direction == "off":
        return native_number_inducing_mask(clean_logits, injected_logits)
    raise ValueError("direction must be 'on' or 'off'")


def formal_train_pair_filter(direction: str) -> str:
    """Return the locked population label for one selection direction."""

    if direction == "on":
        return GATE_ON_TRAIN_PAIR_FILTER
    if direction == "off":
        return GATE_OFF_TRAIN_PAIR_FILTER
    raise ValueError("direction must be 'on' or 'off'")


def hard_topk_mask(scores: torch.Tensor, *, top_k: int) -> torch.Tensor:
    """Return an exactly-``top_k`` binary mask with the shape of ``scores``."""

    if scores.dim() != 2:
        raise ValueError("head scores must have shape [layers, heads]")
    if not 0 < top_k <= scores.numel():
        raise ValueError(
            f"top_k must be in [1, {scores.numel()}], got {top_k}"
        )
    flat_mask = torch.zeros_like(scores).flatten()
    indices = scores.detach().flatten().topk(top_k).indices
    flat_mask.scatter_(0, indices, 1.0)
    return flat_mask.view_as(scores)


def ste_topk_mask(
    scores: torch.Tensor,
    *,
    top_k: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return straight-through, soft, and hard exactly-Top-k masks."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    soft = torch.sigmoid(scores / temperature)
    hard = hard_topk_mask(soft, top_k=top_k)
    straight_through = hard + soft - soft.detach()
    return straight_through, soft, hard


class TopKHeadMask(nn.Module):
    """One global exactly-Top-k mask over searched layer/head cells."""

    def __init__(
        self,
        layers: Sequence[int],
        *,
        n_heads: int,
        top_k: int,
        device: torch.device | str,
        seed: int,
        initialization_std: float = 1e-3,
    ) -> None:
        super().__init__()
        normalized_layers = tuple(int(layer) for layer in layers)
        if not normalized_layers or len(set(normalized_layers)) != len(
            normalized_layers
        ):
            raise ValueError("layers must be non-empty and unique")
        if n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if initialization_std < 0:
            raise ValueError("initialization_std must be non-negative")
        cell_count = len(normalized_layers) * int(n_heads)
        if not 0 < top_k <= cell_count:
            raise ValueError(f"top_k must be in [1, {cell_count}], got {top_k}")
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        initial = torch.randn(
            len(normalized_layers), int(n_heads), generator=generator
        ) * float(initialization_std)
        self.layers = normalized_layers
        self.n_heads = int(n_heads)
        self.top_k = int(top_k)
        self.scores = nn.Parameter(initial.to(device=device))

    def forward(
        self, *, temperature: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return ste_topk_mask(
            self.scores,
            top_k=self.top_k,
            temperature=temperature,
        )

    @torch.no_grad()
    def selected_components(self) -> list[tuple[int, int]]:
        """Return selected ``(layer, head)`` pairs by descending score."""

        flat_indices = self.scores.flatten().topk(self.top_k).indices.tolist()
        return [
            (self.layers[index // self.n_heads], index % self.n_heads)
            for index in flat_indices
        ]

    @torch.no_grad()
    def selection_rows(self) -> list[dict]:
        """Return a stable per-head score and selection table."""

        probabilities = torch.sigmoid(self.scores).detach().float().cpu()
        scores = self.scores.detach().float().cpu()
        hard = hard_topk_mask(scores, top_k=self.top_k).cpu().bool()
        rank_by_component = {
            component: rank
            for rank, component in enumerate(
                self.selected_components(), start=1
            )
        }
        rows: list[dict] = []
        for layer_offset, layer in enumerate(self.layers):
            for head in range(self.n_heads):
                component = (layer, head)
                rows.append(
                    {
                        "layer": layer,
                        "head": head,
                        "score": float(scores[layer_offset, head]),
                        "sigmoid_score": float(
                            probabilities[layer_offset, head]
                        ),
                        "selected": int(hard[layer_offset, head]),
                        "selection_rank": rank_by_component.get(component),
                    }
                )
        rows.sort(
            key=lambda row: (
                -int(row["selected"]),
                row["selection_rank"]
                or self.n_heads * len(self.layers) + 1,
                int(row["layer"]),
                int(row["head"]),
            )
        )
        return rows


@dataclass
class TransitionStats:
    """Paired source-to-target conversion counts for one intervention."""

    n: int = 0
    source: int = 0
    converted: int = 0
    target_before: int = 0
    target_after: int = 0

    def update(
        self,
        baseline_logits: torch.Tensor,
        patched_logits: torch.Tensor,
        *,
        direction: str,
    ) -> None:
        if direction not in {"on", "off"}:
            raise ValueError(f"unknown transition direction: {direction}")
        baseline_predictions = baseline_logits.detach().argmax(dim=-1)
        patched_predictions = patched_logits.detach().argmax(dim=-1)
        baseline_number = baseline_predictions.lt(10)
        patched_number = patched_predictions.lt(10)
        if direction == "on":
            source = ~baseline_number
            target_before = baseline_number
            target_after = patched_number
        else:
            source = baseline_number
            target_before = ~baseline_number
            target_after = ~patched_number
        self.n += int(baseline_predictions.numel())
        self.source += int(source.sum())
        self.converted += int((source & target_after).sum())
        self.target_before += int(target_before.sum())
        self.target_after += int(target_after.sum())

    def tensor(self, device: torch.device | str) -> torch.Tensor:
        return torch.tensor(
            [
                self.n,
                self.source,
                self.converted,
                self.target_before,
                self.target_after,
            ],
            dtype=torch.int64,
            device=device,
        )

    @classmethod
    def from_tensor(cls, values: torch.Tensor) -> "TransitionStats":
        n, source, converted, target_before, target_after = (
            values.detach().cpu().tolist()
        )
        return cls(
            n=int(n),
            source=int(source),
            converted=int(converted),
            target_before=int(target_before),
            target_after=int(target_after),
        )

    def row(self, *, direction: str, router: str) -> dict[str, object]:
        if self.n <= 0 or self.source <= 0:
            raise RuntimeError("cannot summarize a transition without source trials")
        transition = "none_to_number" if direction == "on" else "number_to_none"
        target_before = self.target_before / self.n
        target_after = self.target_after / self.n
        return {
            "transition": transition,
            "router": router,
            "n_trials": self.n,
            "source_prediction_trials": self.source,
            "converted_trials": self.converted,
            "conversion_rate": self.converted / self.source,
            "target_accuracy_before": target_before,
            "target_accuracy_after": target_after,
            "target_accuracy_delta": target_after - target_before,
        }


def masked_final_token_head_hooks(
    model,
    *,
    layers: Sequence[int],
    donor_z_by_layer: dict[int, torch.Tensor],
    mask: torch.Tensor,
) -> list[tuple[str, object]]:
    """Replace selected final-token head outputs from a paired donor run."""

    normalized_layers = tuple(int(layer) for layer in layers)
    if mask.dim() != 2 or mask.shape[0] != len(normalized_layers):
        raise ValueError(
            "mask must have shape [len(layers), heads], got "
            f"{tuple(mask.shape)} for {len(normalized_layers)} layers"
        )
    hooks: list[tuple[str, object]] = []
    for layer_offset, layer in enumerate(normalized_layers):
        if layer not in donor_z_by_layer:
            raise ValueError(f"Missing donor head outputs for layer {layer}")
        donor = donor_z_by_layer[layer]
        if donor.dim() != 4 or donor.shape[1] != 1:
            raise ValueError(
                "donor head outputs must have shape [batch, 1, heads, d_head]"
            )
        if donor.shape[2] != mask.shape[1]:
            raise ValueError(
                f"Layer {layer} donor has {donor.shape[2]} heads but mask has "
                f"{mask.shape[1]}"
            )

        def patch(
            activation: torch.Tensor,
            hook,
            *,
            source=donor,
            row=mask[layer_offset],
            layer_index=layer,
        ) -> torch.Tensor:
            del hook
            if activation.dim() != 4 or activation.shape[1] != 1:
                raise ValueError(
                    "KV-cached head mask expected [batch, 1, heads, d_head], "
                    f"got {tuple(activation.shape)}"
                )
            if source.shape[0] != activation.shape[0]:
                raise ValueError(
                    f"Layer {layer_index} donor and recipient batch sizes differ"
                )
            if source.shape[2:] != activation.shape[2:]:
                raise ValueError(
                    f"Layer {layer_index} donor and recipient head shapes differ: "
                    f"{tuple(source.shape[2:])} != {tuple(activation.shape[2:])}"
                )
            weight = row.to(
                device=activation.device, dtype=activation.dtype
            ).view(1, 1, -1, 1)
            desired = source.to(
                device=activation.device, dtype=activation.dtype
            ).detach()
            return activation + weight * (desired - activation)

        hooks.append((model.attn_hook_name(layer, "z"), patch))
    return hooks


def directional_head_gate_bce(
    patched_logits: torch.Tensor,
    *,
    target_is_number: bool,
    temperature: float,
) -> torch.Tensor:
    """BCE-only objective for the formal on/off head-selection protocol."""

    score = gate_logit(patched_logits, temperature=temperature)
    target = torch.full_like(score, float(target_is_number))
    return F.binary_cross_entropy_with_logits(score, target)


def incremental_one_hot_router_hook(
    model,
    *,
    layer: int,
    heads: Sequence[int],
    key_positions: torch.Tensor,
) -> tuple[str, object]:
    """Force the final cached query to one target routing key per row."""

    selected_heads = tuple(int(head) for head in heads)
    if not selected_heads or len(set(selected_heads)) != len(selected_heads):
        raise ValueError("heads must be non-empty and unique")
    keys_cpu = key_positions.detach().cpu().long()

    def hook(pattern: torch.Tensor, hook) -> torch.Tensor:
        del hook
        if pattern.dim() != 4 or pattern.shape[2] != 1:
            raise ValueError(
                "Incremental router expected [batch, heads, 1, keys], got "
                f"{tuple(pattern.shape)}"
            )
        invalid_heads = [
            head
            for head in selected_heads
            if not 0 <= head < pattern.shape[1]
        ]
        if invalid_heads:
            raise IndexError(
                f"Router heads outside [0, {pattern.shape[1]}): {invalid_heads}"
            )
        keys = keys_cpu.to(pattern.device)
        if keys.shape != (pattern.shape[0],):
            raise ValueError("Router keys must contain one key per batch row")
        if bool(((keys < 0) | (keys >= pattern.shape[-1])).any()):
            raise ValueError("Router key lies outside the cached sequence")
        if bool(keys.eq(pattern.shape[-1] - 1).any()):
            raise ValueError(
                "Forced router keys must lie in the cached prefix, not at the "
                "current final-query token"
            )
        output = pattern.clone()
        output[:, list(selected_heads), 0, :] = 0
        rows = torch.arange(pattern.shape[0], device=pattern.device)[:, None]
        head_indices = torch.tensor(
            selected_heads, dtype=torch.long, device=pattern.device
        )[None, :]
        output[rows, head_indices, 0, keys[:, None]] = 1
        return output

    return model.attn_hook_name(int(layer), "pattern"), hook


def load_head_mask_checkpoint(
    path: Path,
    *,
    expected_top_k: int | None = FORMAL_TOP_K,
) -> dict:
    """Load a formal train-split on/off STE Top-k checkpoint.

    The default preserves Top32-only behavior for established downstream
    experiments. Pass ``expected_top_k=None`` for experiments that explicitly
    compare mask cardinalities.
    """

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != HEAD_MASK_SCHEMA_VERSION:
        raise ValueError("Unsupported head-mask checkpoint schema_version")
    if payload.get("artifact_type") != "ste_topk_head_gate":
        raise ValueError("Checkpoint is not an STE Top-k head-gate artifact")
    if payload.get("source_split") != "train":
        raise ValueError("Head mask must have been trained on the train split")
    protocol = payload.get("formal_training_protocol")
    if protocol is not None and protocol != FORMAL_TRAINING_PROTOCOL:
        raise ValueError(
            f"Unsupported formal_training_protocol: {protocol!r}"
        )
    direction = str(payload.get("selection_direction", ""))
    if direction not in {"on", "off"}:
        raise ValueError("Formal head mask must have selection_direction on/off")
    if protocol is None and "lambda_position" not in payload:
        raise ValueError("Legacy head mask does not record lambda_position")
    if float(payload.get("lambda_position", 0.0)) != 0.0:
        raise ValueError("Formal head mask must use BCE only (lambda_position=0)")
    if protocol is not None and payload.get("loss") != (
        "binary_cross_entropy_with_logits"
    ):
        raise ValueError("Formal head mask must record the BCE-only loss")
    router_mode = str(
        payload.get(
            "training_router_mode", payload.get("bce_router_mode", "")
        )
    )
    expected_router = "forced_target_key" if direction == "on" else "native"
    if router_mode != expected_router:
        raise ValueError(
            f"Formal {direction} head mask requires training_router_mode="
            f"{expected_router}, got {router_mode!r}"
        )
    router_layer = payload.get("training_router_layer")
    router_heads = payload.get("training_router_heads")
    if direction == "on":
        if not isinstance(router_layer, int):
            raise ValueError("Formal gate-on checkpoint has no forced-router layer")
        if not isinstance(router_heads, list) or not router_heads:
            raise ValueError("Formal gate-on checkpoint has no forced-router heads")
        normalized_router_heads = [int(head) for head in router_heads]
        if len(set(normalized_router_heads)) != len(normalized_router_heads):
            raise ValueError("Formal gate-on checkpoint has duplicate router heads")
        payload["training_router_heads"] = normalized_router_heads
    elif router_layer is not None or router_heads is not None:
        raise ValueError("Formal gate-off checkpoint must not define a forced router")

    train_pair_filter = str(payload.get("train_pair_filter", ""))
    expected_filter = formal_train_pair_filter(direction)
    legacy_filters = (
        {"successful_baseline_pairs"}
        if direction == "on" and protocol is None
        else set()
    )
    if train_pair_filter != expected_filter and train_pair_filter not in legacy_filters:
        raise ValueError(
            f"Formal {direction} head mask requires train_pair_filter="
            f"{expected_filter}, got {train_pair_filter!r}"
        )
    payload["normalized_train_pair_filter"] = expected_filter

    layers = payload.get("layers")
    components = payload.get("selected_components")
    if not isinstance(layers, list) or not layers:
        raise ValueError("Head-mask checkpoint has no layers")
    if not isinstance(components, list) or not components:
        raise ValueError("Head-mask checkpoint has no selected components")
    normalized_layers = [int(layer) for layer in layers]
    if normalized_layers != sorted(set(normalized_layers)):
        raise ValueError("Head-mask layers must be sorted and unique")
    if direction == "on" and int(router_layer) <= max(normalized_layers):
        raise ValueError(
            "Formal gate-on forced router must follow every selected-head layer"
        )
    normalized = [(int(layer), int(head)) for layer, head in components]
    if len(set(normalized)) != len(normalized):
        raise ValueError("Head-mask checkpoint contains duplicate components")
    top_k = int(payload.get("top_k", -1))
    if expected_top_k is not None and top_k != expected_top_k:
        raise ValueError(f"Formal head mask must use Top{expected_top_k}")
    if top_k != len(normalized):
        raise ValueError("top_k does not match selected component count")
    n_heads = int(payload.get("n_heads", -1))
    scores = payload.get("scores")
    hard_mask = payload.get("hard_mask")
    expected_shape = (len(normalized_layers), n_heads)
    if n_heads <= 0 or not isinstance(scores, torch.Tensor) or tuple(
        scores.shape
    ) != expected_shape:
        raise ValueError("Head-mask scores do not match layers and n_heads")
    if not isinstance(hard_mask, torch.Tensor) or tuple(
        hard_mask.shape
    ) != expected_shape:
        raise ValueError("Hard mask does not match layers and n_heads")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("Head-mask scores contain non-finite values")
    if not bool(((hard_mask == 0) | (hard_mask == 1)).all()):
        raise ValueError("Hard mask must be binary")
    hard_mask = hard_mask.bool()
    if int(hard_mask.sum()) != len(normalized):
        raise ValueError("Hard-mask cardinality does not match top_k")
    hard_components = {
        (normalized_layers[layer_offset], int(head))
        for layer_offset, head in hard_mask.nonzero(as_tuple=False).tolist()
    }
    if hard_components != set(normalized):
        raise ValueError("Hard mask and selected components disagree")
    expected_hard_mask = hard_topk_mask(scores.float(), top_k=top_k).bool()
    if not torch.equal(hard_mask, expected_hard_mask):
        raise ValueError(f"Hard mask is not the Top{top_k} of the saved scores")

    input_hashes = payload.get("input_sha256")
    if not isinstance(input_hashes, dict):
        raise ValueError("Head-mask checkpoint has no input_sha256 metadata")
    for name in ("train_cluster_csv", "train_concept_vectors"):
        value = input_hashes.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"Head-mask checkpoint has no valid {name} hash")
    if direction == "on":
        outcomes_hash = input_hashes.get("train_outcomes_csv")
        if not isinstance(outcomes_hash, str) or len(outcomes_hash) != 64:
            raise ValueError(
                "Formal gate-on checkpoint has no valid train_outcomes_csv hash"
            )

    required_scalar_fields = (
        "model",
        "injection_layer",
        "strength",
        "scale_mode",
        "prompt_template",
        "prompt_preamble",
        "gate_temperature",
    )
    missing = [name for name in required_scalar_fields if payload.get(name) is None]
    if missing:
        raise ValueError(f"Head-mask checkpoint is missing metadata: {missing}")
    if int(payload["injection_layer"]) < 0:
        raise ValueError("Checkpoint injection_layer must be non-negative")
    if int(payload["injection_layer"]) >= min(normalized_layers):
        raise ValueError("Checkpoint selected heads must follow injection_layer")
    if float(payload["strength"]) <= 0 or float(payload["gate_temperature"]) <= 0:
        raise ValueError("Checkpoint strength and gate_temperature must be positive")
    if not isinstance(payload["model"], str) or not payload["model"]:
        raise ValueError("Checkpoint model must be a non-empty string")
    if payload["scale_mode"] not in {"unit", "relative_hidden_norm"}:
        raise ValueError("Checkpoint has an unsupported scale_mode")
    if payload["prompt_preamble"] not in {"none", "user", "system"}:
        raise ValueError("Checkpoint has an unsupported prompt_preamble")
    if protocol is not None:
        if int(payload.get("full_coordinate_count", -1)) != FORMAL_COORDINATE_COUNT:
            raise ValueError(
                f"Formal checkpoint must record {FORMAL_COORDINATE_COUNT} "
                "full-grid coordinates"
            )
        expected_pool = (
            "locked_native_injected_exact_targets"
            if direction == "on"
            else "full_coordinate_grid"
        )
        if payload.get("train_candidate_pool") != expected_pool:
            raise ValueError(
                f"Formal {direction} checkpoint requires train_candidate_pool="
                f"{expected_pool}"
            )
        candidate_count = int(payload.get("candidate_coordinate_count", -1))
        if not 0 < candidate_count <= FORMAL_COORDINATE_COUNT:
            raise ValueError("Formal checkpoint has an invalid candidate count")
        if direction == "off" and candidate_count != FORMAL_COORDINATE_COUNT:
            raise ValueError("Formal gate-off must train from the full grid")
        final_train_count = int(payload.get("final_epoch_train_count", -1))
        if not 0 < final_train_count <= candidate_count:
            raise ValueError("Formal checkpoint has an invalid final train count")
    payload["selected_components"] = normalized
    payload["layers"] = normalized_layers
    return payload
