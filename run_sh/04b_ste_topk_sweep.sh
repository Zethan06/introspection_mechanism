#!/usr/bin/env bash
# Enumerate STE mask cardinalities and summarize both directional transitions.
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_token0_9_prompt
require_frozen_injection
require_var STE_LAYERS ROUTER_LAYER ROUTER_HEADS STE_GPU

# Optional launch-time overrides avoid editing the checked-in model profiles.
STE_SWEEP_GPUS="${STE_SWEEP_GPUS:-$STE_GPU}"
STE_SWEEP_BATCH_SIZE="${STE_SWEEP_BATCH_SIZE:-$BATCH_SIZE}"
export CUDA_VISIBLE_DEVICES="$STE_SWEEP_GPUS"
IFS=',' read -r -a STE_SWEEP_GPU_IDS <<< "$STE_SWEEP_GPUS"
STE_SWEEP_WORLD_SIZE="${#STE_SWEEP_GPU_IDS[@]}"
STE_SWEEP_DISTRIBUTED="${STE_SWEEP_DISTRIBUTED:-${STE_DISTRIBUTED:-true}}"
if [[ "$STE_SWEEP_DISTRIBUTED" != "true" && "$STE_SWEEP_DISTRIBUTED" != "false" ]]; then
  echo "STE_SWEEP_DISTRIBUTED must be true or false" >&2
  exit 2
fi

run_ste_python() {
  if [[ "$STE_SWEEP_DISTRIBUTED" == "true" ]] && (( STE_SWEEP_WORLD_SIZE > 1 )); then
    "$PYTHON_BIN" -m torch.distributed.run \
      --standalone \
      --nproc_per_node="$STE_SWEEP_WORLD_SIZE" \
      "$@" \
      --distributed
  else
    "$PYTHON_BIN" "$@"
  fi
}

read -r -a TOP_K_VALUES <<< "${STE_SWEEP_TOP_K_VALUES:-1 4 8 16 32 48 64}"
read -r -a SUMMARY_TOP_K_VALUES <<< \
  "${STE_SWEEP_SUMMARY_TOP_K_VALUES:-${TOP_K_VALUES[*]}}"
if (( ${#TOP_K_VALUES[@]} == 0 || ${#SUMMARY_TOP_K_VALUES[@]} == 0 )); then
  echo "STE sweep Top-k lists must not be empty" >&2
  exit 2
fi
SWEEP_DIR="$MODEL_RESULTS_DIR/ste_topk_sweep"
PREPARED="$MODEL_RESULTS_DIR/prepared_splits"
TRAIN_VECTORS="$PREPARED/prepared_train_split/concept_vectors.pt"
TRAIN_OUTCOMES="$PREPARED/prepared_train_split/outcomes.csv"
VALIDATION_VECTORS="$PREPARED/prepared_validation_split/concept_vectors.pt"

require_file "$TRAIN_VECTORS" "$TRAIN_OUTCOMES" "$VALIDATION_VECTORS"
mkdir -p "$SWEEP_DIR"

for TOP_K in "${TOP_K_VALUES[@]}"; do
  TOP_K_DIR="$SWEEP_DIR/top$TOP_K"
  echo "[Top$TOP_K] Training gate-on mask"
  run_ste_python scripts/train_ste_topk_head_gate.py \
    --direction on \
    --top_k "$TOP_K" \
    --model "$MODEL_ID" \
    --train_cluster_csv "$MODEL_DATA_DIR/clusters/train.csv" \
    --train_concept_vectors "$TRAIN_VECTORS" \
    --train_outcomes_csv "$TRAIN_OUTCOMES" \
    --output_dir "$TOP_K_DIR/train_on" \
    --layers "$STE_LAYERS" \
    --injection_layer "$INJECTION_LAYER" \
    --strength "$INJECTION_STRENGTH" \
    --forced_router_layer "$ROUTER_LAYER" \
    --forced_router_heads $ROUTER_HEADS \
    --batch_size "$STE_SWEEP_BATCH_SIZE" \
    --prompt_template "$PROMPT_TEMPLATE" \
    --prompt_preamble "$PROMPT_PREAMBLE" \
    --scale_mode "$SCALE_MODE" \
    --dtype "$DTYPE" \
    --seed "$SEED" \
    --overwrite \
    ${TRUST_REMOTE_CODE_UNDERSCORE_ARGS[@]+"${TRUST_REMOTE_CODE_UNDERSCORE_ARGS[@]}"}

  echo "[Top$TOP_K] Training gate-off mask"
  run_ste_python scripts/train_ste_topk_head_gate.py \
    --direction off \
    --top_k "$TOP_K" \
    --model "$MODEL_ID" \
    --train_cluster_csv "$MODEL_DATA_DIR/clusters/train.csv" \
    --train_concept_vectors "$TRAIN_VECTORS" \
    --output_dir "$TOP_K_DIR/train_off" \
    --layers "$STE_LAYERS" \
    --injection_layer "$INJECTION_LAYER" \
    --strength "$INJECTION_STRENGTH" \
    --batch_size "$STE_SWEEP_BATCH_SIZE" \
    --prompt_template "$PROMPT_TEMPLATE" \
    --prompt_preamble "$PROMPT_PREAMBLE" \
    --scale_mode "$SCALE_MODE" \
    --dtype "$DTYPE" \
    --seed "$SEED" \
    --overwrite \
    ${TRUST_REMOTE_CODE_UNDERSCORE_ARGS[@]+"${TRUST_REMOTE_CODE_UNDERSCORE_ARGS[@]}"}

  for DIRECTION in on off; do
    echo "[Top$TOP_K] Validating $DIRECTION mask"
    run_ste_python scripts/test_ste_topk_head_gate.py \
      --model "$MODEL_ID" \
      --head_mask "$TOP_K_DIR/train_$DIRECTION/head_mask.pt" \
      --test_cluster_csv "$MODEL_DATA_DIR/clusters/validation.csv" \
      --test_concept_vectors "$VALIDATION_VECTORS" \
      --dataset_split validation \
      --output_dir "$TOP_K_DIR/validation_$DIRECTION" \
      --batch_size "$STE_SWEEP_BATCH_SIZE" \
      --dtype "$DTYPE" \
      --seed "$SEED" \
      --overwrite \
      ${TRUST_REMOTE_CODE_UNDERSCORE_ARGS[@]+"${TRUST_REMOTE_CODE_UNDERSCORE_ARGS[@]}"}
  done
done

"$PYTHON_BIN" scripts/summarize_ste_topk_sweep.py \
  --sweep_dir "$SWEEP_DIR" \
  --result_split validation \
  --top_k "${SUMMARY_TOP_K_VALUES[@]}"

# Gate-head table: the Top-32 heads of each direction and their overlap.
if [[ -f "$SWEEP_DIR/top32/train_on/selected_heads.json" ]]; then
  "$PYTHON_BIN" scripts/analyze_ste_head_selection.py \
    --selection "on=$SWEEP_DIR/top32/train_on/selected_heads.json" \
                "off=$SWEEP_DIR/top32/train_off/selected_heads.json" \
    --output-json "$SWEEP_DIR/top32/selection_summary.json"
fi

echo "STE Top-k validation sweep complete: $SWEEP_DIR/validation_transition_summary.csv"
