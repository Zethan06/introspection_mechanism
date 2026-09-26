#!/usr/bin/env bash
# Patch every attention head in every model layer on one requested GPU.
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_frozen_injection

PATCH_GPU="${HEAD_PATCH_GPU_IDS:-${VISUALIZATION_GPU:-${STE_GPU:-}}}"
if [[ ! "$PATCH_GPU" =~ ^[0-9]+$ ]]; then
  echo "Whole-model head sweep requires exactly one numeric GPU id: $PATCH_GPU" >&2
  exit 2
fi

OUTPUT_DIR="${HEAD_PATCH_MODEL_SWEEP_OUTPUT_DIR:-$MODEL_RESULTS_DIR/validation_clean_head_model_sweep}"
LOG_DIR="${HEAD_PATCH_MODEL_SWEEP_LOG_DIR:-$MODEL_LOG_DIR/validation_clean_head_model_sweep}"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "Starting whole-model clean-head sweep for $MODEL_SLUG on GPU $PATCH_GPU"
HEAD_PATCH_SHARD_INDEX=0 \
HEAD_PATCH_SHARD_COUNT=1 \
HEAD_PATCH_GPU_IDS="$PATCH_GPU" \
HEAD_PATCH_OUTPUT_DIR="$OUTPUT_DIR" \
  bash run_sh/08_validation_clean_head_patch.sh "$ENV_FILE" 2>&1 | tee "$LOG_DIR/sweep.log"

"$PYTHON_BIN" scripts/plot_validation_clean_head_model_sweep.py \
  --effects "$OUTPUT_DIR/condition_effects.csv" \
  --output_stem "$OUTPUT_DIR/all_head_causal_effect"

echo "Validation clean-head whole-model sweep complete: $OUTPUT_DIR"
