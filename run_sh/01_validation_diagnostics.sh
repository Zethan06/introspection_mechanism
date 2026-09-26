#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_token0_9_prompt
: "${VISUALIZATION_GPU:?Set VISUALIZATION_GPU in $ENV_FILE}"
export CUDA_VISIBLE_DEVICES="$VISUALIZATION_GPU"
require_frozen_injection

DIAG_DIR="$MODEL_RESULTS_DIR/validation_diagnostics"

# One canonical vector payload containing the fixed train/validation/test
# concept population. All downstream fits select their own split from it.
"$PYTHON_BIN" scripts/extract_concept_vector_payload.py \
  --model "$MODEL_ID" \
  --concepts_json "$MODEL_DATA_DIR/concepts/selected.json" \
  --output_dir "$DIAG_DIR/state_vectors" \
  --layer "$INJECTION_LAYER" \
  --batch_size "$EXTRACTION_BATCH_SIZE" \
  --dtype "$DTYPE" \
  ${TRUST_REMOTE_CODE_ARGS[@]+"${TRUST_REMOTE_CODE_ARGS[@]}"}

# Fit position K-means on train only and evaluate once on validation.
"$PYTHON_BIN" scripts/train_test_latent_position_clustering.py \
  --model "$MODEL_ID" \
  --train-concepts-json "$MODEL_DATA_DIR/concepts/train.json" \
  --train-cluster-csv "$MODEL_DATA_DIR/clusters/train.csv" \
  --test-concepts-json "$MODEL_DATA_DIR/concepts/validation.json" \
  --test-cluster-csv "$MODEL_DATA_DIR/clusters/validation.csv" \
  --state-vectors "$STATE_VECTOR_FILE" \
  --output-dir "$DIAG_DIR/position_kmeans" \
  --injection-layer "$INJECTION_LAYER" \
  --start-layer 0 \
  --strength "$INJECTION_STRENGTH" \
  --cluster-count 30 \
  --concept-batch-size "$BATCH_SIZE" \
  --prompt-template "$PROMPT_TEMPLATE" \
  --prompt-preamble "$PROMPT_PREAMBLE" \
  --scale-mode "$SCALE_MODE" \
  --seed "$SEED" \
  --dtype "$DTYPE" \
  ${TRUST_REMOTE_CODE_ARGS[@]+"${TRUST_REMOTE_CODE_ARGS[@]}"}
