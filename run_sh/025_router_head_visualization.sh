#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_token0_9_prompt
: "${VISUALIZATION_GPU:?Set VISUALIZATION_GPU in $ENV_FILE}"
: "${ROUTER_LAYER:?Set ROUTER_LAYER in $ENV_FILE}"
export CUDA_VISIBLE_DEVICES="$VISUALIZATION_GPU"
require_frozen_injection
require_file "$STATE_VECTOR_FILE"

"$PYTHON_BIN" scripts/run_token_representation_projection.py \
  --model "$MODEL_ID" \
  --cluster_csv "$MODEL_DATA_DIR/clusters/validation.csv" \
  --cluster_count 30 \
  --concepts_json "$MODEL_DATA_DIR/concepts/validation.json" \
  --state_vectors "$STATE_VECTOR_FILE" \
  --injection_layer "$INJECTION_LAYER" \
  --coeffs "$INJECTION_STRENGTH" \
  --start_layer 0 \
  --head_layer "$ROUTER_LAYER" \
  --scale_mode "$SCALE_MODE" \
  --prompt_template "$PROMPT_TEMPLATE" \
  --prompt_preamble "$PROMPT_PREAMBLE" \
  --concept_batch_size "$BATCH_SIZE" \
  --dtype "$DTYPE" \
  --results_dir "$MODEL_RESULTS_DIR" \
  --seed "$SEED" \
  ${TRUST_REMOTE_CODE_UNDERSCORE_ARGS[@]+"${TRUST_REMOTE_CODE_UNDERSCORE_ARGS[@]}"}

# Single-canvas copy of the per-head gallery, exportable as vector PDF (head PCA figures).
"$PYTHON_BIN" scripts/redraw_latent_html_from_saved.py \
  "$MODEL_RESULTS_DIR/visualizations/validation_head_ov.html" \
  "$MODEL_RESULTS_DIR/visualizations/validation_head_ov.vector.html" \
  --artifact head-ov
