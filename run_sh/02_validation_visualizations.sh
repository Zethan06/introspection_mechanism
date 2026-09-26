#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_token0_9_prompt
: "${VISUALIZATION_GPU:?Set VISUALIZATION_GPU in $ENV_FILE}"
export CUDA_VISIBLE_DEVICES="$VISUALIZATION_GPU"
require_frozen_injection

# Both figures reuse the population-wide payload from Stage 01, sliced to the
# validation concepts by name. Re-extracting here would recompute the same
# contrastive baseline twice more.
require_file "$STATE_VECTOR_FILE"

# Every layer and every attention head on validation.
"$PYTHON_BIN" scripts/build_position_averaged_attention_visualization.py \
  --model "$MODEL_ID" --task token_localization \
  --cluster_file "$MODEL_DATA_DIR/clusters/validation.csv" \
  --concepts_json "$MODEL_DATA_DIR/concepts/validation.json" \
  --concept_csv "$MODEL_DATA_DIR/concepts/validation.csv" \
  --state_vectors "$STATE_VECTOR_FILE" \
  --average_all_concepts \
  --calibration_selection "$MODEL_RESULTS_DIR/calibration/selection.json" \
  --injection_layer "$INJECTION_LAYER" --strength "$INJECTION_STRENGTH" \
  --scale_mode "$SCALE_MODE" --prompt_template "$PROMPT_TEMPLATE" \
  --prompt_preamble "$PROMPT_PREAMBLE" \
  --position_index_start "$POSITION_INDEX_START" \
  --concept_batch_size "$BATCH_SIZE" --vector_batch_size "$EXTRACTION_BATCH_SIZE" \
  --dtype "$DTYPE" --results_dir "$MODEL_RESULTS_DIR" --seed "$SEED" \
  ${TRUST_REMOTE_CODE_UNDERSCORE_ARGS[@]+"${TRUST_REMOTE_CODE_UNDERSCORE_ARGS[@]}"}

# Whole-model latent representation from layer zero through the final layer.
"$PYTHON_BIN" scripts/run_token_representation_projection.py \
  --model "$MODEL_ID" \
  --cluster_csv "$MODEL_DATA_DIR/clusters/validation.csv" --cluster_count 30 \
  --concepts_json "$MODEL_DATA_DIR/concepts/validation.json" \
  --state_vectors "$STATE_VECTOR_FILE" \
  --injection_layer "$INJECTION_LAYER" --coeffs "$INJECTION_STRENGTH" \
  --start_layer 0 \
  --scale_mode "$SCALE_MODE" --prompt_template "$PROMPT_TEMPLATE" \
  --prompt_preamble "$PROMPT_PREAMBLE" --concept_batch_size "$BATCH_SIZE" \
  --dtype "$DTYPE" --results_dir "$MODEL_RESULTS_DIR" --seed "$SEED" \
  ${TRUST_REMOTE_CODE_UNDERSCORE_ARGS[@]+"${TRUST_REMOTE_CODE_UNDERSCORE_ARGS[@]}"}

# Single-canvas copy of the latent gallery, exportable as vector PDF (latent PCA figures).
"$PYTHON_BIN" scripts/redraw_latent_html_from_saved.py \
  "$MODEL_RESULTS_DIR/visualizations/validation_latent.html" \
  "$MODEL_RESULTS_DIR/visualizations/validation_latent.vector.html" \
  --artifact latent
