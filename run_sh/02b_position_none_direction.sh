#!/usr/bin/env bash
# Figure 2a: final-token position--none direction on held-out validation trials.
#
# The injected validation trials are split into a fit half and a held-out
# half: validation concepts 50/50 and validation clusters 15/15, so neither
# concept identities nor prompt contexts leak from fit to eval. On the fit half,
#   d = norm( mean_{position report} norm(h) - mean_{none} norm(h) )
# over injected final-token residuals, at every layer; each held-out trial is
# scored by its cosine with d. plot_manuscript_figures.py reads
# $DIST_DIR/test_scores.npz.
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_token0_9_prompt
: "${VISUALIZATION_GPU:?Set VISUALIZATION_GPU in $ENV_FILE}"
export CUDA_VISIBLE_DEVICES="$VISUALIZATION_GPU"
require_frozen_injection
require_file "$STATE_VECTOR_FILE"

DIST_DIR="$MODEL_RESULTS_DIR/position_none_direction"
mkdir -p "$DIST_DIR"

echo "[02b:1/3] Split validation concepts and clusters into fit/eval halves"
"$PYTHON_BIN" scripts/split_csv_halves.py \
  --input "$MODEL_DATA_DIR/concepts/validation.csv" \
  --output-fit "$DIST_DIR/validation_fit_concepts.csv" \
  --output-eval "$DIST_DIR/validation_eval_concepts.csv" \
  --key-column concept \
  --seed "$SEED"
"$PYTHON_BIN" scripts/split_csv_halves.py \
  --input "$MODEL_DATA_DIR/clusters/validation.csv" \
  --output-fit "$DIST_DIR/validation_fit_clusters.csv" \
  --output-eval "$DIST_DIR/validation_eval_clusters.csv" \
  --key-column cluster_key \
  --seed "$SEED"

COMMON_ARGS=(
  --model "$MODEL_ID"
  --state-vectors "$STATE_VECTOR_FILE"
  --injection-layer "$INJECTION_LAYER"
  --start-layer 0
  --strength "$INJECTION_STRENGTH"
  --representation injected
  --capture-position final_token
  --prompt-template "$PROMPT_TEMPLATE"
  --prompt-preamble "$PROMPT_PREAMBLE"
  --scale-mode "$SCALE_MODE"
  --batch-size "$BATCH_SIZE"
  --dtype "$DTYPE"
  ${TRUST_REMOTE_CODE_ARGS[@]+"${TRUST_REMOTE_CODE_ARGS[@]}"}
)

echo "[02b:2/3] Fit the direction on the fit half"
"$PYTHON_BIN" scripts/fit_position_none_direction.py \
  "${COMMON_ARGS[@]}" \
  --train-concept-csv "$DIST_DIR/validation_fit_concepts.csv" \
  --train-cluster-csv "$DIST_DIR/validation_fit_clusters.csv" \
  --output "$DIST_DIR/train_direction_components.pt"

echo "[02b:3/3] Score the eval half once"
"$PYTHON_BIN" scripts/score_position_none_direction.py \
  "${COMMON_ARGS[@]}" \
  --train-concept-csv "$DIST_DIR/validation_fit_concepts.csv" \
  --train-cluster-csv "$DIST_DIR/validation_fit_clusters.csv" \
  --training-components "$DIST_DIR/train_direction_components.pt" \
  --test-concept-csv "$DIST_DIR/validation_eval_concepts.csv" \
  --test-cluster-csv "$DIST_DIR/validation_eval_clusters.csv" \
  --output "$DIST_DIR/test_scores.npz"

echo "Stage 02b complete: $DIST_DIR/test_scores.npz"
