#!/usr/bin/env bash
# Random-k control for the Top-k selection figure.
#
# For each k, draw k heads uniformly from the STE search window (the layers of
# the Top-32 masks) ten times, patch them exactly as the trained masks are
# patched, and evaluate both gate directions on the validation grid. Writes
# ste_topk_sweep/random_topk_control/random_topk_transition_summary.csv, which
# plot_ste_topk_gate_effect.py draws beside the Stage 04b curve.
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_token0_9_prompt
require_frozen_injection
require_var STE_GPU

export CUDA_VISIBLE_DEVICES="$STE_GPU"
IFS=',' read -r -a GPU_LIST <<< "$STE_GPU"
WORLD_SIZE="${#GPU_LIST[@]}"
read -r -a TOP_K_VALUES <<< "${STE_SWEEP_TOP_K_VALUES:-1 4 8 16 32 48 64}"

SWEEP_DIR="$MODEL_RESULTS_DIR/ste_topk_sweep"
VALIDATION_VECTORS="$MODEL_RESULTS_DIR/prepared_splits/prepared_validation_split/concept_vectors.pt"
require_file "$SWEEP_DIR/top32/train_on/head_mask.pt" "$SWEEP_DIR/top32/train_off/head_mask.pt" \
  "$VALIDATION_VECTORS"

ARGS=(
  scripts/evaluate_random_topk_head_controls.py
  --model "$MODEL_ID"
  --reference_head_mask_on "$SWEEP_DIR/top32/train_on/head_mask.pt"
  --reference_head_mask_off "$SWEEP_DIR/top32/train_off/head_mask.pt"
  --test_cluster_csv "$MODEL_DATA_DIR/clusters/validation.csv"
  --test_concept_vectors "$VALIDATION_VECTORS"
  --dataset_split validation
  --output_dir "$SWEEP_DIR/random_topk_control"
  --top_k "${TOP_K_VALUES[@]}"
  --repeats 10
  --batch_size "$BATCH_SIZE"
  --dtype "$DTYPE"
  --seed "$SEED"
  --overwrite
  ${TRUST_REMOTE_CODE_UNDERSCORE_ARGS[@]+"${TRUST_REMOTE_CODE_UNDERSCORE_ARGS[@]}"}
)
if (( WORLD_SIZE > 1 )); then
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$WORLD_SIZE" \
    "${ARGS[@]}" --distributed
else
  "$PYTHON_BIN" "${ARGS[@]}"
fi

echo "Stage 04d complete: $SWEEP_DIR/random_topk_control/random_topk_transition_summary.csv"
