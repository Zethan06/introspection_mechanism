#!/usr/bin/env bash
# Figure 3 and Table 2 on the test grid with the Top-32 masks from Stage 04b.
#
#   top32/test_{on,off}                      Figure 3a/b: gate patch, native router
#   ste_topk_sweep/transition_summary.csv    the Figure 3a/b rates
#   top32/test_{on,off}_clean_router_pin     Figure 3c/d: router heads pinned clean
#   top32/test_{on,off}_env_output_patch     Figure 3c/d gate+router cells and
#                                            Table 2: gate donor i x router donor j,
#                                            router heads from the injected run at j
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_token0_9_prompt
require_frozen_injection
require_var ROUTER_LAYER ROUTER_HEADS STE_GPU

FIG3_BATCH_SIZE="${FIG3_BATCH_SIZE:-$BATCH_SIZE}"
export CUDA_VISIBLE_DEVICES="$STE_GPU"
IFS=',' read -r -a GPU_LIST <<< "$STE_GPU"
WORLD_SIZE="${#GPU_LIST[@]}"

run_test() {
  if (( WORLD_SIZE > 1 )); then
    "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$WORLD_SIZE" \
      scripts/test_ste_topk_head_gate.py "$@" --distributed
  else
    "$PYTHON_BIN" scripts/test_ste_topk_head_gate.py "$@"
  fi
}

SWEEP_DIR="$MODEL_RESULTS_DIR/ste_topk_sweep"
TOP32="$SWEEP_DIR/top32"
TEST_VECTORS="$MODEL_RESULTS_DIR/prepared_splits/prepared_test_split/concept_vectors.pt"
require_file "$TOP32/train_on/head_mask.pt" "$TOP32/train_off/head_mask.pt" "$TEST_VECTORS"

COMMON=(
  --model "$MODEL_ID"
  --test_cluster_csv "$MODEL_DATA_DIR/clusters/test.csv"
  --test_concept_vectors "$TEST_VECTORS"
  --batch_size "$FIG3_BATCH_SIZE"
  --dtype "$DTYPE"
  --seed "$SEED"
  --overwrite
  ${TRUST_REMOTE_CODE_UNDERSCORE_ARGS[@]+"${TRUST_REMOTE_CODE_UNDERSCORE_ARGS[@]}"}
)
ROUTER=(--forced_router_layer "$ROUTER_LAYER" --forced_router_heads $ROUTER_HEADS)

for DIRECTION in on off; do
  MASK="$TOP32/train_$DIRECTION/head_mask.pt"
  echo "[04f] gate $DIRECTION: native router"
  run_test "${COMMON[@]}" --head_mask "$MASK" --output_dir "$TOP32/test_$DIRECTION"

  echo "[04f] gate $DIRECTION: router pinned to the clean run"
  run_test "${COMMON[@]}" --head_mask "$MASK" "${ROUTER[@]}" --clean_router_patch \
    --output_dir "$TOP32/test_${DIRECTION}_clean_router_pin"

  echo "[04f] gate $DIRECTION: gate donor i x injected router donor j"
  run_test "${COMMON[@]}" --head_mask "$MASK" "${ROUTER[@]}" \
    --router_position_mode all \
    --router_intervention injected_output_patch \
    --router_donor_state injected \
    --output_dir "$TOP32/test_${DIRECTION}_env_output_patch"
done

"$PYTHON_BIN" scripts/summarize_ste_topk_sweep.py \
  --sweep_dir "$SWEEP_DIR" \
  --result_split test \
  --top_k 32

echo "Stage 04f complete: $SWEEP_DIR/transition_summary.csv and $TOP32/test_*"
