#!/usr/bin/env bash
# Lexical replacement control (appendix table): no activation injection.
#
# Validation concepts x 30 test clusters x ten slots, each slot replaced by the
# matching concept word or a seeded random vocabulary word, scored under all six
# label arms. Writes results/lexical_replacement/<model>/<arm>/ and pools every
# model already present into results/lexical_replacement/summary.csv.
#
# LEXICAL_ARMS narrows the arms; LEXICAL_WORKERS splits each arm over that many
# processes on the GPUs in STE_GPU (round robin).
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_var STE_GPU

LEXICAL_ARMS="${LEXICAL_ARMS:-digits_identity letters_identity words_identity digits_shuffled letters_shuffled words_shuffled}"
LEXICAL_BATCH_SIZE="${LEXICAL_BATCH_SIZE:-$BATCH_SIZE}"
IFS=',' read -r -a GPU_LIST <<< "$STE_GPU"
LEXICAL_WORKERS="${LEXICAL_WORKERS:-${#GPU_LIST[@]}}"
ROOT="$RESULTS_ROOT/lexical_replacement"
PLAN_DIR="$ROOT/$MODEL_SLUG"
require_file "$MODEL_DATA_DIR/concepts/validation.json" "$MODEL_DATA_DIR/clusters/test.csv" \
  "$MODEL_DATA_DIR/vocabulary/english.csv"

if [[ ! -f "$PLAN_DIR/plan.json" ]]; then
  "$PYTHON_BIN" scripts/run_lexical_replacement_control.py plan \
    --dataset_dir "$MODEL_DATA_DIR" \
    --output_dir "$PLAN_DIR" \
    --seed "$SEED"
fi

for ARM in $LEXICAL_ARMS; do
  echo "[05] $MODEL_SLUG $ARM"
  PIDS=()
  for (( WORKER = 0; WORKER < LEXICAL_WORKERS; WORKER++ )); do
    CUDA_VISIBLE_DEVICES="${GPU_LIST[$(( WORKER % ${#GPU_LIST[@]} ))]}" \
      "$PYTHON_BIN" scripts/run_lexical_replacement_control.py run \
        --arm "$ARM" \
        --model "$MODEL_ID" \
        --plan "$PLAN_DIR/plan.json" \
        --output_dir "$PLAN_DIR/$ARM" \
        --worker "$WORKER" \
        --workers "$LEXICAL_WORKERS" \
        --batch_size "$LEXICAL_BATCH_SIZE" \
        --dtype "$DTYPE" &
    PIDS+=($!)
  done
  for PID in "${PIDS[@]}"; do
    wait "$PID"
  done
done

"$PYTHON_BIN" scripts/run_lexical_replacement_control.py summarize --results_root "$ROOT"
echo "Stage 05 complete: $ROOT/summary.csv"
