#!/usr/bin/env bash
# Stage 09: Table 1 localization and clean-none accuracy under other label sets.
#
# Holds the experiment fixed (concept vectors, concepts, clusters, injection
# layer and strength) and changes only the label printed beside each candidate.
# Every concept x cluster x position trial of a split is counted. The ordered
# digit column comes from the Stage 04f test grid.
#
#   LABEL_ARMS   (default: letters words digits:shuffled letters:shuffled words:shuffled)
#   LABEL_SPLITS (default: test)
#
# Ordered arms write label_accuracy/<template>/; shuffled arms, which print one
# derangement of the labels per cluster (seed 42), write
# label_shuffle/shuffled_<label set>/. summarize_task_performance.py reads both.
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_frozen_injection

LABEL_ARMS="${LABEL_ARMS:-letters words digits:shuffled letters:shuffled words:shuffled}"
read -r -a LABEL_SPLITS <<< "${LABEL_SPLITS:-test}"
LABEL_EVAL_GPU="${LABEL_EVAL_GPU:-$VISUALIZATION_GPU}"
LABEL_EVAL_BATCH_SIZE="${LABEL_EVAL_BATCH_SIZE:-$BATCH_SIZE}"
require_var LABEL_EVAL_GPU LABEL_EVAL_BATCH_SIZE
export CUDA_VISIBLE_DEVICES="$LABEL_EVAL_GPU"
require_file "$STATE_VECTOR_FILE"

declare -a ARM_SPECS=()
for ARM in $LABEL_ARMS; do
  ARM_SPECS+=("$(parse_label_arm "$ARM")")
done

LABEL_EVAL_OVERWRITE_ARGS=()
if [[ "${LABEL_EVAL_OVERWRITE:-false}" == "true" ]]; then
  LABEL_EVAL_OVERWRITE_ARGS=(--overwrite)
fi

for ARM_SPEC in "${ARM_SPECS[@]}"; do
  read -r LABEL_SET TEMPLATE PERMUTATION <<<"$ARM_SPEC"
  if [[ "$PERMUTATION" == "identity" ]]; then
    OUTPUT_DIR="$MODEL_RESULTS_DIR/label_accuracy/$TEMPLATE"
  else
    case "$LABEL_SET" in
      letters) LABEL_KEY=letters_a_j ;;
      words)   LABEL_KEY=numwords_one_ten ;;
      digits)  LABEL_KEY=tokens_0_9 ;;
    esac
    OUTPUT_DIR="$MODEL_RESULTS_DIR/label_shuffle/shuffled_$LABEL_KEY"
  fi
  mkdir -p "$OUTPUT_DIR"
  for SPLIT in "${LABEL_SPLITS[@]}"; do
    require_file "$MODEL_DATA_DIR/concepts/$SPLIT.json" "$MODEL_DATA_DIR/clusters/$SPLIT.csv"
    echo "[09] ${LABEL_SET}_${PERMUTATION} $SPLIT -> $OUTPUT_DIR"
    "$PYTHON_BIN" scripts/evaluate_split_label_accuracy.py \
      --model "$MODEL_ID" \
      --cluster_csv "$MODEL_DATA_DIR/clusters/$SPLIT.csv" \
      --concepts_json "$MODEL_DATA_DIR/concepts/$SPLIT.json" \
      --concept_vectors "$STATE_VECTOR_FILE" \
      --output_dir "$OUTPUT_DIR" \
      --split_name "$SPLIT" \
      --injection_layer "$INJECTION_LAYER" \
      --strength "$INJECTION_STRENGTH" \
      --batch_size "$LABEL_EVAL_BATCH_SIZE" \
      --prompt_template "$TEMPLATE" \
      --label_permutation "$PERMUTATION" \
      --prompt_preamble "$PROMPT_PREAMBLE" \
      --scale_mode "$SCALE_MODE" \
      --dtype "$DTYPE" \
      --seed "$SEED" \
      ${LABEL_EVAL_OVERWRITE_ARGS[@]+"${LABEL_EVAL_OVERWRITE_ARGS[@]}"} \
      ${TRUST_REMOTE_CODE_UNDERSCORE_ARGS[@]+"${TRUST_REMOTE_CODE_UNDERSCORE_ARGS[@]}"}
  done
done

echo "Stage 09 complete for arms: $LABEL_ARMS"
