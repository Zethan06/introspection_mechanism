#!/usr/bin/env bash
# Stage 03: build the three prepared split banks.
#
# Each bank is a self-contained pair:
#
#   prepared_<split>_split/concept_vectors.pt   split vectors, in split order
#   prepared_<split>_split/outcomes.csv         natural injected outcome per
#                                               (concept, cluster, position)
#
# Both files share one positional concept index, which is what every causal
# experiment in stages 04b-10 keys on. Always pass the pair from the same bank.
#
# The splits are disjoint subsets of concepts/selected.json and share one
# baseline word list, so vectors are sliced by name out of the canonical
# state_vectors.pt from Stage 01 rather than re-extracted. Only the outcome
# labels need forward passes.
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_token0_9_prompt
require_frozen_injection
require_var STE_GPU
export CUDA_VISIBLE_DEVICES="$STE_GPU"

OUTCOME_DIR="$MODEL_RESULTS_DIR/prepared_splits"
require_file "$STATE_VECTOR_FILE"
mkdir -p "$OUTCOME_DIR"

for SPLIT in train validation test; do
  require_file "$MODEL_DATA_DIR/concepts/$SPLIT.json" "$MODEL_DATA_DIR/clusters/$SPLIT.csv"
  if [[ -f "$OUTCOME_DIR/prepared_${SPLIT}_split/outcomes.csv" ]]; then
    echo "[03] reuse prepared_${SPLIT}_split"
    continue
  fi
  echo "[03] $SPLIT"
  "$PYTHON_BIN" scripts/prepare_injected_outcomes.py \
    --model "$MODEL_ID" \
    --cluster_csv "$MODEL_DATA_DIR/clusters/$SPLIT.csv" \
    --concept_vectors "$STATE_VECTOR_FILE" \
    --concepts_json "$MODEL_DATA_DIR/concepts/$SPLIT.json" \
    --output_dir "$OUTCOME_DIR" \
    --split_name "$SPLIT" \
    --injection_layer "$INJECTION_LAYER" \
    --strength "$INJECTION_STRENGTH" \
    --batch_size "$BATCH_SIZE" \
    --prompt_template "$PROMPT_TEMPLATE" \
    --prompt_preamble "$PROMPT_PREAMBLE" \
    --scale_mode "$SCALE_MODE" \
    --dtype "$DTYPE" \
    --seed "$SEED" \
    ${TRUST_REMOTE_CODE_UNDERSCORE_ARGS[@]+"${TRUST_REMOTE_CODE_UNDERSCORE_ARGS[@]}"}
done

echo "Stage 03 complete: $OUTCOME_DIR"
