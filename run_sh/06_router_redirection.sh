#!/usr/bin/env bash
# Stage 06: cross-position attention redirection in the router heads.
#
# Inject a concept at position i and set the router heads' post-softmax
# attention at the final prompt position to one-hot on the successor token t_j
# of candidate j, for every i and j, on 100 test concepts x 30 test prompts.
# The redirection target is structural (the " TOKEN" that follows candidate j,
# or the newline after the last candidate), so the same run is repeated under
# the six label arms of Table 1.
#
#   ROUTER_LABEL_ARMS (default: all six) "<digits|letters|words>[:shuffled]"
#
# The digits identity arm writes router_sweeps/forced_i<i>/; every other arm
# writes router_label_sweeps/<label_set>_<permutation>/forced_i<i>/.
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_token0_9_prompt
require_frozen_injection
require_var ROUTER_LAYER ROUTER_HEADS STE_GPU
export CUDA_VISIBLE_DEVICES="$STE_GPU"

ROUTER_LABEL_ARMS="${ROUTER_LABEL_ARMS:-digits letters words digits:shuffled letters:shuffled words:shuffled}"
# Resolve every arm before the first model load.
declare -a ARM_SPECS=()
for ARM in $ROUTER_LABEL_ARMS; do
  ARM_SPECS+=("$(parse_label_arm "$ARM")")
done

CLUSTER_FILE="$MODEL_DATA_DIR/clusters/test.csv"
CONCEPTS_JSON="$MODEL_DATA_DIR/concepts/test.json"
CONCEPT_VECTORS="$MODEL_RESULTS_DIR/prepared_splits/prepared_test_split/concept_vectors.pt"
require_file "$CLUSTER_FILE" "$CONCEPTS_JSON" "$CONCEPT_VECTORS"

for ARM_SPEC in "${ARM_SPECS[@]}"; do
  read -r ARM_LABEL_SET ARM_TEMPLATE ARM_PERMUTATION <<<"$ARM_SPEC"
  if [[ "$ARM_LABEL_SET" == "digits" && "$ARM_PERMUTATION" == "identity" ]]; then
    ROUTER_DIR="$MODEL_RESULTS_DIR/router_sweeps"
  else
    ROUTER_DIR="$MODEL_RESULTS_DIR/router_label_sweeps/${ARM_LABEL_SET}_${ARM_PERMUTATION}"
  fi
  mkdir -p "$ROUTER_DIR"
  echo "[06] arm=${ARM_LABEL_SET}_${ARM_PERMUTATION} heads=L$ROUTER_LAYER H$ROUTER_HEADS -> $ROUTER_DIR"
  for POSITION in $(seq 0 9); do
    "$PYTHON_BIN" scripts/run_attention_routing_sweep.py \
      --model "$MODEL_ID" \
      --cluster_file "$CLUSTER_FILE" \
      --concepts_json "$CONCEPTS_JSON" \
      --concept_vectors_file "$CONCEPT_VECTORS" \
      --injection_layer "$INJECTION_LAYER" \
      --strength "$INJECTION_STRENGTH" \
      --attention_layer "$ROUTER_LAYER" \
      --heads $ROUTER_HEADS \
      --prompt_template "$ARM_TEMPLATE" \
      --label_permutation "$ARM_PERMUTATION" \
      --prompt_preamble "$PROMPT_PREAMBLE" \
      --scale_mode "$SCALE_MODE" \
      --batch_size "$BATCH_SIZE" \
      --dtype "$DTYPE" \
      --results_dir "$ROUTER_DIR" \
      --run_name "forced_i${POSITION}" \
      --seed "$SEED" \
      --injection_position "$POSITION" \
      ${TRUST_REMOTE_CODE_UNDERSCORE_ARGS[@]+"${TRUST_REMOTE_CODE_UNDERSCORE_ARGS[@]}"}
  done
done

echo "Stage 06 complete for arms: $ROUTER_LABEL_ARMS"
