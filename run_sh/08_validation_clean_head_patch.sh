#!/usr/bin/env bash
# Head-wise clean patching (router-head heatmaps): for one shard of all
# (layer, head) cells, replace the head's final-position output in the injected
# validation run with its clean-run value and record the drop in correct-position
# accuracy. Called by 08c (one GPU) and 08d (four GPUs); HEAD_PATCH_SHARD_INDEX
# and HEAD_PATCH_SHARD_COUNT select the shard.
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_frozen_injection

VALIDATION_DIR="$MODEL_RESULTS_DIR/prepared_splits/prepared_validation_split"
OUTPUT_DIR="${HEAD_PATCH_OUTPUT_DIR:-$MODEL_RESULTS_DIR/validation_clean_head_patch}"
require_file "$MODEL_DATA_DIR/clusters/validation.csv"
require_file "$VALIDATION_DIR/concept_vectors.pt"
require_file "$VALIDATION_DIR/metadata.json"

# Use the prompt recorded with the prepared validation bank (Stage 03).
mapfile -t PATCH_PROMPT_FIELDS < <(
  "$PYTHON_BIN" - \
    "$VALIDATION_DIR/metadata.json" \
    "$MODEL_ID" \
    "$INJECTION_LAYER" \
    "$INJECTION_STRENGTH" \
    "$SCALE_MODE" <<'PY'
import json
import math
from pathlib import Path
import sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected_model, expected_layer, expected_strength, expected_scale = sys.argv[2:]
if payload.get("model") != expected_model:
    raise ValueError("prepared validation metadata model does not match MODEL_ID")
if int(payload.get("injection_layer", -1)) != int(expected_layer):
    raise ValueError("prepared validation metadata injection layer does not match")
if not math.isclose(float(payload.get("strength", "nan")), float(expected_strength)):
    raise ValueError("prepared validation metadata strength does not match")
if payload.get("scale_mode") != expected_scale:
    raise ValueError("prepared validation metadata scale mode does not match")

template = str(payload["prompt_template"])
preamble = str(payload["prompt_preamble"])
choice_suffix = str(payload.get("choice_suffix", ""))
position_index_start = int(payload.get("position_index_start") or 0)
if position_index_start != 0:
    raise ValueError("prepared validation position_index_start must be 0")
print(template)
print(preamble)
print(position_index_start)
print(choice_suffix)
PY
)
if (( ${#PATCH_PROMPT_FIELDS[@]} != 4 )); then
  echo "Invalid prompt provenance in $VALIDATION_DIR/metadata.json" >&2
  exit 2
fi
PATCH_PROMPT_TEMPLATE="${PATCH_PROMPT_FIELDS[0]}"
PATCH_PROMPT_PREAMBLE="${PATCH_PROMPT_FIELDS[1]}"
PATCH_POSITION_INDEX_START="${PATCH_PROMPT_FIELDS[2]}"
PATCH_CHOICE_SUFFIX="${PATCH_PROMPT_FIELDS[3]}"

PATCH_SELECTION_ARGS=(
  --sweep_model
  --head_shard_index "${HEAD_PATCH_SHARD_INDEX:-0}"
  --head_shard_count "${HEAD_PATCH_SHARD_COUNT:-1}"
)

PATCH_LIMIT_ARGS=()
if [[ "${HEAD_PATCH_CANDIDATE_ONLY:-false}" == "true" ]]; then
  PATCH_LIMIT_ARGS+=(--candidate_only)
fi
if [[ -n "${HEAD_PATCH_MAX_CONCEPTS:-}" ]]; then
  PATCH_LIMIT_ARGS+=(--max_concepts "$HEAD_PATCH_MAX_CONCEPTS")
fi
if [[ -n "${HEAD_PATCH_MAX_CLUSTERS:-}" ]]; then
  PATCH_LIMIT_ARGS+=(--max_clusters "$HEAD_PATCH_MAX_CLUSTERS")
fi
if [[ -n "${HEAD_PATCH_MAX_TRIALS:-}" ]]; then
  PATCH_LIMIT_ARGS+=(--max_trials "$HEAD_PATCH_MAX_TRIALS")
fi

PATCH_GPU_IDS="${HEAD_PATCH_GPU_IDS:-${VISUALIZATION_GPU:-${STE_GPU:-}}}"
if [[ -z "$PATCH_GPU_IDS" ]]; then
  echo "Set HEAD_PATCH_GPU_IDS, VISUALIZATION_GPU, or STE_GPU in $ENV_FILE" >&2
  exit 2
fi
if [[ "$PATCH_GPU_IDS" == *";"* ]]; then
  echo "Head patch GPU selection must be one comma-separated group: $PATCH_GPU_IDS" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$PATCH_GPU_IDS"

"$PYTHON_BIN" scripts/run_validation_clean_head_patch.py \
  --model "$MODEL_ID" \
  --cluster_csv "$MODEL_DATA_DIR/clusters/validation.csv" \
  --concept_vectors "$VALIDATION_DIR/concept_vectors.pt" \
  --output_dir "$OUTPUT_DIR" \
  "${PATCH_SELECTION_ARGS[@]}" \
  --injection_layer "$INJECTION_LAYER" \
  --strength "$INJECTION_STRENGTH" \
  --batch_size "${HEAD_PATCH_BATCH_SIZE:-$BATCH_SIZE}" \
  --reference_batch_size "${HEAD_PATCH_REFERENCE_BATCH_SIZE:-32}" \
  --bootstrap_samples "${HEAD_PATCH_BOOTSTRAP_SAMPLES:-10000}" \
  --prompt_template "$PATCH_PROMPT_TEMPLATE" \
  --prompt_preamble "$PATCH_PROMPT_PREAMBLE" \
  --position_index_start "$PATCH_POSITION_INDEX_START" \
  --choice_suffix "$PATCH_CHOICE_SUFFIX" \
  --scale_mode "$SCALE_MODE" \
  --dtype "$DTYPE" \
  --seed "$SEED" \
  --overwrite \
  ${PATCH_LIMIT_ARGS[@]+"${PATCH_LIMIT_ARGS[@]}"} \
  ${TRUST_REMOTE_CODE_UNDERSCORE_ARGS[@]+"${TRUST_REMOTE_CODE_UNDERSCORE_ARGS[@]}"}

echo "Validation clean-head patch complete: $OUTPUT_DIR"
