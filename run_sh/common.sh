#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ $# -lt 1 || -z "${1:-}" ]]; then
  echo "Usage: bash $0 path/to/model.env" >&2
  exit 2
fi
ENV_FILE="$1"
if [[ "$ENV_FILE" != /* ]]; then
  ENV_FILE="$REPO_ROOT/$ENV_FILE"
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing model environment: $ENV_FILE" >&2
  exit 2
fi

set -a
source "$ENV_FILE"
set +a

: "${MODEL_ID:?MODEL_ID is required}"
: "${MODEL_SLUG:?MODEL_SLUG is required}"

source "$(dirname "${BASH_SOURCE[0]}")/python_env.sh"

MODEL_DATA_DIR="$DATASET_ROOT/$MODEL_SLUG"
MODEL_RESULTS_DIR="$RESULTS_ROOT/$MODEL_SLUG"
MODEL_LOG_DIR="$LOGS_ROOT/$MODEL_SLUG"
MODEL_TMP_DIR="$TMP_ROOT/$MODEL_SLUG"
STATE_VECTOR_FILE="$MODEL_RESULTS_DIR/validation_diagnostics/state_vectors/state_vectors.pt"

TRUST_REMOTE_CODE_ARGS=()
if [[ "${TRUST_REMOTE_CODE:-false}" == "true" ]]; then
  TRUST_REMOTE_CODE_ARGS=(--trust-remote-code)
fi

TRUST_REMOTE_CODE_UNDERSCORE_ARGS=()
if [[ "${TRUST_REMOTE_CODE:-false}" == "true" ]]; then
  TRUST_REMOTE_CODE_UNDERSCORE_ARGS=(--trust_remote_code)
fi

require_token0_9_prompt() {
  local expected_template="semantic_highinj_posref_gate_balanced_disrupts"
  if [[ "${NUM_CHOICES:-}" != "10" || "${POSITION_INDEX_START:-}" != "0" \
        || "${PROMPT_TEMPLATE:-}" != "$expected_template" ]]; then
    echo "This workflow requires 10 choices labeled TOKEN 0..TOKEN 9 and the original registered prompt template" >&2
    exit 2
  fi
}

# The three label sets that print the same ten-candidate list with the same
# system prompt wording and differ only in the label beside each candidate.
# `digits` is the frozen experiment every other stage is calibrated on.
resolve_label_template() {
  local label_set="$1"
  case "$label_set" in
    digits)
      echo "semantic_highinj_posref_gate_balanced_disrupts"
      ;;
    letters)
      echo "semantic_highinj_posref_gate_balanced_disrupts_letters_a_j"
      ;;
    words)
      echo "semantic_highinj_posref_gate_balanced_disrupts_numwords_one_ten"
      ;;
    *)
      echo "Unknown label set '$label_set'; expected digits, letters or words" >&2
      return 2
      ;;
  esac
}

# Split a "<label_set>[:shuffled]" arm into its template and permutation, so a
# stage can sweep label sets without each one needing its own env file.
parse_label_arm() {
  local arm="$1"
  local label_set="${arm%%:*}"
  local permutation="identity"
  if [[ "$arm" == *:* ]]; then
    permutation="${arm#*:}"
  fi
  if [[ "$permutation" != "identity" && "$permutation" != "shuffled" ]]; then
    echo "Unknown label permutation '$permutation' in arm '$arm'; expected identity or shuffled" >&2
    return 2
  fi
  local template
  template="$(resolve_label_template "$label_set")" || return 2
  echo "$label_set" "$template" "$permutation"
}

# Fail with one clear message naming the env file, instead of letting a Python
# entry point die on an empty argument several minutes into model loading.
require_var() {
  local name
  for name in "$@"; do
    if [[ -z "${!name:-}" ]]; then
      echo "Set $name in $ENV_FILE" >&2
      exit 2
    fi
  done
}

require_file() {
  local path
  for path in "$@"; do
    if [[ ! -f "$path" ]]; then
      echo "Missing required input: $path" >&2
      exit 2
    fi
  done
}

require_frozen_injection() {
  local selection_file="$MODEL_RESULTS_DIR/calibration/selection.json"
  if [[ ! -f "$selection_file" ]]; then
    echo "Missing frozen calibration selection: $selection_file" >&2
    exit 2
  fi

  local selected_layer selected_strength
  read -r selected_layer selected_strength < <(
    "$PYTHON_BIN" - "$selection_file" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(payload["injection_layer"], payload["strength"])
PY
  )

  if [[ -n "${INJECTION_LAYER:-}" && "$INJECTION_LAYER" != "$selected_layer" ]]; then
    echo "INJECTION_LAYER=$INJECTION_LAYER disagrees with $selection_file ($selected_layer)" >&2
    exit 2
  fi
  if [[ -n "${INJECTION_STRENGTH:-}" && "$INJECTION_STRENGTH" != "$selected_strength" ]]; then
    echo "INJECTION_STRENGTH=$INJECTION_STRENGTH disagrees with $selection_file ($selected_strength)" >&2
    exit 2
  fi

  export INJECTION_LAYER="$selected_layer"
  export INJECTION_STRENGTH="$selected_strength"
}
