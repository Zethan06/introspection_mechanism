#!/usr/bin/env bash
# Fig 3 under other label sets: apply the frozen ordered-digit Top-32 gate
# masks, without re-optimizing them, to the letters / number-word labels and
# to the shuffled-label arms of Table 1.
#
# MODES selects which Fig 3 runs to queue (space separated):
#   gate        Fig 3a/b: gate patch under the native router      -> test_<dir>
#   router_all  Fig 3c/d and Table 2: complete donor i x router j
#               grid, router heads patched from the natural injected
#               run at j (as in the digit Fig 3d grid)            -> test_<dir>_env_output_patch
#   router_pin  Fig 3c/d: router heads pinned to the clean run     -> test_<dir>_clean_router_pin
# Router heads (as in the configs): Qwen L24 H29,31; LLaMA L17 H24; Gemma L29 H1,11.
#
# One worker per GPU. Workers on any node share one job list and claim jobs
# with an atomic mkdir under the shared results tree, longest model first; a
# worker skips models its node does not hold under MODEL_ROOT and jobs whose
# summary.json already exists, so relaunching resumes.
#
# Usage: run_sh/04e_gate_label_transfer.sh <gpu_id>
#   MODES="gate router_all router_pin"  MODELS="gemma llama qwen"
#   ARMS="letters words digits:shuffled ..."  DIRECTIONS="off on"
#   MODEL_ROOT (optional): directory holding local checkpoints; unset loads
#   every model from the Hugging Face Hub.
# A failed job leaves <claim>/failed; delete its claim directory to retry it.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:?usage: $0 <gpu_id>}"
source "$(dirname "${BASH_SOURCE[0]}")/python_env.sh"
MODES="${MODES:-gate router_all router_pin}"
MODELS="${MODELS:-gemma llama qwen}"
ARMS="${ARMS:-letters words digits:shuffled letters:shuffled words:shuffled}"
DIRECTIONS="${DIRECTIONS:-off on}"
SEED="${SEED:-42}"
CLAIM_ROOT="results/label_transfer_claims"
LOG_ROOT="logs/gate_label_transfer"
mkdir -p "$CLAIM_ROOT" "$LOG_ROOT"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" TOKENIZERS_PARALLELISM=false

label_template() {
  case "$1" in
    digits)  echo "semantic_highinj_posref_gate_balanced_disrupts" ;;
    letters) echo "semantic_highinj_posref_gate_balanced_disrupts_letters_a_j" ;;
    words)   echo "semantic_highinj_posref_gate_balanced_disrupts_numwords_one_ten" ;;
    *) echo "unknown label set: $1" >&2; return 2 ;;
  esac
}

# slug, model path, batch size, router layer, router heads (comma separated).
# The all-position router grid needs a batch of at least ten trials.
model_config() {
  case "$1" in
    qwen)  echo "qwen3-4b-instruct-2507 Qwen/Qwen3-4B-Instruct-2507 32 24 29,31" ;;
    llama) echo "llama3.1-8b-instruct meta-llama/Llama-3.1-8B-Instruct 32 17 24" ;;
    gemma) echo "gemma3-12b-it google/gemma-3-12b-it 10 29 1,11" ;;
    *) echo "unknown model: $1" >&2; return 2 ;;
  esac
}

# Output suffix of each mode, matching the digit runs in ste_topk_sweep/top32.
mode_suffix() {
  case "$1" in
    gate)       echo "" ;;
    router_all) echo "_env_output_patch" ;;
    router_pin) echo "_clean_router_pin" ;;
    *) echo "unknown mode: $1" >&2; return 2 ;;
  esac
}

# Resolve everything up front so a typo fails before any model load.
for ARM in $ARMS; do
  label_template "${ARM%%:*}" >/dev/null
  if [[ "$ARM" == *:* && "${ARM#*:}" != "shuffled" ]]; then
    echo "unknown label permutation in arm '$ARM'; expected <set> or <set>:shuffled" >&2
    exit 2
  fi
done
for NAME in $MODELS; do model_config "$NAME" >/dev/null; done
for MODE in $MODES; do mode_suffix "$MODE" >/dev/null; done

for NAME in $MODELS; do
  read -r SLUG MODEL BATCH ROUTER_LAYER ROUTER_HEADS <<<"$(model_config "$NAME")"
  MODEL="${MODEL_ROOT:+$MODEL_ROOT/}$MODEL"
  if [[ -n "${MODEL_ROOT:-}" && ! -d "$MODEL" ]]; then
    echo "[gpu$GPU@$(hostname)] skip $SLUG: $MODEL not on this node"
    continue
  fi
  TOP32="results/$SLUG/ste_topk_sweep/top32"
  for MODE in $MODES; do
    SUFFIX="$(mode_suffix "$MODE")"
    case "$MODE" in
      gate)       MODE_ARGS=() ;;
      router_all) MODE_ARGS=(--forced_router_layer "$ROUTER_LAYER"
                             --forced_router_heads ${ROUTER_HEADS//,/ }
                             --router_position_mode all
                             --router_intervention injected_output_patch
                             --router_donor_state injected) ;;
      router_pin) MODE_ARGS=(--forced_router_layer "$ROUTER_LAYER"
                             --forced_router_heads ${ROUTER_HEADS//,/ }
                             --clean_router_patch) ;;
    esac
    for ARM in $ARMS; do
      LABEL_SET="${ARM%%:*}"
      PERMUTATION="identity"
      [[ "$ARM" == *:* ]] && PERMUTATION="${ARM#*:}"
      TEMPLATE="$(label_template "$LABEL_SET")"
      ARM_NAME="${LABEL_SET}_${PERMUTATION}"
      for DIRECTION in $DIRECTIONS; do
        JOB="${SLUG}__${ARM_NAME}__${DIRECTION}"
        [[ "$MODE" != "gate" ]] && JOB="${JOB}__${MODE}"
        OUT="$TOP32/label_transfer/$ARM_NAME/test_${DIRECTION}${SUFFIX}"
        [[ -f "$OUT/summary.json" ]] && continue
        mkdir "$CLAIM_ROOT/$JOB" 2>/dev/null || continue
        echo "$(hostname) gpu$GPU $(date -Is)" > "$CLAIM_ROOT/$JOB/owner"
        LOG="$LOG_ROOT/$JOB.log"
        echo "[gpu$GPU@$(hostname)] $(date -Is) start $JOB -> $OUT"
        if "$PYTHON_BIN" scripts/test_ste_topk_head_gate.py \
            --model "$MODEL" \
            --head_mask "$TOP32/train_$DIRECTION/head_mask.pt" \
            --test_cluster_csv "data/dataset/$SLUG/clusters/test.csv" \
            --test_concept_vectors "results/$SLUG/prepared_splits/prepared_test_split/concept_vectors.pt" \
            --output_dir "$OUT" \
            --prompt_template "$TEMPLATE" \
            --label_permutation "$PERMUTATION" \
            ${MODE_ARGS[@]+"${MODE_ARGS[@]}"} \
            --batch_size "$BATCH" \
            --dtype bfloat16 \
            --seed "$SEED" \
            --overwrite >"$LOG" 2>&1; then
          date -Is > "$CLAIM_ROOT/$JOB/done"
          echo "[gpu$GPU@$(hostname)] $(date -Is) done  $JOB"
        else
          date -Is > "$CLAIM_ROOT/$JOB/failed"
          echo "[gpu$GPU@$(hostname)] $(date -Is) FAILED $JOB (see $LOG)"
        fi
      done
    done
  done
done
echo "[gpu$GPU@$(hostname)] $(date -Is) no more claimable jobs"
