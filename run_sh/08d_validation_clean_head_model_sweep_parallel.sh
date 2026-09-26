#!/usr/bin/env bash
# Divide all layer/head interventions over four independent single-GPU jobs.
set -euo pipefail
source "$(dirname "$0")/common.sh"

GPU_SPEC="${HEAD_PATCH_GPU_IDS:?Set HEAD_PATCH_GPU_IDS to four comma-separated GPU indices}"
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_SPEC"
if (( ${#GPU_ARRAY[@]} != 4 )); then
  echo "Exactly four GPUs are required: $GPU_SPEC" >&2
  exit 2
fi
declare -A SEEN_GPUS=()
for gpu in "${GPU_ARRAY[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ || -n "${SEEN_GPUS[$gpu]:-}" ]]; then
    echo "GPU indices must be unique nonnegative integers: $GPU_SPEC" >&2
    exit 2
  fi
  SEEN_GPUS[$gpu]=1
done

OUTPUT_DIR="${HEAD_PATCH_MODEL_SWEEP_OUTPUT_DIR:-$MODEL_RESULTS_DIR/validation_clean_head_model_sweep_parallel}"
LOG_DIR="${HEAD_PATCH_MODEL_SWEEP_LOG_DIR:-$MODEL_LOG_DIR/validation_clean_head_model_sweep_parallel}"
mkdir -p "$OUTPUT_DIR/shards" "$LOG_DIR"

# Validate every target before launching any background process. A retry must
# never silently overwrite partial work or leave earlier shards running.
if [[ -d "$OUTPUT_DIR/merged" && -n "$(ls -A "$OUTPUT_DIR/merged")" ]]; then
  echo "Refusing to overwrite merged output: $OUTPUT_DIR/merged" >&2
  exit 2
fi
for index in "${!GPU_ARRAY[@]}"; do
  shard="$OUTPUT_DIR/shards/shard_$index"
  if [[ -e "$shard" && ( ! -d "$shard" || -n "$(ls -A "$shard")" ) ]]; then
    echo "Refusing to overwrite existing shard data: $shard" >&2
    exit 2
  fi
done

pids=()
for index in "${!GPU_ARRAY[@]}"; do
  shard="$OUTPUT_DIR/shards/shard_$index"
  echo "Starting head shard $index/4 on GPU ${GPU_ARRAY[$index]}"
    HEAD_PATCH_CANDIDATE_ONLY=true \
  HEAD_PATCH_SHARD_INDEX="$index" \
  HEAD_PATCH_SHARD_COUNT=4 \
  HEAD_PATCH_GPU_IDS="${GPU_ARRAY[$index]}" \
  HEAD_PATCH_OUTPUT_DIR="$shard" \
  HEAD_PATCH_BATCH_SIZE="${HEAD_PATCH_BATCH_SIZE:-8}" \
  HEAD_PATCH_REFERENCE_BATCH_SIZE="${HEAD_PATCH_REFERENCE_BATCH_SIZE:-8}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash run_sh/08_validation_clean_head_patch.sh "$ENV_FILE" \
      >"$LOG_DIR/shard_$index.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if (( failed )); then
  echo "At least one shard failed; inspect $LOG_DIR" >&2
  exit 1
fi

SHARDS=()
for index in "${!GPU_ARRAY[@]}"; do
  SHARDS+=("$OUTPUT_DIR/shards/shard_$index")
done
"$PYTHON_BIN" scripts/merge_validation_clean_head_model_sweep.py \
  --shards "${SHARDS[@]}" \
  --output_dir "$OUTPUT_DIR/merged"
"$PYTHON_BIN" scripts/plot_validation_clean_head_model_sweep.py \
  --effects "$OUTPUT_DIR/merged/condition_effects.csv" \
  --output_stem "$OUTPUT_DIR/merged/all_head_causal_effect"
echo "Whole-model four-GPU sweep complete: $OUTPUT_DIR/merged"
