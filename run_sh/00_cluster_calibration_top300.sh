#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_token0_9_prompt

# "all" (or "auto"/empty) leaves --layer_end unset, and the sweep
# then scans every layer of the model.
SWEEP_LAYER_END_ARGS=()
if [[ -n "${SWEEP_LAYER_END:-}" && "$SWEEP_LAYER_END" != "all" \
      && "$SWEEP_LAYER_END" != "auto" ]]; then
  SWEEP_LAYER_END_ARGS=(--layer_end "$SWEEP_LAYER_END")
fi

echo "[00:1/5] Build the complete clean-token prior"
"$PYTHON_BIN" scripts/run_vocab_token_clean_prior.py \
  --model "$MODEL_ID" \
  --results_dir "$MODEL_RESULTS_DIR/token_prior" \
  --work_dir "$MODEL_TMP_DIR/token_prior" \
  --logs_dir "$MODEL_LOG_DIR/token_prior" \
  --gpus "$GPU_IDS" \
  --seed "$SEED" \
  --batch_size "$BATCH_SIZE" \
  --num_choices "$NUM_CHOICES" \
  --position_index_start "$POSITION_INDEX_START" \
  --min_word_len 1 \
  --max_word_len 32 \
  --case_filter all \
  --prompt_preamble "$PROMPT_PREAMBLE" \
  --dtype "$DTYPE" \
  ${TRUST_REMOTE_CODE_UNDERSCORE_ARGS[@]+"${TRUST_REMOTE_CODE_UNDERSCORE_ARGS[@]}"}

echo "[00:2/5] Construct four disjoint 30-cluster banks"
for BANK in calibration train validation test; do
  "$PYTHON_BIN" scripts/search_balanced_token_clusters.py \
    --model "$MODEL_ID" \
    --prior_summary "$MODEL_RESULTS_DIR/token_prior/summary_by_token.csv" \
    --dataset_base "$DATASET_ROOT" \
    --dataset_name "$MODEL_SLUG" \
    --work_root "$TMP_ROOT" \
    --logs_root "$LOGS_ROOT" \
    --gpus "$GPU_IDS" \
    --candidate_bank "$BANK" \
    --candidate_token_count 4000 \
    --candidate_min_len 3 \
    --candidate_max_len 12 \
    --candidate_min_mean_prior 0.0 \
    --candidate_max_mean_prior 0.25 \
    --candidate_max_argmax_rate 0.35 \
    --candidate_max_position_range 999.0 \
    --max_seed_tokens 1000 \
    --neighbor_pool 72 \
    --variants_per_seed 4 \
    --variant_stride 5 \
    --permutations_per_cluster 30 \
    --num_choices "$NUM_CHOICES" \
    --position_index_start "$POSITION_INDEX_START" \
    --dataset_clusters 30 \
    --batch_size "$BATCH_SIZE" \
    --prompt_preamble "$PROMPT_PREAMBLE" \
    --dtype "$DTYPE" \
    --seed "$SEED"
done
"$PYTHON_BIN" scripts/validate_cluster_split.py "$MODEL_DATA_DIR"

echo "[00:3/5] Exhaustively calibrate the layer/strength grid on all 1,000 concepts"
"$PYTHON_BIN" scripts/run_sweep_injection_parallel.py \
  --model "$MODEL_ID" \
  --concepts_json "$CALIBRATION_CONCEPTS_JSON" \
  --max_concepts 1000 \
  --cluster_csv "$MODEL_DATA_DIR/clusters/calibration.csv" \
  --baseline_mode full_english \
  --baseline_min_word_len 1 \
  --baseline_max_word_len 32 \
  --baseline_case_filter all \
  --layer_start "$SWEEP_LAYER_START" \
  ${SWEEP_LAYER_END_ARGS[@]+"${SWEEP_LAYER_END_ARGS[@]}"} \
  --layer_step "$SWEEP_LAYER_STEP" \
  --strengths $SWEEP_STRENGTHS \
  --prompt_template "$PROMPT_TEMPLATE" \
  --preamble "$PROMPT_PREAMBLE" \
  --scale_mode "$SCALE_MODE" \
  --dtype "$DTYPE" \
  --results_dir "$MODEL_RESULTS_DIR" \
  --work_dir "$MODEL_TMP_DIR/calibration" \
  --logs_dir "$MODEL_LOG_DIR/calibration" \
  --gpus "$GPU_IDS" \
  --batch_size "$BATCH_SIZE" \
  --extraction_batch_size "$EXTRACTION_BATCH_SIZE" \
  --seed "$SEED"

require_frozen_injection
echo "[00] Frozen setting: layer=$INJECTION_LAYER strength=$INJECTION_STRENGTH"

echo "[00:4/5] Screen the complete calibration-disjoint vocabulary on all positions to top 3,000"
"$PYTHON_BIN" scripts/search_vocab_injection_words.py \
  --model "$MODEL_ID" \
  --cluster_csv "$MODEL_DATA_DIR/clusters/calibration.csv" \
  --dataset_dir "$MODEL_DATA_DIR" \
  --screening_stage coarse \
  --results_dir "$MODEL_RESULTS_DIR/screening/coarse" \
  --work_dir "$MODEL_TMP_DIR/screening/coarse" \
  --logs_dir "$MODEL_LOG_DIR/screening/coarse" \
  --baseline_mode full_english \
  --calibration_concepts_json "$CALIBRATION_CONCEPTS_JSON" \
  --expected_candidate_count 3000 \
  --gpus "$GPU_IDS" \
  --layer "$INJECTION_LAYER" \
  --strength "$INJECTION_STRENGTH" \
  --batch_size "$BATCH_SIZE" \
  --extraction_batch_size "$EXTRACTION_BATCH_SIZE" \
  --dtype "$DTYPE" \
  --expected_clusters 3 \
  --min_word_len 1 \
  --max_word_len 32 \
  --case_filter all \
  --seed "$SEED" \
  --prompt_templates "$PROMPT_TEMPLATE" \
  ${TRUST_REMOTE_CODE_UNDERSCORE_ARGS[@]+"${TRUST_REMOTE_CODE_UNDERSCORE_ARGS[@]}"}

echo "[00:5/5] Screen all 3,000 shortlisted concepts on all 30 calibration clusters to top 300"
"$PYTHON_BIN" scripts/search_vocab_injection_words.py \
  --model "$MODEL_ID" \
  --cluster_csv "$MODEL_DATA_DIR/clusters/calibration.csv" \
  --dataset_dir "$MODEL_DATA_DIR" \
  --screening_stage fine \
  --results_dir "$MODEL_RESULTS_DIR/screening/fine" \
  --work_dir "$MODEL_TMP_DIR/screening/fine" \
  --logs_dir "$MODEL_LOG_DIR/screening/fine" \
  --baseline_mode full_english \
  --calibration_concepts_json "$CALIBRATION_CONCEPTS_JSON" \
  --candidate_words_csv "$MODEL_DATA_DIR/concepts/shortlist.csv" \
  --candidate_column concept \
  --expected_candidate_count 3000 \
  --gpus "$GPU_IDS" \
  --layer "$INJECTION_LAYER" \
  --strength "$INJECTION_STRENGTH" \
  --batch_size "$BATCH_SIZE" \
  --extraction_batch_size "$EXTRACTION_BATCH_SIZE" \
  --dtype "$DTYPE" \
  --expected_clusters 30 \
  --min_word_len 1 \
  --max_word_len 32 \
  --case_filter all \
  --seed "$SEED" \
  --prompt_templates "$PROMPT_TEMPLATE" \
  ${TRUST_REMOTE_CODE_UNDERSCORE_ARGS[@]+"${TRUST_REMOTE_CODE_UNDERSCORE_ARGS[@]}"}

echo "Stage 00 complete: fresh top-300 concepts and fixed 100/100/100 splits are ready"
