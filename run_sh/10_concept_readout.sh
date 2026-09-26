#!/usr/bin/env bash
# Section 4: how gate heads read different concept injections (QK and OV).
#
# C_intro is the 100 validation concepts; C_nonintro is a seeded random 100 of
# the 300 lowest-ranked concepts within the Stage 00 shortlist of 3,000,
# ranked by fine-screening accuracy. Both groups are
# injected at all ten positions of the 30 test clusters, and every analysis
# uses the 32 gate heads of the Stage 04b Top-32 masks (ste_topk_sweep/top32).
#
# Outputs, all under $MODEL_RESULTS_DIR/concept_readout/:
#   source/                      frozen population read by every capture below
#   head_selections/             heads.csv of the gate-on and gate-off masks
#   qk_successor/                key-change SVD at the ten successor positions
#     first_mode_summary/          sigma_1, |v_1^T q|, R_1, U_i1, target response
#     context_summary/             query norms and per-head context modes
#   qk_kl/summary/               attention KL after removing the query or key term
#   qk_score_ablation/summary/   correct-position rate after the same removal
#   ov/output_svd/               SVD of M = W_O dV^T: sizes, alignments, energies
#   ov/causal_terms|top5|ksweep/ subtracting dV / da terms or leading modes
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_token0_9_prompt
require_frozen_injection
require_var STE_GPU
export CUDA_VISIBLE_DEVICES="${STE_GPU%%,*}"

CR="$MODEL_RESULTS_DIR/concept_readout"
GATE_DIR="$MODEL_RESULTS_DIR/ste_topk_sweep/top32"
READOUT_BATCH_SIZE="${READOUT_BATCH_SIZE:-$BATCH_SIZE}"
CLUSTERS=($(seq 0 29))
require_file "$MODEL_RESULTS_DIR/screening/fine/metrics.csv" "$STATE_VECTOR_FILE" \
  "$MODEL_DATA_DIR/concepts/shortlist.csv" \
  "$GATE_DIR/train_on/head_mask.pt" "$GATE_DIR/train_off/head_mask.pt"
mkdir -p "$CR"

echo "[10:1/8] Sample C_nonintro from the bottom of the 3,000-concept shortlist"
"$PYTHON_BIN" scripts/sample_low_accuracy_concepts.py \
  --screening_metrics "$MODEL_RESULTS_DIR/screening/fine/metrics.csv" \
  --dataset_dir "$MODEL_DATA_DIR" \
  --expected_candidate_count 3000 \
  --seed "$SEED"

echo "[10:2/8] Concept vectors for both groups under one frozen baseline"
if [[ ! -f "$CR/vector_bank/concept_vectors.pt" ]]; then
  "$PYTHON_BIN" scripts/prepare_ste_group_vectors.py \
    --model "$MODEL_ID" \
    --dataset_dir "$MODEL_DATA_DIR" \
    --reference_vectors "$STATE_VECTOR_FILE" \
    --layer "$INJECTION_LAYER" \
    --batch_size "$EXTRACTION_BATCH_SIZE" \
    --results_dir "$CR/vector_bank"
fi

echo "[10:3/8] Frozen source directory and gate-head tables"
"$PYTHON_BIN" scripts/prepare_concept_group_source.py \
  --model_results "$MODEL_RESULTS_DIR" \
  --dataset_dir "$MODEL_DATA_DIR" \
  --concept_vectors "$CR/vector_bank/concept_vectors.pt" \
  --output_dir "$CR/source"
ON_HEADS="$CR/head_selections/metrics/heads.csv"
OFF_HEADS="$CR/head_selections/gate_off_metrics/heads.csv"
"$PYTHON_BIN" scripts/export_gate_heads.py \
  --head_mask "$MODEL_SLUG=$GATE_DIR/train_on/head_mask.pt" --output "$ON_HEADS"
"$PYTHON_BIN" scripts/export_gate_heads.py \
  --head_mask "$MODEL_SLUG=$GATE_DIR/train_off/head_mask.pt" --output "$OFF_HEADS"

echo "[10:4/8] QK: capture key changes, queries and scores at the successor positions"
"$PYTHON_BIN" scripts/capture_ste_context_modes.py \
  --source "$CR/source" \
  --training-config "$GATE_DIR/train_on/configuration.json" \
  --query-state injected \
  --key-positions successor \
  --save-score-rows \
  --save-first-mode-rows \
  --all-clusters \
  --expected-clusters 30 \
  --resume \
  --batch_size "$READOUT_BATCH_SIZE" \
  --results_dir "$CR/qk_successor"
QK_CAPTURES=()
for CLUSTER in "${CLUSTERS[@]}"; do
  QK_CAPTURES+=("$CR/qk_successor/injected/cluster_$(printf %02d "$CLUSTER")")
done
"$PYTHON_BIN" scripts/analyze_successor_first_mode.py \
  --capture-root "$CR/qk_successor" \
  --on-heads "$ON_HEADS" \
  --model-id "$MODEL_SLUG" \
  --expected-clusters 30 \
  --results_dir "$CR/qk_successor/first_mode_summary"
"$PYTHON_BIN" scripts/analyze_ste_context_modes.py \
  --captures "${QK_CAPTURES[@]}" \
  --average-clusters \
  --expected-clusters 30 \
  --on-heads "$ON_HEADS" \
  --off-heads "$OFF_HEADS" \
  --model-id "$MODEL_SLUG" \
  --results_dir "$CR/qk_successor/context_summary"

echo "[10:5/8] QK: attention KL after removing the query or the key term"
"$PYTHON_BIN" scripts/capture_qk_attention_kl.py \
  --source "$CR/qk_successor/injected/cluster_00" \
  --prompt-reference-root "$CR/qk_successor/injected" \
  --clusters "${CLUSTERS[@]}" \
  --batch_size "$READOUT_BATCH_SIZE" \
  --results_dir "$CR/qk_kl/captures/$MODEL_SLUG"
"$PYTHON_BIN" scripts/summarize_qk_attention_kl.py \
  --captures "$CR/qk_kl/captures/$MODEL_SLUG" \
  --expected-clusters "${CLUSTERS[@]}" \
  --results_dir "$CR/qk_kl/summary"

echo "[10:6/8] QK: correct-position rate after the same score ablations"
"$PYTHON_BIN" scripts/run_ste_qk_score_ablation.py \
  --source "$CR/qk_successor/injected/cluster_00" \
  --all-clusters \
  --resume \
  --batch_size "$READOUT_BATCH_SIZE" \
  --results_dir "$CR/qk_score_ablation/$MODEL_SLUG"
"$PYTHON_BIN" scripts/summarize_ste_qk_score_ablation.py \
  --results_dir "$CR/qk_score_ablation" \
  --models "$MODEL_SLUG" \
  --expected-clusters 30

OV="$CR/ov"
echo "[10:7/8] OV: SVD of M = W_O dV^T over the full context"
"$PYTHON_BIN" scripts/capture_ov_output_svd.py \
  --source "$CR/source" \
  --training-config "$CR/source/training_configuration.json" \
  --head-selection "$ON_HEADS" \
  --model-slug "$MODEL_SLUG" \
  --svd-device cpu \
  --clusters "${CLUSTERS[@]}" \
  --batch_size "$READOUT_BATCH_SIZE" \
  --results_dir "$OV/output_svd/shards/$MODEL_SLUG"
"$PYTHON_BIN" - "$OV/output_svd" "$MODEL_SLUG" <<'PY'
import json, sys
from pathlib import Path
root, slug = Path(sys.argv[1]), sys.argv[2]
root.mkdir(parents=True, exist_ok=True)
(root / "jobs.json").write_text(json.dumps(
    [{"model": slug, "output": str(root / "shards" / slug), "clusters": list(range(30))}], indent=2))
PY
"$PYTHON_BIN" scripts/summarize_ov_output_svd.py \
  --root "$OV/output_svd"

echo "[10:8/8] OV: causal subtraction of the dV / da terms and of the leading modes"
OV_ABLATION=(
  scripts/run_ov_causal_ablation.py
  --source "$CR/source"
  --heads "$ON_HEADS"
  --model-slug "$MODEL_SLUG"
  --training-config "$CR/source/training_configuration.json"
  --clusters "${CLUSTERS[@]}"
  --batch-size "$READOUT_BATCH_SIZE"
)
"$PYTHON_BIN" "${OV_ABLATION[@]}" --results_dir "$OV/causal_terms"
"$PYTHON_BIN" "${OV_ABLATION[@]}" --top-k 5 --results_dir "$OV/causal_top5"
"$PYTHON_BIN" "${OV_ABLATION[@]}" --top-ks 1 2 10 --results_dir "$OV/causal_ksweep"

echo "Stage 10 complete: $CR"
