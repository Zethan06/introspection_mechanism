#!/usr/bin/env bash
# Cross-model figures and tables, read from the per-model stage outputs.
#
#   paper/manuscript_panels/fig2_layerwise.*        Figure 2 (Stages 01, 02b)
#   paper/manuscript_panels/fig3_interventions_mean6.*
#                                                   Figure 3, mean over the six
#                                                   label settings (Stages 04e, 04f)
#   paper/manuscript_panels/fig3_intervals.json     Figure 3 Wilson intervals
#   paper/ste_topk_gate_effect/                     Top-k selection vs random-k
#                                                   (Stages 04b, 04d)
#   paper/task_performance.*                        Table 1 (Stages 04f, 09)
#   paper/uncertainty/                              Table 2 and redirection-table
#                                                   intervals (Stages 04f, 06)
#
# Usage: bash run_sh/11_paper_figures_and_tables.sh
#   RESULTS_ROOT (default results), PYTHON_BIN (default project .venv/bin/python)
set -euo pipefail
cd "$(dirname "$0")/.."

source "$(dirname "${BASH_SOURCE[0]}")/python_env.sh"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
OUT="$RESULTS_ROOT/paper"
PANELS="$OUT/manuscript_panels"
MODELS=(qwen3-4b-instruct-2507 llama3.1-8b-instruct gemma3-12b-it)
# Panel labels; their slugs name the files the manuscript includes.
LABELS=("Qwen3-4B" "Llama3.1-8B" "Gemma-3-12B-IT")
mkdir -p "$OUT"

echo "[11:1/5] Figure 2"
"$PYTHON_BIN" scripts/plot_manuscript_figures.py \
  --results-root "$RESULTS_ROOT" \
  --output-dir "$PANELS"

echo "[11:2/5] Figure 3 over the six label settings"
"$PYTHON_BIN" scripts/summarize_gate_label_transfer.py \
  --results_root "$RESULTS_ROOT" \
  --output_dir "$OUT/gate_label_transfer"
"$PYTHON_BIN" scripts/summarize_router_label_transfer.py \
  --results_root "$RESULTS_ROOT" \
  --output_dir "$OUT/router_label_transfer"
"$PYTHON_BIN" scripts/plot_fig3_label_average.py \
  --gate_summary "$OUT/gate_label_transfer/gate_label_transfer.json" \
  --router_summary "$OUT/router_label_transfer/router_label_transfer.json" \
  --output_dir "$PANELS"
"$PYTHON_BIN" scripts/fig3_label_average_intervals.py \
  --provenance "$PANELS/provenance.json" \
  --output_json "$PANELS/fig3_intervals.json"

echo "[11:3/5] Top-k selection against the random-k control"
TOPK_ARGS=()
for INDEX in "${!MODELS[@]}"; do
  SWEEP="$RESULTS_ROOT/${MODELS[$INDEX]}/ste_topk_sweep"
  TOPK_ARGS+=(--input "${LABELS[$INDEX]}=$SWEEP/validation_transition_summary.csv")
  TOPK_ARGS+=(--random_input "${LABELS[$INDEX]}=$SWEEP/random_topk_control/random_topk_transition_summary.csv")
done
"$PYTHON_BIN" scripts/plot_ste_topk_gate_effect.py \
  "${TOPK_ARGS[@]}" \
  --evaluation_split validation \
  --split_directions \
  --skip_overview \
  --hide_titles \
  --output_dir "$OUT/ste_topk_gate_effect"

echo "[11:4/5] Table 1"
"$PYTHON_BIN" scripts/summarize_task_performance.py \
  --results_root "$RESULTS_ROOT" \
  --output_dir "$OUT"

echo "[11:5/5] Table 2 and redirection-table intervals"
"$PYTHON_BIN" scripts/summarize_error_estimates.py \
  --results_root "$RESULTS_ROOT" \
  --output_dir "$OUT/uncertainty"

echo "Stage 11 complete: $OUT"
