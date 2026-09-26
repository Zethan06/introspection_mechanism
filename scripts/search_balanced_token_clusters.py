#!/usr/bin/env python3
"""CLI entry point for balanced_cluster_search.

Runner (default): spawns one worker process per GPU.
Worker: scores clusters for its assigned shard and writes CSV files.

Example — deterministic calibration candidate bank:
    python scripts/search_balanced_token_clusters.py \\
        --model "$MODEL_ID" \\
        --prior_summary "$MODEL_RESULTS_DIR/token_prior/summary_by_token.csv" \\
        --dataset_name "$MODEL_SLUG" \\
        --candidate_bank calibration \\
        --candidate_token_count 4000 \\
        --max_seed_tokens 1000 \\
        --seed 42 \\
        --gpus "$GPU_IDS"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from introspection_core.balanced_cluster_search import main

if __name__ == "__main__":
    main()
