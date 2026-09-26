#!/usr/bin/env python3
"""Draw the low-accuracy concept group (C_nonintro) from the fine screening.

Candidates are restricted to concepts/shortlist.csv (3,000 concepts by default).
The fine screening ranks every shortlisted concept by calibration-cluster
localization accuracy. The introspective group is the top 300 split 100/100/100;
the comparison group is a seeded random 100 of the bottom 300, so it is fixed
before any head-level measurement and never re-ranked on test outcomes.

Writes, next to the selected concepts:
  concepts/bottom300.csv / bottom300.json   the 300 lowest-ranked concepts
  concepts/bottom100_seed42.csv             random.Random(seed).sample(bottom300, 100)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from introspection_core.concept_groups import read_population
from scripts.search_vocab_injection_words import _sort_concept_metric_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--screening_metrics",
        type=Path,
        required=True,
        help="results/<model>/screening/fine/metrics.csv",
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help="data/dataset/<model>; reads and writes its concepts/ folder",
    )
    parser.add_argument("--expected_candidate_count", type=int, default=3000)
    parser.add_argument("--pool_size", type=int, default=300)
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.sample_size <= args.pool_size:
        parser.error("need 0 < --sample_size <= --pool_size")

    concepts_dir = args.dataset_dir / "concepts"
    candidates = read_population(concepts_dir / "shortlist.csv")
    if len(candidates) != args.expected_candidate_count:
        raise ValueError(
            f"shortlist must contain {args.expected_candidate_count} concepts; "
            f"got {len(candidates)}"
        )
    metrics = pd.read_csv(args.screening_metrics, keep_default_na=False)
    metrics = metrics.loc[metrics["concept"].isin(candidates)]
    if metrics["concept"].duplicated().any():
        raise ValueError("screening contains duplicate shortlisted concepts")
    missing = set(candidates) - set(metrics["concept"])
    if missing:
        raise ValueError(f"screening is missing {len(missing)} shortlisted concepts")
    metrics = _sort_concept_metric_frame(metrics)
    selected = json.loads((concepts_dir / "selected.json").read_text())
    if len(metrics) < len(selected["concept_vector_words"]) + args.pool_size:
        raise ValueError("screening is too small for disjoint top and bottom pools")
    bottom = metrics.iloc[-args.pool_size:]
    names = bottom["concept"].astype(str).tolist()
    if set(names) & set(selected["concept_vector_words"]):
        raise ValueError("bottom pool overlaps the selected concepts")

    bottom.to_csv(concepts_dir / f"bottom{args.pool_size}.csv", index=False)
    (concepts_dir / f"bottom{args.pool_size}.json").write_text(
        json.dumps(
            {
                "concept_vector_words": names,
                "baseline_words": selected["baseline_words"],
                "baseline_mode": selected["baseline_mode"],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    sample = random.Random(args.seed).sample(names, args.sample_size)
    sample_path = concepts_dir / f"bottom{args.sample_size}_seed{args.seed}.csv"
    pd.DataFrame({"concept": sample}).to_csv(sample_path, index=False)
    print(sample_path)


if __name__ == "__main__":
    main()
