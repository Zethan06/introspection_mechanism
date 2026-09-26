#!/usr/bin/env python3
"""Attach 95% intervals to Table 2 and the router-redirection table.

Every interval is built from trial counts: a Wilson score interval that treats
trials as independent, and, since both tables pool injection-position pairs, a
bootstrap that resamples the ten source positions with all of their pairs.
Neither resamples concepts or prompts. Figure 3 intervals come from
fig3_label_average_intervals.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from introspection_core.analysis_statistics import (
    cluster_bootstrap_rate, wilson_interval, write_csv,
)


MODELS = ("qwen3-4b-instruct-2507", "llama3.1-8b-instruct", "gemma3-12b-it")
POSITIONS = 10
# Redirection arms in table column order; Stage 06 writes the digits identity
# arm to router_sweeps/ and every other arm to router_label_sweeps/<arm>/.
REDIRECTION_ARMS = [
    ("digits_identity", "router_sweeps"),
    ("letters_identity", "router_label_sweeps/letters_identity"),
    ("words_identity", "router_label_sweeps/words_identity"),
    ("digits_shuffled", "router_label_sweeps/digits_shuffled"),
    ("letters_shuffled", "router_label_sweeps/letters_shuffled"),
    ("words_shuffled", "router_label_sweeps/words_shuffled"),
]
REDIRECTION_OUTCOMES = ["to_j", "to_i", "other", "none"]
CROSS_OUTCOMES = ["output_i", "output_j", "other"]


def read_rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def count(rate: float | str, total: int) -> int:
    """Recover an integer count from a saved rate, refusing inexact ones."""

    value = float(rate) * total
    if abs(value - round(value)) > 1e-6 * max(total, 1):
        raise ValueError(f"rate {rate} is not a count out of {total}")
    return int(round(value))


def cross_position_counts(root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Counts [gate source i, outcome] and trials [i] over pairs with j != i."""

    path = root / "ste_topk_sweep/top32/test_on_env_output_patch/position_results.csv"
    successes = np.zeros((POSITIONS, len(CROSS_OUTCOMES)), dtype=np.int64)
    trials = np.zeros(POSITIONS, dtype=np.int64)
    pairs = set()
    for row in read_rows(path):
        if row["intervention"] != "clean_with_injected_top32" or \
                row["position_relation"] != "mismatch":
            continue
        i, j, total = int(row["donor_position"]), int(row["router_position"]), int(row["n_trials"])
        if i == j:
            raise ValueError("mismatch row with equal positions")
        pairs.add((i, j))
        trials[i] += total
        successes[i] += [count(row["overall_donor_accuracy"], total),
                         count(row["overall_router_accuracy"], total),
                         count(row["other_number_rate"], total)]
    if len(pairs) != POSITIONS * (POSITIONS - 1):
        raise ValueError(f"expected 90 cross-position pairs, found {len(pairs)}")
    return successes, trials


def redirection_counts(root: Path, arm_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Counts [injection i, outcome] and trials [i] over readout pairs j != i."""

    successes = np.zeros((POSITIONS, len(REDIRECTION_OUTCOMES)), dtype=np.int64)
    trials = np.zeros(POSITIONS, dtype=np.int64)
    pairs = set()
    for path in sorted((root / arm_dir).glob("forced_i*/sweep_grid.csv")):
        for row in read_rows(path):
            if row["condition"] != "onehot_injected" or row["is_diagonal"] == "1":
                continue
            i, j = int(row["injection_position"]), int(row["attention_position"])
            total = int(row["n_trials"])
            predictions = json.loads(row["prediction_counts"])
            if sum(predictions.values()) != total:
                raise ValueError(f"{path}: prediction counts do not sum to n_trials")
            # Prediction keys are the arm's labels, so the run's own counts
            # identify which label belongs to positions j and i.
            to_j, to_i = int(row["n_follow_attention"]), int(row["n_correct"])
            none = predictions["none"]
            pairs.add((i, j))
            trials[i] += total
            successes[i] += [to_j, to_i, total - to_j - to_i - none, none]
    if len(pairs) != POSITIONS * (POSITIONS - 1):
        raise ValueError(f"{root / arm_dir}: expected 90 pairs, found {len(pairs)}")
    return successes, trials


def percentile_interval(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.percentile(samples, 2.5, axis=0), np.percentile(samples, 97.5, axis=0)


def fmt_interval(estimate: float, low: float, high: float, digits: int) -> str:
    return f"{estimate:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", type=Path, default=Path("results"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    cross_rows, redirection_rows = [], []
    for model in MODELS:
        root = args.results_root / model

        successes, trials = cross_position_counts(root)
        boot = cluster_bootstrap_rate(
            successes, np.repeat(trials[:, None], len(CROSS_OUTCOMES), axis=1),
            rng=rng, bootstrap_samples=args.bootstrap_samples)
        boot_low, boot_high = percentile_interval(boot)
        for k, outcome in enumerate(CROSS_OUTCOMES):
            hits, total = int(successes[:, k].sum()), int(trials.sum())
            low, high = wilson_interval(hits, total)
            cross_rows.append(dict(model=model, outcome=outcome, successes=hits, trials=total,
                                   rate=hits / total, wilson_low=low, wilson_high=high,
                                   position_bootstrap_low=boot_low[k],
                                   position_bootstrap_high=boot_high[k]))

        arm_counts = [redirection_counts(root, arm_dir) for _, arm_dir in REDIRECTION_ARMS]
        successes = np.stack([s for s, _ in arm_counts], axis=1)  # [i, arm, outcome]
        trials = np.stack([t for _, t in arm_counts], axis=1)     # [i, arm]
        trials = np.repeat(trials[:, :, None], len(REDIRECTION_OUTCOMES), axis=2)
        boot = cluster_bootstrap_rate(successes, trials, rng=rng,
                                      bootstrap_samples=args.bootstrap_samples)
        arm_low, arm_high = percentile_interval(boot)
        mean_low, mean_high = percentile_interval(boot.mean(axis=1))
        rates = successes.sum(axis=0) / trials.sum(axis=0)
        for k, outcome in enumerate(REDIRECTION_OUTCOMES):
            for a, (arm, _) in enumerate(REDIRECTION_ARMS):
                hits, total = int(successes[:, a, k].sum()), int(trials[:, a, k].sum())
                low, high = wilson_interval(hits, total)
                redirection_rows.append(dict(
                    model=model, outcome=outcome, arm=arm, successes=hits, trials=total,
                    rate=rates[a, k], wilson_low=low, wilson_high=high,
                    position_bootstrap_low=arm_low[a, k],
                    position_bootstrap_high=arm_high[a, k]))
            # Mean of six independent binomial arms: normal interval on the
            # average, next to the bootstrap that resamples positions jointly.
            standard_error = np.sqrt(np.sum(rates[:, k] * (1 - rates[:, k])
                                            / trials.sum(axis=0)[:, k])) / len(REDIRECTION_ARMS)
            mean = rates[:, k].mean()
            redirection_rows.append(dict(
                model=model, outcome=outcome, arm="mean", successes="", trials="",
                rate=mean, wilson_low=mean - 1.96 * standard_error,
                wilson_high=mean + 1.96 * standard_error,
                position_bootstrap_low=mean_low[k], position_bootstrap_high=mean_high[k]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "cross_position_intervals.csv", cross_rows)
    write_csv(args.output_dir / "redirection_intervals.csv", redirection_rows)
    (args.output_dir / "tables.tex").write_text(
        render_tables(cross_rows, redirection_rows))
    (args.output_dir / "metadata.json").write_text(json.dumps(dict(
        results_root=str(args.results_root), bootstrap_samples=args.bootstrap_samples,
        seed=args.seed, bootstrap_unit="source injection position (10), all pairs retained",
        wilson="95% score interval, trials treated as independent",
    ), indent=2) + "\n")
    print(args.output_dir)


def render_tables(cross_rows: list[dict], redirection_rows: list[dict]) -> str:
    """LaTeX bodies for the two appendix tables (percent / proportion)."""

    models = list(MODELS)
    lines = ["% Table 2"]
    names = dict(zip(models, ["Qwen3-4B-IT", "LLaMA-3.1-8B-IT", "Gemma-3-12B-IT"]))
    for model in models:
        rows = [next(r for r in cross_rows if r["model"] == model and r["outcome"] == o)
                for o in CROSS_OUTCOMES]
        wilson = [fmt_interval(100 * r["rate"], 100 * r["wilson_low"],
                               100 * r["wilson_high"], 1) for r in rows]
        boot = [f"[{100 * r['position_bootstrap_low']:.1f}, "
                f"{100 * r['position_bootstrap_high']:.1f}]" for r in rows]
        lines.append(f"\\multirow{{2}}{{*}}{{{names[model]}}} & Wilson & "
                     + " & ".join(wilson) + " \\\\")
        lines.append(" & Position bootstrap & " + " & ".join(boot) + " \\\\")
    lines.append("% Redirection table")
    labels = dict(to_j="$\\to j$", to_i="$\\to i$", other="Other", none="\\texttt{none}")
    arms = [arm for arm, _ in REDIRECTION_ARMS] + ["mean"]
    for model in models:
        lines.append(f"\\multirow{{4}}{{*}}{{{names[model]}}}")
        for outcome in REDIRECTION_OUTCOMES:
            cells = []
            for arm in arms:
                row = next(r for r in redirection_rows if r["model"] == model
                           and r["outcome"] == outcome and r["arm"] == arm)
                cells.append(f"[{row['position_bootstrap_low']:.3f}, "
                             f"{row['position_bootstrap_high']:.3f}]")
            lines.append(f"  & {labels[outcome]} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
