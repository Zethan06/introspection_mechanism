#!/usr/bin/env python3
"""Parallel launcher for exhaustive layer/strength calibration.

The complete concept panel and strength grid are evaluated at every requested
layer. Layers are split across the given GPUs, and the final setting is selected
from the complete grid.

Example
-------
python scripts/run_sweep_injection_parallel.py \\
    --model "$MODEL_ID" \\
    --concepts_json "$CALIBRATION_CONCEPTS_JSON" \\
    --results_dir "$MODEL_RESULTS_DIR" \\
    --work_dir "$MODEL_TMP_DIR/calibration" \\
    --gpus 0,1,2,3,4,5,6,7

All flags except ``--gpus``, ``--work_dir``, and ``--logs_dir`` are forwarded
verbatim to ``sweep_injection_localization.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from introspection_core.data_parallel import (
    WorkerSpec,
    build_worker_env,
    current_python,
    parse_gpu_groups,
    run_worker_pool,
)
from introspection_core.results import make_run_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_launcher_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Parse launcher controls and return remaining worker arguments."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--gpus",
        default="0,1,2,3,4,5,6,7",
        help=(
            "comma-separated GPU ids, or semicolon-separated multi-GPU "
            "worker groups such as 2,3;5,6"
        ),
    )
    p.add_argument(
        "--logs_dir",
        type=Path,
        help="Canonical log directory for this calibration stage",
    )
    p.add_argument(
        "--work_dir",
        type=Path,
        required=True,
        help="Temporary calibration worker directory outside results",
    )
    launcher_args, worker_argv = p.parse_known_args(argv)
    return launcher_args, worker_argv


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------

def merge_partial_grids(work_dir: Path, num_workers: int) -> list[dict]:
    """Read temporary worker grids and return their combined rows."""
    rows: list[dict] = []
    for wid in range(num_workers):
        partial = work_dir / "workers" / f"worker_{wid}.csv"
        if not partial.exists():
            print(f"[launcher] WARNING: missing partial file {partial}", flush=True)
            continue
        with partial.open(newline="") as fh:
            reader = csv.DictReader(fh)
            rows.extend(list(reader))
    return rows


def write_merged_grid(run_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    out_path = run_dir / "sweep.csv"
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[launcher] Merged grid → {out_path} ({len(rows)} rows)", flush=True)


def find_best(rows: list[dict]) -> dict:
    best_row = sorted(
        rows,
        key=lambda row: (
            -float(row["accuracy"]),
            -float(row["mean_correct_prob"]),
            float(row["strength"]),
            int(row["injection_layer"]),
        ),
    )[0]
    return {
        "injection_layer": int(best_row["injection_layer"]),
        "extraction_layer": int(best_row["extraction_layer"]),
        "strength": float(best_row["strength"]),
        "accuracy": float(best_row["accuracy"]),
        "mean_correct_prob": float(best_row["mean_correct_prob"]),
        "n_correct": int(best_row["n_correct"]),
        "n_trials": int(best_row["n_trials"]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if "--help" in argv or "-h" in argv:
        sweep_script = str(
            Path(__file__).resolve().parent
            / "sweep_injection_localization.py"
        )
        subprocess.run(
            [current_python(), sweep_script, "--help"],
            check=True,
            env=build_worker_env("cpu"),
        )
        print(
            "\nLauncher-only options:\n"
            "  --gpus IDS       GPU ids, optionally grouped as 2,3;5,6\n"
            "  --work_dir PATH  required temporary directory outside results\n"
            "  --logs_dir PATH  calibration log directory",
            flush=True,
        )
        return

    launcher_args, worker_argv = parse_launcher_args(argv)
    gpus = parse_gpu_groups(launcher_args.gpus)
    num_workers = len(gpus)

    # Pull the canonical model results root from worker argv.
    # Workers receive --run_dir pointing to the temporary calibration area.
    tmp_p = argparse.ArgumentParser(add_help=False)
    tmp_p.add_argument("--results_dir", type=Path, required=True)
    tmp_p.add_argument("--run_name")
    tmp_args, _ = tmp_p.parse_known_args(worker_argv)
    if tmp_args.run_name not in (None, "calibration"):
        raise ValueError("The canonical calibration directory is fixed")

    run_dir = make_run_dir(tmp_args.results_dir, "calibration")
    if any(path.is_file() for path in run_dir.rglob("*")):
        raise FileExistsError(
            f"Calibration output already exists: {run_dir}; "
            "use a new model run root"
        )
    print(f"[launcher] Run directory: {run_dir}", flush=True)
    print(f"[launcher] {num_workers} workers on GPUs: {gpus}", flush=True)

    work_dir = launcher_args.work_dir.resolve()
    if work_dir == run_dir.resolve() or run_dir.resolve() in work_dir.parents:
        raise ValueError("--work_dir must be outside the results directory")
    if work_dir.exists() and any(work_dir.rglob("*")):
        raise FileExistsError(
            f"Calibration temporary output already exists: {work_dir}"
        )
    work_dir.mkdir(parents=True, exist_ok=True)

    cleaned: list[str] = []
    skip_next = False
    for token in worker_argv:
        if skip_next:
            skip_next = False
            continue
        if token in ("--results_dir", "--run_name"):
            skip_next = True
            continue
        if token.startswith(("--results_dir=", "--run_name=")):
            continue
        cleaned.append(token)
    worker_base_argv = cleaned + [
        "--run_dir",
        str(work_dir),
        "--num_workers",
        str(num_workers),
    ]

    worker_specs: list[WorkerSpec] = []
    sweep_script = str(
        Path(__file__).resolve().parent / "sweep_injection_localization.py"
    )
    for worker_id, gpu in enumerate(gpus):
        command = [
            current_python(),
            sweep_script,
            *worker_base_argv,
            "--worker_id",
            str(worker_id),
            "--device",
            "cuda",
        ]
        worker_specs.append(
            WorkerSpec(
                name=f"gpu{gpu}_worker{worker_id}",
                command=command,
                env=build_worker_env(gpu),
            )
        )

    log_dir = launcher_args.logs_dir or (
        Path("logs")
        / "auto_workflow"
        / tmp_args.results_dir.name
        / "calibration"
    )
    run_worker_pool(worker_specs, log_dir=log_dir)

    rows = merge_partial_grids(work_dir, num_workers)
    if not rows:
        raise RuntimeError("no partial grid files found")
    rows.sort(
        key=lambda row: (
            int(row["injection_layer"]),
            float(row["strength"]),
        )
    )
    write_merged_grid(run_dir, rows)

    best = find_best(rows)
    (run_dir / "selection.json").write_text(json.dumps(best, indent=2) + "\n")

    worker_metadata = work_dir / "metadata.json"
    if worker_metadata.exists():
        metadata = json.loads(worker_metadata.read_text())
        metadata["calibration_strategy"] = {
            "name": "exhaustive_grid",
            "selection_source": "sweep.csv",
        }
        (run_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
    shutil.rmtree(work_dir)

    print(
        f"\n[launcher] Best: layer={best['injection_layer']}  "
        f"strength={best['strength']}  acc={best['accuracy']:.4f}  "
        f"({best['n_correct']}/{best['n_trials']})",
        flush=True,
    )
    print(f"[launcher] Done. Results in {run_dir}", flush=True)


if __name__ == "__main__":
    main()
