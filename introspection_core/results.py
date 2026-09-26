"""Result-writing conventions for stable, semantic artifact layouts.

Every run gets its own directory containing:
  - metadata.json  : model, args (verbatim CLI args), seed, date, GPUs used
  - results.csv (or another primary table — name is caller's choice)
  - optional plots/ subdirectory

No wall-clock reads happen inside this module — the caller passes
`date`/`seed` explicitly so a run's metadata is fully determined by its
inputs and reproducible from a recorded command line.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path

import pandas as pd


def make_run_dir(results_root: Path, run_name: str) -> Path:
    """Create and return a semantic run directory.

    Numerical experiment settings belong in metadata, not in path names.
    Reject common encodings such as ``l4``, ``head_12``, ``strength3``, or
    ``top200`` so older orchestration code cannot silently reintroduce
    parameterized paths.
    """
    parameter_token = re.compile(
        r"(?:^|[_-])(?:l|s|h|k|layer|strength|head|top|panel|prompt|"
        r"concepts?|clusters?|train|validation|test)[_-]?[+-]?\d",
        re.IGNORECASE,
    )
    if parameter_token.search(run_name):
        raise ValueError(
            "run_name must describe the artifact semantically and must not "
            f"encode layer/strength/head/k values: {run_name!r}"
        )
    run_dir = results_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _jsonable(value):
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_metadata(
    run_dir: Path,
    *,
    model_name: str,
    args: dict,
    seed: int | None = None,
    date: str | None = None,
    gpus: list[str] | None = None,
    extra: dict | None = None,
    filename: str = "metadata.json",
) -> Path:
    """Write a JSON provenance record in ``run_dir``.

    Args:
        run_dir: directory created by :func:`make_run_dir`.
        model_name: model identifier used for this run.
        args: the run's CLI args, as a plain dict (e.g. ``vars(parsed_args)``).
        seed: RNG seed used, if any (pass explicitly — this module never
            reads or sets global RNG state).
        date: ISO date string for the run, supplied by the caller (this
            module does not call any wall-clock API).
        gpus: which GPU ids the run used (e.g. from
            ``data_parallel.parse_gpu_list``).
        extra: any additional bookkeeping fields to merge in verbatim.
        filename: stable semantic filename, relative to ``run_dir``.

    Returns:
        Path to the written metadata file.
    """
    metadata = {
        "model": model_name,
        "args": {k: _jsonable(v) for k, v in args.items()},
        "seed": seed,
        "date": date,
        "gpus": gpus,
    }
    if extra:
        metadata.update({k: _jsonable(v) for k, v in extra.items()})

    path = run_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, default=str))
    return path


def write_table(run_dir: Path, name: str, rows: list[dict]) -> Path:
    """Write ``rows`` as ``run_dir/<name>.csv`` (creates parent dirs as needed).

    ``rows`` is a list of flat dicts (one per output row) — the convention
    used for per-example / per-component
    result tables (``results.csv``, ``per_head.csv``, ``choice_tokens.csv``, ...).
    """
    path = run_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def plots_dir(run_dir: Path) -> Path:
    """Return (creating if needed) ``run_dir/plots/``."""
    path = run_dir / "plots"
    path.mkdir(parents=True, exist_ok=True)
    return path
