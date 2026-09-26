"""The two frozen concept groups of Section 4: C_intro and C_nonintro."""

from __future__ import annotations

import csv
from pathlib import Path


GROUPS = ("validation100", "bottom100")


def read_population(path: Path) -> list[str]:
    """Read all members without pandas' string-to-NA conversions."""
    with path.open(newline="", encoding="utf-8") as handle:
        names = [row["concept"] for row in csv.DictReader(handle)]
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError(f"population must contain nonempty unique concepts: {path}")
    return names


def load_populations(high_path: Path, low_path: Path, expected_count: int) -> tuple[list[str], list[str]]:
    """Require the supplied population identities; never re-rank by outcomes."""
    high, low = read_population(high_path), read_population(low_path)
    if len(high) != expected_count or len(low) != expected_count:
        raise ValueError("population size differs from the explicit expected count")
    if set(high) & set(low):
        raise ValueError("high and low concept populations overlap")
    return high + low, [GROUPS[0]] * len(high) + [GROUPS[1]] * len(low)
