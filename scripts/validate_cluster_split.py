#!/usr/bin/env python3
"""Validate four frozen cluster banks and write their shared manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from introspection_core.cluster_split import write_cluster_split_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Directory containing all four candidate, cluster, and summary files",
    )
    args = parser.parse_args()
    path = write_cluster_split_manifest(args.dataset_dir)
    print(f"validated cluster split: {path}")


if __name__ == "__main__":
    main()
