"""Pure tests for train-only cluster-to-position assignment."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_test_latent_position_clustering.py"
SPEC = importlib.util.spec_from_file_location("latent_position_clustering", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LatentPositionClusteringTests(unittest.TestCase):
    def test_hungarian_mapping_recovers_permuted_cluster_ids(self) -> None:
        positions = np.tile(np.arange(3), 4)
        permutation = np.asarray([2, 0, 1])
        cluster_ids = permutation[positions]

        mapping, contingency = MODULE._fit_position_mapping(
            cluster_ids, positions, position_count=3
        )

        np.testing.assert_array_equal(mapping[cluster_ids], positions)
        self.assertEqual(int(contingency.sum()), len(positions))


if __name__ == "__main__":
    unittest.main()
