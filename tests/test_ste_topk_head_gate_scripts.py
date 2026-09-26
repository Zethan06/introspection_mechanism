"""Pure orchestration tests for the formal STE train/test entry points."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import torch

from introspection_core.cluster_split import file_sha256
from scripts.test_ste_topk_head_gate import (
    OutputStats,
    _assert_test_split,
    _condition_labels,
    _validate_model_configuration,
    parse_args as parse_test_args,
    sharded_batches,
)
from scripts.train_ste_topk_head_gate import _distributed_epoch_layout, parse_args


def _required_train_args(direction: str) -> list[str]:
    args = [
        "--direction",
        direction,
        "--model",
        "model",
        "--train_cluster_csv",
        "train.csv",
        "--train_concept_vectors",
        "vectors.pt",
        "--output_dir",
        "out",
        "--layers",
        "3-4",
        "--injection_layer",
        "1",
        "--strength",
        "3",
    ]
    if direction == "on":
        args.extend(["--train_outcomes_csv", "outcomes.csv"])
    return args


class FormalTrainingCliTests(unittest.TestCase):
    def test_top_k_defaults_to_32_and_accepts_sweep_value(self) -> None:
        self.assertEqual(parse_args(_required_train_args("off")).top_k, 32)
        parsed = parse_args([*_required_train_args("off"), "--top_k", "4"])
        self.assertEqual(parsed.top_k, 4)
        with self.assertRaises(SystemExit):
            parse_args([*_required_train_args("off"), "--top_k", "0"])

    def test_distributed_epoch_layout_keeps_safe_partial_batch(self) -> None:
        self.assertEqual(
            _distributed_epoch_layout(
                30_000, per_rank_batch_size=32, world_size=1
            ),
            (30_000, 938),
        )
        self.assertEqual(
            _distributed_epoch_layout(101, per_rank_batch_size=16, world_size=4),
            (100, 2),
        )

    def test_gate_on_requires_forced_router(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(_required_train_args("on"))
        args = parse_args(
            [
                *_required_train_args("on"),
                "--forced_router_layer",
                "5",
                "--forced_router_heads",
                "1",
                "2",
            ]
        )
        self.assertEqual(args.direction, "on")

    def test_gate_on_requires_locked_outcomes(self) -> None:
        args = _required_train_args("on")
        index = args.index("--train_outcomes_csv")
        del args[index : index + 2]
        args.extend(
            [
                "--forced_router_layer",
                "5",
                "--forced_router_heads",
                "1",
            ]
        )
        with self.assertRaises(SystemExit):
            parse_args(args)

    def test_gate_off_rejects_forced_router(self) -> None:
        args = parse_args(_required_train_args("off"))
        self.assertEqual(args.direction, "off")
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    *_required_train_args("off"),
                    "--forced_router_layer",
                    "5",
                    "--forced_router_heads",
                    "1",
                ]
            )
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    *_required_train_args("off"),
                    "--train_outcomes_csv",
                    "outcomes.csv",
                ]
            )


class TrainTestSeparationTests(unittest.TestCase):
    def test_directional_condition_labels_cover_both_topk_patches(self) -> None:
        self.assertEqual(
            _condition_labels(recipient="clean", top_k=32),
            ("clean_baseline", "off", "clean_with_injected_top32", "on"),
        )
        self.assertEqual(
            _condition_labels(recipient="injected", top_k=32),
            ("injected_baseline", "on", "injected_with_clean_top32", "off"),
        )

    def test_all_router_positions_require_forced_router(self) -> None:
        required = [
            "--model",
            "model",
            "--head_mask",
            "mask.pt",
            "--test_cluster_csv",
            "test.csv",
            "--test_concept_vectors",
            "vectors.pt",
            "--output_dir",
            "out",
            "--router_position_mode",
            "all",
        ]
        with self.assertRaises(SystemExit):
            parse_test_args(required)
        parsed = parse_test_args(
            [
                *required,
                "--forced_router_layer",
                "5",
                "--forced_router_heads",
                "1",
            ]
        )
        self.assertEqual(parsed.router_position_mode, "all")

    def test_injected_router_patch_requires_all_position_grid(self) -> None:
        required = [
            "--model",
            "model",
            "--head_mask",
            "mask.pt",
            "--test_cluster_csv",
            "test.csv",
            "--test_concept_vectors",
            "vectors.pt",
            "--output_dir",
            "out",
            "--forced_router_layer",
            "5",
            "--forced_router_heads",
            "1",
            "--router_intervention",
            "injected_output_patch",
        ]
        with self.assertRaises(SystemExit):
            parse_test_args(required)
        parsed = parse_test_args(
            [*required, "--router_position_mode", "all"]
        )
        self.assertEqual(parsed.router_intervention, "injected_output_patch")

    def test_test_split_rejects_reused_train_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_cluster = root / "train_cluster.csv"
            train_vectors = root / "train_vectors.pt"
            test_cluster = root / "test_cluster.csv"
            test_vectors = root / "test_vectors.pt"
            for path, value in (
                (train_cluster, "train cluster"),
                (train_vectors, "train vectors"),
                (test_cluster, "test cluster"),
                (test_vectors, "test vectors"),
            ):
                path.write_text(value, encoding="utf-8")
            checkpoint = {
                "input_sha256": {
                    "train_cluster_csv": file_sha256(train_cluster),
                    "train_concept_vectors": file_sha256(train_vectors),
                }
            }
            with self.assertRaisesRegex(ValueError, "concept_vectors"):
                _assert_test_split(
                    checkpoint,
                    cluster_csv=test_cluster,
                    concept_vectors_file=train_vectors,
                )

    def test_batch_sharding_is_complete_and_disjoint(self) -> None:
        items = list(range(11))
        shards = [
            sharded_batches(items, batch_size=3, rank=rank, world_size=2)
            for rank in range(2)
        ]
        flattened = [item for shard in shards for batch in shard for item in batch]
        self.assertEqual(sorted(flattened), items)
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_evaluation_validates_checkpoint_model_shape_without_forced_router(
        self,
    ) -> None:
        args = argparse.Namespace(
            forced_router_layer=None,
            forced_router_heads=None,
        )
        checkpoint = {
            "layers": [3, 4],
            "injection_layer": 1,
            "n_heads": 19,
            "selection_direction": "off",
        }

        class _Config:
            n_layers = 6
            n_heads = 20

        class _Model:
            cfg = _Config()

        with self.assertRaisesRegex(ValueError, "head count"):
            _validate_model_configuration(
                args, checkpoint=checkpoint, model=_Model()
            )

    def test_injected_output_patch_accepts_env_heads_independent_of_training_router(
        self,
    ) -> None:
        args = argparse.Namespace(
            forced_router_layer=5,
            forced_router_heads=[1, 2, 3],
            router_intervention="injected_output_patch",
        )
        checkpoint = {
            "layers": [3, 4],
            "injection_layer": 1,
            "n_heads": 4,
            "selection_direction": "on",
            "training_router_layer": 5,
            "training_router_heads": [3],
        }

        class _Config:
            n_layers = 6
            n_heads = 4

        class _Model:
            cfg = _Config()

        _validate_model_configuration(args, checkpoint=checkpoint, model=_Model())


class OutputStatsTests(unittest.TestCase):
    def test_reports_number_exact_and_none_rates(self) -> None:
        logits = torch.full((3, 11), -5.0)
        logits[0, 2] = 5.0
        logits[1, 7] = 5.0
        logits[2, 10] = 5.0
        stats = OutputStats()
        stats.update(
            logits,
            positions=torch.tensor([2, 3, 4]),
            temperature=1.0,
        )
        row = stats.row(
            condition="patched",
            gate_state="off",
            router="native",
        )
        self.assertEqual(row["gate_state"], "off")
        self.assertAlmostEqual(float(row["number_rate"]), 2 / 3)
        self.assertAlmostEqual(float(row["exact_target_accuracy"]), 1 / 3)
        self.assertNotIn("exact_given_number", row)


if __name__ == "__main__":
    unittest.main()
