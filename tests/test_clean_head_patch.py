"""CPU tests for validation clean-head patch metrics and controls."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import torch

from introspection_core.clean_head_patch import (
    CleanHeadPatchAccumulator,
    build_patch_conditions,
    concept_bootstrap_interval,
    format_accuracy_change_sentence,
    layer_matched_control_groups,
    parse_head_components,
    resolve_candidate_layout,
)
from scripts.run_validation_clean_head_patch import build_sweep_specs
from scripts.merge_validation_clean_head_model_sweep import main as merge_model_sweep_main
from introspection_core.model import HookedModel


class _Example:
    positions = (1, 2, 3)
    clean_target_label = "none"
    candidate_token_ids = {"1": 11, "2": 12, "3": 13, "none": 99}


class CleanHeadPatchTests(unittest.TestCase):
    def test_candidate_only_unembedding_matches_full_projection(self) -> None:
        torch.manual_seed(42)
        linear = torch.nn.Linear(5, 17)
        model = object.__new__(HookedModel)
        model.bridge = SimpleNamespace(unembed=SimpleNamespace(original_component=linear))
        model.cfg = SimpleNamespace(output_logits_soft_cap=0.0)
        normalized = torch.randn(4, 1, 5)
        candidate_ids = [2, 7, 11]
        expected = linear(normalized)[:, 0, candidate_ids].float()
        actual = model._candidate_unembed_logits(normalized, candidate_ids)
        torch.testing.assert_close(actual, expected)

    def test_merge_model_sweep_requires_all_heads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shards = []
            for index in range(2):
                shard = root / f"shard_{index}"
                shard.mkdir()
                shards.append(shard)
                summary = {
                    "experiment": "validation_clean_head_model_sweep_shard",
                    "head_shard_count": 2,
                    "head_shard_index": index,
                    "component_count": 2,
                    "n_trials": 6,
                    "concept_count": 1,
                    "cluster_count": 1,
                    "position_labels": [0],
                    "clean_exact_number_accuracy": 0.0,
                    "injected_exact_number_accuracy": 0.5,
                    "patch_site": "final token",
                }
                (shard / "summary.json").write_text(json.dumps(summary))
                metadata = {
                    "model": "synthetic", "n_layers": 2, "n_heads": 2,
                    "candidate_labels": ["0", "none"],
                    "args": {"model": "synthetic", "head_shard_index": index},
                }
                (shard / "metadata.json").write_text(json.dumps(metadata))
                with (shard / "condition_effects.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=[
                        "heads", "layer", "head", "n_trials", "accuracy_drop",
                        "accuracy_drop_pp", "rank_by_accuracy_drop",
                    ])
                    writer.writeheader()
                    for layer in range(2):
                        head = index
                        writer.writerow({
                            "heads": f"L{layer}H{head}", "layer": layer,
                            "head": head, "n_trials": 6,
                            "accuracy_drop": 0.1 * (layer + head),
                            "accuracy_drop_pp": 10 * (layer + head),
                            "rank_by_accuracy_drop": 1,
                        })
                with (shard / "per_concept_effects.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["heads", "concept_index", "n_trials"])
                    writer.writeheader()
                    for layer in range(2):
                        writer.writerow({
                            "heads": f"L{layer}H{index}", "concept_index": 0, "n_trials": 6,
                        })
            output = root / "merged"
            merge_model_sweep_main([
                "--shards", *map(str, shards), "--output_dir", str(output),
            ])
            merged = json.loads((output / "summary.json").read_text())
            self.assertEqual(merged["component_count"], 4)
            self.assertEqual(merged["top_head"], "L1H1")
            self.assertEqual(len(json.loads((output / "metadata.json").read_text())["shards"]), 2)
            changed = json.loads((shards[1] / "metadata.json").read_text())
            changed["args"]["model"] = "different"
            (shards[1] / "metadata.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "experiment arguments"):
                merge_model_sweep_main([
                    "--shards", *map(str, shards), "--output_dir", str(root / "mixed"),
                ])
            changed["args"]["model"] = "synthetic"
            (shards[1] / "metadata.json").write_text(json.dumps(changed))
            concept_csv = shards[1] / "per_concept_effects.csv"
            original_concepts = concept_csv.read_text()
            concept_csv.write_text(original_concepts.replace("L1H1,0,6", "L0H1,0,6"))
            with self.assertRaisesRegex(ValueError, "per-concept coverage"):
                merge_model_sweep_main([
                    "--shards", *map(str, shards), "--output_dir", str(root / "duplicate"),
                ])
            concept_csv.write_text(original_concepts)
            with self.assertRaisesRegex(ValueError, "shard coverage"):
                merge_model_sweep_main([
                    "--shards", str(shards[0]),
                    "--output_dir", str(root / "incomplete"),
                ])

    def test_whole_model_sweep_shards_flattened_layer_head_grid(self) -> None:
        specs = build_sweep_specs(
            n_layers=3,
            n_heads=4,
            sweep_layer=None,
            sweep_model=True,
            shard_index=1,
            shard_count=3,
        )
        self.assertEqual(
            [spec.components[0] for spec in specs],
            [(0, 1), (1, 0), (1, 3), (2, 2)],
        )
        self.assertTrue(all(spec.kind == "model_head" for spec in specs))

    def test_candidate_layout_supports_one_based_prompts(self) -> None:
        labels, token_ids, clean_index = resolve_candidate_layout([_Example(), _Example()])
        self.assertEqual(labels, [1, 2, 3])
        self.assertEqual(token_ids, [11, 12, 13, 99])
        self.assertEqual(clean_index, 3)

    def test_parse_and_layer_matched_controls(self) -> None:
        target = parse_head_components("L24H29,L24H31,L20H2")
        controls = layer_matched_control_groups(
            target, n_heads=32, count=12, seed=7
        )
        self.assertEqual(len(controls), 12)
        self.assertEqual(len({tuple(group) for group in controls}), 12)
        for group in controls:
            self.assertEqual([layer for layer, _head in group].count(24), 2)
            self.assertEqual([layer for layer, _head in group].count(20), 1)
            self.assertTrue(set(group).isdisjoint(target))

    def test_accumulator_reports_exact_accuracy_drop(self) -> None:
        clean = torch.tensor([[0.0, 0.0, 5.0], [0.0, 0.0, 5.0]])
        injected = torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
        patched = torch.tensor([[0.0, 0.0, 5.0], [0.0, 5.0, 0.0]])
        accumulator = CleanHeadPatchAccumulator(concept_count=2)
        accumulator.update(
            clean_logits=clean,
            injected_logits=injected,
            patched_logits=patched,
            target_indices=torch.tensor([0, 1]),
            clean_index=2,
            concept_indices=torch.tensor([0, 1]),
        )
        row = accumulator.row()
        self.assertEqual(row["injected_exact_number_accuracy"], 1.0)
        self.assertEqual(row["patched_exact_number_accuracy"], 0.5)
        self.assertEqual(row["accuracy_drop_pp"], 50.0)
        self.assertEqual(row["patched_clean_prediction_rate"], 0.5)
        torch.testing.assert_close(
            accumulator.concept_accuracy_drops(), torch.tensor([1.0, 0.0], dtype=torch.float64)
        )

    def test_concept_bootstrap_is_deterministic(self) -> None:
        effects = torch.tensor([0.0, 0.5, 1.0])
        self.assertEqual(
            concept_bootstrap_interval(effects, samples=100, seed=4),
            concept_bootstrap_interval(effects, samples=100, seed=4),
        )

    def test_conditions_are_typed_and_layer_matched(self) -> None:
        conditions = build_patch_conditions(
            [(24, 29), (24, 31)], n_heads=32, control_count=3, seed=7
        )
        self.assertEqual(conditions[0].name, "target_group")
        self.assertEqual(conditions[0].components, ((24, 29), (24, 31)))
        self.assertEqual(
            [condition.kind for condition in conditions].count("matched_control"),
            3,
        )

    def test_accuracy_change_sentence_respects_effect_sign(self) -> None:
        reduction = format_accuracy_change_sentence(
            heads="L24H29",
            accuracy_drop_pp=4.0,
            ci_low=0.01,
            ci_high=0.07,
        )
        increase = format_accuracy_change_sentence(
            heads="L24H29",
            accuracy_drop_pp=-3.0,
            ci_low=-0.05,
            ci_high=-0.01,
        )
        unchanged = format_accuracy_change_sentence(
            heads="L24H29",
            accuracy_drop_pp=0.0,
            ci_low=-0.01,
            ci_high=0.01,
        )
        self.assertIn("reduces", reduction)
        self.assertIn("increases", increase)
        self.assertIn("does not measurably change", unchanged)


if __name__ == "__main__":
    unittest.main()
