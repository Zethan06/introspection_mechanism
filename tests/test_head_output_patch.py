"""Pure tests for final-token single-head patch sweep helpers."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from introspection_core.head_output_patch import (
    assign_direction_ranks,
    final_token_head_patch_hook,
    full_sequence_final_token_head_group_patch_hooks,
    HeadEffectAccumulator,
    head_group_spec,
    load_head_selection,
    mode_shift,
    parse_head_group_spec,
    parse_layer_spec,
    select_top_heads,
)
from introspection_core.model import HookedModel


class HeadOutputPatchTests(unittest.TestCase):
    def test_full_sequence_group_patch_changes_only_last_token_selected_heads(self) -> None:
        model = mock.Mock()
        model.attn_hook_name.return_value = "L2.z"
        source = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        hooks = full_sequence_final_token_head_group_patch_hooks(
            model,
            components=[(2, 1)],
            source_z_by_layer={2: source},
        )
        destination = torch.zeros(2, 5, 3, 4)
        patched = hooks[0][1](destination, None)
        torch.testing.assert_close(patched[:, -1, 1], source[:, 1])
        torch.testing.assert_close(patched[:, :-1], destination[:, :-1])
        torch.testing.assert_close(patched[:, -1, 0], destination[:, -1, 0])
        torch.testing.assert_close(patched[:, -1, 2], destination[:, -1, 2])

    def test_parse_layer_spec(self) -> None:
        self.assertEqual(parse_layer_spec("17-19,22,19"), [17, 18, 19, 22])
        with self.assertRaises(ValueError):
            parse_layer_spec("20-17")

    def test_parse_head_group_spec(self) -> None:
        name, components = parse_head_group_spec("top2=L21H19,L22H5")
        self.assertEqual(name, "top2")
        self.assertEqual(components, [(21, 19), (22, 5)])

    def test_patch_replaces_only_one_head_at_single_token(self) -> None:
        destination = torch.zeros(2, 1, 3, 4)
        source = torch.arange(24, dtype=torch.float32).reshape(2, 1, 3, 4)
        patched = final_token_head_patch_hook(source, head=1)(destination, None)
        self.assertTrue(torch.equal(patched[:, :, 1, :], source[:, :, 1, :]))
        self.assertTrue(torch.equal(patched[:, :, 0, :], destination[:, :, 0, :]))
        self.assertTrue(torch.equal(patched[:, :, 2, :], destination[:, :, 2, :]))

    def test_mode_shift_is_patched_minus_base(self) -> None:
        base = torch.zeros(1, 11)
        patched = base.clone()
        patched[0, 0] = 2.0
        self.assertGreater(float(mode_shift(patched, base, temperature=0.1)), 0.0)

    def test_direction_ranks_use_opposite_orders(self) -> None:
        rows = [
            {
                "clean_from_injected_mean_delta_mode": 2.0,
                "injected_from_clean_mean_delta_mode": -1.0,
            },
            {
                "clean_from_injected_mean_delta_mode": 1.0,
                "injected_from_clean_mean_delta_mode": -3.0,
            },
        ]
        assign_direction_ranks(rows)
        self.assertEqual(rows[0]["clean_from_injected_rank_desc"], 1)
        self.assertEqual(rows[1]["injected_from_clean_rank_asc"], 1)

    def test_bidirectional_selection_uses_rank_sum(self) -> None:
        rows = [
            {
                "layer": 20,
                "head": 1,
                "clean_from_injected_rank_desc": 1,
                "injected_from_clean_rank_asc": 4,
                "clean_from_injected_mean_delta_mode": 4.0,
                "injected_from_clean_mean_delta_mode": -1.0,
            },
            {
                "layer": 21,
                "head": 2,
                "clean_from_injected_rank_desc": 2,
                "injected_from_clean_rank_asc": 1,
                "clean_from_injected_mean_delta_mode": 3.0,
                "injected_from_clean_mean_delta_mode": -4.0,
            },
            {
                "layer": 22,
                "head": 3,
                "clean_from_injected_rank_desc": 3,
                "injected_from_clean_rank_asc": 2,
                "clean_from_injected_mean_delta_mode": 2.0,
                "injected_from_clean_mean_delta_mode": -3.0,
            },
        ]
        selected = select_top_heads(rows, top_k=2)
        self.assertEqual(
            [(row["layer"], row["head"]) for row in selected],
            [(21, 2), (22, 3)],
        )
        self.assertEqual(head_group_spec("top2", selected), "top2=L21H2,L22H3")

    def test_load_head_selection_checks_serialized_group(self) -> None:
        payload = {
            "schema_version": 1,
            "model": "model",
            "source_split": "validation",
            "group_name": "top2",
            "gate_group": "top2=L21H2,L22H3",
            "top_k": 2,
            "heads": [{"layer": 21, "head": 2}, {"layer": 22, "head": 3}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selected_heads.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(load_head_selection(path)["group_name"], "top2")
            payload["gate_group"] = "top2=L21H2"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_head_selection(path)

    def test_bidirectional_selection_excludes_wrong_signs(self) -> None:
        rows = [
            {
                "layer": 20,
                "head": 1,
                "clean_from_injected_rank_desc": 1,
                "injected_from_clean_rank_asc": 1,
                "clean_from_injected_mean_delta_mode": 2.0,
                "injected_from_clean_mean_delta_mode": 1.0,
            },
            {
                "layer": 21,
                "head": 2,
                "clean_from_injected_rank_desc": 2,
                "injected_from_clean_rank_asc": 2,
                "clean_from_injected_mean_delta_mode": 1.0,
                "injected_from_clean_mean_delta_mode": -1.0,
            },
        ]
        selected = select_top_heads(rows, top_k=2)
        self.assertEqual([(row["layer"], row["head"]) for row in selected], [(21, 2)])

    def test_clone_legacy_cache_does_not_alias(self) -> None:
        cache = ((torch.ones(1, 2), torch.zeros(1, 2)),)
        cloned = HookedModel.clone_kv_cache(cache)
        cloned[0][0][0, 0] = 9
        self.assertEqual(float(cache[0][0][0, 0]), 1.0)

    def test_reverse_direction_treats_negative_shift_as_favorable(self) -> None:
        accumulator = HeadEffectAccumulator()
        accumulator.update(
            torch.tensor([-2.0, 1.0]),
            torch.tensor([10, 4]),
            direction="injected_from_clean",
            positions=torch.tensor([3, 4]),
        )
        row = accumulator.row()
        self.assertEqual(row["favorable_delta_rate"], 0.5)
        self.assertEqual(row["target_success_rate"], 0.5)
        self.assertEqual(row["exact_number_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
