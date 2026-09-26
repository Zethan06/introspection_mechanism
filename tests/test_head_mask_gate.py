"""Pure tests for the formal STE Top-k head-selection core."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from introspection_core.attention_inputs import AttentionExample
from introspection_core.injected_trials import InjectedTrial
from introspection_core.head_mask_gate import (
    FORMAL_TRAINING_PROTOCOL,
    HEAD_MASK_SCHEMA_VERSION,
    HeadGateDataset,
    TransitionStats,
    _ordered_concepts_from_payload,
    directional_head_gate_bce,
    formal_training_pair_mask,
    hard_topk_mask,
    incremental_one_hot_router_hook,
    labeled_candidate_layout,
    load_head_mask_checkpoint,
    masked_final_token_head_hooks,
    native_number_inducing_mask,
    ste_topk_mask,
)


class _FakeModel:
    @staticmethod
    def attn_hook_name(layer: int, kind: str) -> str:
        return f"blocks.{layer}.attn.hook_{kind}"


class HeadMaskTests(unittest.TestCase):
    def test_reads_ordered_concepts_from_held_out_rows(self) -> None:
        self.assertEqual(
            _ordered_concepts_from_payload(
                {"rows": [{"word": "alpha"}, {"word": "beta"}]}
            ),
            ("alpha", "beta"),
        )

    def test_rejects_duplicate_held_out_row_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            _ordered_concepts_from_payload(
                {"rows": [{"word": "alpha"}, {"word": "alpha"}]}
            )

    def test_hard_topk_has_exact_cardinality(self) -> None:
        scores = torch.tensor([[0.1, 0.9, 0.3], [0.8, 0.2, 0.7]])
        mask = hard_topk_mask(scores, top_k=3)
        self.assertEqual(int(mask.sum()), 3)
        self.assertEqual(
            mask.tolist(), [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]
        )

    def test_ste_is_hard_forward_and_soft_backward(self) -> None:
        scores = torch.tensor(
            [[-0.4, 0.2], [0.1, 0.8]],
            dtype=torch.float32,
            requires_grad=True,
        )
        mask, soft, hard = ste_topk_mask(
            scores, top_k=2, temperature=0.7
        )
        torch.testing.assert_close(mask, hard)
        self.assertFalse(torch.equal(soft, hard))
        coefficients = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        (mask * coefficients).sum().backward()
        self.assertIsNotNone(scores.grad)
        self.assertTrue(scores.grad.ne(0).all())

    def test_reports_paired_none_to_number_conversion(self) -> None:
        baseline = torch.full((3, 11), -5.0)
        patched = torch.full((3, 11), -5.0)
        baseline[0, 10] = 5.0
        baseline[1, 10] = 5.0
        baseline[2, 2] = 5.0
        patched[0, 1] = 5.0
        patched[1, 10] = 5.0
        patched[2, 2] = 5.0
        stats = TransitionStats()
        stats.update(baseline, patched, direction="on")
        row = stats.row(direction="on", router="native")
        self.assertEqual(row["transition"], "none_to_number")
        self.assertEqual(row["source_prediction_trials"], 2)
        self.assertEqual(row["converted_trials"], 1)
        self.assertAlmostEqual(float(row["conversion_rate"]), 0.5)
        self.assertAlmostEqual(float(row["target_accuracy_before"]), 1 / 3)
        self.assertAlmostEqual(float(row["target_accuracy_after"]), 2 / 3)
        self.assertAlmostEqual(float(row["target_accuracy_delta"]), 1 / 3)

    def test_reports_paired_number_to_none_conversion(self) -> None:
        baseline = torch.full((3, 11), -5.0)
        patched = torch.full((3, 11), -5.0)
        baseline[0, 1] = 5.0
        baseline[1, 2] = 5.0
        baseline[2, 10] = 5.0
        patched[0, 10] = 5.0
        patched[1, 2] = 5.0
        patched[2, 10] = 5.0
        stats = TransitionStats()
        stats.update(baseline, patched, direction="off")
        row = stats.row(direction="off", router="native")
        self.assertEqual(row["transition"], "number_to_none")
        self.assertEqual(row["source_prediction_trials"], 2)
        self.assertEqual(row["converted_trials"], 1)
        self.assertAlmostEqual(float(row["conversion_rate"]), 0.5)
        self.assertAlmostEqual(float(row["target_accuracy_before"]), 1 / 3)
        self.assertAlmostEqual(float(row["target_accuracy_after"]), 2 / 3)
        self.assertAlmostEqual(float(row["target_accuracy_delta"]), 1 / 3)
    def test_patch_uses_dynamic_recipient_and_paired_donor(self) -> None:
        donor = {4: torch.full((1, 1, 3, 2), 10.0)}
        mask = torch.tensor([[1.0, 0.0, 1.0]], requires_grad=True)
        name, hook = masked_final_token_head_hooks(
            _FakeModel(), layers=[4], donor_z_by_layer=donor, mask=mask
        )[0]
        self.assertEqual(name, "blocks.4.attn.hook_z")
        recipient = torch.arange(6, dtype=torch.float32).reshape(1, 1, 3, 2)
        patched = hook(recipient, None)
        torch.testing.assert_close(patched[:, :, 0], donor[4][:, :, 0])
        torch.testing.assert_close(patched[:, :, 1], recipient[:, :, 1])
        torch.testing.assert_close(patched[:, :, 2], donor[4][:, :, 2])
        patched.sum().backward()
        self.assertIsNotNone(mask.grad)


class HeadGateObjectiveTests(unittest.TestCase):
    @staticmethod
    def _logits(*, number: float, none: float) -> torch.Tensor:
        logits = torch.full((2, 11), -4.0)
        logits[:, 3] = number
        logits[:, 10] = none
        return logits

    def test_bce_only_objective_has_opposite_on_off_targets(self) -> None:
        number_logits = self._logits(number=5.0, none=-3.0)
        none_logits = self._logits(number=-3.0, none=5.0)
        good_on = directional_head_gate_bce(
            number_logits, target_is_number=True, temperature=1.0
        )
        bad_on = directional_head_gate_bce(
            none_logits, target_is_number=True, temperature=1.0
        )
        good_off = directional_head_gate_bce(
            none_logits, target_is_number=False, temperature=1.0
        )
        bad_off = directional_head_gate_bce(
            number_logits, target_is_number=False, temperature=1.0
        )
        self.assertLess(float(good_on), float(bad_on))
        self.assertLess(float(good_off), float(bad_off))

    def test_native_filter_accepts_any_injected_number(self) -> None:
        clean = torch.full((4, 11), -5.0)
        injected = torch.full((4, 11), -5.0)
        clean[0, 10] = 5.0
        injected[0, 3] = 5.0
        clean[1, 10] = 5.0
        injected[1, 8] = 5.0
        clean[2, 10] = 5.0
        injected[2, 10] = 5.0
        clean[3, 4] = 5.0
        injected[3, 7] = 5.0
        self.assertEqual(
            native_number_inducing_mask(clean, injected).tolist(),
            [True, True, False, False],
        )

    def test_formal_filters_gate_on_exact_but_gate_off_any_number(self) -> None:
        clean = torch.full((3, 11), -5.0)
        injected = torch.full((3, 11), -5.0)
        clean[:, 10] = 5.0
        injected[0, 2] = 7.0
        injected[1, 7] = 7.0
        injected[2, 10] = 7.0
        positions = torch.tensor([2, 3, 4])
        self.assertEqual(
            formal_training_pair_mask(
                clean, injected, positions=positions, direction="on"
            ).tolist(),
            [True, False, False],
        )
        self.assertEqual(
            formal_training_pair_mask(
                clean, injected, positions=positions, direction="off"
            ).tolist(),
            [True, True, False],
        )

    def test_forced_router_is_one_hot_at_the_target_key(self) -> None:
        name, hook = incremental_one_hot_router_hook(
            _FakeModel(),
            layer=24,
            heads=[1, 3],
            key_positions=torch.tensor([4, 2]),
        )
        self.assertEqual(name, "blocks.24.attn.hook_pattern")
        pattern = torch.full((2, 4, 1, 6), 1 / 6)
        patched = hook(pattern, hook=None)
        self.assertEqual(float(patched[0, 1, 0, 4]), 1.0)
        self.assertEqual(float(patched[1, 3, 0, 2]), 1.0)
        self.assertEqual(float(patched[0, 1, 0].sum()), 1.0)
        self.assertTrue(torch.equal(patched[:, 0], pattern[:, 0]))

    def test_forced_router_rejects_the_current_query_token_as_a_key(self) -> None:
        _name, hook = incremental_one_hot_router_hook(
            _FakeModel(),
            layer=24,
            heads=[1],
            key_positions=torch.tensor([5]),
        )
        with self.assertRaisesRegex(ValueError, "cached prefix"):
            hook(torch.full((1, 4, 1, 6), 1 / 6), hook=None)


class HeadMaskCheckpointTests(unittest.TestCase):
    @staticmethod
    def _payload(direction: str, router: str) -> dict:
        scores = torch.arange(40, dtype=torch.float32).reshape(2, 20)
        hard_mask = torch.zeros(2, 20, dtype=torch.bool)
        hard_mask.flatten()[8:] = True
        selected_components = [
            (layer, head)
            for layer_offset, layer in enumerate((3, 4))
            for head in range(20)
            if bool(hard_mask[layer_offset, head])
        ]
        return {
            "schema_version": HEAD_MASK_SCHEMA_VERSION,
            "artifact_type": "ste_topk_head_gate",
            "formal_training_protocol": FORMAL_TRAINING_PROTOCOL,
            "loss": "binary_cross_entropy_with_logits",
            "source_split": "train",
            "selection_direction": direction,
            "training_router_mode": router,
            "training_router_layer": 5 if direction == "on" else None,
            "training_router_heads": [1] if direction == "on" else None,
            "train_pair_filter": (
                "native_clean_none_injected_exact_target"
                if direction == "on"
                else "native_clean_none_injected_any_number"
            ),
            "lambda_position": 0.0,
            "layers": [3, 4],
            "n_heads": 20,
            "top_k": 32,
            "scores": scores,
            "hard_mask": hard_mask,
            "selected_components": selected_components,
            "model": "model",
            "injection_layer": 1,
            "strength": 3.0,
            "scale_mode": "relative_hidden_norm",
            "prompt_template": "template",
            "prompt_preamble": "system",
            "gate_temperature": 0.1,
            "input_sha256": {
                "train_cluster_csv": "a" * 64,
                "train_concept_vectors": "b" * 64,
                **(
                    {"train_outcomes_csv": "c" * 64}
                    if direction == "on"
                    else {}
                ),
            },
            "train_candidate_pool": (
                "locked_native_injected_exact_targets"
                if direction == "on"
                else "full_coordinate_grid"
            ),
            "full_coordinate_count": 30_000,
            "candidate_coordinate_count": 30_000,
            "final_epoch_train_count": 10_000,
        }

    def test_loader_accepts_only_formal_router_direction_pairing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "head_mask.pt"
            for direction, router in (
                ("on", "forced_target_key"),
                ("off", "native"),
            ):
                torch.save(self._payload(direction, router), path)
                loaded = load_head_mask_checkpoint(path)
                self.assertEqual(loaded["selection_direction"], direction)

            torch.save(self._payload("on", "native"), path)
            with self.assertRaisesRegex(ValueError, "forced_target_key"):
                load_head_mask_checkpoint(path)
            payload = self._payload("off", "native")
            payload["lambda_position"] = 0.1
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "BCE only"):
                load_head_mask_checkpoint(path)

    def test_loader_rejects_non_top32_or_non_topk_saved_mask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "head_mask.pt"
            payload = self._payload("off", "native")
            payload["top_k"] = 31
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "Top32"):
                load_head_mask_checkpoint(path)

            payload = self._payload("off", "native")
            payload["scores"][0, 0] = 100.0
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "saved scores"):
                load_head_mask_checkpoint(path)

    def test_loader_can_opt_in_to_non_default_top_k(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "head_mask.pt"
            payload = self._payload("off", "native")
            payload["top_k"] = 1
            payload["hard_mask"] = hard_topk_mask(
                payload["scores"], top_k=1
            ).bool()
            payload["selected_components"] = [(4, 19)]
            torch.save(payload, path)
            loaded = load_head_mask_checkpoint(path, expected_top_k=None)
            self.assertEqual(loaded["top_k"], 1)


_LETTERS = tuple("ABCDEFGHIJ")


def _labeled_example(key: str, display: tuple[str, ...]) -> AttentionExample:
    candidates = {label: 100 + index for index, label in enumerate(_LETTERS)}
    candidates["none"] = 999
    return AttentionExample(
        key=key,
        prompt="",
        input_ids=torch.zeros(1, 4, dtype=torch.long),
        positions=tuple(range(10)),
        injection_spans={},
        candidate_token_ids=candidates,
        expected_candidate_by_position=dict(enumerate(display)),
        item_labels={},
        item_token_indices=(),
        records=(),
    )


class LabeledCandidateLayoutTests(unittest.TestCase):
    def test_ascending_labels_need_no_remap(self) -> None:
        token_ids, columns = labeled_candidate_layout(
            [_labeled_example("a", _LETTERS), _labeled_example("b", _LETTERS)]
        )
        self.assertEqual(token_ids, (*range(100, 110), 999))
        self.assertIsNone(columns)

    def test_shuffled_labels_map_each_slot_to_its_label_column(self) -> None:
        shuffled = tuple("HIDACJBEFG")
        token_ids, columns = labeled_candidate_layout(
            [_labeled_example("a", _LETTERS), _labeled_example("b", shuffled)]
        )
        self.assertEqual(token_ids, (*range(100, 110), 999))
        assert columns is not None
        self.assertEqual(columns[0].tolist(), list(range(11)))
        self.assertEqual(columns[1].tolist(), [7, 8, 3, 0, 2, 9, 1, 4, 5, 6, 10])

    def test_rejects_repeated_slot_label(self) -> None:
        with self.assertRaises(ValueError):
            labeled_candidate_layout([_labeled_example("a", tuple("AACDEFGHIJ"))])

    def test_slot_order_logits_moves_the_labelled_column_to_its_slot(self) -> None:
        shuffled = tuple("HIDACJBEFG")
        _, columns = labeled_candidate_layout(
            [_labeled_example("a", _LETTERS), _labeled_example("b", shuffled)]
        )
        dataset = HeadGateDataset(
            concepts=("x",),
            examples=(),
            candidate_token_ids=(),
            base_tokens=torch.zeros(0),
            injection_token_positions=torch.zeros(0),
            concept_vectors=torch.zeros(0),
            trials=(),
            slot_columns=columns,
        )
        # Cluster 1 prints label D (column 3) beside slot 2.
        logits = torch.zeros(2, 11)
        logits[:, 3] = 5.0
        trials = [
            InjectedTrial(0, cluster, 2, False, False) for cluster in (0, 1)
        ]
        ordered = dataset.slot_order_logits(logits, trials)
        self.assertEqual(ordered.argmax(dim=-1).tolist(), [3, 2])
        self.assertTrue(torch.equal(ordered[:, 10], logits[:, 10]))


if __name__ == "__main__":
    unittest.main()
