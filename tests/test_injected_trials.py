"""Tests for the shared injected-trial helpers."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import torch

from introspection_core.injected_trials import gate_logit, load_injected_trials, parse_layers


class GateLogitTests(unittest.TestCase):
    def test_gate_logit_matches_definition(self) -> None:
        logits = torch.full((1, 11), -8.0)
        logits[0, 4] = 3.0
        logits[0, 10] = 1.0
        score = gate_logit(logits, temperature=0.1)
        expected = 0.1 * torch.logsumexp(logits[:, :10] / 0.1, dim=-1)
        expected -= 0.1 * math.log(10)
        expected -= logits[:, 10]
        torch.testing.assert_close(score, expected)

    def test_gate_logit_is_neutral_when_all_candidates_are_equal(self) -> None:
        logits = torch.full((3, 11), 2.5)
        score = gate_logit(logits, temperature=0.1)
        torch.testing.assert_close(score, torch.zeros_like(score), atol=1e-6, rtol=0)


class InjectedTrialTests(unittest.TestCase):
    def test_loader_excludes_wrong_position_outputs(self) -> None:
        body = "\n".join(
            [
                "concept_index,cluster_index,position,condition,correct,predicted_none",
                "0,0,0,injected,0,1",
                "1,0,0,injected,1,0",
                "2,0,0,injected,0,0",
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trials.csv"
            path.write_text(body + "\n", encoding="utf-8")
            trials = load_injected_trials(path)
        self.assertEqual(len(trials), 2)

    def test_parse_layers(self) -> None:
        self.assertEqual(parse_layers("8-10,12,10"), [8, 9, 10, 12])


if __name__ == "__main__":
    unittest.main()
