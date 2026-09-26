"""Regression tests for model-agnostic localization sweep helpers."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import torch

from introspection_core.extraction import _last_non_padding_indices
from introspection_core.attention_aggregation import (
    _capture_target_hooks,
    average_attention,
)
from introspection_core.attention_inputs import AttentionExample
from introspection_core.layer_cache import resolve_resid_hook_name
from introspection_core.localization import Span
from introspection_core.model import HookedModel
from introspection_core.patching import patch
from introspection_core.projection import ProjectionResult
from introspection_core.scatter_html import (
    CLEAN_ORIGIN_STYLE,
    TOKEN_POSITION_HEX,
    _render_view_html,
    token_position_style,
    write_scatter_html,
)

class LastNonPaddingIndicesTests(unittest.TestCase):
    def test_handles_right_and_left_padding(self) -> None:
        attention_mask = torch.tensor(
            [
                [1, 1, 0, 0],
                [0, 0, 1, 1],
                [1, 1, 1, 1],
            ]
        )

        indices = _last_non_padding_indices(attention_mask)

        torch.testing.assert_close(indices, torch.tensor([1, 3, 3]))

    def test_rejects_all_padding_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "all-padding"):
            _last_non_padding_indices(torch.tensor([[0, 0]]))


class FinalLogitTransformTests(unittest.TestCase):
    def test_applies_configured_soft_cap(self) -> None:
        model = HookedModel.__new__(HookedModel)
        model.cfg = SimpleNamespace(output_logits_soft_cap=2.0)
        logits = torch.tensor([[-4.0, 0.0, 4.0]])

        transformed = model._apply_final_logit_transforms(logits)

        torch.testing.assert_close(transformed, 2.0 * torch.tanh(logits / 2.0))

    def test_leaves_uncapped_logits_unchanged(self) -> None:
        model = HookedModel.__new__(HookedModel)
        model.cfg = SimpleNamespace()
        logits = torch.tensor([[-4.0, 0.0, 4.0]])

        transformed = model._apply_final_logit_transforms(logits)

        self.assertIs(transformed, logits)


class TrialLevelAttentionLocalizationTests(unittest.TestCase):
    def test_counts_each_injected_trial_before_averaging_attention(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.cfg = SimpleNamespace(n_layers=1)
                self.bridge = SimpleNamespace(
                    cfg=SimpleNamespace(device="cpu", dtype=torch.float32)
                )
                self.calls = 0

            def attention_sums_and_candidate_logits(
                self,
                tokens,
                *,
                layers,
                candidate_token_ids,
                localization_token_indices,
            ):
                del layers, candidate_token_ids
                self.calls += 1
                batch_size = int(tokens.shape[0])
                attention = torch.zeros(1, 2, 4, 4)
                logits = torch.zeros(batch_size, 2)
                if self.calls == 1:
                    localization = torch.tensor([[[0, 1]]])
                else:
                    localization = torch.tensor([[[0, 1]], [[1, 0]]])
                self.assert_localization_indices = tuple(localization_token_indices)
                return attention, logits, localization

        example = AttentionExample(
            key="cluster_000",
            prompt="prompt",
            input_ids=torch.tensor([[1, 2, 3, 4]]),
            positions=(0, 1),
            injection_spans={
                0: Span(0, 1, "a", [1]),
                1: Span(1, 2, "b", [2]),
            },
            candidate_token_ids={"0": 10, "1": 11},
            expected_candidate_by_position={0: "0", 1: "1"},
            item_labels={0: "a", 1: "b"},
            item_token_indices=(0, 1),
            records=(),
        )
        model = FakeModel()

        with mock.patch(
            "introspection_core.attention_aggregation.inject",
            return_value=mock.MagicMock(
                __enter__=lambda self: self,
                __exit__=lambda self, *args: False,
            ),
        ):
            result = average_attention(
                model,
                examples=[example],
                concept_vectors=torch.eye(2),
                injection_layer=0,
                positions=[0],
                concept_batch_size=2,
                layers=[0],
            )

        torch.testing.assert_close(
            result.clean_attention_correct[0], torch.tensor([[2, 0]])
        )
        torch.testing.assert_close(
            result.injected_attention_correct[0], torch.tensor([[1, 1]])
        )
        self.assertEqual(model.assert_localization_indices, (0, 1))


class QkScorePatchTests(unittest.TestCase):
    def test_resolves_pre_softmax_hook_name(self) -> None:
        model = HookedModel.__new__(HookedModel)
        expected = "blocks.17.attn.hook_attn_scores"
        model._hook_names = {expected}

        self.assertEqual(model.attn_hook_name(17, "qk_scores"), expected)

    def test_overwrites_only_selected_head_and_query_row(self) -> None:
        captured = {}

        class Hooks:
            def __init__(self, fwd_hooks):
                captured["hook_name"], captured["hook_fn"] = fwd_hooks[0]

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

        bridge = SimpleNamespace(hooks=lambda *, fwd_hooks: Hooks(fwd_hooks))
        model = SimpleNamespace(
            bridge=bridge,
            attn_hook_name=lambda layer, kind: f"L{layer}.{kind}",
        )
        clean_row = torch.full((2, 1, 1, 5), 7.0)
        injected_scores = torch.arange(2 * 4 * 5 * 5).reshape(2, 4, 5, 5)

        with patch(
            model,
            layer=17,
            kind="qk_scores",
            source=clean_row,
            heads=[2],
            query_spans=[4],
        ):
            result = captured["hook_fn"](injected_scores, None)

        expected = injected_scores.clone()
        expected[:, 2, 4, :] = 7
        self.assertEqual(captured["hook_name"], "L17.qk_scores")
        torch.testing.assert_close(result, expected)


class OvPatchTests(unittest.TestCase):
    def test_overwrites_only_selected_head_and_token(self) -> None:
        captured = {}

        class Hooks:
            def __init__(self, fwd_hooks):
                captured["hook_name"], captured["hook_fn"] = fwd_hooks[0]

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

        bridge = SimpleNamespace(hooks=lambda *, fwd_hooks: Hooks(fwd_hooks))
        model = SimpleNamespace(
            bridge=bridge,
            attn_hook_name=lambda layer, kind: f"L{layer}.{kind}",
        )
        clean_z = torch.full((2, 5, 4, 3), 7.0)
        injected_z = torch.arange(2 * 5 * 4 * 3).reshape(2, 5, 4, 3)

        with patch(
            model,
            layer=17,
            kind="z",
            source=clean_z,
            heads=[2],
            query_spans=[(4, 5), (4, 5)],
        ):
            result = captured["hook_fn"](injected_z, None)

        expected = injected_z.clone()
        expected[:, 4, 2, :] = 7
        self.assertEqual(captured["hook_name"], "L17.z")
        torch.testing.assert_close(result, expected)


class KvInputPatchTests(unittest.TestCase):
    def test_overwrites_only_selected_kv_head_at_each_rows_newline(self) -> None:
        captured = {}

        class Hooks:
            def __init__(self, fwd_hooks):
                captured["hook_name"], captured["hook_fn"] = fwd_hooks[0]

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

        bridge = SimpleNamespace(hooks=lambda *, fwd_hooks: Hooks(fwd_hooks))
        model = SimpleNamespace(
            bridge=bridge,
            attn_hook_name=lambda layer, kind: f"L{layer}.{kind}",
        )
        clean_k = torch.full((2, 5, 2, 3), 7.0)
        injected_k = torch.arange(2 * 5 * 2 * 3).reshape(2, 5, 2, 3)

        with patch(
            model,
            layer=17,
            kind="k",
            source=clean_k,
            heads=[1],
            query_spans=[(2, 3), (4, 5)],
        ):
            result = captured["hook_fn"](injected_k, None)

        expected = injected_k.clone()
        expected[0, 2, 1, :] = 7
        expected[1, 4, 1, :] = 7
        torch.testing.assert_close(result, expected)


class ResidualHookResolutionTests(unittest.TestCase):
    def test_resolves_pre_and_post_hooks(self) -> None:
        model = SimpleNamespace(
            resid_hook_name=lambda layer: f"post_{layer}",
            resid_pre_hook_name=lambda layer: f"pre_{layer}",
        )

        self.assertEqual(resolve_resid_hook_name(model, 5), "post_5")
        self.assertEqual(
            resolve_resid_hook_name(model, 5, "resid_pre"), "pre_5"
        )

    def test_rejects_unknown_hook_point(self) -> None:
        model = SimpleNamespace()
        with self.assertRaisesRegex(ValueError, "Unsupported residual hook"):
            resolve_resid_hook_name(model, 5, "unknown")


class LazyPlotSerializationTests(unittest.TestCase):
    def test_escapes_closing_script_tags_in_labels(self) -> None:
        malicious_label = 'unsafe</script><script>alert("x")</script>'
        result = ProjectionResult(
            coords=torch.tensor([[0.0, 0.0, 0.0]]),
            method="pca",
            n_components=3,
            variance_ratio=None,
        )

        html = _render_view_html(
            {"layer": 0, "method": "pca", "result": result},
            rows=[{"condition": "test"}],
            group_legend=[{
                "label": malicious_label,
                "predicate": lambda row: True,
                "color": "#000000",
                "marker": "circle",
                "size": 3,
                "opacity": 1.0,
            }],
            div_id="plot-0",
            include_plotlyjs=False,
        )

        self.assertEqual(html.count("</script>"), 1)
        self.assertIn(r"unsafe<\/script><script>", html)

    def test_gallery_embeds_feedback_safe_camera_sync(self) -> None:
        result = ProjectionResult(
            coords=torch.tensor([[0.0, 0.0, 0.0]]),
            method="pca",
            n_components=3,
            variance_ratio=None,
        )
        group_legend = [{
            "label": "point",
            "predicate": lambda row: True,
            "color": "#000000",
            "marker": "circle",
            "size": 3,
            "opacity": 1.0,
        }]

        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "gallery.html"
            write_scatter_html(
                output_path,
                title="Camera sync",
                views=[{"layer": 0, "method": "pca", "result": result}],
                rows=[{}],
                group_legend=group_legend,
            )
            html = output_path.read_text(encoding="utf-8")

        self.assertIn('id="sync-rotation" checked', html)
        self.assertIn('id="camera-azimuth" value="45"', html)
        self.assertIn('id="camera-elevation" value="35.3"', html)
        self.assertIn('id="point-size" value="60"', html)
        self.assertIn('id="point-size-value"', html)
        self.assertIn("var pointSizeScale = 0.6;", html)
        self.assertIn("function plotData(spec)", html)
        self.assertIn("function displayConfig(spec)", html)
        self.assertIn("scaled.marker.size = scaledMarkerSize", html)
        self.assertIn('pointSizeInput.addEventListener("input"', html)
        self.assertIn('"width": 1200', html)
        self.assertIn('"height": 1200', html)
        self.assertIn("function setAllCameraAngles(azimuth, elevation)", html)
        self.assertIn("showCameraAngles(lastCamera);", html)
        self.assertIn("Date.now() + 500", html)
        self.assertIn('azimuthInput.addEventListener("change"', html)
        self.assertIn('elevationInput.addEventListener("change"', html)
        self.assertIn('class="export-plot"', html)
        self.assertIn("async function exportPlotPdf(button)", html)
        self.assertIn("function vectorPoints(source, spec)", html)
        self.assertIn("function drawVectorMarker(pdf, point)", html)
        self.assertIn("function pdfRgb(color)", html)
        self.assertIn("function setPdfMarkerOpacity(pdf, opacity)", html)
        self.assertIn("var rgb = pdfRgb(point.color);", html)
        self.assertIn("new pdf.GState({ opacity: alpha })", html)
        self.assertIn("pdf.setGState(pdf.__markerOpacityStates[key]);", html)
        self.assertNotIn("(1 - alpha) * 255", html)
        self.assertIn("points.forEach(function (point)", html)
        self.assertIn("function exportMillimetersPerPixel(source)", html)
        self.assertIn("function displayedAxisBounds(source, axisName, points)", html)
        self.assertIn("function projectScenePoint(point, eye", html)
        self.assertIn("var traces = plotData(spec).filter", html)
        self.assertIn("millimetersPerPixel / 2", html)
        self.assertIn("var projectedCube = [];", html)
        self.assertNotIn("2 * (point.x - bounds.x.middle)", html)
        self.assertNotIn("pdf.addImage", html)
        self.assertNotIn("Plotly.toImage(exportDiv", html)
        self.assertIn("displayed.visible = false;", html)
        self.assertIn("jspdf@2.5.2", html)
        self.assertIn("function ensureJsPdf()", html)
        self.assertNotIn('<script src="https://cdn.jsdelivr.net/npm/jspdf', html)
        self.assertNotIn('<script src="https://cdn.plot.ly', html)
        self.assertIn("window.Plotly", html)
        self.assertIn('"visible": false', html)
        self.assertIn("function cleanDisplayLayout(layout)", html)
        self.assertIn('button.textContent = "Export Vector PDF";', html)
        self.assertEqual(html.count("// Lazy Plotly renderer"), 1)
        self.assertEqual(html.count("async function exportPlotPdf(button)"), 1)
        self.assertIn("var expectedCameras = new WeakMap();", html)
        self.assertIn("if (syncingDivs.has(div)) return;", html)
        self.assertIn('Plotly.relayout(div, { "scene.camera":', html)

    def test_point_filters_preserve_trace_grouping_and_keep_references(self) -> None:
        result = ProjectionResult(
            coords=torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0],
                ]
            ),
            method="pca",
            n_components=3,
            variance_ratio=None,
        )
        rows = [
            {"condition": "clean", "correct": ""},
            {"condition": "injected", "correct": 1},
            {"condition": "injected", "correct": 0},
        ]
        html = _render_view_html(
            {"layer": 0, "method": "pca", "result": result},
            rows=rows,
            group_legend=[{
                "label": "same original color",
                "predicate": lambda row: True,
                "color": "#123456",
                "marker": "circle",
                "size": 3,
                "opacity": 1.0,
            }],
            div_id="plot-0",
            include_plotlyjs=False,
            point_filter_groups=[
                {
                    "id": "correct",
                    "label": "Correct",
                    "predicate": lambda row: row["correct"] == 1,
                },
                {
                    "id": "incorrect",
                    "label": "Incorrect",
                    "predicate": lambda row: row["correct"] == 0,
                },
            ],
        )
        spec_text = html.split('data-target="plot-0">', 1)[1].rsplit(
            "</script>", 1
        )[0]
        spec = json.loads(spec_text)

        self.assertEqual(len(spec["data"]), 4)
        self.assertTrue(
            all(
                trace["marker"]["color"] == "#123456"
                for trace in spec["data"]
            )
        )
        self.assertEqual(
            spec["pointFilters"], [None, None, "correct", "incorrect"]
        )

    def test_gallery_adds_global_point_filter_controls(self) -> None:
        result = ProjectionResult(
            coords=torch.tensor([[0.0, 0.0, 0.0]]),
            method="pca",
            n_components=3,
            variance_ratio=None,
        )
        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "gallery.html"
            write_scatter_html(
                output_path,
                title="Point filters",
                views=[{"layer": 0, "method": "pca", "result": result}],
                rows=[{"correct": 1}],
                group_legend=[{
                    "label": "point",
                    "predicate": lambda row: True,
                    "color": "#000000",
                    "marker": "circle",
                    "size": 3,
                    "opacity": 1.0,
                }],
                point_filter_groups=[{
                    "id": "correct",
                    "label": "Correct answer",
                    "predicate": lambda row: row["correct"] == 1,
                }],
            )
            html = output_path.read_text(encoding="utf-8")

        self.assertIn('data-point-filter="correct" checked', html)
        self.assertIn("Correct answer", html)
        self.assertIn("Plotly.react(div, plotData(spec)", html)

    def test_gallery_adds_exclusive_point_filter_controls(self) -> None:
        result = ProjectionResult(
            coords=torch.tensor([[0.0, 0.0, 0.0]]),
            method="pca",
            n_components=3,
            variance_ratio=None,
        )
        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "gallery.html"
            write_scatter_html(
                output_path,
                title="Exclusive point filters",
                views=[{"layer": 0, "method": "pca", "result": result}],
                rows=[{"correct": 1}],
                group_legend=[{
                    "label": "point",
                    "predicate": lambda row: True,
                    "color": "#000000",
                    "marker": "diamond",
                    "size": 3,
                    "opacity": 1.0,
                }],
                point_filter_groups=[
                    {
                        "id": "correct",
                        "label": "Correct answer",
                        "predicate": lambda row: row["correct"] == 1,
                    },
                    {
                        "id": "incorrect",
                        "label": "Incorrect answer",
                        "predicate": lambda row: row["correct"] == 0,
                    },
                ],
                point_filter_mode="exclusive",
            )
            html = output_path.read_text(encoding="utf-8")

        self.assertIn(
            'type="radio" name="point-filter-selection" data-point-filter-all checked',
            html,
        )
        self.assertIn('data-point-filter="correct"> Correct answer', html)
        self.assertIn('data-point-filter="incorrect"> Incorrect answer', html)
        self.assertIn("function readPointFilters()", html)

    def test_shared_token_position_style_matches_reference_gallery(self) -> None:
        self.assertEqual(
            token_position_style(0),
            {
                "color": "#1f77b4",
                "marker": "circle",
                "size": 4.0,
                "opacity": 0.75,
            },
        )
        self.assertEqual(token_position_style(10)["color"], TOKEN_POSITION_HEX[0])
        self.assertEqual(
            CLEAN_ORIGIN_STYLE,
            {
                "color": "#000000",
                "marker": "circle",
                "size": 6.0,
                "opacity": 1.0,
            },
        )



if __name__ == "__main__":
    unittest.main()
