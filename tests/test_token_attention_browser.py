"""Regression tests for the standalone attention browser."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "introspection_core"
    / "token_attention_browser.py"
)
_SPEC = importlib.util.spec_from_file_location("token_attention_browser", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
make_token_attention_browser = _MODULE.make_token_attention_browser


class TokenAttentionBrowserTests(unittest.TestCase):
    def test_includes_current_view_pdf_export(self) -> None:
        html = make_token_attention_browser({"title": "unsafe </script> title"})

        self.assertIn('id="exportPdf"', html)
        self.assertIn("window.print()", html)
        self.assertIn("@page{size:A4 landscape", html)
        self.assertIn("print-color-adjust:exact", html)
        self.assertIn(
            "token_attention_L${l}_H${h}_P${p}_${mode}_Q${q}_K${start}-${end}_C${lineChars.value}",
            html,
        )
        self.assertIn('id="rangeStart"', html)
        self.assertIn('id="rangeEnd"', html)
        self.assertIn('id="lineChars"', html)
        self.assertIn("function updateVisibleRange(changed)", html)
        self.assertIn("function updateLineWidth()", html)
        self.assertIn("function normalizeLineWidth()", html)
        self.assertIn('lineChars.addEventListener("change",normalizeLineWidth)', html)
        self.assertIn("--chars-per-line", html)
        self.assertIn(".tok[hidden]{display:none}", html)
        self.assertIn(".titlebar,.controls,.footer,.legend{display:none}", html)
        self.assertIn("@page{size:A4 landscape;margin:0}", html)
        self.assertIn(".tok.topkey::after{display:none}", html)
        self.assertIn(
            r'function visible(text){return String(text).replace(/\n/g,"\\n\n")}',
            html,
        )
        self.assertNotIn("unsafe </script>", html)


if __name__ == "__main__":
    unittest.main()
