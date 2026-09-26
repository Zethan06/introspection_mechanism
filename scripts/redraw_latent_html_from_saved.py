#!/usr/bin/env python3
"""Rebuild a single-canvas projection gallery from a saved scatter HTML.

This is an artifact-only operation: the PCA coordinates and trace metadata are
read from the saved HTML, so no model checkpoint or GPU is needed.
"""

from __future__ import annotations

import argparse
import json
import re
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAZY_PLOTS_JS = ROOT / "introspection_core" / "assets" / "lazy_plots.js"


def _saved_parts(source_html: str) -> tuple[str, list[tuple[int, str, str]]]:
    script_bodies = re.findall(r"<script>(.*?)</script>", source_html, re.DOTALL)
    plotly_js = next(
        (body for body in script_bodies if "plotly.js v" in body),
        "",
    )
    if not plotly_js:
        raise ValueError("saved HTML does not contain an embedded Plotly bundle")

    views: list[tuple[int, str, str]] = []
    pattern = re.compile(
        r'<script type="application/json" class="plot-spec" '
        r'data-target="plot-(\d+)">(.*?)</script>',
        re.DOTALL,
    )
    for layer_text, spec_text in pattern.findall(source_html):
        spec = json.loads(spec_text)
        label = str(
            spec.get("layout", {}).get("title", {}).get(
                "text", f"Layer {layer_text} — PCA"
            )
        )
        views.append((int(layer_text), label, spec_text))
    if not views:
        raise ValueError("saved HTML does not contain latent plot specifications")
    return plotly_js, views


def redraw_from_saved(
    source: Path,
    output: Path,
    *,
    artifact: str = "latent",
) -> Path:
    """Write a one-WebGL-canvas HTML using coordinates embedded in ``source``."""
    artifact_settings = {
        "latent": ("latent", "Layer", "layer"),
        "head-ov": ("head OV", "Head", "head"),
    }
    if artifact not in artifact_settings:
        raise ValueError(f"unsupported artifact: {artifact}")
    artifact_label, selector_label, card_prefix = artifact_settings[artifact]
    plotly_js, views = _saved_parts(source.read_text(encoding="utf-8"))
    lazy_js = LAZY_PLOTS_JS.read_text(encoding="utf-8")
    model_name = next(
        (
            parent.name
            for parent in source.resolve().parents
            if parent.parent.name == "results"
        ),
        source.stem,
    )
    first_layer, first_label, _ = views[0]
    options = "".join(
        f'<option value="{layer}">{escape(label)}</option>'
        for layer, label, _ in views
    )
    specs = []
    for index, (layer, label, spec_text) in enumerate(views):
        active = ' class="plot-spec" data-target="plot-main"' if index == 0 else (
            ' class="layer-plot-spec"'
        )
        specs.append(
            f'<script type="application/json"{active} data-layer="{layer}" '
            f'data-label="{escape(label, quote=True)}">{spec_text}</script>'
        )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Validation {escape(artifact_label)} representation — {escape(model_name)}</title>
  <style>
    :root {{ --ink:#172033; --muted:#667085; --line:#d0d5dd; --panel:#fff; --bg:#f5f7fa; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; color:var(--ink); background:var(--bg); font-family:Inter,system-ui,sans-serif; }}
    main {{ max-width:1120px; margin:auto; padding:24px; }}
    h1 {{ margin:0 0 6px; font-size:24px; }}
    .subtitle {{ margin:0 0 18px; color:var(--muted); }}
    .controls {{ display:flex; flex-wrap:wrap; gap:12px 20px; align-items:end; padding:14px;
      border:1px solid var(--line); border-radius:10px; background:var(--panel); }}
    label,fieldset {{ color:var(--muted); font-size:12px; }}
    label.control {{ display:grid; gap:5px; }}
    select,input[type=number] {{ min-height:34px; border:1px solid var(--line); border-radius:6px;
      padding:6px 9px; background:#fff; color:var(--ink); }}
    fieldset {{ display:flex; gap:12px; border:0; padding:0; margin:0; }}
    fieldset legend {{ margin-bottom:5px; }}
    .card {{ margin-top:16px; padding:16px; border:1px solid var(--line); border-radius:12px;
      background:var(--panel); box-shadow:0 4px 18px rgba(16,24,40,.05); }}
    .card-head {{ display:flex; gap:16px; align-items:center; justify-content:space-between; }}
    .card-head h2 {{ margin:0; font-size:17px; }}
    .metrics {{ display:flex; gap:10px; align-items:center; color:var(--muted); font-size:12px; }}
    button {{ border:1px solid #175cd3; border-radius:7px; padding:8px 12px; color:#fff;
      background:#175cd3; cursor:pointer; }}
    button:disabled {{ opacity:.6; cursor:wait; }}
    #plot-main {{ width:100%; height:760px; }}
    .point-size-control output {{ min-width:36px; display:inline-block; }}
  </style>
</head>
<body>
<main>
  <h1>Validation {escape(artifact_label)} representation — {escape(model_name)}</h1>
  <p class="subtitle">Reconstructed from saved PCA coordinates; no model inference or PCA recomputation.</p>
  <section class="controls">
    <label class="control" for="layer-select">{escape(selector_label)}
      <select id="layer-select">{options}</select>
    </label>
    <label class="control" for="camera-azimuth">Azimuth
      <input type="number" id="camera-azimuth" value="45" step="1" min="-180" max="180">
    </label>
    <label class="control" for="camera-elevation">Elevation
      <input type="number" id="camera-elevation" value="35.3" step="1" min="-89.9" max="89.9">
    </label>
    <label class="point-size-control" for="point-size">Point size
      <input type="range" id="point-size" value="60" step="5" min="20" max="150">
      <output id="point-size-value" for="point-size">60%</output>
    </label>
    <fieldset class="point-filters"><legend>Show points</legend>
      <label><input type="checkbox" data-point-filter="injected-correct" checked> Correct</label>
      <label><input type="checkbox" data-point-filter="injected-incorrect" checked> Incorrect</label>
    </fieldset>
    <input type="checkbox" id="sync-rotation" checked hidden>
  </section>
  <section class="card" id="{card_prefix}-{first_layer}-pca">
    <div class="card-head">
      <h2 id="view-title">{escape(first_label)}</h2>
      <div class="metrics"><span>1,001 points</span>
        <button type="button" class="export-plot" data-export-target="plot-main">Export Vector PDF</button>
      </div>
    </div>
    <div id="plot-main" class="lazy-plot" data-plot-id="plot-main"></div>
  </section>
  {''.join(specs)}
</main>
<script>{plotly_js}</script>
<script>{lazy_js}</script>
<script>
(function () {{
  var select = document.getElementById("layer-select");
  var plot = document.getElementById("plot-main");
  var title = document.getElementById("view-title");
  var card = plot.closest(".card");
  select.addEventListener("change", function () {{
    var current = document.querySelector('script.plot-spec[data-target="plot-main"]');
    var next = document.querySelector('script[data-layer="' + select.value + '"]');
    if (!next || next === current) return;
    current.className = "layer-plot-spec";
    current.removeAttribute("data-target");
    next.className = "plot-spec";
    next.dataset.target = "plot-main";
    title.textContent = next.dataset.label;
    card.id = "{card_prefix}-" + select.value + "-pca";
    if (plot.dataset.rendered === "1") {{
      document.querySelector("input[data-point-filter]").dispatchEvent(new Event("change"));
    }}
  }});
}})();
</script>
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="saved scatter HTML")
    parser.add_argument("output", type=Path, help="new standalone HTML")
    parser.add_argument(
        "--artifact",
        choices=("latent", "head-ov"),
        default="latent",
        help="projection type used for page labels (default: latent)",
    )
    args = parser.parse_args()
    written = redraw_from_saved(args.source, args.output, artifact=args.artifact)
    print(f"Wrote {written}")


if __name__ == "__main__":
    main()
