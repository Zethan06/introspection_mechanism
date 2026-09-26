"""Self-contained interactive 3D scatter gallery for latent and head-output PCA.

Renders one Plotly Scatter3d card per (layer, method) view. Points are colored
by injected position, with clean runs in black.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any, Callable

import torch
from plotly.offline import get_plotlyjs

from .projection import ProjectionResult

_ASSETS_DIR = Path(__file__).parent / "assets"

TOKEN_POSITION_HEX = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)

CLEAN_ORIGIN_STYLE: dict[str, Any] = {
    "color": "#000000",
    "marker": "circle",
    "size": 6.0,
    "opacity": 1.0,
}


def injection_outcome_filter_groups() -> list[dict[str, Any]]:
    """Return the shared correct/incorrect controls for injection galleries."""
    return [
        {
            "id": "injected-correct",
            "label": "Correct injection (answer = i)",
            "predicate": lambda row: (
                row["condition"] == "injected"
                and int(row["correct"]) == 1
            ),
        },
        {
            "id": "injected-incorrect",
            "label": "Incorrect answer (answer != i)",
            "predicate": lambda row: (
                row["condition"] == "injected"
                and int(row["correct"]) == 0
            ),
        },
    ]


def token_position_style(position: int) -> dict[str, Any]:
    """Return the shared token-position marker style used by galleries."""
    return {
        "color": TOKEN_POSITION_HEX[position % len(TOKEN_POSITION_HEX)],
        "marker": "circle",
        "size": 4.0,
        "opacity": 0.75,
    }


# label, predicate, color, marker symbol, size, opacity -- verbatim from
# token_representation_pca.py::plot_layer's six matplotlib groups, translated to Plotly's
# marker vocabulary ("square"/"circle"/"diamond" stand in for matplotlib's "s"/"o"/"^").
DEFAULT_GROUP_LEGEND: list[dict[str, Any]] = [
    {
        "label": "clean not chosen",
        "predicate": lambda row: row["condition"] == "clean" and int(row["is_chosen"]) == 0,
        "color": "#bdbdbd",
        "marker": "square",
        "size": 3,
        "opacity": 0.30,
    },
    {
        "label": "clean chosen",
        "predicate": lambda row: row["condition"] == "clean" and int(row["is_chosen"]) == 1,
        "color": "#e6550d",
        "marker": "square",
        "size": 5,
        "opacity": 0.85,
    },
    {
        "label": "uninjected not chosen",
        "predicate": lambda row: (
            row["condition"] == "injected"
            and row["token_role"] == "uninjected"
            and int(row["is_chosen"]) == 0
        ),
        "color": "#6baed6",
        "marker": "circle",
        "size": 2.5,
        "opacity": 0.18,
    },
    {
        "label": "uninjected chosen",
        "predicate": lambda row: (
            row["condition"] == "injected"
            and row["token_role"] == "uninjected"
            and int(row["is_chosen"]) == 1
        ),
        "color": "#e6550d",
        "marker": "circle",
        "size": 3.5,
        "opacity": 0.65,
    },
    {
        "label": "injected not chosen",
        "predicate": lambda row: (
            row["condition"] == "injected"
            and row["token_role"] == "injected"
            and int(row["is_chosen"]) == 0
        ),
        "color": "#2171b5",
        "marker": "diamond",
        "size": 5.5,
        "opacity": 0.55,
    },
    {
        "label": "injected chosen",
        "predicate": lambda row: (
            row["condition"] == "injected"
            and row["token_role"] == "injected"
            and int(row["is_chosen"]) == 1
        ),
        "color": "#d94801",
        "marker": "diamond",
        "size": 6.5,
        "opacity": 0.90,
    },
]


def sampled_indices(count: int, max_points: int, seed: int) -> list[int]:
    """Same seeded subsample technique as token_representation_pca.py::sampled_indices."""
    if max_points <= 0 or count <= max_points:
        return list(range(count))
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(count, generator=generator)[:max_points].sort().values.tolist()


def _view_title(view: dict[str, Any]) -> str:
    result: ProjectionResult = view["result"]
    variance = result.variance_ratio
    var_suffix = ""
    if variance:
        var_suffix = " (" + ", ".join(f"PC{i + 1} {v * 100:.1f}%" for i, v in enumerate(variance)) + ")"
    # views may carry an explicit "label" (e.g. "Head 3") instead of the default "Layer N"
    label = view.get("label", f"Layer {view['layer']}")
    return f"{label} — {result.method}{var_suffix}"


def _render_view_html(
    view: dict[str, Any],
    rows: list[dict[str, Any]],
    group_legend: list[dict[str, Any]],
    div_id: str,
    include_plotlyjs: bool | str,
    point_filter_groups: list[dict[str, Any]] | None = None,
) -> str:
    """Emit a placeholder div plus an inline JSON spec for lazy rendering.

    Instead of embedding a live Plotly plot per view (which exhausts the
    browser's ~16 WebGL-context limit once a gallery has many 3-D scenes and
    silently blanks the earliest plots), we register each figure's spec in a
    global registry and let an IntersectionObserver (injected by
    write_gallery_html) render it only while scrolled into view, purging the
    WebGL context when it scrolls away.
    """
    import plotly.graph_objects as go

    result: ProjectionResult = view["result"]
    coords = result.coords.numpy()
    n_components = result.n_components
    hover_fields = list(view.get("hover_fields", []))

    fig = go.Figure()
    trace_point_filters: list[str | None] = []

    def add_trace(
        idxs: list[int],
        group: dict[str, Any],
        *,
        legend_count: int,
        show_legend: bool,
        point_filter_id: str | None,
    ) -> None:
        xs = coords[idxs, 0]
        ys = coords[idxs, 1] if n_components > 1 else [0.0] * len(idxs)
        zs = coords[idxs, 2] if n_components > 2 else [0.0] * len(idxs)
        hover_text = None
        hover_template = None
        if hover_fields:
            hover_text = [
                "<br>".join(
                    f"{field}: {rows[index].get(field, '')}"
                    for field in hover_fields
                )
                for index in idxs
            ]
            hover_template = "%{text}<extra>%{fullData.name}</extra>"
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="markers",
                text=hover_text,
                hovertemplate=hover_template,
                marker=dict(
                    size=group["size"],
                    color=group["color"],
                    symbol=group["marker"],
                    opacity=group["opacity"],
                ),
                name=f"{group['label']} (n={legend_count})",
                legendgroup=group["label"],
                showlegend=show_legend,
            )
        )
        trace_point_filters.append(point_filter_id)

    for group in group_legend:
        predicate: Callable[[dict[str, Any]], bool] = group["predicate"]
        idxs = [i for i, row in enumerate(rows) if predicate(row)]
        if not idxs:
            continue
        if not point_filter_groups:
            add_trace(
                idxs,
                group,
                legend_count=len(idxs),
                show_legend=True,
                point_filter_id=None,
            )
            continue

        buckets: dict[str | None, list[int]] = {None: []}
        for filter_group in point_filter_groups:
            buckets[str(filter_group["id"])] = []
        for index in idxs:
            matches = [
                str(filter_group["id"])
                for filter_group in point_filter_groups
                if filter_group["predicate"](rows[index])
            ]
            if len(matches) > 1:
                raise ValueError(
                    f"row {index} matches multiple point filter groups: {matches}"
                )
            # Rows outside the configured groups (for example the clean
            # origin) are reference points and remain visible under every
            # filter selection.
            buckets[matches[0] if matches else None].append(index)

        populated = [
            (point_filter_id, bucket)
            for point_filter_id, bucket in buckets.items()
            if bucket
        ]
        if len(populated) == 1:
            point_filter_id, bucket = populated[0]
            add_trace(
                bucket,
                group,
                legend_count=len(idxs),
                show_legend=True,
                point_filter_id=point_filter_id,
            )
            continue

        # Multiple outcome buckets retain one copy of the original legend
        # entry. The empty marker trace owns that entry; the point-bearing
        # traces keep the same color/shape and are filtered independently.
        fig.add_trace(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                hoverinfo="skip",
                marker=dict(
                    size=group["size"],
                    color=group["color"],
                    symbol=group["marker"],
                    opacity=group["opacity"],
                ),
                name=f"{group['label']} (n={len(idxs)})",
                legendgroup=group["label"],
                showlegend=True,
            )
        )
        trace_point_filters.append(None)
        for point_filter_id, bucket in populated:
            add_trace(
                bucket,
                group,
                legend_count=len(idxs),
                show_legend=False,
                point_filter_id=point_filter_id,
            )

    scene_axis = dict(visible=False)
    fig.update_layout(
        title=dict(text=_view_title(view), font=dict(size=13)),
        height=640,
        margin=dict(l=0, r=0, t=66, b=0),
        legend=dict(font=dict(size=9), itemsizing="constant"),
        scene=dict(
            xaxis=scene_axis,
            yaxis=scene_axis,
            zaxis=scene_axis,
            aspectmode="cube",
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    # Serialize the figure spec (data + layout) and defer rendering to the
    # lazy IntersectionObserver script. div_id is the render target.
    spec: dict[str, Any] = {
        "data": json.loads(fig.to_json())["data"],
        "layout": json.loads(fig.to_json())["layout"],
        "config": {
            "responsive": True,
            "toImageButtonOptions": {
                "format": "png",
                "width": 1200,
                "height": 1200,
                "scale": 2,
            },
        },
    }
    if point_filter_groups:
        spec["pointFilters"] = trace_point_filters
    # A literal closing script tag terminates application/json elements in
    # HTML parsing, even though the content is not executable JavaScript.
    spec_json = json.dumps(spec).replace("</", r"<\/")
    return (
        f'<div id="{div_id}" class="lazy-plot" '
        f'style="width:100%;height:660px;" data-plot-id="{div_id}"></div>\n'
        f'<script type="application/json" class="plot-spec" '
        f'data-target="{div_id}">{spec_json}</script>'
    )


def write_gallery_html(
    output_path: Path,
    page_title: str,
    meta: dict[str, Any],
    sections: list[dict[str, Any]],
    summary_table: dict[str, Any] | None = None,
    nav_title: str | None = None,
    include_meta: bool = True,
    foot_html: str = "",
    point_filter_groups: list[dict[str, Any]] | None = None,
    point_filter_mode: str = "multi",
) -> Path:
    """Page chrome, ported from pca_gallery_html.py::write_yes_no_gallery_html, reading its
    CSS from assets/gallery.css instead of an inline <style> string."""

    def _anchor(item: dict[str, Any]) -> str:
        return str(item.get("anchor", f"section-{item['index']}"))

    def _nav_label(item: dict[str, Any]) -> str:
        return str(item.get("nav_label", item.get("heading", f"#{item['index']}")))

    nav_links = " ".join(
        f'<a href="#{escape(_anchor(item))}">{escape(_nav_label(item))}</a>' for item in sections
    )
    nav_title = nav_title or "Jump to view"
    page_description = meta.get("page_description", "")
    point_filter_html = ""
    if point_filter_groups:
        if point_filter_mode not in {"multi", "exclusive"}:
            raise ValueError(
                "point_filter_mode must be either 'multi' or 'exclusive'"
            )
        if point_filter_mode == "exclusive":
            point_filter_inputs = (
                '<label><input type="radio" name="point-filter-selection" '
                'data-point-filter-all checked> All points</label>'
                + "".join(
                    f'<label><input type="radio" name="point-filter-selection" '
                    f'data-point-filter="{escape(str(group["id"]))}"> '
                    f'{escape(str(group["label"]))}</label>'
                    for group in point_filter_groups
                )
            )
        else:
            point_filter_inputs = "".join(
                f'<label><input type="checkbox" data-point-filter="{escape(str(group["id"]))}" '
                f'checked> {escape(str(group["label"]))}</label>'
                for group in point_filter_groups
            )
        point_filter_html = (
            '<fieldset class="point-filters">'
            '<legend>Show points</legend>'
            f'{point_filter_inputs}'
            '</fieldset>'
        )

    summary_html = ""
    if summary_table is not None:
        columns = summary_table.get("columns", [])
        rows = summary_table.get("rows", [])
        header_html = "".join(f"<th>{escape(str(column))}</th>" for column in columns)
        body_html = "".join(
            "<tr>" + "".join(f"<td>{escape(str(cell))}</td>" for cell in row) + "</tr>" for row in rows
        )
        summary_html = f"""
<section class="card">
  <div class="card-head">
    <h2>{escape(str(summary_table.get("title", "Summary")))}</h2>
  </div>
  <div class="table-wrap">
    <table>
      <thead><tr>{header_html}</tr></thead>
      <tbody>{body_html}</tbody>
    </table>
  </div>
</section>
"""

    meta_json = escape(json.dumps(meta, ensure_ascii=False, indent=2))
    meta_html = f'<pre class="meta">{meta_json}</pre>' if include_meta else ""

    cards = []
    for item in sections:
        heading = item.get("heading", f"Section {item['index']}")
        anchor = item.get("anchor", f"section-{item['index']}")
        metrics_html = "".join(
            f"<span>{escape(str(metric['label']))}: {escape(str(metric['value']))}</span>"
            for metric in item.get("metrics", [])
        )
        export_target = item.get("export_target")
        export_html = (
            f'<button type="button" class="export-plot" '
            f'data-export-target="{escape(str(export_target))}">Export Vector PDF</button>'
            if export_target
            else ""
        )
        cards.append(
            f"""
<section class="card" id="{escape(anchor)}">
  <div class="card-head">
    <h2>{escape(heading)}</h2>
    <div class="metrics">
      {metrics_html}
      {export_html}
    </div>
  </div>
  {item["html_block"]}
</section>
"""
        )

    css = (_ASSETS_DIR / "gallery.css").read_text()
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{escape(page_title)}</title>
  <style>
{css}
  </style>
</head>
<body>
  <div class="page">
    <header class="hero">
      <h1>{escape(page_title)}</h1>
      <p>{escape(page_description)}</p>
    </header>
    <div class="nav">
      <strong>{escape(nav_title)}</strong>
      <label class="sync-toggle">
        <input type="checkbox" id="sync-rotation" checked>
        Sync 3D rotation across views
      </label>
      <label class="angle-control" for="camera-azimuth">
        Azimuth
        <input type="number" id="camera-azimuth" value="45" step="1"
          min="-180" max="180" inputmode="decimal">
        <span aria-hidden="true">°</span>
      </label>
      <label class="angle-control" for="camera-elevation">
        Elevation
        <input type="number" id="camera-elevation" value="35.3" step="1"
          min="-89.9" max="89.9" inputmode="decimal">
        <span aria-hidden="true">°</span>
      </label>
      <label class="point-size-control" for="point-size">
        Point size
        <input type="range" id="point-size" value="60" step="5"
          min="20" max="150">
        <output id="point-size-value" for="point-size">60%</output>
      </label>
      {point_filter_html}
      <div class="links">{nav_links}</div>
    </div>
    {meta_html}
    {summary_html}
    {''.join(cards)}
  </div>
  {foot_html}
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def write_scatter_html(
    path: Path,
    *,
    title: str,
    views: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    group_legend: list[dict[str, Any]] | None = None,
    max_points: int = 12000,
    seed: int = 0,
    page_description: str = "",
    extra_meta: dict[str, Any] | None = None,
    point_filter_groups: list[dict[str, Any]] | None = None,
    point_filter_mode: str = "multi",
) -> Path:
    """Samples `rows` down to max_points once (shared across every view), then builds one
    Plotly-based card per view in `views` (each {"layer": int, "method": str, "result":
    ProjectionResult} -- result.coords must have exactly len(rows) rows, sampled the same way).

    `group_legend` defaults to DEFAULT_GROUP_LEGEND (the six clean/injected/chosen groups from
    token_representation_pca.py::plot_layer); pass a custom list of
    {"label", "predicate", "color", "marker", "size", "opacity"} dicts to override.

    `point_filter_groups` adds controls to the sticky navigation. Each {"id", "label",
    "predicate"} group controls matching points without changing their original trace style.
    `point_filter_mode="exclusive"` adds a mutually exclusive All/group radio selector;
    the default `"multi"` mode uses checked-by-default checkboxes. Rows that match no filter
    group remain visible as reference points.
    """
    group_legend = group_legend if group_legend is not None else DEFAULT_GROUP_LEGEND

    sample_idx = sampled_indices(len(rows), max_points, seed)
    sampled_rows = [rows[i] for i in sample_idx]
    index_tensor = torch.tensor(sample_idx, dtype=torch.long)

    sections = []
    for i, view in enumerate(views):
        result: ProjectionResult = view["result"]
        if result.coords.shape[0] != len(rows):
            raise ValueError(
                f"view {i} (layer={view.get('layer')}, method={view.get('method')}) has "
                f"{result.coords.shape[0]} coords, expected {len(rows)} to match rows"
            )
        sampled_result = ProjectionResult(
            coords=result.coords.index_select(0, index_tensor),
            method=result.method,
            n_components=result.n_components,
            variance_ratio=result.variance_ratio,
            extra=result.extra,
        )
        sampled_view = {**view, "result": sampled_result}
        anchor = f"layer-{view['layer']}-{view['method']}"
        html_block = _render_view_html(
            sampled_view,
            sampled_rows,
            group_legend,
            div_id=f"plot-{i}",
            include_plotlyjs="cdn" if i == 0 else False,
            point_filter_groups=point_filter_groups,
        )
        # views may carry a short "nav_label" override (e.g. "H3 pca" for head views)
        default_nav = f"L{view['layer']} {view['method']}"
        nav_label = view.get("nav_label", default_nav)
        sections.append(
            {
                "index": i,
                "anchor": anchor,
                "heading": _view_title(sampled_view),
                "nav_label": nav_label,
                "metrics": [{"label": "points", "value": len(sampled_rows)}],
                "export_target": f"plot-{i}",
                "html_block": html_block,
            }
        )

    meta = {"point_count": len(rows), "sampled_point_count": len(sampled_rows), "views": len(views)}
    if extra_meta:
        meta.update(extra_meta)
    if page_description:
        meta["page_description"] = page_description

    # Embed Plotly so opening a generated result never depends on CDN access.
    # The lazy observer still caps concurrent WebGL contexts for large galleries.
    plotly_js = get_plotlyjs()
    lazy_js = (_ASSETS_DIR / "lazy_plots.js").read_text()
    foot_html = (
        f"<script>{plotly_js}</script>\n"
        f"<script>{lazy_js}</script>"
    )

    return write_gallery_html(
        Path(path),
        page_title=title,
        meta=meta,
        sections=sections,
        nav_title="Jump to view",
        foot_html=foot_html,
        point_filter_groups=point_filter_groups,
        point_filter_mode=point_filter_mode,
    )
