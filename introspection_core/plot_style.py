"""Shared styling helpers for repository line plots.

The style follows the project's compact paper-figure convention: white
background, no grid, black boxed axes, legends on the left, and series that
remain distinguishable through colour, line style, and marker shape.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure


# Okabe-Ito: a compact, colour-blind-safe qualitative palette.
LINE_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
)
LINE_STYLES = ("-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)))
LINE_MARKERS = ("o", "s", "^", "D", "v", "P")


def configure_line_plot_style() -> None:
    """Apply the repository-wide Matplotlib defaults for line figures."""

    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "lines.linewidth": 1.8,
            "axes.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def style_line_axis(axis: Axes) -> None:
    """Remove chart clutter and keep ticks only on the left and bottom."""

    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#1A1A1A")
        spine.set_linewidth(1.2)
    axis.tick_params(
        axis="both",
        which="both",
        direction="out",
        colors="#1A1A1A",
        width=1.0,
        length=4,
        top=False,
        right=False,
    )
    axis.xaxis.set_ticks_position("bottom")
    axis.yaxis.set_ticks_position("left")


def plot_line_series(
    axis: Axes,
    x_values: Sequence[float],
    y_values: Sequence[float],
    *,
    label: str,
    style_index: int,
    markevery: int | None = None,
    color: str | None = None,
    linestyle: object | None = None,
    marker: str | None = None,
) -> None:
    """Draw one consistently encoded series on ``axis``."""

    if len(x_values) == 0:
        raise ValueError(f"series {label!r} is empty")
    if len(x_values) != len(y_values):
        raise ValueError(f"series {label!r} has unequal x/y lengths")
    index = style_index % len(LINE_COLORS)
    axis.plot(
        x_values,
        y_values,
        label=label,
        color=color or LINE_COLORS[index],
        linestyle=linestyle or LINE_STYLES[index],
        marker=marker or LINE_MARKERS[index],
        markersize=3.8,
        markeredgewidth=0.0,
        markevery=markevery,
        linewidth=1.8,
    )


def add_left_legend(axis: Axes, *, location: str = "upper left") -> None:
    """Place a frameless, compact legend against the left edge."""

    axis.legend(
        loc=location,
        frameon=False,
        handlelength=2.6,
        handletextpad=0.7,
        borderaxespad=0.5,
    )


def save_figure(
    figure: Figure,
    output_stem: Path,
    *,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    dpi: int = 300,
) -> tuple[Path, ...]:
    """Save vector paper outputs plus a high-resolution preview."""

    output_stem = output_stem.expanduser().resolve()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for raw_format in formats:
        output_format = raw_format.lower().lstrip(".")
        if output_format not in {"pdf", "svg", "png"}:
            raise ValueError(f"unsupported output format: {raw_format}")
        output_path = output_stem.with_suffix(f".{output_format}")
        figure.savefig(
            output_path,
            dpi=dpi if output_format == "png" else None,
            bbox_inches="tight",
        )
        written.append(output_path)
    return tuple(written)
