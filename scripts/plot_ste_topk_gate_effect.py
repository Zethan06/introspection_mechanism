#!/usr/bin/env python3
"""Plot gate-off and gate-on outcome rates across STE Top-k sweeps.

An optional random-k control is drawn alongside each model's curve. It
patches the same number of heads drawn uniformly from the STE candidate
pool, so the gap between the two arms is what the selection buys over
patching any k heads at the same depth.

By default each model gets one panel carrying both gate directions. With
``--split_directions`` each model gets one panel per direction instead, so a
panel holds only the selected arm and the random arm it is compared against.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from introspection_core.plot_style import (  # noqa: E402
    configure_line_plot_style,
    plot_line_series,
    save_figure,
    style_line_axis,
)


GATE_OFF_COLOR = "#0072B2"
GATE_ON_COLOR = "#D62728"
# The random arm reuses each direction's hue so the comparison reads within a
# direction rather than across them.
RANDOM_ALPHA = 0.55
RANDOM_BAND_ALPHA = 0.16
# In a one-direction panel the two arms are what differ, not the direction, so
# the random arm drops the direction hue for a neutral grey.
SPLIT_RANDOM_COLOR = "#6E6E6E"
SPLIT_RANDOM_BAND_ALPHA = 0.22


@dataclass(frozen=True)
class SweepCurve:
    """The two requested intervention outcomes and their natural baselines."""

    label: str
    source: Path
    evaluation_split: str
    top_k: tuple[int, ...]
    gate_off_none: tuple[float, ...]
    gate_on_number: tuple[float, ...]
    clean_none_baseline: float
    injected_number_baseline: float


@dataclass(frozen=True)
class RandomControl:
    """Mean and interval of the random-k arm, over the k values it covers."""

    label: str
    source: Path
    n_draws: int
    top_k: tuple[int, ...]
    gate_off_none: tuple[float, ...]
    gate_off_low: tuple[float, ...]
    gate_off_high: tuple[float, ...]
    gate_on_number: tuple[float, ...]
    gate_on_low: tuple[float, ...]
    gate_on_high: tuple[float, ...]


@dataclass(frozen=True)
class Direction:
    """One gate direction, and where to read it on a curve and a control."""

    key: str
    title: str
    # A two-word form, for titles too narrow to carry the full phrase.
    short_title: str
    color: str
    marker: str
    # Attribute names, so a panel can be drawn from the direction alone.
    curve_field: str
    control_field: str
    control_low_field: str
    control_high_field: str
    baseline_field: str
    baseline_title: str
    short_baseline: str
    arm_title: str


DIRECTIONS: tuple[Direction, ...] = (
    Direction(
        key="gate_on",
        title="Gate on: Inject \u2192 Clean, output Number",
        short_title="Gate on",
        color=GATE_ON_COLOR,
        marker="s",
        curve_field="gate_on_number",
        control_field="gate_on_number",
        control_low_field="gate_on_low",
        control_high_field="gate_on_high",
        baseline_field="injected_number_baseline",
        baseline_title="original Inject output Number",
        short_baseline="Inject",
        arm_title="Output Number rate",
    ),
    Direction(
        key="gate_off",
        title="Gate off: Clean \u2192 Inject, output none",
        short_title="Gate off",
        color=GATE_OFF_COLOR,
        marker="o",
        curve_field="gate_off_none",
        control_field="gate_off_none",
        control_low_field="gate_off_low",
        control_high_field="gate_off_high",
        baseline_field="clean_none_baseline",
        baseline_title="Clean output none",
        short_baseline="Clean",
        arm_title="Output none rate",
    ),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="LABEL=CSV",
        help="Model label and transition_summary.csv; repeat for each model.",
    )
    parser.add_argument(
        "--random_input",
        action="append",
        default=[],
        metavar="LABEL=CSV",
        help="Model label and random_topk_transition_summary.csv; the label "
        "must match the --input label of the same model.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--evaluation_split",
        choices=("validation", "test"),
        default="test",
        help="Split represented by every supplied sweep summary.",
    )
    parser.add_argument(
        "--output_name", default="ste_topk_gate_effect", help="Output stem suffix."
    )
    parser.add_argument(
        "--hide_titles",
        action="store_true",
        help="Omit model titles for LaTeX subfigures with captions below each panel.",
    )
    parser.add_argument(
        "--individual_legend",
        choices=("all", "first", "none"),
        default="all",
        help="Choose which separately saved model panels contain a legend.",
    )
    parser.add_argument(
        "--individual_ylabel",
        choices=("all", "first", "none"),
        default="all",
        help="Choose which separately saved model panels contain a y-axis label.",
    )
    parser.add_argument(
        "--skip_overview",
        action="store_true",
        help="Write only the per-model panels, for LaTeX subfigures that "
        "assemble the grid themselves.",
    )
    parser.add_argument(
        "--split_directions",
        action="store_true",
        help="Draw one panel per gate direction, each comparing the Top-k "
        "selected heads against the random-k control, instead of one panel "
        "per model carrying both directions.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args(argv)


def _parse_inputs(values: Sequence[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    labels: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"--input must use LABEL=CSV: {value!r}")
        label, raw_path = value.split("=", 1)
        if not label or not raw_path or label in labels:
            raise ValueError(f"invalid or duplicate --input: {value!r}")
        labels.add(label)
        parsed.append((label, Path(raw_path)))
    return parsed


def _read_rate(row: dict[str, str], field: str, source: Path) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{source} has invalid {field}") from error
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{source} has out-of-range {field}: {value}")
    return value


def _read_bound(row: dict[str, str], field: str, source: Path) -> float:
    """Read one interval endpoint, clamped into the plottable range.

    A Student-t interval on a rate near 0 or 1 can fall outside [0, 1] — with
    ten draws it routinely does — so an endpoint is not itself a rate and is
    only required to be finite. The band is drawn on the rate axis, so the
    endpoint is clamped rather than rejected.
    """

    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{source} has invalid {field}") from error
    if not math.isfinite(value):
        raise ValueError(f"{source} has non-finite {field}: {value}")
    return min(max(value, 0.0), 1.0)


def _constant(values: Sequence[float], *, name: str, source: Path) -> float:
    if not values:
        raise ValueError(f"{source} contains no {name} values")
    first = values[0]
    if any(not math.isclose(value, first, abs_tol=1e-12) for value in values[1:]):
        raise ValueError(f"{source} contains inconsistent {name} values")
    return first


def load_curve(
    label: str,
    source: Path,
    *,
    evaluation_split: str = "test",
) -> SweepCurve:
    """Load one sweep and map its transition fields to requested output rates."""

    source = source.expanduser().resolve()
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty sweep summary: {source}")

    parsed_rows: list[tuple[int, float, float, float, float]] = []
    for row in rows:
        try:
            top_k = int(row["top_k"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{source} has invalid top_k") from error

        # test_off is an injected recipient patched with clean head outputs.
        gate_off_none = _read_rate(
            row, "number_to_none_target_accuracy_after", source
        )
        # test_on is a clean recipient patched with injected head outputs.
        gate_on_number = _read_rate(
            row, "none_to_number_target_accuracy_after", source
        )
        clean_none_baseline = 1.0 - _read_rate(
            row, "none_to_number_target_accuracy_before", source
        )
        injected_number_baseline = 1.0 - _read_rate(
            row, "number_to_none_target_accuracy_before", source
        )
        parsed_rows.append(
            (
                top_k,
                gate_off_none,
                gate_on_number,
                clean_none_baseline,
                injected_number_baseline,
            )
        )

    parsed_rows.sort(key=lambda item: item[0])
    top_k = tuple(item[0] for item in parsed_rows)
    if len(set(top_k)) != len(top_k):
        raise ValueError(f"{source} contains duplicate top_k values")
    # The swept k values are whatever the sweep was configured to run, so take
    # them from the summary rather than pinning one list here.
    if any(value <= 0 for value in top_k):
        raise ValueError(f"{source} has nonpositive top_k values: {top_k}")

    clean_none_baseline = _constant(
        [item[3] for item in parsed_rows],
        name="clean-none baseline",
        source=source,
    )
    injected_number_baseline = _constant(
        [item[4] for item in parsed_rows],
        name="injected-number baseline",
        source=source,
    )

    return SweepCurve(
        label=label,
        source=source,
        evaluation_split=evaluation_split,
        top_k=(0, *top_k),
        # At k=0 no head is patched, so each recipient remains in its
        # original state. These complements are the corresponding native
        # Inject-none and Clean-number output rates.
        gate_off_none=(
            1.0 - injected_number_baseline,
            *(item[1] for item in parsed_rows),
        ),
        gate_on_number=(
            1.0 - clean_none_baseline,
            *(item[2] for item in parsed_rows),
        ),
        clean_none_baseline=clean_none_baseline,
        injected_number_baseline=injected_number_baseline,
    )


def load_random_control(label: str, source: Path) -> RandomControl:
    """Load one random-k control summary written by the control evaluator."""

    source = source.expanduser().resolve()
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty random control summary: {source}")

    parsed: list[tuple[int, tuple[float, float, float], tuple[float, float, float]]] = []
    draw_counts: set[int] = set()
    for row in rows:
        try:
            top_k = int(row["top_k"])
            draw_counts.add(int(row["n_draws"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{source} has invalid top_k or n_draws") from error
        gate_off = (
            _read_rate(row, "number_to_none_target_accuracy_after_mean", source),
            *(
                _read_bound(
                    row, f"number_to_none_target_accuracy_after_{statistic}", source
                )
                for statistic in ("ci_low", "ci_high")
            ),
        )
        gate_on = (
            _read_rate(row, "none_to_number_target_accuracy_after_mean", source),
            *(
                _read_bound(
                    row, f"none_to_number_target_accuracy_after_{statistic}", source
                )
                for statistic in ("ci_low", "ci_high")
            ),
        )
        parsed.append((top_k, gate_off, gate_on))

    parsed.sort(key=lambda item: item[0])
    top_k = tuple(item[0] for item in parsed)
    if len(set(top_k)) != len(top_k):
        raise ValueError(f"{source} contains duplicate top_k values")
    if any(value <= 0 for value in top_k):
        raise ValueError(f"{source} has nonpositive top_k values: {top_k}")
    if len(draw_counts) != 1:
        raise ValueError(f"{source} mixes draw counts: {sorted(draw_counts)}")
    return RandomControl(
        label=label,
        source=source,
        n_draws=next(iter(draw_counts)),
        top_k=top_k,
        gate_off_none=tuple(item[1][0] for item in parsed),
        gate_off_low=tuple(item[1][1] for item in parsed),
        gate_off_high=tuple(item[1][2] for item in parsed),
        gate_on_number=tuple(item[2][0] for item in parsed),
        gate_on_low=tuple(item[2][1] for item in parsed),
        gate_on_high=tuple(item[2][2] for item in parsed),
    )


def _random_positions(curve: SweepCurve, control: RandomControl) -> list[int]:
    """Place the control's k values on the curve's categorical x axis.

    The control may cover fewer cardinalities than the sweep, so it is matched
    by k value rather than by position.
    """

    missing = [value for value in control.top_k if value not in curve.top_k]
    if missing:
        raise ValueError(
            f"{control.source} has k values outside the {curve.label} sweep: "
            f"{missing}"
        )
    return [curve.top_k.index(value) for value in control.top_k]


def _draw_random_control(
    axis: Axes,
    curve: SweepCurve,
    control: RandomControl,
    *,
    compact_legend: bool,
) -> None:
    """Overlay the random-k mean and its interval for both directions."""

    positions = _random_positions(curve, control)
    # At k=0 nothing is patched, so the random arm starts from the identical
    # recipient state and the two arms must meet there.
    zero_index = curve.top_k.index(0)
    series = (
        (
            "off",
            GATE_OFF_COLOR,
            curve.gate_off_none[zero_index],
            control.gate_off_none,
            control.gate_off_low,
            control.gate_off_high,
        ),
        (
            "on",
            GATE_ON_COLOR,
            curve.gate_on_number[zero_index],
            control.gate_on_number,
            control.gate_on_low,
            control.gate_on_high,
        ),
    )
    for direction, color, origin, mean, low, high in series:
        x_values = [zero_index, *positions]
        axis.fill_between(
            x_values,
            [origin, *low],
            [origin, *high],
            color=color,
            alpha=RANDOM_BAND_ALPHA,
            linewidth=0.0,
            label="_nolegend_",
            zorder=1,
        )
        axis.plot(
            x_values,
            [origin, *mean],
            color=color,
            alpha=RANDOM_ALPHA,
            linestyle=(0, (3, 1.6)),
            linewidth=1.4,
            marker="^",
            markersize=3.2,
            markeredgewidth=0.0,
            # Direction is already encoded by hue in the Top-k entries, so the
            # two random lines share one neutral legend entry below rather than
            # doubling the legend height.
            label="_nolegend_",
            zorder=1,
        )
    axis.plot(
        [],
        [],
        color="#6E6E6E",
        linestyle=(0, (3, 1.6)),
        linewidth=1.4,
        marker="^",
        markersize=3.2,
        markeredgewidth=0.0,
        label=(
            "Random-k"
            if compact_legend
            else f"Random-k: mean of {control.n_draws} draws, 95% CI"
        ),
    )


def _draw_curve(
    axis: Axes,
    curve: SweepCurve,
    *,
    show_legend: bool,
    show_title: bool = True,
    compact_legend: bool = False,
    random_control: RandomControl | None = None,
) -> None:
    split_label = curve.evaluation_split.capitalize()
    x_positions = tuple(range(len(curve.top_k)))
    plot_line_series(
        axis,
        x_positions,
        curve.gate_off_none,
        label="Gate off" if compact_legend else "Gate off: Clean → Inject, output none",
        style_index=0,
        color=GATE_OFF_COLOR,
        linestyle="-",
        marker="o",
    )
    plot_line_series(
        axis,
        x_positions,
        curve.gate_on_number,
        label="Gate on" if compact_legend else "Gate on: Inject → Clean, output Number",
        style_index=1,
        color=GATE_ON_COLOR,
        linestyle="-",
        marker="s",
    )
    axis.axhline(
        curve.clean_none_baseline,
        color=GATE_OFF_COLOR,
        linestyle=(0, (1.5, 1.5)),
        linewidth=1.15,
        alpha=0.8,
        label=(
            "_nolegend_"
            if compact_legend
            else f"{split_label} Clean output none ({curve.clean_none_baseline:.1%})"
        ),
        zorder=0,
    )
    axis.axhline(
        curve.injected_number_baseline,
        color=GATE_ON_COLOR,
        linestyle=(0, (5, 2, 1, 2)),
        linewidth=1.15,
        alpha=0.8,
        label=(
            "_nolegend_"
            if compact_legend
            else f"{split_label} original Inject output Number "
            f"({curve.injected_number_baseline:.1%})"
        ),
        zorder=0,
    )
    if random_control is not None:
        _draw_random_control(
            axis, curve, random_control, compact_legend=compact_legend
        )
    if show_title:
        axis.set_title(curve.label)
    axis.set_xlim(-0.35, len(curve.top_k) - 0.65)
    axis.set_xticks(x_positions, labels=[str(value) for value in curve.top_k])
    axis.set_ylim(0.0, 1.01)
    axis.set_yticks([index / 5 for index in range(6)])
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    style_line_axis(axis)
    if show_legend:
        legend = axis.legend(
            loc="upper left" if compact_legend else "best",
            frameon=True,
            fontsize=8.2 if compact_legend else 6.4,
            handlelength=2.0 if compact_legend else 2.7,
            handletextpad=0.6,
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("none")
        legend.get_frame().set_alpha(0.9)


def _draw_direction_panel(
    axis: Axes,
    curve: SweepCurve,
    direction: Direction,
    *,
    random_control: RandomControl | None,
    show_legend: bool,
    show_title: bool,
    title: str | None = None,
) -> None:
    """Draw one gate direction: the selected arm against the random arm.

    Only one direction lives on the panel, so the two lines differ by which
    heads were patched rather than by which direction was measured. That is the
    comparison the random control exists to make.
    """

    x_positions = tuple(range(len(curve.top_k)))
    selected = getattr(curve, direction.curve_field)
    plot_line_series(
        axis,
        x_positions,
        selected,
        label="Selected $k$ heads",
        style_index=0,
        color=direction.color,
        linestyle="-",
        marker=direction.marker,
    )
    if random_control is not None:
        positions = _random_positions(curve, random_control)
        # At k=0 nothing is patched, so both arms start from the identical
        # recipient state and the band must close to a point there.
        zero_index = curve.top_k.index(0)
        origin = selected[zero_index]
        x_values = [zero_index, *positions]
        mean = getattr(random_control, direction.control_field)
        low = getattr(random_control, direction.control_low_field)
        high = getattr(random_control, direction.control_high_field)
        axis.fill_between(
            x_values,
            [origin, *low],
            [origin, *high],
            color=SPLIT_RANDOM_COLOR,
            alpha=SPLIT_RANDOM_BAND_ALPHA,
            linewidth=0.0,
            label=f"Random $k$ (95% CI, {random_control.n_draws} draws)",
            zorder=1,
        )
        axis.plot(
            x_values,
            [origin, *mean],
            color=SPLIT_RANDOM_COLOR,
            linestyle=(0, (3, 1.6)),
            linewidth=1.6,
            marker="^",
            markersize=3.6,
            markeredgewidth=0.0,
            label="Random $k$ (mean)",
            zorder=2,
        )
    baseline = getattr(curve, direction.baseline_field)
    axis.axhline(
        baseline,
        color=direction.color,
        linestyle=(0, (1.5, 1.5)),
        linewidth=1.15,
        alpha=0.8,
        label=f"Unpatched {direction.short_baseline} ({baseline:.1%})",
        zorder=0,
    )
    if show_title:
        axis.set_title(title if title is not None else direction.title)
    axis.set_xlim(-0.35, len(curve.top_k) - 0.65)
    axis.set_xticks(x_positions, labels=[str(value) for value in curve.top_k])
    axis.set_ylim(0.0, 1.01)
    axis.set_yticks([index / 5 for index in range(6)])
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    style_line_axis(axis)
    if show_legend:
        legend = axis.legend(
            # Where a panel leaves room depends on where its curves rise, so
            # the placement is left to Matplotlib.
            loc="best",
            frameon=True,
            fontsize=5.8,
            handlelength=2.0,
            handletextpad=0.5,
            borderpad=0.35,
            labelspacing=0.35,
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("none")
        legend.get_frame().set_alpha(0.9)


def build_direction_figure(
    curve: SweepCurve,
    direction: Direction,
    *,
    random_control: RandomControl | None = None,
    show_title: bool = True,
    show_legend: bool = True,
    show_ylabel: bool = True,
):
    """Build one single-direction panel for a single model."""

    configure_line_plot_style()
    figure, axis = plt.subplots(figsize=(3.2, 2.4), constrained_layout=True)
    _draw_direction_panel(
        axis,
        curve,
        direction,
        random_control=random_control,
        show_legend=show_legend,
        show_title=show_title,
        title=f"{curve.label} \u2014 {direction.short_title}",
    )
    axis.set_xlabel("Number of patched heads ($k$)")
    if show_ylabel:
        axis.set_ylabel(direction.arm_title)
    return figure


def build_split_overview_figure(
    curves: Sequence[SweepCurve],
    *,
    random_controls: dict[str, RandomControl] | None = None,
    show_titles: bool = True,
):
    """Build a direction-per-row, model-per-column grid of single-arm panels."""

    if not curves:
        raise ValueError("cannot plot an empty set of curves")
    configure_line_plot_style()
    n_rows, n_columns = len(DIRECTIONS), len(curves)
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(3.4 * n_columns, 2.6 * n_rows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    controls = random_controls or {}
    for row, direction in enumerate(DIRECTIONS):
        for column, curve in enumerate(curves):
            axis = axes[row][column]
            _draw_direction_panel(
                axis,
                curve,
                direction,
                random_control=controls.get(curve.label),
                show_legend=False,
                # The direction is the row, named by the y label, so only the
                # top row needs a title and it names the model alone.
                show_title=show_titles and row == 0,
                title=curve.label,
            )
            if row == n_rows - 1:
                axis.set_xlabel("Number of patched heads ($k$)")
            if column == 0:
                # Only the grid needs the direction on the axis; a standalone
                # panel gets it from its subfigure caption.
                axis.set_ylabel(f"{direction.short_title}: {direction.arm_title}")
            if not show_titles:
                panel = chr(ord("a") + row * n_columns + column)
                axis.text(
                    0.5,
                    -0.30,
                    f"({panel}) {curve.label}, {direction.short_title}",
                    transform=axis.transAxes,
                    ha="center",
                    va="top",
                    fontsize=8,
                )
    # Each row carries its own direction hue and baseline, so one shared legend
    # would have to name both; the per-row legend is drawn on the first column.
    for row, direction in enumerate(DIRECTIONS):
        handles, labels = axes[row][0].get_legend_handles_labels()
        # The baseline value differs by model, so the grid legend names its role.
        labels[-1] = f"Unpatched {direction.short_baseline} (baseline)"
        legend = axes[row][0].legend(
            handles,
            labels,
            # Panels differ in where they leave room, so let Matplotlib place it.
            loc="best",
            frameon=True,
            fontsize=6.2,
            handlelength=2.4,
            handletextpad=0.6,
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("none")
        legend.get_frame().set_alpha(0.9)
    figure.subplots_adjust(
        bottom=0.20 if show_titles else 0.26, hspace=0.34, wspace=0.18
    )
    return figure


def build_model_figure(
    curve: SweepCurve,
    *,
    show_title: bool = True,
    show_legend: bool = True,
    show_ylabel: bool = True,
    compact_legend: bool = False,
    random_control: RandomControl | None = None,
):
    """Build one repository-standard figure for a single model."""

    configure_line_plot_style()
    figure, axis = plt.subplots(figsize=(3.0, 2.25), constrained_layout=True)
    _draw_curve(
        axis,
        curve,
        show_legend=show_legend,
        show_title=show_title,
        compact_legend=compact_legend,
        random_control=random_control,
    )
    axis.set_xlabel("Number of patched heads (k)")
    if show_ylabel:
        axis.set_ylabel("Output rate")
    return figure


def build_overview_figure(
    curves: Sequence[SweepCurve],
    *,
    show_titles: bool = True,
    random_controls: dict[str, RandomControl] | None = None,
):
    """Build a compact multi-model overview with shared scales."""

    if not curves:
        raise ValueError("cannot plot an empty set of curves")
    configure_line_plot_style()
    n_columns = 3 if len(curves) == 3 else (2 if len(curves) > 1 else 1)
    n_rows = math.ceil(len(curves) / n_columns)
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(10.2 if n_columns == 3 else (7.2 if n_columns == 2 else 4.1),
                 2.75 * n_rows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    flat_axes = axes.ravel()
    for index, (axis, curve) in enumerate(zip(flat_axes, curves, strict=False)):
        _draw_curve(
            axis,
            curve,
            show_legend=False,
            show_title=show_titles,
            random_control=(random_controls or {}).get(curve.label),
        )
        if index // n_columns == n_rows - 1:
            axis.set_xlabel("Number of patched heads (k)")
        if index % n_columns == 0:
            axis.set_ylabel("Output rate")
        if not show_titles:
            panel = chr(ord("a") + index)
            axis.text(
                0.5,
                -0.31,
                f"({panel}) {curve.label}",
                transform=axis.transAxes,
                ha="center",
                va="top",
                fontsize=9,
            )
    for axis in flat_axes[len(curves) :]:
        axis.set_visible(False)

    handles, labels = flat_axes[0].get_legend_handles_labels()
    # Baseline values vary by model, so the overview legend names their role only.
    split_labels = {curve.evaluation_split for curve in curves}
    if len(split_labels) != 1:
        raise ValueError("all curves must use the same evaluation split")
    split_label = next(iter(split_labels)).capitalize()
    labels[2] = f"{split_label} Clean output none (baseline)"
    labels[3] = f"{split_label} original Inject output Number (baseline)"
    legend_columns = 2
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=legend_columns,
        frameon=False,
        fontsize=7.6,
        handlelength=2.8,
    )
    # The random-k entry adds a legend row, which has to come out of the axes
    # rather than overlap them.
    legend_rows = math.ceil(len(labels) / legend_columns)
    bottom = (0.34 if not show_titles and n_rows == 1 else 0.16) + 0.20 * (
        legend_rows - 2
    ) / n_rows
    figure.subplots_adjust(bottom=bottom, hspace=0.34, wspace=0.22)
    return figure


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def write_plot_data(
    curves: Sequence[SweepCurve],
    output_path: Path,
    *,
    random_controls: dict[str, RandomControl] | None = None,
) -> Path:
    """Write the exact plotted values and their source paths."""

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    controls = random_controls or {}
    fieldnames = [
        "model",
        "top_k",
        "gate_off_clean_to_inject_none_rate",
        "gate_on_inject_to_clean_number_rate",
        "random_gate_off_none_rate_mean",
        "random_gate_off_none_rate_ci_low",
        "random_gate_off_none_rate_ci_high",
        "random_gate_on_number_rate_mean",
        "random_gate_on_number_rate_ci_low",
        "random_gate_on_number_rate_ci_high",
        "random_n_draws",
        "random_source",
        "evaluation_split",
        "clean_none_baseline",
        "original_inject_number_baseline",
        "source",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for curve in curves:
            control = controls.get(curve.label)
            # The control may cover fewer cardinalities than the sweep, so its
            # cells are looked up by k and left blank where it has none.
            random_by_k = (
                {
                    value: index
                    for index, value in enumerate(control.top_k)
                }
                if control is not None
                else {}
            )
            for top_k, gate_off, gate_on in zip(
                curve.top_k,
                curve.gate_off_none,
                curve.gate_on_number,
                strict=True,
            ):
                random_index = random_by_k.get(top_k)
                random_fields: dict[str, object] = {
                    "random_gate_off_none_rate_mean": "",
                    "random_gate_off_none_rate_ci_low": "",
                    "random_gate_off_none_rate_ci_high": "",
                    "random_gate_on_number_rate_mean": "",
                    "random_gate_on_number_rate_ci_low": "",
                    "random_gate_on_number_rate_ci_high": "",
                    "random_n_draws": "" if control is None else control.n_draws,
                    "random_source": "" if control is None else str(control.source),
                }
                if control is not None and random_index is not None:
                    random_fields.update(
                        {
                            "random_gate_off_none_rate_mean": (
                                control.gate_off_none[random_index]
                            ),
                            "random_gate_off_none_rate_ci_low": (
                                control.gate_off_low[random_index]
                            ),
                            "random_gate_off_none_rate_ci_high": (
                                control.gate_off_high[random_index]
                            ),
                            "random_gate_on_number_rate_mean": (
                                control.gate_on_number[random_index]
                            ),
                            "random_gate_on_number_rate_ci_low": (
                                control.gate_on_low[random_index]
                            ),
                            "random_gate_on_number_rate_ci_high": (
                                control.gate_on_high[random_index]
                            ),
                        }
                    )
                writer.writerow(
                    {
                        "model": curve.label,
                        "top_k": top_k,
                        "gate_off_clean_to_inject_none_rate": gate_off,
                        "gate_on_inject_to_clean_number_rate": gate_on,
                        **random_fields,
                        "evaluation_split": curve.evaluation_split,
                        "clean_none_baseline": curve.clean_none_baseline,
                        "original_inject_number_baseline": (
                            curve.injected_number_baseline
                        ),
                        "source": str(curve.source),
                    }
                )
    return output_path


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    curves = [
        load_curve(label, source, evaluation_split=args.evaluation_split)
        for label, source in _parse_inputs(args.input)
    ]
    random_controls = {
        label: load_random_control(label, source)
        for label, source in _parse_inputs(args.random_input)
    }
    unknown = sorted(set(random_controls) - {curve.label for curve in curves})
    if unknown:
        raise ValueError(
            f"--random_input labels have no matching --input: {unknown}"
        )
    for curve in curves:
        control = random_controls.get(curve.label)
        if control is not None:
            # Fail here rather than silently dropping points from the figure.
            _random_positions(curve, control)
    # The overview shares one categorical x axis across models.
    grids = {curve.top_k for curve in curves}
    if len(grids) != 1:
        raise ValueError(
            "every --input must sweep the same top_k values; got "
            + "; ".join(f"{curve.label}={curve.top_k[1:]}" for curve in curves)
        )
    output_paths: list[Path] = []
    for index, curve in enumerate(curves):
        show_legend = args.individual_legend == "all" or (
            args.individual_legend == "first" and index == 0
        )
        show_ylabel = args.individual_ylabel == "all" or (
            args.individual_ylabel == "first" and index == 0
        )
        if args.split_directions:
            for direction in DIRECTIONS:
                figure = build_direction_figure(
                    curve,
                    direction,
                    random_control=random_controls.get(curve.label),
                    show_title=not args.hide_titles,
                    show_legend=show_legend,
                    show_ylabel=show_ylabel,
                )
                output_paths.extend(
                    save_figure(
                        figure,
                        args.output_dir
                        / f"{_slug(curve.label)}_{args.output_name}"
                        f"_{direction.key}",
                        dpi=args.dpi,
                    )
                )
                plt.close(figure)
            continue
        figure = build_model_figure(
            curve,
            show_title=not args.hide_titles,
            show_legend=show_legend,
            show_ylabel=show_ylabel,
            compact_legend=args.individual_legend == "first",
            random_control=random_controls.get(curve.label),
        )
        output_paths.extend(
            save_figure(
                figure,
                args.output_dir / f"{_slug(curve.label)}_{args.output_name}",
                dpi=args.dpi,
            )
        )
        plt.close(figure)

    if args.skip_overview:
        overview = None
    elif args.split_directions:
        overview = build_split_overview_figure(
            curves,
            random_controls=random_controls,
            show_titles=not args.hide_titles,
        )
    else:
        overview = build_overview_figure(
            curves,
            show_titles=not args.hide_titles,
            random_controls=random_controls,
        )
    if overview is not None:
        output_paths.extend(
            save_figure(
                overview,
                args.output_dir / f"all_models_{args.output_name}",
                dpi=args.dpi,
            )
        )
        plt.close(overview)
    output_paths.append(
        write_plot_data(
            curves,
            args.output_dir / f"{args.output_name}_data.csv",
            random_controls=random_controls,
        )
    )
    print("wrote " + ", ".join(str(path) for path in output_paths), flush=True)


if __name__ == "__main__":
    main()
