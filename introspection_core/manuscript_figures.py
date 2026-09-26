"""Compact, shared typography for the manuscript's empirical figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from .distribution_figures import plot_density_ridges


BLUE = "#0072B2"
ORANGE = "#D55E00"


def configure_manuscript_style() -> None:
    """Use readable Times-compatible text at the final 5.5-inch width."""
    plt.rcParams.update({
        "font.family": "Liberation Serif", "font.size": 8,
        "mathtext.fontset": "stix", "axes.labelsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": .6, "pdf.fonttype": 42,
        "svg.fonttype": "none", "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def draw_layerwise_panels(panels: list[dict], *, width: float = 5.5):
    """Align model columns, preserving the supplied score groups and KDE rule."""
    fig = plt.figure(figsize=(width, 3.75))
    grid = fig.add_gridspec(2, 3, left=.085, right=.99, bottom=.105,
                           top=.84, hspace=.55, wspace=.28,
                           height_ratios=[1.6, 1])
    details = []
    classes = ((False, "None report", ORANGE, "--"),
               (True, "Position report", BLUE, "-"))
    fig.text(.085, .985, "(a) Response-group separation", va="top", weight="bold")
    fig.legend(handles=[Line2D([], [], color=c, ls=s, label=n)
                        for _, n, c, s in classes],
               loc="upper right", bbox_to_anchor=(1, 1.002),
               ncol=2, frameon=False, handlelength=1.6,
               columnspacing=.9, handletextpad=.4)
    for col, panel in enumerate(panels):
        ax = fig.add_subplot(grid[0, col])
        _, kde = plot_density_ridges(
            panel["layers"], panel["scores"], panel["labels"],
            selected_layers=panel["ridge_layers"], title="",
            axis=ax, decorations=False, classes=classes)
        details.append(kde)
        ax.set_ylim(-.15, len(panel["ridge_layers"]) - .05)
        ax.set_xticks([0, .5, 1], ["0", "0.5", "1"])
        ax.tick_params(length=2, pad=2)
        ax.set_title(panel["title"], fontsize=8.5, pad=8)
        for line in ax.lines:
            line.set_linewidth(.85)
        if col == 0:
            ax.set_ylabel("Layer", labelpad=3)
        if col == 1:
            ax.set_xlabel("Normalized projection", labelpad=2)

        ax = fig.add_subplot(grid[1, col])
        x, y = panel["accuracy_layers"], panel["accuracy"]
        ax.plot(x, y, color=BLUE, linewidth=1.1)
        ax.axhline(10, color="#999999", ls=":", lw=.8)
        layer = panel["marked_layer"]
        value = float(y[list(x).index(layer)])
        ax.axvline(layer, color="#777777", ls="--", lw=.6, zorder=0)
        ax.plot(layer, value, marker="*", markersize=7,
                color=ORANGE, markeredgecolor="white", markeredgewidth=.35)
        ax.annotate(f"L{layer}: {value:.1f}%", (layer, value),
                    xytext=(-3, -14), textcoords="offset points",
                    ha="right", va="top", fontsize=8)
        ax.set(xlim=(min(x), max(x)), ylim=(0, 103), xlabel="Layer")
        ax.set_xticks([0, 10, 20, 30] if max(x) < 40 else [0, 15, 30, 45])
        ax.set_yticks([0, 50, 100])
        if col == 0:
            ax.set_ylabel("Accuracy (%)", labelpad=3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, color="#E6E6E6", linewidth=.5)
        ax.tick_params(length=2, pad=2)
    lower_top = fig.axes[1].get_position().y1
    fig.text(.085, lower_top + .035, "(b) Position-clustering accuracy",
             weight="bold", va="bottom")
    return fig, details


def draw_intervention_panels(models: list[str], values: np.ndarray,
                             factorial: np.ndarray, *, width: float = 5.5,
                             height: float = 2.2):
    """Draw gate transfer and gate-by-router factorial panels in one row.

    ``values`` is [direction, model, before/gate/target] and ``factorial`` is
    [direction, model, unmodified/router/gate/gate+router], direction 0 being
    the injected run (gate off) and 1 the clean run (gate on).
    """
    values = np.asarray(values, dtype=float)
    factorial = np.asarray(factorial, dtype=float)
    for name, array, cells in (("values", values, 3), ("factorial", factorial, 4)):
        if array.shape != (2, len(models), cells) or not np.isfinite(array).all():
            raise ValueError(f"expected finite [2, model, {cells}] {name} percentages")
        if np.any((array < 0) | (array > 100)):
            raise ValueError(f"{name} percentages must be in [0, 100]")
    # Shared bar grammar: gray keeps the run's own gate, orange swaps the gate,
    # hatching also swaps the router. Each panel group carries its own labels.
    unmodified = dict(facecolor="#D5DADF", edgecolor="#66717C", linewidth=.5)
    router = dict(unmodified, hatch="////")
    gate = dict(facecolor=ORANGE, edgecolor=ORANGE, linewidth=.5)
    both = dict(facecolor="white", edgecolor=ORANGE, linewidth=.9, hatch="////")
    target = dict(facecolor="white", edgecolor="#333333", linewidth=.8)
    cross = "\u00d7"
    panels = [
        ("(a) Gate off", "None (%)", values[0], [unmodified, gate, target], None),
        ("(b) Gate on", "Position (%)", values[1], [unmodified, gate, target], None),
        ("(c) Clean run", "Position (%)", factorial[1], [unmodified, router, gate, both],
         ["Clean run", f"Gate clean {cross} router injected",
          f"Gate injected {cross} router clean", f"Gate injected {cross} router injected"]),
        ("(d) Injected run", "None (%)", factorial[0], [unmodified, router, gate, both],
         ["Injected run", f"Gate injected {cross} router clean",
          f"Gate clean {cross} router injected", f"Gate clean {cross} router clean"]),
    ]
    plt.rcParams["hatch.linewidth"] = .6
    fig = plt.figure(figsize=(width, height))
    left, right, gap, bar = .066, .997, .062, .215
    bottom, top, legend_bottom = .175, .745, .83
    w = (right - left - gap * (len(panels) - 1)) / len(panels)
    lefts = [left + i * (w + gap) for i in range(len(panels))]
    legend_style = dict(frameon=False, handlelength=1.3, handleheight=.8,
                        handletextpad=.4, labelspacing=.22, fontsize=6.4,
                        borderaxespad=0, borderpad=0)
    # (a,b) share one legend flush with the left edge; (c) and (d) each label
    # their own four cells. All legends sit on one baseline above the titles.
    fig.legend(handles=[Patch(**style, label=label) for style, label in
                        ((unmodified, "Before patching"), (gate, "Gate patch"),
                         (target, "Target run"))],
               loc="lower left", bbox_to_anchor=(lefts[0] - .02, legend_bottom),
               **legend_style)
    for (title, ylabel, data, styles, labels), x0 in zip(panels, lefts):
        ax = fig.add_axes([x0, bottom, w, top - bottom])
        if labels is not None:
            fig.legend(handles=[Patch(**style, label=label)
                                for style, label in zip(styles, labels)],
                       loc="lower left", bbox_to_anchor=(x0 - .02, legend_bottom),
                       **{**legend_style, "fontsize": 5.7, "handlelength": 1.1,
                          "handleheight": .7, "labelspacing": .18})
        k = len(styles)
        group = 5 * bar
        x = np.arange(len(models)) * group
        for j, style in enumerate(styles):
            bars = ax.bar(x + (j - (k - 1) / 2) * bar, data[:, j], width=bar * .93,
                          zorder=3, **style)
            ax.bar_label(bars, fmt="%.1f", fontsize=5.6, padding=1.5, rotation=90,
                         weight="normal" if style is unmodified or style is target
                         else "bold")
        ax.set_title(title, loc="left", fontsize=7.8, pad=2)
        ax.set(xlim=(-group / 2, x[-1] + group / 2), ylim=(0, 118))
        ax.set_yticks([0, 50, 100])
        ax.set_ylabel(ylabel, labelpad=2, fontsize=7.5)
        ax.set_xticks(x, models, fontsize=7)
        ax.tick_params(length=2, pad=2, labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color="#E6E6E6", linewidth=.5)
    return fig
