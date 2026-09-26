"""Density ridges with shared score coordinates across layers."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import gaussian_kde

from .plot_style import configure_line_plot_style


CLASSES = ((False, "None", "#D55E00", "--"), (True, "Number", "#0072B2", "-"))


def _legend(axis) -> None:
    axis.legend(handles=[Line2D([], [], color=c, ls=s, label=n)
                         for _, n, c, s in CLASSES],
                frameon=False, ncols=2, loc="upper left")


def plot_density_ridges(layers: np.ndarray, scores: np.ndarray,
                        labels: np.ndarray, *, selected_layers: list[int], title: str,
                        axis: Axes | None = None, decorations: bool = True,
                        classes=CLASSES, equal_area: bool = False):
    """Overlay KDEs; share bandwidth and peak scale within each layer."""
    count = len(selected_layers)
    if axis is None:
        configure_line_plot_style()
        figure, axis = plt.subplots(figsize=(5.7, 1.35 + .45 * count), constrained_layout=True)
    else:
        figure = axis.figure
    details = []
    for row, layer in enumerate(selected_layers):
        offset = list(layers).index(layer)
        groups = [scores[labels == code, offset] for code, *_ in classes]
        groups = [v[np.isfinite(v)] for v in groups]
        if any(len(v) < 2 for v in groups):
            raise ValueError(f"layer {layer} needs two finite observations per class")
        pooled = np.concatenate(groups)
        bandwidth = float(np.std(pooled, ddof=1) * len(pooled) ** (-.2))
        baseline = count - row - 1
        if bandwidth == 0:
            # A zero training direction yields identical scores. Show a point
            # mass instead of inventing variation to make KDE invertible.
            location = float(pooled[0])
            axis.hlines(baseline, 0, 1, color="#DDDDDD", lw=.6)
            axis.vlines(location, baseline, baseline + .78, color="#666666", lw=1)
            axis.text(location + .025, baseline + .2, "identical scores", fontsize=6, color="#666666")
            details.append({"layer": layer, "bandwidth": 0.,
                            "shared_density_peak": None, "density_areas": None,
                            "point_mass": location,
                            "grid_limits": [location - .04, location + .04]})
            continue
        # Adaptive grid resolves narrow early-layer peaks without widening them.
        grid = np.linspace(pooled.min() - 4 * bandwidth,
                           pooled.max() + 4 * bandwidth, 900)
        densities = [gaussian_kde(v, bw_method=bandwidth / np.std(v, ddof=1))(grid)
                     if np.std(v, ddof=1) > 0 else
                     np.exp(-.5 * ((grid - v[0]) / bandwidth) ** 2) / (bandwidth * np.sqrt(2 * np.pi))
                     for v in groups]
        integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        if equal_area:
            densities = [d / integrate(d, grid) for d in densities]
        peak = max(d.max() for d in densities)
        axis.hlines(baseline, 0, 1, color="#DDDDDD", lw=.6)
        for density, (_, _, color, style) in zip(densities, classes):
            height = .78 * density / peak
            axis.fill_between(grid, baseline, baseline + height,
                              color=color, alpha=.22, linewidth=0)
            axis.plot(grid, baseline + height, color=color, ls=style, lw=1.3)
        details.append({"layer": layer, "bandwidth": bandwidth,
                        "shared_density_peak": float(peak),
                        "density_areas": [float(integrate(d, grid)) for d in densities],
                        "grid_limits": [float(grid[0]), float(grid[-1])]})
    axis.set(xlim=(-.04, 1.04), ylim=(-.15, count + .5), title=title)
    if equal_area:
        axis.set_xlim(min(d["grid_limits"][0] for d in details),
                      max(d["grid_limits"][1] for d in details))
    axis.set_yticks(np.arange(count), [str(v) for v in selected_layers[::-1]])
    axis.set_xticks(np.linspace(0, 1, 6))
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    if decorations:
        axis.set(xlabel="Normalized projection", ylabel="Residual stream layer")
        _legend(axis)
        axis.text(.98, .97, "Density height scaled within each layer",
                  transform=axis.transAxes, ha="right", va="top", fontsize=7)
    return figure, details
