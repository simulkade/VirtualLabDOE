"""Shared plotting helpers, centered on the project's visual contract:
*Mechanistic Truth vs Virtual Experiment* (and, where relevant, the recovered
or optimized estimate).
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


# ── Default styling ────────────────────────────────────────────────────────
_TRUTH_STYLE = dict(color="black", lw=2.0, zorder=3)
_OBS_STYLE = dict(color="steelblue", alpha=0.65, s=12, zorder=2)
_EST_STYLE = dict(color="crimson", lw=1.8, ls="--", zorder=4)
_BAND_STYLE = dict(color="crimson", alpha=0.15)


def plot_truth_vs_experiment(
    x: np.ndarray,
    truth: np.ndarray,
    observed: np.ndarray,
    *,
    estimate: np.ndarray | None = None,
    uncertainty_band: np.ndarray | None = None,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    ax: Axes | None = None,
) -> Axes:
    """Overlay the mechanistic truth, the noisy observation, and an optional estimate.

    Parameters
    ----------
    x:
        Shared abscissa (e.g. time or volume).
    truth:
        Noise-free model output ("Mechanistic Truth").
    observed:
        Perturbed output ("Virtual Experiment").
    estimate:
        Optional fitted/optimized curve (e.g. inverse-modeling result).
    uncertainty_band:
        Optional ±1σ half-width around *estimate* for uncertainty shading.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    ax.plot(x, truth, label="Mechanistic Truth", **_TRUTH_STYLE)
    ax.scatter(x, observed, label="Virtual Experiment", **_OBS_STYLE)

    if estimate is not None:
        ax.plot(x, estimate, label="Estimate / Fitted", **_EST_STYLE)
        if uncertainty_band is not None:
            ax.fill_between(
                x,
                estimate - uncertainty_band,
                estimate + uncertainty_band,
                label="±1σ epistemic",
                **_BAND_STYLE,
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(framealpha=0.8)
    ax.grid(True, alpha=0.3)
    return ax


def plot_response_surface(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    response: np.ndarray,
    *,
    factor_names: Sequence[str] = ("factor 1", "factor 2"),
    response_name: str = "response",
    ax: Axes | None = None,
) -> Axes:
    """Filled contour of a 2-factor response surface (Phases 2 & 4).

    Parameters
    ----------
    grid_x, grid_y:
        2-D coordinate grids (output of :func:`numpy.meshgrid`).
    response:
        2-D response values on the same grid.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    cs = ax.contourf(grid_x, grid_y, response, levels=25, cmap="viridis")
    plt.colorbar(cs, ax=ax, label=response_name)
    ax.contour(grid_x, grid_y, response, levels=25, colors="white", linewidths=0.4, alpha=0.4)
    ax.set_xlabel(factor_names[0])
    ax.set_ylabel(factor_names[1])
    return ax


def plot_pareto_effects(effects: dict[str, float], *, ax: Axes | None = None) -> Axes:
    """Horizontal Pareto bar chart of effect magnitudes (Phase 2)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, max(3, len(effects) * 0.4)))

    names = list(effects.keys())
    vals = [abs(v) for v in effects.values()]
    order = np.argsort(vals)
    ax.barh([names[i] for i in order], [vals[i] for i in order], color="steelblue")
    ax.set_xlabel("|Effect|")
    ax.grid(axis="x", alpha=0.3)
    return ax


def plot_chromatogram(
    t: np.ndarray,
    c_outlet: np.ndarray,
    *,
    t_pool_start: float | None = None,
    t_pool_end: float | None = None,
    ylabel: str = "concentration (g/L)",
    ax: Axes | None = None,
) -> Axes:
    """Plot a single-component outlet chromatogram with optional pool shading."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    ax.plot(t, c_outlet, "k-", lw=1.5)
    if t_pool_start is not None and t_pool_end is not None:
        mask = (t >= t_pool_start) & (t <= t_pool_end)
        ax.fill_between(t, 0, c_outlet, where=mask, alpha=0.3, color="green", label="Pool")
    ax.set_xlabel("time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return ax
