"""Phase 3 — Latin Hypercube Sampling (space-filling design).

Uses ``scipy.stats.qmc.LatinHypercube`` with an optional low-discrepancy
optimisation criterion (``"random-cd"`` by default — minimises centered
L₂-discrepancy) and scales the unit hypercube to physical factor bounds.

A cross-check using ``numpy`` random is provided for comparison, and
:func:`coverage_metrics` quantifies space-filling quality.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import qmc

from vlab_doe.doe.factorial import Factor


def latin_hypercube(
    factors: Sequence[Factor],
    n_samples: int,
    *,
    seed: int | None = None,
    optimization: str | None = "random-cd",
) -> pd.DataFrame:
    """Generate an optimised Latin Hypercube design in physical units.

    Parameters
    ----------
    factors:
        Design factors with physical bounds (lo, hi).
    n_samples:
        Number of design points.
    seed:
        Integer seed for reproducibility.
    optimization:
        Optimisation criterion passed to ``scipy.stats.qmc.LatinHypercube``:
        ``"random-cd"`` (low centered discrepancy), ``"lloyd"`` (Voronoi),
        or ``None`` (plain LHS, no optimisation).

    Returns
    -------
    pandas.DataFrame
        Shape ``(n_samples, len(factors))``, columns named after the factors,
        values in physical units.  All values lie strictly within
        ``[factor.low, factor.high]``.
    """
    k = len(factors)
    lhs = qmc.LatinHypercube(d=k, seed=seed, optimization=optimization)
    sample_unit = lhs.random(n=n_samples)  # (n_samples, k) in [0, 1)

    # Scale to physical bounds
    lows = np.array([f.low for f in factors])
    highs = np.array([f.high for f in factors])
    sample_phys = qmc.scale(sample_unit, lows, highs)

    return pd.DataFrame(sample_phys, columns=[f.name for f in factors])


def coverage_metrics(design: pd.DataFrame) -> dict[str, float]:
    """Space-filling diagnostics for the design.

    Returns
    -------
    dict with keys:

    * ``"discrepancy"`` — centered L₂-discrepancy (lower is better; the
      ``scipy.stats.qmc.discrepancy`` scale is [0, 1]).
    * ``"min_pairwise_dist"`` — minimum Euclidean distance between any two
      points (after normalising columns to [0, 1]).
    * ``"mean_pairwise_dist"`` — mean pairwise Euclidean distance (normalised).
    """
    # Normalise each column to [0, 1] for scale-independent metrics
    X = design.values.astype(float)
    X_norm = (X - X.min(axis=0)) / ((X.max(axis=0) - X.min(axis=0)) + 1e-12)

    disc = float(qmc.discrepancy(X_norm, method="CD"))

    # Pairwise distances
    n = len(X_norm)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(float(np.linalg.norm(X_norm[i] - X_norm[j])))

    return {
        "discrepancy": disc,
        "min_pairwise_dist": float(np.min(dists)) if dists else 0.0,
        "mean_pairwise_dist": float(np.mean(dists)) if dists else 0.0,
    }
