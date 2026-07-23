"""Chromatographic performance and separation metrics."""

from __future__ import annotations

import numpy as np


def pool_metrics(
    t: np.ndarray,
    c_outlet: np.ndarray,
    *,
    cut_start: float,
    cut_end: float,
    target_index: int = 0,
) -> dict[str, float]:
    """Compute **yield / purity / productivity** for a pool between cut-points.

    The pool is the fraction of the outlet chromatogram between *cut_start* and
    *cut_end* (both in seconds).  For a single-component run, purity is
    trivially 1.0; purity is meaningful when *c_outlet* contains multiple
    components (rows) — the target component is selected by *target_index*.

    Parameters
    ----------
    t:
        Time axis (s), shape ``(n_t,)``.
    c_outlet:
        Outlet concentrations, shape ``(n_components, n_t)``.
    cut_start, cut_end:
        Pool collection window (s).
    target_index:
        Row index of the target component in *c_outlet*.
    """
    t = np.asarray(t, dtype=float)
    c_outlet = np.atleast_2d(np.asarray(c_outlet, dtype=float))

    mask = (t >= cut_start) & (t <= cut_end)

    if mask.sum() < 2:
        return {"yield": 0.0, "purity": 0.0, "productivity": 0.0}

    # Trapezoid integration over pool window and over full chromatogram
    total_mass = np.trapezoid(c_outlet[target_index], t)
    pool_mass = np.trapezoid(c_outlet[target_index, mask], t[mask])
    all_pool_mass = np.sum(
        [np.trapezoid(c_outlet[i, mask], t[mask]) for i in range(c_outlet.shape[0])]
    )

    protein_yield = pool_mass / total_mass if total_mass > 0 else 0.0
    purity = pool_mass / all_pool_mass if all_pool_mass > 0 else 1.0
    dt = t[-1] - t[0]
    productivity = pool_mass / dt if dt > 0 else 0.0

    return {
        "yield": float(np.clip(protein_yield, 0.0, 1.0)),
        "purity": float(np.clip(purity, 0.0, 1.0)),
        "productivity": float(productivity),
    }


# ── Peak characterisation ─────────────────────────────────────────────────────

def peak_moments(t: np.ndarray, c: np.ndarray) -> dict[str, float]:
    """Statistical moments of a single elution peak.

    Returns the peak **area**, the **retention time** (first temporal moment), the
    **variance** (second central moment), the apex time/height, and the standard
    deviation ``sigma``.  Moments are robust to gradient peaks where a Gaussian fit
    would be biased.
    """
    t = np.asarray(t, dtype=float)
    c = np.asarray(c, dtype=float)
    c = np.clip(c, 0.0, None)

    area = float(np.trapezoid(c, t))
    if area <= 0.0:
        return {
            "area": 0.0,
            "retention_time": float("nan"),
            "variance": 0.0,
            "sigma": 0.0,
            "apex_time": float("nan"),
            "apex_height": 0.0,
        }

    t_r = float(np.trapezoid(t * c, t) / area)
    variance = float(np.trapezoid((t - t_r) ** 2 * c, t) / area)
    apex = int(np.argmax(c))

    return {
        "area": area,
        "retention_time": t_r,
        "variance": max(variance, 0.0),
        "sigma": float(np.sqrt(max(variance, 0.0))),
        "apex_time": float(t[apex]),
        "apex_height": float(c[apex]),
    }


def plate_count(t: np.ndarray, c: np.ndarray) -> float:
    """Apparent number of theoretical plates ``N = t_R² / σ²`` for a peak."""
    mom = peak_moments(t, c)
    sigma = mom["sigma"]
    if sigma <= 0.0 or not np.isfinite(mom["retention_time"]):
        return 0.0
    return float((mom["retention_time"] / sigma) ** 2)


def resolution(t: np.ndarray, c_a: np.ndarray, c_b: np.ndarray) -> float:
    """Chromatographic resolution ``Rs = 2·(t_R,b − t_R,a)/(w_a + w_b)``.

    Baseline widths are taken as ``w = 4σ`` from the peak moments, so the result is
    a moment-based resolution that does not assume Gaussian peaks.  Returns ``0.0`` if
    either peak is empty.
    """
    ma = peak_moments(t, c_a)
    mb = peak_moments(t, c_b)
    w_a, w_b = 4.0 * ma["sigma"], 4.0 * mb["sigma"]
    if w_a + w_b <= 0.0 or not (np.isfinite(ma["retention_time"]) and np.isfinite(mb["retention_time"])):
        return 0.0
    return float(2.0 * abs(mb["retention_time"] - ma["retention_time"]) / (w_a + w_b))
