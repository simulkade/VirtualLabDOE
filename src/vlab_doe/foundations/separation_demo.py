"""The separation science of Chapter 2, in the smallest runnable form.

Four ideas underpin every later chapter and are built here from scratch:

#. **The adsorption isotherm** -- how much solute the resin holds at
   equilibrium as a function of what is in the liquid.  The linear (Henry) and
   Langmuir forms of :func:`langmuir` are the constitutive heart of every
   chromatography model in the book.
#. **The tanks-in-series column** -- the simplest model that turns an isotherm
   into a *breakthrough curve* and shows why a column has a finite capacity and
   a finite sharpness.  See :func:`tanks_in_series_breakthrough`.
#. **Peak shape and resolution** -- the Gaussian band a column of ``N``
   theoretical plates produces, and the resolution :math:`R_s` between two such
   bands.  See :func:`gaussian_peak` and :func:`resolution`.
#. **Batch microbial growth** -- the Monod/logistic picture of a fermentation,
   the living unit operation of Chapter 5.  See :func:`monod_batch`.

These are teaching reductions of the full models in
:mod:`vlab_doe.models`; they share the vocabulary but not the machinery.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ── 1. Adsorption isotherms ───────────────────────────────────────────────────

def langmuir(c: np.ndarray, q_max: float, b: float) -> np.ndarray:
    """The Langmuir isotherm ``q* = q_max · b · c / (1 + b · c)``.

    ``c`` is the mobile-phase concentration, ``q*`` the equilibrium loading on
    the resin.  Two regimes live inside this one curve:

    * **Dilute (linear) limit**, ``b·c ≪ 1``: the denominator is one and
      ``q* ≈ (q_max·b)·c = H·c``.  The slope ``H = q_max·b`` is the *Henry
      constant* -- where a dilute band sits is fixed entirely by ``H``.
    * **Overload (saturation) limit**, ``b·c ≫ 1``: ``q* → q_max``, the finite
      monolayer capacity.  The bend between the two is what makes preparative
      chromatography nonlinear and lets one species displace another.
    """
    c = np.asarray(c, dtype=float)
    return q_max * b * c / (1.0 + b * c)


def henry(c: np.ndarray, H: float) -> np.ndarray:
    """The linear (dilute-limit) isotherm ``q* = H · c``."""
    c = np.asarray(c, dtype=float)
    return H * c


# ── 2. The column as a chain of equilibrium stages ────────────────────────────

@dataclass(frozen=True)
class Breakthrough:
    """A breakthrough curve: outlet/inlet concentration ratio versus volume."""

    cv: np.ndarray                 # throughput in column volumes
    c_out_ratio: np.ndarray        # c_outlet / c_feed in [0, 1]
    n_stages: int


def tanks_in_series_breakthrough(
    n_stages: int = 30,
    retention_factor: float = 5.0,
    cv_max: float = 12.0,
    n_points: int = 600,
) -> Breakthrough:
    """Breakthrough of a column modelled as ``n_stages`` stirred tanks in series.

    A real column is a continuum, but a remarkably faithful caricature is a
    chain of ``N`` well-mixed equilibrium stages (the "theoretical plates").
    Feeding a step of solute into the chain, the outlet of the last tank rises
    along the regularised lower incomplete gamma function

    .. math::

        \\frac{c_\\text{out}}{c_\\text{feed}}(V)
            = P\\!\\left(N,\\; N\\,\\frac{V}{V_R}\\right),

    where :math:`V_R = (1+k)\\,V_\\text{col}` is the volume at which the solute
    front emerges -- later for a more strongly retained solute (larger ``k``),
    the retention factor set by the isotherm slope.  As ``N`` grows the front
    sharpens toward the ideal step: *plate count is column efficiency made
    quantitative.*

    Implementation note: we integrate the chain in closed form through the gamma
    CDF rather than by stepping ``N`` coupled ODEs, which is both exact and
    instant.
    """
    from scipy.special import gammainc  # regularised lower incomplete gamma

    cv = np.linspace(0.0, cv_max, n_points)
    v_r = 1.0 + retention_factor                 # retention volume in column volumes
    # Mean number of "tank volumes" of throughput seen by the solute front.
    x = n_stages * cv / v_r
    ratio = gammainc(n_stages, x)
    return Breakthrough(cv=cv, c_out_ratio=ratio, n_stages=n_stages)


# ── 3. Peak shape and resolution ──────────────────────────────────────────────

def gaussian_peak(
    t: np.ndarray, t_r: float, n_plates: float, area: float = 1.0
) -> np.ndarray:
    """A Gaussian elution band of retention time ``t_r`` and ``n_plates`` plates.

    Plate count and peak width are two views of one quantity.  By definition
    :math:`N = t_R^2/\\sigma^2`, so a column of ``n_plates`` plates produces a
    band of standard deviation :math:`\\sigma = t_R/\\sqrt N`.  The taller and
    narrower the band, the more plates -- the more *efficient* the column.
    """
    t = np.asarray(t, dtype=float)
    sigma = t_r / np.sqrt(n_plates)
    return area / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((t - t_r) / sigma) ** 2)


def resolution(t_r1: float, t_r2: float, n_plates: float) -> float:
    """Chromatographic resolution :math:`R_s` between two Gaussian bands.

    With both peaks at the same plate count, :math:`\\sigma = t_R/\\sqrt N` and
    the baseline width is :math:`w = 4\\sigma`, giving

    .. math::

        R_s = \\frac{2\\,|t_{R,2}-t_{R,1}|}{w_1 + w_2}.

    The rule of thumb is that :math:`R_s \\gtrsim 1.5` is *baseline*
    separation -- the two bands barely touch.
    """
    sigma1 = t_r1 / np.sqrt(n_plates)
    sigma2 = t_r2 / np.sqrt(n_plates)
    w1, w2 = 4 * sigma1, 4 * sigma2
    return 2.0 * abs(t_r2 - t_r1) / (w1 + w2)


# ── 4. Batch microbial growth ─────────────────────────────────────────────────

@dataclass(frozen=True)
class GrowthCurve:
    """A simulated batch fermentation: biomass and substrate over time."""

    t: np.ndarray
    biomass: np.ndarray
    substrate: np.ndarray
    mu_max: float


def monod_batch(
    mu_max: float = 0.8,
    k_s: float = 0.5,
    yield_xs: float = 0.5,
    x0: float = 0.02,
    s0: float = 10.0,
    lag: float = 1.5,
    t_max: float = 16.0,
    n_points: int = 400,
) -> GrowthCurve:
    """Integrate a lag–exponential–stationary batch growth on one substrate.

    The specific growth rate follows Monod's saturating law
    :math:`\\mu = \\mu_\\max\\, S/(K_S+S)`: fast while food is plentiful,
    throttled as the substrate ``S`` runs out.  Biomass ``X`` and substrate are
    coupled through a yield ``Y_{X/S}`` (biomass formed per unit substrate
    consumed):

    .. math::

        \\frac{dX}{dt} = \\alpha(t)\\,\\mu(S)\\,X, \\qquad
        \\frac{dS}{dt} = -\\frac{1}{Y_{X/S}}\\frac{dX}{dt},

    with a smooth lag gate :math:`\\alpha(t)` that switches growth on around
    ``lag`` hours.  Together they reproduce the four textbook phases -- lag,
    exponential, deceleration as ``S`` falls, and a stationary plateau once it is
    exhausted -- with explicit Euler stepping so the algorithm is transparent.
    """
    t = np.linspace(0.0, t_max, n_points)
    dt = t[1] - t[0]
    X = np.empty(n_points)
    S = np.empty(n_points)
    X[0], S[0] = x0, s0

    for i in range(n_points - 1):
        alpha = 1.0 / (1.0 + np.exp(-(t[i] - lag) / 0.3))   # smooth lag switch
        mu = mu_max * S[i] / (k_s + S[i]) if S[i] > 0 else 0.0
        dX = alpha * mu * X[i]
        X[i + 1] = X[i] + dX * dt
        S[i + 1] = max(S[i] - dX * dt / yield_xs, 0.0)
    return GrowthCurve(t=t, biomass=X, substrate=S, mu_max=mu_max)


__all__ = [
    "langmuir",
    "henry",
    "Breakthrough",
    "tanks_in_series_breakthrough",
    "gaussian_peak",
    "resolution",
    "GrowthCurve",
    "monod_batch",
]
