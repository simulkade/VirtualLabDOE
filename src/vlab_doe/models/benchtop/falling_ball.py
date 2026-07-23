"""Benchtop model 2 — Falling-ball (falling-sphere) viscometer.

A dense sphere of diameter :math:`d` and density :math:`\\rho_s` is released in a
vertical tube of internal diameter :math:`D` filled with a Newtonian test fluid
of density :math:`\\rho_f` and viscosity :math:`\\mu`.  Once it reaches terminal
velocity we time its passage over a marked fall distance :math:`L`.  The
measured fall **time** is the response; the viscosity is what we ultimately want
to infer from it.

Physics
-------
At terminal velocity the buoyant weight balances the drag:

.. math::

    \\underbrace{\\tfrac{\\pi}{6} d^3 (\\rho_s-\\rho_f) g}_{\\text{net weight}}
      = \\underbrace{C_D\\,\\tfrac12 \\rho_f v^2 \\tfrac{\\pi}{4} d^2}_{\\text{drag}} .

In the creeping-flow (Stokes) limit :math:`C_D = 24/\\mathrm{Re}` and this solves
in closed form,

.. math::

    v_{\\text{Stokes}} = \\frac{g\\,d^2 (\\rho_s-\\rho_f)}{18\\,\\mu},
    \\qquad \\mathrm{Re}=\\frac{\\rho_f v d}{\\mu}.

Two corrections make the virtual instrument realistic --- and make it a good
lesson in *where in the design space the measurement is trustworthy*:

* **Wall effect.** In a finite tube the ball falls more slowly.  We apply the
  Ladenburg--Faxén factor :math:`K_w = 1 - 2.104\\,\\beta + 2.09\\,\\beta^3 -
  0.95\\,\\beta^5` with :math:`\\beta = d/D`.
* **Inertia.** Away from :math:`\\mathrm{Re}\\ll 1` Stokes under-predicts drag.
  We use the Schiller--Naumann correlation
  :math:`C_D = (24/\\mathrm{Re})(1 + 0.15\\,\\mathrm{Re}^{0.687})` and solve the
  force balance numerically, so the naive Stokes reading acquires a *bias* that
  grows with ball size and density.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

_G = 9.80665  # gravitational acceleration (m/s²)


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FallingBallConfig:
    """A single falling-ball run.

    Parameters
    ----------
    ball_diameter:
        Sphere diameter *d* (m).
    ball_density:
        Sphere density :math:`\\rho_s` (kg/m³), e.g. ~7800 for steel.
    fluid_density:
        Test-fluid density :math:`\\rho_f` (kg/m³).
    fluid_viscosity:
        Test-fluid dynamic viscosity :math:`\\mu` (Pa·s) — the quantity the
        instrument measures.
    tube_diameter:
        Internal tube diameter *D* (m); sets the wall correction.
    fall_distance:
        Timed fall distance *L* (m).
    """

    ball_diameter: float
    ball_density: float
    fluid_density: float
    fluid_viscosity: float
    tube_diameter: float = 0.025
    fall_distance: float = 0.10


# ── Sub-models ────────────────────────────────────────────────────────────────

def wall_factor(ball_diameter: float, tube_diameter: float) -> float:
    """Ladenburg--Faxén wall-correction factor :math:`K_w \\in (0,1]`.

    The terminal velocity in a tube is :math:`K_w` times the unbounded-fluid
    value.  Returns ``1`` in the limit of an infinitely wide tube.
    """
    beta = float(ball_diameter) / float(tube_diameter)
    beta = min(beta, 0.6)  # correlation is only valid for slender balls
    return 1.0 - 2.104 * beta + 2.09 * beta ** 3 - 0.95 * beta ** 5


def stokes_velocity(config: FallingBallConfig) -> float:
    """Unbounded creeping-flow terminal velocity (m/s), no wall correction."""
    d = config.ball_diameter
    return _G * d ** 2 * (config.ball_density - config.fluid_density) / (
        18.0 * config.fluid_viscosity
    )


def terminal_velocity(config: FallingBallConfig) -> float:
    """Terminal velocity (m/s) from the full Schiller--Naumann force balance.

    Solves ``net_weight = drag`` for *v* with a Reynolds-dependent drag
    coefficient, then applies the wall correction.  Reduces to
    :func:`stokes_velocity` (times the wall factor) as ``Re → 0``.
    """
    d = config.ball_diameter
    rho_f = config.fluid_density
    mu = config.fluid_viscosity
    net_weight = (np.pi / 6.0) * d ** 3 * (config.ball_density - rho_f) * _G
    if net_weight <= 0.0:
        return 0.0
    area = (np.pi / 4.0) * d ** 2

    def residual(v: float) -> float:
        re = max(rho_f * v * d / mu, 1e-12)
        cd = (24.0 / re) * (1.0 + 0.15 * re ** 0.687)
        drag = cd * 0.5 * rho_f * v ** 2 * area
        return drag - net_weight

    v_stokes = stokes_velocity(config)
    # Bracket the root generously around the Stokes estimate.
    lo, hi = 1e-9, max(10.0 * v_stokes, 1e-6)
    while residual(hi) < 0.0 and hi < 1e4:
        hi *= 10.0
    v = brentq(residual, lo, hi, xtol=1e-12, rtol=1e-10)
    return v * wall_factor(d, config.tube_diameter)


def reynolds_number(config: FallingBallConfig) -> float:
    """Particle Reynolds number at terminal velocity."""
    v = terminal_velocity(config)
    return config.fluid_density * v * config.ball_diameter / config.fluid_viscosity


# ── Response ──────────────────────────────────────────────────────────────────

def fall_time(config: FallingBallConfig) -> float:
    """Time (s) to traverse ``fall_distance`` at terminal velocity."""
    v = terminal_velocity(config)
    if v <= 0.0:
        return float("inf")
    return config.fall_distance / v


def infer_viscosity(
    measured_fall_time: float,
    config: FallingBallConfig,
    *,
    stokes_only: bool = True,
) -> float:
    """Back out the fluid viscosity from a measured fall time.

    With ``stokes_only=True`` (the usual working assumption of the instrument)
    the classic wall-corrected Stokes formula is inverted analytically::

        μ = g d² (ρ_s − ρ_f) K_w L / (18 · v_measured)

    where ``v_measured = L / t``.  Comparing this estimate with the ``μ`` that
    was actually used to *generate* ``t`` (via the full drag law) exposes the
    inertial bias — the core lesson of the falling-ball exercise.

    With ``stokes_only=False`` the full non-linear model is inverted numerically
    (root-find on :func:`fall_time`), recovering the true viscosity.
    """
    v_measured = config.fall_distance / measured_fall_time
    if stokes_only:
        kw = wall_factor(config.ball_diameter, config.tube_diameter)
        return (
            _G * config.ball_diameter ** 2
            * (config.ball_density - config.fluid_density)
            * kw / (18.0 * v_measured)
        )

    from dataclasses import replace

    def residual(mu: float) -> float:
        return fall_time(replace(config, fluid_viscosity=mu)) - measured_fall_time

    return brentq(residual, 1e-5, 1e3, xtol=1e-12, rtol=1e-8)
