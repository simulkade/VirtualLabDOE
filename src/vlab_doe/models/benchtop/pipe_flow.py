"""Benchtop model 1 — Pressure drop for flow in a circular pipe.

A pump pushes a Newtonian liquid (water, by default) through a straight
horizontal pipe of length :math:`L` and internal diameter :math:`d`, and we
measure the pressure drop :math:`\\Delta P` across it as a function of the
volumetric flow rate :math:`Q`.  This is about the simplest mechanistic model in
the package, yet it is a rich sandbox for experimental design because the
response spans two flow regimes with very different sensitivities.

Governing relations
--------------------
The mean velocity and Reynolds number are

.. math::

    v = \\frac{Q}{A}, \\qquad A = \\frac{\\pi d^2}{4}, \\qquad
    \\mathrm{Re} = \\frac{\\rho v d}{\\mu}.

The Darcy--Weisbach equation writes the pressure drop in terms of a
dimensionless friction factor :math:`f`,

.. math::

    \\Delta P = f\\,\\frac{L}{d}\\,\\frac{\\rho v^2}{2}.

In **laminar** flow (:math:`\\mathrm{Re}\\lesssim 2300`) the friction factor is
exactly :math:`f = 64/\\mathrm{Re}`, and substituting recovers the
**Hagen--Poiseuille** law,

.. math::

    \\Delta P = \\frac{128\\,\\mu L Q}{\\pi d^4},

a *linear* function of flow rate with a ferocious :math:`d^{-4}` dependence on
diameter.  In **turbulent** flow (:math:`\\mathrm{Re}\\gtrsim 4000`) a smooth-pipe
Blasius correlation :math:`f = 0.316\\,\\mathrm{Re}^{-1/4}` gives
:math:`\\Delta P \\propto Q^{1.75}`.  Between the two we interpolate the friction
factor smoothly (in log--log space) so the response is continuous.

Because every relation is a power law, the model is *linear in the logarithms*
of the factors --- which makes it the ideal system on which to teach why a
log transform turns a tangle of interactions into clean, additive main effects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ── Physical constants ────────────────────────────────────────────────────────
_G = 9.80665           # gravitational acceleration (m/s²) — unused but documented
_RE_LAM = 2300.0       # upper end of the laminar regime
_RE_TURB = 4000.0      # lower end of the fully-turbulent regime
_BLASIUS_C = 0.316     # Blasius smooth-pipe coefficient
_BAR_TO_PA = 1.0e5


# ── Fluid property correlations for water ─────────────────────────────────────

def water_viscosity(temp_c: float) -> float:
    """Dynamic viscosity of liquid water (Pa·s) via the Vogel correlation.

    :math:`\\mu = A\\,\\exp\\!\\big(B/(T-C)\\big)` with *T* in kelvin and
    ``A=2.414e-5``, ``B=247.8``, ``C=140.0``.  Accurate to well under 1 % over
    0–100 °C; at 20 °C it returns ≈ 1.0×10⁻³ Pa·s.
    """
    temp_k = float(temp_c) + 273.15
    return 2.414e-5 * 10.0 ** (247.8 / (temp_k - 140.0))


def water_density(temp_c: float) -> float:
    """Density of liquid water (kg/m³) from the standard 0–40 °C correlation.

    Uses the Tanaka/Kell-style rational fit; returns ≈ 998.2 kg/m³ at 20 °C.
    """
    t = float(temp_c)
    return 1000.0 * (
        1.0
        - (t + 288.9414) / (508929.2 * (t + 68.12963)) * (t - 3.9863) ** 2
    )


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PipeFlowConfig:
    """A single pressure-drop measurement condition.

    Parameters
    ----------
    flow_rate:
        Volumetric flow rate *Q* (m³/s).
    diameter:
        Internal pipe diameter *d* (m).
    length:
        Pipe length *L* (m).
    temperature:
        Fluid temperature (°C); sets ``density`` and ``viscosity`` when those
        are left as ``None`` (the water default).
    density, viscosity:
        Optional explicit fluid properties (kg/m³, Pa·s).  When ``None`` the
        water correlations at ``temperature`` are used.
    """

    flow_rate: float
    diameter: float
    length: float
    temperature: float = 20.0
    density: float | None = None
    viscosity: float | None = None

    def rho(self) -> float:
        return water_density(self.temperature) if self.density is None else self.density

    def mu(self) -> float:
        return water_viscosity(self.temperature) if self.viscosity is None else self.viscosity


# ── Sub-models ────────────────────────────────────────────────────────────────

def mean_velocity(config: PipeFlowConfig) -> float:
    """Cross-sectional mean velocity *v = Q/A* (m/s)."""
    area = np.pi * config.diameter ** 2 / 4.0
    return config.flow_rate / area


def reynolds_number(config: PipeFlowConfig) -> float:
    """Pipe Reynolds number :math:`\\mathrm{Re}=\\rho v d/\\mu`."""
    return config.rho() * mean_velocity(config) * config.diameter / config.mu()


def friction_factor(reynolds: float) -> float:
    """Darcy friction factor with a smooth laminar→turbulent transition.

    * ``Re <= 2300``: laminar, :math:`f = 64/\\mathrm{Re}`.
    * ``Re >= 4000``: turbulent, Blasius :math:`f = 0.316\\,\\mathrm{Re}^{-1/4}`.
    * in between: log–log interpolation between the two endpoint values so the
      friction factor (and hence the pressure drop) is continuous.
    """
    re = max(float(reynolds), 1e-9)
    f_lam = 64.0 / re
    f_turb = _BLASIUS_C * re ** -0.25
    if re <= _RE_LAM:
        return f_lam
    if re >= _RE_TURB:
        return f_turb
    # Continuous blend of the two branches across the transition band.
    f_lam_edge = 64.0 / _RE_LAM
    f_turb_edge = _BLASIUS_C * _RE_TURB ** -0.25
    w = (np.log(re) - np.log(_RE_LAM)) / (np.log(_RE_TURB) - np.log(_RE_LAM))
    log_f = (1.0 - w) * np.log(f_lam_edge) + w * np.log(f_turb_edge)
    return float(np.exp(log_f))


# ── Response ──────────────────────────────────────────────────────────────────

def pressure_drop(config: PipeFlowConfig) -> float:
    """Pressure drop :math:`\\Delta P` across the pipe (Pa).

    Uses Darcy--Weisbach with the regime-aware :func:`friction_factor`.  In the
    laminar regime this is algebraically identical to Hagen--Poiseuille.
    """
    v = mean_velocity(config)
    re = reynolds_number(config)
    f = friction_factor(re)
    return f * (config.length / config.diameter) * (config.rho() * v ** 2 / 2.0)


def hagen_poiseuille(config: PipeFlowConfig) -> float:
    """Analytic laminar pressure drop :math:`128\\mu L Q/(\\pi d^4)` (Pa).

    Provided for validation and teaching; equals :func:`pressure_drop` whenever
    the flow is laminar.
    """
    return 128.0 * config.mu() * config.length * config.flow_rate / (
        np.pi * config.diameter ** 4
    )


def flow_curve(
    flow_rates: np.ndarray,
    diameter: float,
    length: float,
    *,
    temperature: float = 20.0,
    density: float | None = None,
    viscosity: float | None = None,
) -> dict[str, np.ndarray]:
    """Sweep flow rate and return the pressure-drop characteristic of a pipe.

    Returns a dict with keys ``"flow_rate"`` (m³/s), ``"reynolds"``,
    ``"friction_factor"``, ``"pressure_drop"`` (Pa) and ``"pressure_drop_bar"``.
    """
    flow_rates = np.asarray(flow_rates, dtype=float)
    re = np.empty_like(flow_rates)
    ff = np.empty_like(flow_rates)
    dp = np.empty_like(flow_rates)
    for i, q in enumerate(flow_rates):
        cfg = PipeFlowConfig(
            flow_rate=float(q), diameter=diameter, length=length,
            temperature=temperature, density=density, viscosity=viscosity,
        )
        re[i] = reynolds_number(cfg)
        ff[i] = friction_factor(re[i])
        dp[i] = pressure_drop(cfg)
    return {
        "flow_rate": flow_rates,
        "reynolds": re,
        "friction_factor": ff,
        "pressure_drop": dp,
        "pressure_drop_bar": dp / _BAR_TO_PA,
    }
