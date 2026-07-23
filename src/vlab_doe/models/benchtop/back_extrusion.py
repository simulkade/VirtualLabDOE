"""Benchtop model 3 — Back-extrusion probe rheometry of a yogurt-like gel.

To measure the texture of a semi-solid dairy product (set yogurt, custard,
labneh) a texture analyser pushes a flat cylindrical **probe** of radius
:math:`R_p` straight down at constant speed *V* into a cylindrical **cup** of
radius :math:`R_c` filled to depth *H*.  The displaced product is forced up
through the thin annular gap between probe and cup wall, and the instrument
records the compression **force** on the probe.  The peak force (and the work
done) are the classic texture-analysis responses.

Constitutive model
-------------------
Yogurt is shear-thinning with a yield stress, well described by the
**Herschel--Bulkley** law

.. math::

    \\tau = \\tau_0 + K\\,\\dot\\gamma^{\\,n},

with consistency *K*, flow index :math:`n<1`, and yield stress
:math:`\\tau_0`.  The three parameters are what fermentation and formulation
actually change, so they are the natural link to the dairy chapter.

Force model
-----------
Continuity fixes the flow the probe must displace through the annulus,

.. math::

    Q = \\pi R_p^2\\,V .

Treating the annular gap (width :math:`b = R_c-R_p`, mean breadth
:math:`W = \\pi (R_c+R_p)`, length = immersion depth *L*) as a thin slit, the
pressure gradient needed to drive a Herschel--Bulkley fluid at that flow rate is
the sum of a yield term and a power-law (consistency) term,

.. math::

    \\frac{\\Delta P}{L} = \\underbrace{\\frac{2\\tau_0}{b}}_{\\text{yield}}
      + \\underbrace{\\frac{2K}{b}\\!\\left[\\frac{(2n+1)}{n}\\,
        \\frac{2\\,\\bar u}{b}\\right]^{n}}_{\\text{consistency}},
    \\qquad \\bar u = \\frac{Q}{W b}.

The force on the probe is that pressure acting over its face, less buoyancy:

.. math::

    F = \\Delta P\\,\\pi R_p^2 - \\rho g\\,\\pi R_p^2 L .

In the Newtonian limit (:math:`\\tau_0=0`, :math:`n=1`, :math:`K=\\mu`) the
consistency term collapses to an ordinary viscous slit-flow drag, which the
tests check against the closed form.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_G = 9.80665
_R_GAS = 8.314462618  # J/(mol·K)


# ── Rheology ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HerschelBulkley:
    """Herschel--Bulkley parameters of the test product.

    Parameters
    ----------
    consistency:
        Consistency index *K* (Pa·sⁿ).
    flow_index:
        Flow behaviour index *n* (–); ``< 1`` for shear-thinning yogurt.
    yield_stress:
        Yield stress :math:`\\tau_0` (Pa).
    ref_temperature:
        Reference temperature (°C) at which *K* and :math:`\\tau_0` are quoted.
    activation_energy:
        Flow activation energy (J/mol) for the Arrhenius softening of *K* and
        :math:`\\tau_0` with temperature (both scale together, so *n* is fixed).
    """

    consistency: float
    flow_index: float
    yield_stress: float
    ref_temperature: float = 10.0
    activation_energy: float = 45_000.0

    def at_temperature(self, temp_c: float) -> "HerschelBulkley":
        """Return the rheology at ``temp_c`` — warmer yogurt is softer.

        Both *K* and :math:`\\tau_0` are multiplied by the Arrhenius factor
        :math:`\\exp[(E_a/R)(1/T - 1/T_\\text{ref})]`, which is ``<1`` above the
        reference temperature.
        """
        t = float(temp_c) + 273.15
        t_ref = self.ref_temperature + 273.15
        factor = np.exp((self.activation_energy / _R_GAS) * (1.0 / t - 1.0 / t_ref))
        return HerschelBulkley(
            consistency=self.consistency * factor,
            flow_index=self.flow_index,
            yield_stress=self.yield_stress * factor,
            ref_temperature=self.ref_temperature,
            activation_energy=self.activation_energy,
        )


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BackExtrusionConfig:
    """A single back-extrusion measurement.

    Parameters
    ----------
    rheology:
        Herschel--Bulkley parameters at their reference temperature.
    probe_radius:
        Probe radius :math:`R_p` (m).
    cup_radius:
        Cup internal radius :math:`R_c` (m); must exceed ``probe_radius``.
    immersion_depth:
        Depth *L* the probe is pushed in (m).
    probe_speed:
        Cross-head speed *V* (m/s).
    temperature:
        Product temperature (°C).
    density:
        Product density (kg/m³) for the buoyancy term.
    """

    rheology: HerschelBulkley
    probe_radius: float = 0.010
    cup_radius: float = 0.0125
    immersion_depth: float = 0.020
    probe_speed: float = 1.0e-3
    temperature: float = 10.0
    density: float = 1040.0


# ── Sub-models ────────────────────────────────────────────────────────────────

def _gap_geometry(config: BackExtrusionConfig) -> tuple[float, float, float]:
    """Return ``(gap_width b, mean_breadth W, mean_slit_velocity u_bar)``."""
    rp, rc = config.probe_radius, config.cup_radius
    b = rc - rp
    if b <= 0.0:
        raise ValueError("cup_radius must exceed probe_radius")
    w = np.pi * (rc + rp)
    q = np.pi * rp ** 2 * config.probe_speed  # displaced flow (m³/s)
    u_bar = q / (w * b)
    return b, w, u_bar


def pressure_gradient(config: BackExtrusionConfig) -> float:
    """Slit-flow pressure gradient ΔP/L (Pa/m) for the Herschel--Bulkley fluid."""
    hb = config.rheology.at_temperature(config.temperature)
    b, _, u_bar = _gap_geometry(config)
    n = hb.flow_index
    wall_shear_rate = (2.0 * n + 1.0) / n * (2.0 * u_bar / b)
    yield_term = 2.0 * hb.yield_stress / b
    consistency_term = (2.0 * hb.consistency / b) * wall_shear_rate ** n
    return yield_term + consistency_term


def peak_force(config: BackExtrusionConfig) -> float:
    """Peak compression force on the probe (N).

    :math:`F = (\\Delta P/L)\\,L\\,\\pi R_p^2 - \\rho g \\pi R_p^2 L`
    (pressure over the probe face, minus buoyancy of the immersed probe).
    """
    dpdl = pressure_gradient(config)
    dp = dpdl * config.immersion_depth
    face = np.pi * config.probe_radius ** 2
    buoyancy = config.density * _G * face * config.immersion_depth
    return dp * face - buoyancy


def work_of_penetration(config: BackExtrusionConfig, *, n_steps: int = 50) -> float:
    """Work (J) done pushing the probe from the surface to ``immersion_depth``.

    Integrates the force over depth as the immersed length grows from 0 to *L*
    at fixed speed (the area under the force--displacement curve).
    """
    from dataclasses import replace

    depths = np.linspace(0.0, config.immersion_depth, n_steps + 1)[1:]
    forces = np.array([
        peak_force(replace(config, immersion_depth=float(d))) for d in depths
    ])
    return float(np.trapezoid(forces, depths))
