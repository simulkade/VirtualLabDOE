"""Backward-compatible single-component, isocratic model.

This is the original equilibrium-dispersive (ED) formulation, preserved unchanged so that
existing notebooks, tests, and the DoE/optimization/UQ layers keep working.  New work
should prefer :func:`..engine.run_column`, which supports multiple components, every
chromatography mode, and gradient elution.

    R(c)·∂c/∂t = −u·∂c/∂z + D_ap·∂²c/∂z²,   R = 1 + (1−ε)/ε·H(salt, pH)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .geometry import ColumnGeometry
from .isotherms import SMAParameters, sma_henry_constant


@dataclass(frozen=True)
class ChromatographyConfig:
    """Full configuration of an ED chromatography run (single component, isocratic)."""

    geometry: ColumnGeometry
    velocity: float         # interstitial velocity u (m/s)
    dispersion: float       # apparent axial dispersion D_ap (m²/s)
    isotherm: SMAParameters
    salt: float             # mobile-phase salt concentration (mM)
    ph: float               # mobile-phase pH (−)
    load_density: float     # mg protein per mL resin (g/L resin)
    n_cells: int = 100      # spatial discretisation cells


def simulate(config: ChromatographyConfig, t_eval: np.ndarray) -> dict[str, np.ndarray]:
    """Run the ED column model and return outlet chromatograms.

    The single-component protein is injected as a rectangular pulse whose
    duration is set by *load_density* (the total mass loaded per mL resin).
    The column is initially empty.  After the load phase the inlet switches to
    zero concentration (wash / isocratic elution).

    Returns
    -------
    dict with keys ``t``, ``c_outlet`` (shape ``(1, n_t)``), ``c_profile``,
    ``t_load``, ``henry``, ``retardation``.
    """
    N = config.n_cells
    L = config.geometry.length
    d = config.geometry.diameter
    eps = config.geometry.porosity
    u = config.velocity
    D = config.dispersion

    dz = L / N

    # Column and flow geometry
    A_cross = math.pi / 4.0 * d**2          # cross-sectional area (m²)
    V_column = A_cross * L                   # total column volume (m³)
    V_resin = V_column * (1.0 - eps)         # stationary-phase volume (m³)
    Q_flow = u * A_cross * eps               # volumetric flow rate (m³/s)

    # Feed properties
    c_feed = 1.0  # g/L — a representative protein feed concentration
    V_inject = config.load_density * V_resin  # m³  (since g/L = kg/m³ up to factor 1)
    t_load = V_inject / Q_flow if Q_flow > 0 else float("inf")

    # SMA Henry's constant (linearised; single component, index 0)
    H = sma_henry_constant(config.salt, config.ph, config.isotherm, index=0)
    # Retardation factor R = 1 + (1-ε)/ε * H
    R = 1.0 + (1.0 - eps) / eps * H

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        c_in = c_feed if t <= t_load else 0.0

        c_ext = np.empty(N + 2)
        c_ext[0] = c_in          # inlet (Dirichlet)
        c_ext[1 : N + 1] = y
        c_ext[N + 1] = y[-1]     # outlet zero-gradient (Neumann)

        conv = u * (c_ext[1 : N + 1] - c_ext[0:N]) / dz
        disp = D * (c_ext[2 : N + 2] - 2.0 * c_ext[1 : N + 1] + c_ext[0:N]) / dz**2

        return (-conv + disp) / R

    y0 = np.zeros(N)

    sol = solve_ivp(
        rhs,
        t_span=(float(t_eval[0]), float(t_eval[-1])),
        y0=y0,
        method="BDF",
        t_eval=t_eval,
        rtol=1e-4,
        atol=1e-6,
    )

    c_outlet = sol.y[-1, :]  # last spatial cell → outlet chromatogram

    return {
        "t": sol.t,
        "c_outlet": c_outlet[np.newaxis, :],  # shape (1, n_t)
        "c_profile": sol.y[:, -1],            # spatial profile at final time
        "t_load": t_load,
        "henry": H,
        "retardation": R,
    }
