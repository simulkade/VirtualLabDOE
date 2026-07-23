"""Phase 1.2 — Ultrafiltration / Diafiltration (UF/DF) model.

The model couples two sub-models:

1. **Concentration polarisation / gel model (flux)**::

       J = k · ln(C_gel / C_bulk)           (gel-polarised limit)

   where *k* is the film mass-transfer coefficient and C_gel ≈ 500 g/L is the
   protein gel concentration.  *k* follows a Sherwood power-law correlation::

       Sh = a · Re^0.8 · Sc^(1/3)  →  k = k_ref · (v / v_ref)^0.8

   with k_ref = 2×10⁻⁵ m/s at v_ref = 1.0 m/s (typical hollow-fibre values).
   This makes **TMP** and **cross-flow velocity** the operating CPPs (TMP sets
   the pressure-driving force that limits flux when polarisation is weak; at
   high concentration the gel model dominates).

   Combined pressure + gel model::

       J = min(TMP / (μ · R_m),  k · ln(C_gel / C_bulk))

2. **Mass-balance ODEs** for the retentate during a UF step::

       dV/dt = -J · A_membrane
       dC/dt = J · A_membrane · C · (1 − S) / V

   where *S* is the observed sieving coefficient (S = 0 → perfect retention).
   During the DF step (n_diavolumes > 0) the volume is held constant by buffer
   addition at the same rate as permeate removal, so dV/dt = 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MembraneProperties:
    """Membrane and hollow-fibre module characteristics."""

    area: float                  # total membrane area (m²)
    hydraulic_resistance: float  # clean-membrane resistance R_m (1/m)
    sieving_coefficient: float   # observed sieving S (−); 0 = perfect retention


@dataclass(frozen=True)
class UFDFConfig:
    """Configuration of a UF/DF operation."""

    membrane: MembraneProperties
    tmp: float                    # transmembrane pressure (bar)
    crossflow_velocity: float     # cross-flow velocity (m/s)
    feed_concentration: float     # initial protein concentration (g/L)
    feed_volume: float            # initial retentate volume (L)
    target_concentration: float   # desired final concentration after UF (g/L)
    n_diavolumes: float = 0.0     # DF buffer exchange diavolumes (0 = UF only)


# ── Physical constants & model parameters ─────────────────────────────────────
_MU_WATER = 1.0e-3       # dynamic viscosity of water (Pa·s)
_C_GEL = 500.0           # protein gel concentration (g/L) — literature value
_K_REF = 2.0e-5          # reference mass-transfer coeff at v_ref (m/s)
_V_REF = 1.0             # reference cross-flow velocity (m/s)
_BAR_TO_PA = 1.0e5       # unit conversion


# ── Sub-models ────────────────────────────────────────────────────────────────

def mass_transfer_coefficient(crossflow_velocity: float, config: UFDFConfig) -> float:
    """Film mass-transfer coefficient *k* from a power-law Sherwood correlation.

    k(v) = k_ref · (v / v_ref)^0.8

    This is valid for turbulent flow in hollow-fibre membranes.

    Parameters
    ----------
    crossflow_velocity:
        Operating cross-flow velocity (m/s).
    config:
        Process configuration (not used beyond the velocity; included for API
        consistency and future extension to geometry-based correlations).
    """
    return _K_REF * (max(crossflow_velocity, 1e-9) / _V_REF) ** 0.8


def permeate_flux(bulk_concentration: float, config: UFDFConfig) -> float:
    """Instantaneous permeate flux *J* (m/s).

    Uses the combined pressure / gel-polarisation model::

        J_pressure = TMP / (μ · R_m)
        J_gel      = k · ln(C_gel / C_bulk)   (only if C_bulk < C_gel)
        J          = min(J_pressure, J_gel)

    At low concentration the pressure term dominates; at high concentration the
    gel layer dominates and the flux is independent of TMP.

    Parameters
    ----------
    bulk_concentration:
        Retentate bulk protein concentration (g/L).
    config:
        Process configuration (TMP, cross-flow velocity, membrane properties).
    """
    # Pressure-driven flux (Darcy)
    tmp_pa = config.tmp * _BAR_TO_PA
    j_pressure = tmp_pa / (_MU_WATER * config.membrane.hydraulic_resistance)

    # Film-theory / gel-polarisation flux
    c_bulk = max(bulk_concentration, 1e-9)
    if c_bulk >= _C_GEL:
        j_gel = 0.0
    else:
        k = mass_transfer_coefficient(config.crossflow_velocity, config)
        j_gel = k * np.log(_C_GEL / c_bulk)

    return float(min(j_pressure, max(j_gel, 0.0)))


# ── Simulation ─────────────────────────────────────────────────────────────────

def simulate(config: UFDFConfig, t_eval: np.ndarray) -> dict[str, np.ndarray]:
    """Integrate the UF/DF mass balances over time.

    The run has two sequential phases:

    1. **UF** (concentration): integrate until ``target_concentration`` is
       reached or ``t_eval[-1]``, whichever comes first.
    2. **DF** (diafiltration, if ``n_diavolumes > 0``): hold volume constant,
       continue until the diavolume target is met.

    Parameters
    ----------
    config:
        Process configuration.
    t_eval:
        Time grid for output (seconds).

    Returns
    -------
    dict with keys:

    * ``"t"`` — time array (s), shape ``(n_t,)``.
    * ``"flux"`` — permeate flux J (m/s), shape ``(n_t,)``.
    * ``"retentate_concentration"`` — C (g/L), shape ``(n_t,)``.
    * ``"retentate_volume"`` — V (L), shape ``(n_t,)``.
    * ``"permeate_volume_cum"`` — cumulative permeate volume (L), shape ``(n_t,)``.
    * ``"retention"`` — protein retention at each time (−).
    * ``"yield"`` — fraction of input protein remaining in retentate (−).
    """
    t_eval = np.asarray(t_eval, dtype=float)
    A = config.membrane.area          # m²
    S = config.membrane.sieving_coefficient
    V0_m3 = config.feed_volume * 1e-3  # L → m³
    C0 = config.feed_concentration

    # ── ODE system ────────────────────────────────────────────────────────────
    # State: y = [V (m³), C (g/L)]
    # Phase flag passed via closure; DF phase holds V constant.

    def ode_uf(t, y):
        V, C = y
        if V <= 0.0 or C <= 0.0:
            return [0.0, 0.0]
        J = permeate_flux(C, config)
        dV_dt = -J * A
        dC_dt = J * A * C * (1.0 - S) / max(V, 1e-12)
        return [dV_dt, dC_dt]

    def ode_df(t, y):
        V, C = y
        if C <= 0.0 or V <= 0.0:
            return [0.0, 0.0]
        J = permeate_flux(C, config)
        # Permeate removed = buffer added → dV/dt = 0
        dC_dt = -J * A * C * (1.0 - S) / max(V, 1e-12)
        return [0.0, dC_dt]

    # Event to stop UF when target concentration is reached
    def uf_target_event(t, y):
        return y[1] - config.target_concentration

    uf_target_event.terminal = True
    uf_target_event.direction = 1  # C is increasing

    t0 = t_eval[0]
    tf = t_eval[-1]
    y0 = [V0_m3, C0]

    # ── Phase 1: UF ───────────────────────────────────────────────────────────
    sol_uf = solve_ivp(
        ode_uf,
        t_span=(t0, tf),
        y0=y0,
        method="RK45",
        t_eval=t_eval,
        events=uf_target_event,
        rtol=1e-5,
        atol=1e-8,
    )

    V_arr = sol_uf.y[0]  # m³
    C_arr = sol_uf.y[1]  # g/L
    t_arr = sol_uf.t

    # ── Phase 2: DF (if requested and time remains) ───────────────────────────
    if config.n_diavolumes > 0 and len(t_arr) > 0:
        t_uf_end = t_arr[-1]
        V_df0 = float(V_arr[-1])
        C_df0 = float(C_arr[-1])

        # DF operates until n_diavolumes * V_df have been exchanged
        # Volume of one diafiltration volume = V_df0
        # Total permeate for DF = n_diavolumes * V_df0
        # At flux J and area A: t_df = n_diavolumes * V_df0 / (J * A)
        J_df = permeate_flux(C_df0, config)
        if J_df > 0:
            t_df = config.n_diavolumes * V_df0 / (J_df * A)
        else:
            t_df = 0.0

        t_df_end = min(t_uf_end + t_df, tf)

        if t_df_end > t_uf_end:
            t_df_eval = t_eval[(t_eval > t_uf_end) & (t_eval <= t_df_end)]
            if len(t_df_eval) > 0:
                sol_df = solve_ivp(
                    ode_df,
                    t_span=(t_uf_end, t_df_end),
                    y0=[V_df0, C_df0],
                    method="RK45",
                    t_eval=t_df_eval,
                    rtol=1e-5,
                    atol=1e-8,
                )
                # Append DF results (keep only unique time points)
                new_mask = sol_df.t > t_uf_end
                t_arr = np.concatenate([t_arr, sol_df.t[new_mask]])
                V_arr = np.concatenate([V_arr, sol_df.y[0][new_mask]])
                C_arr = np.concatenate([C_arr, sol_df.y[1][new_mask]])

    # ── Derived quantities ─────────────────────────────────────────────────────
    flux_arr = np.array([permeate_flux(c, config) for c in C_arr])
    permeate_vol_cum = (V0_m3 - V_arr) * 1e3  # m³ → L (cumulative permeate)
    retention = 1.0 - S * np.ones_like(C_arr)
    protein_yield = (C_arr * V_arr) / (C0 * V0_m3)  # fraction of input mass

    return {
        "t": t_arr,
        "flux": flux_arr,
        "retentate_concentration": C_arr,
        "retentate_volume": V_arr * 1e3,   # m³ → L
        "permeate_volume_cum": permeate_vol_cum,
        "retention": retention,
        "yield": protein_yield,
    }
