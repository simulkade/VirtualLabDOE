"""Transport-dispersive chromatography engine (multi-component + modulator).

The column is discretised into ``N`` finite-volume cells (method of lines).  Per cell the
state is the mobile concentration ``c_i``, the stationary loading ``q_i`` (one pair per
component ``i``), and the modulator ``m``:

    modulator   :  ∂m/∂t   = −u·∂m/∂z + D·∂²m/∂z²                    (unretained tracer)
    mobile  c_i :  ∂c_i/∂t = −u·∂c_i/∂z + D·∂²c_i/∂z² − φ·∂q_i/∂t     φ = (1−ε)/ε
    stationary  :  ∂q_i/∂t = k_m,i·(q*_i(c, m, pH) − q_i)            (linear driving force)

Carrying ``q`` explicitly (rather than folding the isotherm into a retardation factor)
is what lets the modulator vary in time — gradient elution — without a ``dH/dt`` source
term or a per-cell equilibrium solve.  The equilibrium-dispersive limit is recovered as
``k_m → ∞``.  Convection uses first-order upwinding (stable, non-oscillatory), dispersion
central differences, a Dirichlet inlet and a zero-gradient outlet; time integration is
stiff ``BDF``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix, lil_matrix

from .geometry import ColumnGeometry
from .isotherms import Isotherm
from .program import CompiledProgram, ElutionProgram

_M_FLOOR = 1e-9  # keep the modulator strictly positive for the SMA (Λ/m)^ν law


def _jacobian_sparsity(nc: int, N: int) -> csr_matrix:
    """Sparsity pattern of the RHS Jacobian.

    State order is ``[c (nc·N), q (nc·N), m (N)]`` with cell-major blocks.  ``c_i,j``
    couples to its transport neighbours and (through the competitive isotherm) to every
    component's ``c`` and the modulator in the *same* cell; ``q_i,j`` couples to the same
    cell's species and modulator; ``m_j`` only to its transport neighbours.  Handing this
    pattern to ``BDF`` replaces the dense finite-difference Jacobian (``n`` RHS evals) with
    a handful of graph-coloured evals — the dominant speed-up.
    """
    n = 2 * nc * N + N
    S = lil_matrix((n, n), dtype=np.int8)

    def ci(i: int, j: int) -> int:
        return i * N + j

    def qi(i: int, j: int) -> int:
        return nc * N + i * N + j

    def mi(j: int) -> int:
        return 2 * nc * N + j

    for i in range(nc):
        for j in range(N):
            row_c = ci(i, j)
            for jj in (j - 1, j, j + 1):
                if 0 <= jj < N:
                    S[row_c, ci(i, jj)] = 1          # convection/dispersion in c_i
            for k in range(nc):
                S[row_c, ci(k, j)] = 1               # competition via q*_i,j
            S[row_c, qi(i, j)] = 1
            S[row_c, mi(j)] = 1

            row_q = qi(i, j)
            for k in range(nc):
                S[row_q, ci(k, j)] = 1
            S[row_q, qi(i, j)] = 1
            S[row_q, mi(j)] = 1

    for j in range(N):
        row_m = mi(j)
        for jj in (j - 1, j, j + 1):
            if 0 <= jj < N:
                S[row_m, mi(jj)] = 1                 # modulator transport
    return S.tocsr()


@dataclass
class ColumnSetup:
    """Full specification of a multi-mode chromatography run.

    Parameters
    ----------
    geometry:
        Column dimensions and packing.
    velocity:
        Interstitial velocity u (m/s).
    dispersion:
        Apparent axial dispersion D_ap (m²/s).
    isotherm:
        Mode-specific :class:`~.isotherms.Isotherm` (CEX/AEX/HIC/RP).
    program:
        The :class:`~.program.ElutionProgram` (modulator timeline + injection).
    ph:
        Mobile-phase pH (constant during the run).
    mass_transfer:
        Linear-driving-force coefficient ``k_m`` (1/s); scalar or per-component.
        Larger values → sharper peaks (more plates); ``→∞`` is local equilibrium.
    n_cells:
        Number of spatial finite-volume cells.
    """

    geometry: ColumnGeometry
    velocity: float
    dispersion: float
    isotherm: Isotherm
    program: ElutionProgram
    ph: float = 7.0
    mass_transfer: float | Sequence[float] = 1.0
    n_cells: int = 100


@dataclass
class ChromatogramResult:
    """Outputs of a :func:`run_column` simulation."""

    t: np.ndarray                       # (n_t,)
    c_outlet: np.ndarray                # (n_components, n_t)
    m_outlet: np.ndarray                # (n_t,) modulator trace at the outlet
    c_profile: np.ndarray               # (n_components, n_cells) final mobile profile
    q_profile: np.ndarray               # (n_components, n_cells) final bound profile
    compiled: CompiledProgram           # resolved program (for cut points / plotting)
    henry_load: np.ndarray              # Henry constant at the loading modulator
    segment_bounds_s: np.ndarray = field(default_factory=lambda: np.empty(0))
    segment_names: list[str] = field(default_factory=list)


def run_column(
    setup: ColumnSetup,
    t_eval: np.ndarray | None = None,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> ChromatogramResult:
    """Integrate the transport-dispersive model and return outlet chromatograms.

    Parameters
    ----------
    setup:
        Full column + isotherm + program configuration.
    t_eval:
        Times (s) at which to report the solution.  Defaults to 400 points spanning
        the compiled program duration.
    rtol, atol:
        Solver tolerances passed to ``solve_ivp``.
    """
    geom = setup.geometry
    N = setup.n_cells
    nc = setup.isotherm.n_components
    u = setup.velocity
    D = setup.dispersion
    eps = geom.porosity
    phi = (1.0 - eps) / eps
    dz = geom.length / N
    ph = setup.ph

    compiled = setup.program.compile(geom, u)
    if compiled.n_components != nc:
        raise ValueError(
            f"program feed has {compiled.n_components} components but isotherm has {nc}"
        )

    k_m = np.broadcast_to(np.asarray(setup.mass_transfer, dtype=float), (nc,)).copy()
    k_m = k_m[:, None]  # (nc, 1) for broadcasting over cells

    if t_eval is None:
        t_eval = np.linspace(0.0, compiled.t_end_s, 400)
    t_eval = np.asarray(t_eval, dtype=float)

    n_c = nc * N
    iso = setup.isotherm

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        c = y[:n_c].reshape(nc, N)
        q = y[n_c : 2 * n_c].reshape(nc, N)
        m = np.maximum(y[2 * n_c :], _M_FLOOR)

        m_inlet = compiled.modulator(t)
        feed = compiled.feed_at(t)  # (nc,)

        # ── Modulator transport (unretained) ──
        m_ext = np.empty(N + 2)
        m_ext[0] = m_inlet
        m_ext[1 : N + 1] = m
        m_ext[N + 1] = m[-1]
        conv_m = u * (m_ext[1 : N + 1] - m_ext[0:N]) / dz
        disp_m = D * (m_ext[2 : N + 2] - 2.0 * m_ext[1 : N + 1] + m_ext[0:N]) / dz**2
        dmdt = -conv_m + disp_m

        # ── Stationary phase (linear driving force) ──
        q_star = iso.q_star(c, m, ph)  # (nc, N)
        dqdt = k_m * (q_star - q)

        # ── Mobile phase transport ──
        c_ext = np.empty((nc, N + 2))
        c_ext[:, 0] = feed
        c_ext[:, 1 : N + 1] = c
        c_ext[:, N + 1] = c[:, -1]
        conv = u * (c_ext[:, 1 : N + 1] - c_ext[:, 0:N]) / dz
        disp = D * (c_ext[:, 2 : N + 2] - 2.0 * c_ext[:, 1 : N + 1] + c_ext[:, 0:N]) / dz**2
        dcdt = -conv + disp - phi * dqdt

        return np.concatenate([dcdt.ravel(), dqdt.ravel(), dmdt])

    # Column pre-equilibrated at the starting modulator; empty of protein.
    m0 = np.full(N, max(compiled.modulator(0.0), _M_FLOOR))
    y0 = np.concatenate([np.zeros(n_c), np.zeros(n_c), m0])

    sol = solve_ivp(
        rhs,
        t_span=(float(t_eval[0]), float(t_eval[-1])),
        y0=y0,
        method="BDF",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        jac_sparsity=_jacobian_sparsity(nc, N),
    )

    c_hist = sol.y[:n_c].reshape(nc, N, -1)
    q_hist = sol.y[n_c : 2 * n_c].reshape(nc, N, -1)
    m_hist = sol.y[2 * n_c :]

    return ChromatogramResult(
        t=sol.t,
        c_outlet=c_hist[:, -1, :],          # outlet = last cell
        m_outlet=m_hist[-1, :],
        c_profile=c_hist[:, :, -1],
        q_profile=q_hist[:, :, -1],
        compiled=compiled,
        henry_load=iso.henry(compiled.modulator(compiled.inject_start_s), ph),
        segment_bounds_s=compiled.breakpoints_s,
        segment_names=compiled.segment_names,
    )
