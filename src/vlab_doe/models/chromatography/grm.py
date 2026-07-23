"""General Rate Model (GRM) solved with PyFVTool finite volumes.

This is an alternative, fully-implicit, mass-conservative solver to the
method-of-lines :func:`~.engine.run_column`.  Where the MoL engine lumps all
mass-transfer resistance into a single linear-driving-force coefficient, the GRM
resolves the **mechanistic** two-scale mass transfer:

* an external **film** (boundary layer) around each resin bead, and
* **pore diffusion** inside the porous bead (a radial PDE per axial location).

Governing equations (per component *i*)::

    bulk (axial z):
        dc_i/dt = -u dc_i/dz + D_ax d2c_i/dz2
                  - (1-eps)/eps * (3/R_p) * k_ov,i * (c_i - c_p,i|surf)

    particle (radial r, one sphere per axial cell):
        [eps_p + (1-eps_p) dq_i/dc_p,i] dc_p,i/dt + (1-eps_p) sum_{j!=i} dq_i/dc_p,j dc_p,j/dt
            = eps_p D_p,i (1/r^2) d/dr(r^2 dc_p,i/dr)
        film BC at r=R_p:  eps_p D_p,i dc_p,i/dr = k_f,i (c_i - c_p,i|surf)
        symmetry at r=0:   dc_p,i/dr = 0

with the adsorbed loading ``q_i = q*_i(c_p, m, pH)`` given by the *same*
:class:`~.isotherms.Isotherm` used by the MoL engine, so every mode (CEX/AEX,
HIC, RP) and the competitive multi-component overload case are available.  The
modulator ``m`` (salt / organic fraction) is an unretained tracer; it is
advanced on the bulk mesh and assumed to equilibrate instantly within the pores
(pore modulator = bulk modulator at that axial location), which decouples it
from the protein solve.

Numerics
--------
* Spatial discretisation with PyFVTool: ``convectionUpwindTerm`` /
  ``diffusionTerm`` on a Cartesian :class:`Grid1D` (axial) and a
  :class:`SphericalGrid1D` (radial).
* The film boundary condition is discretised as a series combination of the
  film coefficient and the outer half-cell pore-diffusion resistance, added so
  that the *identical* discrete flux appears in the bulk sink and the particle
  surface source -> exact bulk<->particle mass conservation.
* Fully-implicit (backward-Euler) time stepping; the competitive isotherm
  Jacobian ``dq*/dc_p`` is linearised by Picard iteration each step.
* The bulk and particle fields are coupled into one global sparse system via
  :mod:`.coupling` (a repo-local ``CoupledSystem`` assembler), which owns the
  global index bookkeeping, the film-flux :meth:`~.coupling.Assembler.couple`
  stamp, and the conservative competitive-storage term
  (:func:`~.coupling.conservative_storage`).  The accumulation terms use
  PyFVTool's :func:`~pyfvtool.transientTerm` and the inlet dispersive flux (for
  the mass balance) uses :func:`~pyfvtool.gradientTerm`.  The coupled system is
  solved with
  ``scipy.sparse.linalg.spsolve`` (``solveMatrixPDE`` is single-mesh and does
  not apply across the bulk+particle meshes).

.. note::
   PyFVTool's spherical ``mesh.cellvolume`` is the midpoint approximation
   ``4 pi r_c^2 dr``.  The ``diffusionTerm`` operator conserves against the
   **true shell volume** ``4/3 pi (r_out^3 - r_in^3)``; this module uses the
   true shell volume for all mass accounting and for the film-coupling
   normalisation, which makes the scheme conservative to machine precision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pyfvtool as pf
import scipy.sparse.linalg as spla

from .coupling import CoupledSystem, conservative_storage
from .geometry import ColumnGeometry
from .isotherms import Isotherm
from .program import CompiledProgram, ElutionProgram

_M_FLOOR = 1e-9


# ── Mechanistic mass-transfer parameters ──────────────────────────────────────

def film_coefficient(
    *,
    velocity: float,
    porosity: float,
    particle_radius: float,
    molecular_diffusivity: float,
    density: float = 1000.0,
    viscosity: float = 1.0e-3,
) -> float:
    """External film coefficient ``k_f`` from the Wilson--Geankoplis correlation.

    For packed beds at low Reynolds number (``0.0016 < Re < 55``),

        eps * Sh = 1.09 * Re^(1/3) * Sc^(1/3),

    with ``Re = rho u_s d_p / mu`` (``u_s = eps * u`` the superficial velocity),
    ``Sc = mu / (rho D_m)`` and ``k_f = Sh * D_m / d_p``.

    Parameters
    ----------
    velocity:
        Interstitial velocity ``u`` (m/s).
    porosity:
        Bed (interstitial) porosity ``eps``.
    particle_radius:
        Bead radius ``R_p`` (m); ``d_p = 2 R_p``.
    molecular_diffusivity:
        Free-solution diffusivity ``D_m`` (m^2/s).
    density, viscosity:
        Mobile-phase density (kg/m^3) and dynamic viscosity (Pa s).
    """
    d_p = 2.0 * particle_radius
    u_s = porosity * velocity
    Re = density * u_s * d_p / viscosity
    Sc = viscosity / (density * molecular_diffusivity)
    Sh = (1.09 / porosity) * max(Re, 1e-12) ** (1.0 / 3.0) * Sc ** (1.0 / 3.0)
    return Sh * molecular_diffusivity / d_p


@dataclass
class ParticleProperties:
    """Porous-bead geometry and intraparticle/film transport parameters.

    Parameters
    ----------
    radius:
        Bead radius ``R_p`` (m).
    porosity:
        Intraparticle (pore) porosity ``eps_p``.
    pore_diffusivity:
        Effective pore diffusivity ``D_p`` (m^2/s), scalar or per component.
        If ``None`` it defaults to ``eps_p * D_m / tortuosity`` using the
        setup's ``molecular_diffusivity``.
    film_coeff:
        Film coefficient ``k_f`` (m/s), scalar or per component.  If ``None``
        it is computed from :func:`film_coefficient` (Wilson--Geankoplis).
    tortuosity:
        Pore tortuosity used only for the ``pore_diffusivity`` default.
    n_radial:
        Number of radial finite-volume cells inside the bead.
    """

    radius: float
    porosity: float = 0.5
    pore_diffusivity: float | Sequence[float] | None = None
    film_coeff: float | Sequence[float] | None = None
    tortuosity: float = 2.0
    n_radial: int = 12


@dataclass
class GRMSetup:
    """Full specification of a general-rate-model chromatography run.

    Mirrors :class:`~.engine.ColumnSetup` but adds the particle/film
    description and mobile-phase properties used by the mass-transfer
    correlations.

    Parameters
    ----------
    geometry:
        Column dimensions and packing (interstitial porosity = ``geometry.porosity``).
    velocity:
        Interstitial velocity ``u`` (m/s).
    dispersion:
        Axial dispersion ``D_ax`` (m^2/s).
    isotherm:
        Mode-specific :class:`~.isotherms.Isotherm` giving ``q*(c_p, m, pH)``.
    program:
        :class:`~.program.ElutionProgram` (modulator timeline + injection).
    particle:
        :class:`ParticleProperties`.
    ph:
        Mobile-phase pH (constant).
    molecular_diffusivity:
        Free-solution diffusivity ``D_m`` (m^2/s), used by the film and (default)
        pore-diffusivity correlations.  Scalar or per component.
    density, viscosity:
        Mobile-phase density / viscosity for the film correlation.
    n_cells:
        Number of axial finite-volume cells.
    """

    geometry: ColumnGeometry
    velocity: float
    dispersion: float
    isotherm: Isotherm
    program: ElutionProgram
    particle: ParticleProperties
    ph: float = 7.0
    molecular_diffusivity: float | Sequence[float] = 1.0e-10
    density: float = 1000.0
    viscosity: float = 1.0e-3
    n_cells: int = 40


@dataclass
class GRMResult:
    """Outputs of a :func:`run_grm` simulation (mirrors :class:`ChromatogramResult`)."""

    t: np.ndarray                       # (n_t,)
    c_outlet: np.ndarray                # (n_components, n_t)
    m_outlet: np.ndarray                # (n_t,)
    c_profile: np.ndarray               # (n_components, n_cells) final bulk profile
    cp_profile: np.ndarray              # (n_components, n_cells, n_radial) final pore profile
    q_profile: np.ndarray               # (n_components, n_cells, n_radial) final loading
    compiled: CompiledProgram
    k_film: np.ndarray                  # (n_components,) film coefficients used
    pore_diffusivity: np.ndarray        # (n_components,)
    mass_balance_error: float           # relative (in - out - accumulated)/in
    segment_bounds_s: np.ndarray = field(default_factory=lambda: np.empty(0))
    segment_names: list[str] = field(default_factory=list)


def _vec(x, n: int) -> np.ndarray:
    return np.broadcast_to(np.asarray(x, dtype=float), (n,)).astype(float)


def run_grm(
    setup: GRMSetup,
    t_eval: np.ndarray | None = None,
    *,
    n_steps: int = 600,
    picard_tol: float = 1e-8,
    picard_max: int = 20,
) -> GRMResult:
    """Integrate the general rate model and return outlet chromatograms.

    Parameters
    ----------
    setup:
        Full GRM configuration.
    t_eval:
        Times (s) at which to report the outlet.  Defaults to ``n_steps`` points
        over the compiled program duration.  Time stepping uses a uniform step
        equal to the spacing of ``t_eval``.
    n_steps:
        Number of (uniform) time steps if ``t_eval`` is not given.
    picard_tol, picard_max:
        Convergence tolerance and iteration cap for the per-step Picard
        linearisation of the competitive isotherm (ignored for linear isotherms).
    """
    geom = setup.geometry
    Nz = setup.n_cells
    nc = setup.isotherm.n_components
    Nr = setup.particle.n_radial
    u = setup.velocity
    Dax = setup.dispersion
    eps = geom.porosity
    Rp = setup.particle.radius
    eps_p = setup.particle.porosity
    ph = setup.ph

    compiled = setup.program.compile(geom, u)
    if compiled.n_components != nc:
        raise ValueError(
            f"program feed has {compiled.n_components} components but isotherm has {nc}"
        )

    if t_eval is None:
        t_eval = np.linspace(0.0, compiled.t_end_s, n_steps + 1)
    t_eval = np.asarray(t_eval, dtype=float)
    n_t = len(t_eval)

    # ── mass-transfer parameters (direct or correlation) ──
    D_m = _vec(setup.molecular_diffusivity, nc)
    if setup.particle.pore_diffusivity is None:
        D_p = eps_p * D_m / setup.particle.tortuosity
    else:
        D_p = _vec(setup.particle.pore_diffusivity, nc)
    if setup.particle.film_coeff is None:
        k_f = np.array([
            film_coefficient(
                velocity=u, porosity=eps, particle_radius=Rp,
                molecular_diffusivity=D_m[i], density=setup.density,
                viscosity=setup.viscosity,
            )
            for i in range(nc)
        ])
    else:
        k_f = _vec(setup.particle.film_coeff, nc)

    # ── meshes & static operators ──
    A_col = geom.area
    dz = geom.length / Nz
    V_b = A_col * dz
    mz = pf.Grid1D(Nz, geom.length)
    uf = pf.FaceVariable(mz, u)
    Daxf = pf.FaceVariable(mz, Dax)
    Mconv = pf.convectionUpwindTerm(uf)
    Mdif_ax = pf.diffusionTerm(Daxf)

    mp = pf.SphericalGrid1D(Nr, Rp)
    fr = mp.facecenters.r
    Vshell = 4.0 / 3.0 * np.pi * (fr[1:] ** 3 - fr[:-1] ** 3)   # true conservation weights
    Mdif_p = [pf.diffusionTerm(pf.FaceVariable(mp, eps_p * D_p[i])) for i in range(nc)]
    _cp_bc = pf.CellVariable(mp, 0.0)                            # Neumann both ends
    Mbc_p, RHSbc_p = pf.boundaryConditionsTerm(_cp_bc.BCs)

    nb = Nz + 2
    dr = Rp / Nr
    a_v = 3.0 * (1.0 - eps) / Rp
    A_R = 4.0 * np.pi * Rp ** 2
    Vp = 4.0 / 3.0 * np.pi * Rp ** 3
    Npart = (1.0 - eps) * V_b / Vp                              # beads per axial cell
    V_outer = Vshell[-1]
    # series combination of film + outer half-cell pore diffusion, per component
    k_ov = 1.0 / (1.0 / k_f + (dr / 2.0) / (eps_p * D_p))
    kb = a_v * k_ov / eps                                       # bulk sink coeff (per comp)
    kp = k_ov * A_R / V_outer                                   # particle outer source coeff

    # ── bulk BCs per component (Dirichlet inlet during feed handled via RHS) ──
    # We rebuild the inlet each step (feed concentration changes), so keep a BC
    # template per component and only update .c.
    bulk_bcs = []
    for i in range(nc):
        cvar = pf.CellVariable(mz, 0.0)
        cvar.BCs.left.a[:] = 0.0
        cvar.BCs.left.b[:] = 1.0
        cvar.BCs.left.c[:] = 0.0
        bulk_bcs.append(cvar.BCs)

    # modulator bulk BC (Dirichlet inlet = m_in(t))
    mvar = pf.CellVariable(mz, max(compiled.modulator(0.0), _M_FLOOR))
    mvar.BCs.left.a[:] = 0.0
    mvar.BCs.left.b[:] = 1.0
    mvar.BCs.left.c[:] = compiled.modulator(0.0)

    # ── coupled global layout (replaces hand-rolled bidx/pidx index algebra) ──
    # Fields are registered in a fixed order so each lands at a known global
    # offset; the CoupledSystem owns that bookkeeping (see .coupling).  Bulk:
    # one axial field per component.  Particle: one spherical field per
    # (component, axial cell) -- each is its own small mesh, which keeps the
    # diffusion/BC blocks block-diagonal and makes the film flux a per-cell
    # coupling rather than an index computation.
    sys = CoupledSystem()
    bulk = [sys.add_field(mz, f"c{i}") for i in range(nc)]
    part = [[sys.add_field(mp, f"cp{i}_{j}") for j in range(Nz)] for i in range(nc)]
    N = sys.N

    # interior global-index maps used by the per-step assembly
    bulk_int = np.concatenate([bulk[i].interior for i in range(nc)])      # (nc*Nz,)
    part_interior = np.stack([                                            # (nc, Nz*Nr)
        np.concatenate([part[i][j].interior for j in range(Nz)]) for i in range(nc)
    ])

    # static (geometry) part of the protein matrix: bulk transport, particle
    # diffusion + BC, and the film couplings.  Transient + isotherm capacity and
    # the feed-dependent BCs are added per step.
    asm0 = sys.assembler()
    for i in range(nc):
        asm0.add_block(bulk[i], Mconv - Mdif_ax)
        for j in range(Nz):
            asm0.add_block(part[i][j], -Mdif_p[i] + Mbc_p)
            # film flux: bulk interior cell (1+j) <-> particle outer cell Nr
            asm0.couple(bulk[i], 1 + j, part[i][j], Nr, kb[i])   # bulk sink
            asm0.couple(part[i][j], Nr, bulk[i], 1 + j, kp[i])   # particle source
    M_static = asm0.matrix()

    # modulator matrix (advection-dispersion, Dirichlet inlet); rebuilt RHS each step
    M_mod_static = (Mconv - Mdif_ax).tocsr()

    # ── state ──
    x = np.zeros(N)                       # protein bulk + pore concentrations
    m_state = np.full(nb, max(compiled.modulator(0.0), _M_FLOOR))
    m_state[1:Nz + 1] = max(compiled.modulator(0.0), _M_FLOOR)

    linear = setup.isotherm.linear

    # outputs
    c_out = np.zeros((nc, n_t))
    m_out = np.zeros(n_t)
    # record initial (t=0); outlet = last interior axial cell (Nz) of each bulk field
    outlet_gidx = np.array([bulk[i].offset + Nz for i in range(nc)])
    c_out[:, 0] = x[outlet_gidx]
    m_out[0] = m_state[Nz]

    cum_in = 0.0   # cumulative inlet flux (mol, all components)
    cum_out = 0.0  # cumulative outlet flux (mol, all components)

    def storage(cp_cells, m_cells):
        """Stored solute per unit particle volume, ``eps_p c_p + (1-eps_p) q*``.

        Returns ``(s, q0)`` with shapes ``(nc, Ncells)`` and ``(nc, Ncells)``.
        """
        q0 = setup.isotherm.q_star(cp_cells, m_cells, ph)        # (nc, ncell)
        s = eps_p * cp_cells + (1.0 - eps_p) * q0
        return s, q0

    def capacity_blocks(cp_cells, m_cells):
        """Jacobian of the stored solute, ``C_il = d storage_i / d c_p,l``.

        ``C_il = eps_p delta_il + (1-eps_p) dq*_i/dc_p,l``.  Used to linearise the
        conservative storage balance (see :func:`run_grm`).  Returns
        ``(C, s)`` with shapes ``(Ncells, nc, nc)`` and ``(nc, Ncells)`` where
        ``s`` is the storage at ``cp_cells``.
        """
        s0, q0 = storage(cp_cells, m_cells)
        C = np.zeros((cp_cells.shape[1], nc, nc))
        for l in range(nc):
            dc = np.maximum(1e-6 * np.abs(cp_cells[l]), 1e-9)
            cp_pert = cp_cells.copy()
            cp_pert[l] += dc
            qp = setup.isotherm.q_star(cp_pert, m_cells, ph)
            dqdc = (qp - q0) / dc                                 # dq_i/dc_l
            for i in range(nc):
                C[:, i, l] = (1.0 - eps_p) * dqdc[i]
        for i in range(nc):
            C[:, i, i] += eps_p
        return C, s0

    dt = t_eval[1] - t_eval[0]
    t = t_eval[0]
    out_k = 1
    m_part_prev = np.repeat(np.maximum(m_state[1:Nz + 1], _M_FLOOR), Nr)
    # uniform stepping aligned to t_eval (assumes ~uniform t_eval)
    while out_k < n_t:
        t_target = t_eval[out_k]
        # step to t_target (single step; t_eval assumed uniform)
        dt = t_target - t
        t = t_target

        # ---- advance modulator (unretained, linear) ----
        # accumulation handled by transientTerm (alfa=1): it returns exactly the
        # 1/dt diagonal and the c_old/dt RHS we would otherwise build by hand.
        m_in = max(compiled.modulator(t), _M_FLOOR)
        mvar.BCs.left.c[:] = m_in
        Mbc_m, RHSbc_m = pf.boundaryConditionsTerm(mvar.BCs)
        Mt_m, RHSt_m = pf.transientTerm(pf.CellVariable(mz, m_state), dt, 1.0)
        Mm = M_mod_static + Mbc_m + Mt_m
        RHSm = RHSbc_m + RHSt_m
        m_state = spla.spsolve(Mm.tocsr(), RHSm)
        m_cells = np.maximum(m_state[1:Nz + 1], _M_FLOOR)         # (Nz,)
        # pore modulator = bulk modulator at that axial cell (broadcast over radial)
        m_part = np.repeat(m_cells, Nr)                           # (Nz*Nr,)

        # ---- feed for this step ----
        feed = compiled.feed_at(t)                                # (nc,)

        # bulk inlet BC matrices/RHS (per component; feed-dependent)
        bulk_bc_terms = []
        for i in range(nc):
            bulk_bcs[i].left.c[:] = feed[i]
            bulk_bc_terms.append(pf.boundaryConditionsTerm(bulk_bcs[i]))

        # old stored mass uses the OLD modulator m_part_prev (so a changing
        # gradient is handled exactly by the storage balance below).
        cp_old = x[part_interior]                                     # (nc, Nz*Nr)
        s_old, _ = storage(cp_old, m_part_prev)                       # (nc, Nz*Nr)

        # ---- Picard iteration for the protein system (conservative storage form) ----
        # Particle balance:  [storage(c_p,m) - storage(c_p_old,m_old)]/dt = diffusion(c_p)
        # Linearised about the iterate c_p* with C = d storage/d c_p:
        #   sum_l C_il c_p,l/dt - diffusion_i = [sum_l C_il c_p*,l - storage_i(c_p*) + s_old_i]/dt
        # At convergence (c_p = c_p*) the linearisation cancels and the *exact*
        # storage(c_p) appears -> machine-precision mass conservation, and the
        # modulator change is captured because s_old uses m_old.
        x_old = x.copy()
        x_iter = x.copy()
        for _ in range(1 if linear else picard_max):
            cp_now = x_iter[part_interior]                       # (nc, Nz*Nr)
            C, s_iter = capacity_blocks(cp_now, m_part)          # (Nz*Nr,nc,nc), (nc,Nz*Nr)

            # per-step dynamic terms: feed-dependent bulk BCs, bulk accumulation
            # (transientTerm, alfa=1), particle BC RHS, and the conservative
            # competitive particle storage balance.  The CoupledSystem scatters
            # each contribution to the right global offset.
            asm = sys.assembler()
            for i in range(nc):
                Mbc_b, RHSbc_b = bulk_bc_terms[i]
                asm.add_block(bulk[i], Mbc_b).add_rhs(bulk[i], RHSbc_b)
                Mt_b, RHSt_b = pf.transientTerm(
                    pf.CellVariable(mz, x_old[bulk[i].block]), dt, 1.0)
                asm.add_block(bulk[i], Mt_b).add_rhs(bulk[i], RHSt_b)
                for j in range(Nz):
                    asm.add_rhs(part[i][j], RHSbc_p)
            # particle storage: linearised increment on LHS, exact storage on RHS
            # (uses s_old at the OLD modulator -> machine-precision conservation).
            rows, cols, vals, ridx, rvals = conservative_storage(
                part_interior, C, cp_now, s_iter, s_old, dt)
            asm.add_entries(rows, cols, vals).add_rhs_at(ridx, rvals)

            M = M_static + asm.matrix()
            x_new = spla.spsolve(M.tocsr(), asm.rhs)
            if linear:
                x_iter = x_new
                break
            if np.linalg.norm(x_new - x_iter) <= picard_tol * (np.linalg.norm(x_new) + 1e-30):
                x_iter = x_new
                break
            x_iter = x_new
        x = x_iter

        # cumulative boundary fluxes (real mol): solute travels in the void at the
        # interstitial velocity, so the convective flow uses the void area eps*A_col.
        # Inlet (Dirichlet, upwind face value = feed) carries convection + dispersion;
        # outlet (Neumann, upwind) carries convection only.
        for i in range(nc):
            # inlet total flux = convection (Dirichlet face value = feed) minus
            # dispersion.  gradientTerm gives the FV gradient at the inlet face
            # straight from the solved ghost cell (BC-consistent), so no hand
            # finite difference: -Dax * dc/dz|_inlet == Dax*(feed - c1)/(dz/2).
            grad_in = pf.gradientTerm(pf.CellVariable(mz, x[bulk[i].block]))._xvalue[0]
            cN = x[bulk[i].offset + Nz]
            f_in = u * feed[i] - Dax * grad_in
            cum_in += eps * A_col * f_in * dt
            cum_out += eps * A_col * u * cN * dt

        c_out[:, out_k] = x[outlet_gidx]
        m_out[out_k] = m_state[Nz]
        m_part_prev = m_part
        out_k += 1

    # ── final profiles & loading ──
    c_profile = np.zeros((nc, Nz))
    cp_profile = np.zeros((nc, Nz, Nr))
    for i in range(nc):
        c_profile[i] = x[bulk[i].offset + 1:bulk[i].offset + Nz + 1]
        for j in range(Nz):
            o = part[i][j].offset + 1
            cp_profile[i, j] = x[o:o + Nr]
    # q from isotherm at final pore state
    q_profile = np.zeros((nc, Nz, Nr))
    cp_flat = cp_profile.reshape(nc, Nz * Nr)
    m_flat = np.repeat(np.maximum(m_state[1:Nz + 1], _M_FLOOR), Nr)
    q_flat = setup.isotherm.q_star(cp_flat, m_flat, ph)
    q_profile = q_flat.reshape(nc, Nz, Nr)

    # ── mass balance: accumulated vs injected (outlet integrated) ──
    bulk_mass = float(np.sum(eps * c_profile * V_b))
    part_mass = 0.0
    for i in range(nc):
        store = eps_p * cp_profile[i] + (1.0 - eps_p) * q_profile[i]   # (Nz, Nr)
        part_mass += Npart * float(np.sum(store * Vshell[None, :]))
    accumulated = bulk_mass + part_mass
    mb_err = (cum_in - cum_out - accumulated) / cum_in if cum_in > 0 else 0.0

    return GRMResult(
        t=t_eval,
        c_outlet=c_out,
        m_outlet=m_out,
        c_profile=c_profile,
        cp_profile=cp_profile,
        q_profile=q_profile,
        compiled=compiled,
        k_film=k_f,
        pore_diffusivity=D_p,
        mass_balance_error=mb_err,
        segment_bounds_s=compiled.breakpoints_s,
        segment_names=compiled.segment_names,
    )
