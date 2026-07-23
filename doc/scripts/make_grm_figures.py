"""Figures for the General Rate Model (GRM) vs method-of-lines (MoL) comparison.

Generates (into doc/figures/):
  * grm_vs_mol_validation.png  -- both solvers agree on a benign single-component
    gradient (validates the GRM).
  * grm_conservation.png       -- mass-balance closure error, MoL vs GRM, on a
    stiff strongly-bound gradient where the MoL engine loses all the mass.
  * grm_intraparticle.png      -- radial pore-concentration profiles inside the
    bead at several column positions (the mechanistic content the GRM adds).

Run:  python doc/scripts/make_grm_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIGDIR = HERE.parent / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
SRC = HERE.parent.parent / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False, "figure.autolayout": True,
})
C = {"mol": "#d62728", "grm": "#1f77b4", "mod": "#7f7f7f"}

from vlab_doe.models.chromatography import (  # noqa: E402
    ColumnGeometry, ColumnSetup, ElutionProgram, Injection, cation_exchange,
    run_column,
)
from vlab_doe.models.chromatography.grm import (  # noqa: E402
    GRMSetup, ParticleProperties, run_grm,
)

GEOM = ColumnGeometry(length=0.1, diameter=0.01, porosity=0.4)
U, DAX, EPS = 2e-3, 2e-7, 0.4
A = GEOM.area


def save(fig, name):
    p = FIGDIR / name
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p.relative_to(HERE.parent.parent)}")


def mol_closure(iso, prog, kt, ncell, D):
    s = ColumnSetup(geometry=GEOM, velocity=U, dispersion=D, isotherm=iso,
                    program=prog, ph=5.0, mass_transfer=kt, n_cells=ncell)
    r = run_column(s)
    nc = r.c_outlet.shape[0]
    dz = GEOM.length / ncell
    Vb = A * dz
    injm = sum(np.trapezoid(np.where(
        (r.t >= r.compiled.inject_start_s) & (r.t < r.compiled.inject_end_s), f, 0.0),
        r.t) * U * EPS * A for f in np.atleast_1d(r.compiled.feed))
    el = sum(np.trapezoid(U * EPS * A * r.c_outlet[i], r.t) for i in range(nc))
    ret = sum(np.sum(EPS * r.c_profile[i] * Vb + (1 - EPS) * r.q_profile[i] * Vb)
              for i in range(nc))
    return abs((injm - el - ret) / injm), r


def fig_validation():
    """Benign case: weak-ish binding gradient where MoL is reliable -> both agree."""
    print("[fig] grm_vs_mol_validation")
    iso = cation_exchange(beta=[1.0], nu=[3.0], ionic_capacity=1000.0,
                          q_max=[1.0], linear=True)
    inj = Injection.from_load_density(5.0, feed=1.0, porosity=0.4)
    prog = ElutionProgram.linear_gradient(inj, m_start=300.0, m_end=900.0,
                                          gradient_cv=20.0, strip_cv=4.0)
    # MoL (robust regime: enough dispersion + loose tol)
    s_mol = ColumnSetup(geometry=GEOM, velocity=U, dispersion=1e-6, isotherm=iso,
                        program=prog, ph=5.0, mass_transfer=5.0, n_cells=80)
    r_mol = run_column(s_mol, atol=1e-6, rtol=1e-5)
    # GRM
    part = ParticleProperties(radius=4e-5, porosity=0.5, pore_diffusivity=5e-11,
                              film_coeff=3e-5, n_radial=10)
    s_grm = GRMSetup(geometry=GEOM, velocity=U, dispersion=1e-6, isotherm=iso,
                     program=prog, particle=part, ph=5.0, n_cells=80)
    r_grm = run_grm(s_grm, n_steps=500)

    fig, ax1 = plt.subplots(figsize=(7.4, 4.2))
    ax1.plot(r_mol.t / 60, r_mol.c_outlet[0], color=C["mol"], lw=2.2,
             label="method of lines (LDF)")
    ax1.plot(r_grm.t / 60, r_grm.c_outlet[0], color=C["grm"], lw=1.6, ls="--",
             label="general rate model (PyFVTool)")
    ax1.set_xlabel("time (min)")
    ax1.set_ylabel("outlet concentration (g/L)")
    ax2 = ax1.twinx()
    ax2.plot(r_grm.t / 60, r_grm.m_outlet, color=C["mod"], lw=1.0, ls=":")
    ax2.set_ylabel("salt (mM)  [dotted]", color=C["mod"])
    ax2.tick_params(axis="y", labelcolor=C["mod"])
    ax2.grid(False)
    ax1.set_title("Validation: MoL and GRM agree on a benign gradient")
    ax1.legend(loc="upper left")
    save(fig, "grm_vs_mol_validation.png")


def fig_conservation():
    """Stiff strongly-bound gradient: MoL loses all mass, GRM conserves exactly."""
    print("[fig] grm_conservation")
    iso = cation_exchange(beta=[1.0], nu=[4.0], ionic_capacity=1000.0,
                          q_max=[5.0], linear=True)
    inj = Injection.from_load_density(10.0, feed=1.0, porosity=0.4)
    prog = ElutionProgram.linear_gradient(inj, m_start=150.0, m_end=900.0,
                                          gradient_cv=25.0, strip_cv=4.0)
    err_mol, r_mol = mol_closure(iso, prog, kt=3.0, ncell=120, D=2e-7)
    part = ParticleProperties(radius=4e-5, porosity=0.5, pore_diffusivity=5e-11,
                              film_coeff=3e-5, n_radial=10)
    s_grm = GRMSetup(geometry=GEOM, velocity=U, dispersion=2e-7, isotherm=iso,
                     program=prog, particle=part, ph=5.0, n_cells=80)
    r_grm = run_grm(s_grm, n_steps=500)
    err_grm = abs(r_grm.mass_balance_error)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.5, 4.2),
                                   gridspec_kw={"width_ratios": [1.4, 1]})
    # left: chromatograms
    axL.plot(r_mol.t / 60, r_mol.c_outlet[0], color=C["mol"], lw=2.2,
             label=f"MoL  (loses {err_mol*100:.0f}% of mass)")
    axL.plot(r_grm.t / 60, r_grm.c_outlet[0], color=C["grm"], lw=1.8,
             label="GRM  (conserves)")
    axL.set_xlabel("time (min)")
    axL.set_ylabel("outlet concentration (g/L)")
    axL.set_title("Stiff strongly-bound gradient")
    axL.legend(loc="upper left")
    # right: closure error (log)
    errs = [max(err_mol, 1e-16), max(err_grm, 1e-16)]
    bars = axR.bar(["MoL", "GRM"], errs, color=[C["mol"], C["grm"]])
    axR.set_yscale("log")
    axR.set_ylabel("|mass-balance closure error|")
    axR.set_title("Mass conservation")
    axR.axhline(1.0, color="0.5", ls=":", lw=1)
    for b, e in zip(bars, errs):
        axR.text(b.get_x() + b.get_width() / 2, e * 1.5,
                 f"{e:.0e}" if e < 0.01 else f"{e*100:.0f}%",
                 ha="center", fontsize=9)
    save(fig, "grm_conservation.png")


def fig_intraparticle():
    """Radial pore-concentration profiles inside the bead at several axial positions."""
    print("[fig] grm_intraparticle")
    iso = cation_exchange(beta=[1.0], nu=[0.0], ionic_capacity=1000.0,
                          q_max=[20.0], linear=True)
    inj = Injection(feed=np.array([1.0]), start_cv=0.0, duration_cv=1e6)  # breakthrough
    prog = ElutionProgram.isocratic(100.0, inj, run_cv=6.0)
    part = ParticleProperties(radius=4e-5, porosity=0.5, pore_diffusivity=5e-12,
                              film_coeff=3e-5, n_radial=14)
    s = GRMSetup(geometry=GEOM, velocity=U, dispersion=2e-7, isotherm=iso,
                 program=prog, particle=part, n_cells=40)
    r = run_grm(s, n_steps=300)
    Nr = part.n_radial
    # radial cell-centre positions (fraction of Rp)
    rr = (np.arange(Nr) + 0.5) / Nr
    Nz = s.n_cells
    positions = {"inlet (z=0)": 0, "1/4": Nz // 4, "mid": Nz // 2,
                 "3/4": 3 * Nz // 4}
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for label, j in positions.items():
        ax.plot(rr, r.cp_profile[0, j], "-o", ms=3, label=f"{label}")
    ax.set_xlabel("radial position  r / R$_p$   (0 = centre, 1 = surface)")
    ax.set_ylabel("pore concentration c$_p$ (g/L)")
    ax.set_title("Resolved intraparticle profiles (snapshot during loading)")
    ax.legend(title="column position", loc="upper left")
    save(fig, "grm_intraparticle.png")


def main():
    fig_validation()
    fig_conservation()
    fig_intraparticle()
    print("done.")


if __name__ == "__main__":
    main()
