"""Generate every figure used in the monograph from the *actual* package code.

Each figure is produced inside its own ``try/except`` block so that a single
failure never aborts the whole build; the LaTeX source guards every
``\\includegraphics`` with ``\\IfFileExists`` so a missing figure degrades the
document gracefully rather than breaking the compile.

Run from anywhere::

    python doc/scripts/make_figures.py

Output: ``doc/figures/*.png`` (150 dpi, tight bounding boxes).
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

# Make the package importable whether or not it is pip-installed.
SRC = HERE.parent.parent / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.autolayout": True,
    }
)

C = {
    "cex": "#1f77b4",
    "hic": "#d62728",
    "rp": "#2ca02c",
    "mod": "#7f7f7f",
    "a": "#1f77b4",
    "b": "#ff7f0e",
    "c": "#2ca02c",
}


def save(fig, name: str) -> None:
    path = FIGDIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(HERE.parent.parent)}")


def figure(fn):
    """Decorator: run a figure builder, reporting success/failure independently."""

    def wrapper():
        print(f"[fig] {fn.__name__}")
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - we want to keep going
            print(f"  !! {fn.__name__} failed: {exc!r}")

    return wrapper


# ── 1. Modulator affinity laws b(m) for the three modes ───────────────────────
@figure
def fig_affinity_laws():
    from vlab_doe.models.chromatography.isotherms import (
        LinearSolventStrengthLaw,
        SaltingOutLaw,
        SMALaw,
    )

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))

    salt = np.linspace(20, 1000, 300)
    sma = SMALaw(beta=np.array([1.0]), nu=np.array([4.5]), ionic_capacity=1000.0,
                 nu_ph=np.array([0.0]))
    axes[0].semilogy(salt, sma.b(salt, 7.0)[0], color=C["cex"])
    axes[0].set_title("Ion exchange (SMA)\n$b=\\beta(\\Lambda/m)^{\\nu}$")
    axes[0].set_xlabel("salt $m$ (mM)")
    axes[0].set_ylabel("affinity $b$")

    so = SaltingOutLaw(beta=np.array([0.02]), ks=np.array([0.006]))
    axes[1].semilogy(salt, so.b(salt, 7.0)[0], color=C["hic"])
    axes[1].set_title("HIC (salting-out)\n$b=\\beta\\,e^{K_s m}$")
    axes[1].set_xlabel("salt $m$ (mM)")

    phi = np.linspace(0.0, 1.0, 300)
    lss = LinearSolventStrengthLaw(beta=np.array([5.0e3]), s=np.array([12.0]))
    axes[2].semilogy(phi, lss.b(phi, 7.0)[0], color=C["rp"])
    axes[2].set_title("RP-HPLC (LSS)\n$b=\\beta\\,e^{-S\\varphi}$")
    axes[2].set_xlabel("organic fraction $\\varphi$")

    save(fig, "affinity_laws.png")


# ── 2. Henry constant surface H(salt, pH) for SMA ─────────────────────────────
@figure
def fig_henry_surface():
    from vlab_doe.models.chromatography.isotherms import (
        SMAParameters,
        sma_henry_constant,
    )

    salt = np.linspace(50, 600, 120)
    ph = np.linspace(4.0, 8.0, 120)
    S, P = np.meshgrid(salt, ph)
    params = SMAParameters(
        equilibrium_constant=[5.0],
        characteristic_charge=[4.0],
        steric_factor=[10.0],
        ionic_capacity=1000.0,
        ph_ref=7.0,
        nu_ph=-1.2,
    )
    H = np.vectorize(lambda s, p: sma_henry_constant(s, p, params))(S, P)

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    cf = ax.contourf(S, P, np.log10(H), levels=20, cmap="viridis")
    cs = ax.contour(S, P, np.log10(H), levels=8, colors="white", linewidths=0.6, alpha=0.7)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.0f")
    ax.set_xlabel("salt (mM)")
    ax.set_ylabel("pH")
    ax.set_title("$\\log_{10} H$  (SMA Henry constant, CEX)")
    fig.colorbar(cf, ax=ax, label="$\\log_{10} H$")
    save(fig, "henry_surface.png")


# ── 3. CEX linear-gradient chromatogram with the modulator overlay ────────────
@figure
def fig_cex_gradient():
    from vlab_doe.models.chromatography import (
        ColumnGeometry,
        ColumnSetup,
        ElutionProgram,
        Injection,
        cation_exchange,
        run_column,
    )

    geom = ColumnGeometry(length=0.1, diameter=0.01, porosity=0.4)
    iso = cation_exchange(beta=[1.0], nu=[3.0], ionic_capacity=1000.0,
                          q_max=[1.0], linear=True)
    inj = Injection.from_load_density(8.0, feed=1.0, porosity=0.4)
    prog = ElutionProgram.linear_gradient(
        inj, m_start=300.0, m_end=900.0, gradient_cv=20.0,
        equilibrate_cv=2.0, wash_cv=2.0, strip_cv=5.0,
    )
    setup = ColumnSetup(geometry=geom, velocity=2e-3, dispersion=1e-6,
                        isotherm=iso, program=prog, ph=5.0,
                        mass_transfer=5.0, n_cells=80)
    res = run_column(setup, atol=1e-6, rtol=1e-5)
    t = res.t / 60.0  # minutes

    fig, ax1 = plt.subplots(figsize=(7.2, 4.0))
    ax1.plot(t, res.c_outlet[0], color=C["cex"], lw=2, label="protein (UV)")
    ax1.fill_between(t, res.c_outlet[0], color=C["cex"], alpha=0.18)
    ax1.set_xlabel("time (min)")
    ax1.set_ylabel("outlet concentration (g/L)", color=C["cex"])
    ax1.tick_params(axis="y", labelcolor=C["cex"])

    ax2 = ax1.twinx()
    ax2.plot(t, res.m_outlet, color=C["mod"], lw=1.6, ls="--", label="salt gradient")
    ax2.set_ylabel("salt (mM)", color=C["mod"])
    ax2.tick_params(axis="y", labelcolor=C["mod"])
    ax2.grid(False)
    ax1.set_title("CEX linear salt-gradient elution")
    save(fig, "cex_gradient.png")


# ── 4. Gradient-slope effect: the resolution / dilution trade-off ─────────────
@figure
def fig_gradient_slope():
    from vlab_doe.models.chromatography import (
        ColumnGeometry,
        ColumnSetup,
        ElutionProgram,
        Injection,
        cation_exchange,
        run_column,
    )
    from vlab_doe.models.chromatography.metrics import peak_moments, plate_count

    geom = ColumnGeometry(length=0.1, diameter=0.01, porosity=0.4)
    fig, ax1 = plt.subplots(figsize=(7.4, 4.2))
    ax2 = ax1.twinx()
    ax2.grid(False)

    slopes = [(10.0, C["c"], "steep (10 CV)"),
              (20.0, C["cex"], "medium (20 CV)"),
              (40.0, C["hic"], "shallow (40 CV)")]
    for gcv, col, label in slopes:
        iso = cation_exchange(beta=[1.0], nu=[3.0], ionic_capacity=1000.0,
                              q_max=[1.0], linear=True)
        inj = Injection.from_load_density(8.0, feed=1.0, porosity=0.4)
        prog = ElutionProgram.linear_gradient(
            inj, m_start=300.0, m_end=900.0, gradient_cv=gcv, strip_cv=5.0)
        setup = ColumnSetup(geometry=geom, velocity=2e-3, dispersion=1e-6,
                            isotherm=iso, program=prog, ph=5.0,
                            mass_transfer=5.0, n_cells=80)
        res = run_column(setup, atol=1e-6, rtol=1e-5)
        t = res.t / 60.0
        N = plate_count(res.t, res.c_outlet[0])
        ax1.plot(t, res.c_outlet[0], color=col, lw=2,
                 label=f"{label}, $N\\approx${N:.0f}")
        ax1.fill_between(t, res.c_outlet[0], color=col, alpha=0.12)
        ax2.plot(t, res.m_outlet, color=col, lw=1.0, ls=":", alpha=0.6)

    ax1.set_xlabel("time (min)")
    ax1.set_ylabel("outlet concentration (g/L)")
    ax2.set_ylabel("salt (mM)  [dotted]", color=C["mod"])
    ax2.tick_params(axis="y", labelcolor=C["mod"])
    ax1.set_title("Gradient slope sets the peak: sharper & earlier vs broad & dilute")
    ax1.legend(loc="upper right")
    save(fig, "gradient_slope.png")


# ── 5. LHS vs random space-filling ────────────────────────────────────────────
@figure
def fig_lhs_vs_random():
    from vlab_doe.doe.factorial import Factor
    from vlab_doe.doe.lhs import coverage_metrics, latin_hypercube

    factors = [Factor("pH", 4.0, 8.0), Factor("salt", 50.0, 600.0)]
    n = 20
    lhs = latin_hypercube(factors, n, seed=1)
    rng = np.random.default_rng(1)
    rnd = rng.uniform([4.0, 50.0], [8.0, 600.0], size=(n, 2))

    cov_l = coverage_metrics(lhs)
    import pandas as pd

    cov_r = coverage_metrics(pd.DataFrame(rnd, columns=["pH", "salt"]))

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharey=True)
    axes[0].scatter(rnd[:, 0], rnd[:, 1], c=C["hic"], s=40, edgecolor="k", lw=0.4)
    axes[0].set_title(f"Random\n$D_{{CD}}$={cov_r['discrepancy']:.3f}")
    axes[1].scatter(lhs["pH"], lhs["salt"], c=C["cex"], s=40, edgecolor="k", lw=0.4)
    axes[1].set_title(f"Optimised LHS\n$D_{{CD}}$={cov_l['discrepancy']:.3f}")
    for ax in axes:
        ax.set_xlabel("pH")
        # marginal rug
        ax.set_xticks(np.linspace(4, 8, 5))
    axes[0].set_ylabel("salt (mM)")
    save(fig, "lhs_vs_random.png")


# ── 6. 2^3 full-factorial design cube with centre point ───────────────────────
@figure
def fig_factorial_cube():
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    pts = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)],
                   dtype=float)
    fig = plt.figure(figsize=(5.6, 5.0))
    ax = fig.add_subplot(111, projection="3d")
    # edges
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if np.sum(np.abs(pts[i] - pts[j])) == 2:  # differ in one coord
                ax.plot(*zip(pts[i], pts[j]), color="0.6", lw=1.0)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=70, c=C["cex"],
               depthshade=False, edgecolor="k")
    ax.scatter([0], [0], [0], s=90, c=C["hic"], marker="D",
               edgecolor="k", label="centre point")
    ax.set_xlabel("pH (coded)")
    ax.set_ylabel("salt (coded)")
    ax.set_zlabel("load (coded)")
    ax.set_title("$2^3$ full factorial + centre point")
    ax.legend(loc="upper left")
    ax.view_init(elev=18, azim=-58)
    save(fig, "factorial_cube.png")


# ── 7. Bayesian optimisation vs DoE convergence (synthetic oracle) ────────────
@figure
def fig_bo_convergence():
    # A cheap, deterministic 2-D oracle so the figure is fast and reproducible.
    from vlab_doe.doe.factorial import Factor, full_factorial, run_design
    from vlab_doe.optimization.bayesopt import (
        Objective,
        bayesian_optimization,
        compare_to_doe,
    )

    factors = [Factor("pH", 4.0, 8.0), Factor("salt", 50.0, 600.0)]

    def oracle(point):
        x = (point["pH"] - 6.0) / 2.0
        y = (point["salt"] - 325.0) / 275.0
        val = np.exp(-(x ** 2 + y ** 2)) + 0.15 * np.exp(-((x - 1) ** 2 + (y + 1) ** 2) / 0.1)
        return {"yield": float(val)}

    bo = bayesian_optimization(
        factors, oracle, Objective(maximize="yield"),
        n_initial=5, n_iterations=20, seed=0,
    )
    grid = full_factorial(factors, levels=3, center_points=0)
    doe = run_design(grid, oracle)
    comp = compare_to_doe(bo, doe, Objective(maximize="yield"))

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for method, col in (("BO", C["cex"]), ("DoE", C["hic"])):
        sub = comp[comp["method"] == method]
        ax.plot(sub["n_evaluations"], sub["running_best"], "-o", ms=3,
                color=col, label=method)
    ax.set_xlabel("number of experiments")
    ax.set_ylabel("running-best yield")
    ax.set_title("Adaptive (BO) vs one-shot factorial search")
    ax.legend()
    save(fig, "bo_convergence.png")


# ── 8. GLM analysis of chromatography data: a probabilistic design space ───────
@figure
def fig_chrom_glm():
    """Logistic + Poisson GLMs over a two-CPP CEX charge-variant separation.

    Each design point is a real ``run_column`` simulation of a 4-component
    cation-exchange separation (target + three close charge variants); from the
    pooled chromatogram we record whether the run meets a purity/yield spec
    (a 0/1 outcome) and how many variants co-elute under the target peak (a
    count).  Logistic and Poisson GLMs then map the probabilistic design space.
    """
    import pandas as pd

    from vlab_doe.doe.analysis import fit_glm_response, predict_glm_grid
    from vlab_doe.doe.factorial import Factor
    from vlab_doe.doe.lhs import latin_hypercube
    from vlab_doe.models.chromatography import (
        ColumnGeometry,
        ColumnSetup,
        ElutionProgram,
        Injection,
        cation_exchange,
        run_column,
    )
    from vlab_doe.models.chromatography.metrics import peak_moments, pool_metrics

    geom = ColumnGeometry(length=0.1, diameter=0.01, porosity=0.4)
    nu = [3.00, 2.93, 3.07, 3.14]          # target + 3 close charge variants
    nu_ph = [-1.0, -0.7, -1.25, -1.5]      # differing pH sensitivity => pH is a selectivity lever
    beta = [1.0, 0.95, 1.08, 1.15]
    feed = np.array([1.0, 0.35, 0.35, 0.35])
    PUR_SPEC, YLD_SPEC = 0.95, 0.80

    def evaluate(ph, gcv):
        iso = cation_exchange(beta=beta, nu=nu, ionic_capacity=1000.0, q_max=1.0,
                              nu_ph=nu_ph, ph_ref=7.0, linear=True)
        inj = Injection.from_load_density(8.0, feed=feed, porosity=0.4)
        prog = ElutionProgram.linear_gradient(
            inj, m_start=300.0, m_end=900.0, gradient_cv=gcv,
            equilibrate_cv=2.0, wash_cv=2.0, strip_cv=5.0)
        setup = ColumnSetup(geometry=geom, velocity=2e-3, dispersion=4e-6, isotherm=iso,
                            program=prog, ph=ph, mass_transfer=5.0, n_cells=80)
        res = run_column(setup, atol=1e-6, rtol=1e-5)
        t, ct = res.t, res.c_outlet[0]
        if ct.max() <= 1e-6:
            return 0.0, 0.0, 3
        apex_i = int(np.argmax(ct))
        sigma = max(peak_moments(t, ct)["sigma"], 1.0)
        cs, ce = t[apex_i] - 1.5 * sigma, t[apex_i] + 1.5 * sigma
        pm = pool_metrics(t, res.c_outlet, cut_start=cs, cut_end=ce, target_index=0)
        # count variants co-eluting under the target apex (> 5% of its height)
        k = int(sum(res.c_outlet[i, apex_i] > 0.05 * ct[apex_i] for i in (1, 2, 3)))
        return pm["yield"], pm["purity"], k

    factors = [Factor("pH", 4.5, 6.0), Factor("gradient_cv", 8.0, 40.0)]
    design = latin_hypercube(factors, 64, seed=7)
    rows = []
    for _, r in design.iterrows():
        y, p, k = evaluate(float(r.pH), float(r.gradient_cv))
        rows.append(dict(pH=r.pH, gradient_cv=r.gradient_cv, contam=k,
                         passed=int(p >= PUR_SPEC and y >= YLD_SPEC)))
    df = pd.DataFrame(rows)

    logit = fit_glm_response(df, "passed", ["pH", "gradient_cv"], family="binomial")
    pois = fit_glm_response(df, "contam", ["pH", "gradient_cv"], family="poisson")
    xr, yr = (4.5, 6.0), (8.0, 40.0)
    Xp, Yp, P = predict_glm_grid(logit, "pH", "gradient_cv", xr, yr, n=80)
    Xc, Yc, M = predict_glm_grid(pois, "pH", "gradient_cv", xr, yr, n=80)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # (a) logistic probabilistic design space
    cs1 = ax1.contourf(Xp, Yp, P, levels=np.linspace(0, 1, 11), cmap="Greens")
    fig.colorbar(cs1, ax=ax1, fraction=0.046, pad=0.04, label="$P(\\mathrm{meet\\ spec})$")
    ax1.contour(Xp, Yp, P, levels=[0.9], colors="k", linewidths=2, linestyles="--")
    ok = df.passed == 1
    ax1.scatter(df.pH[ok], df.gradient_cv[ok], c="white", edgecolor="k", s=26,
                label="pass", zorder=3)
    ax1.scatter(df.pH[~ok], df.gradient_cv[~ok], c=C["hic"], marker="x", s=26,
                label="fail", zorder=3)
    ax1.set(xlabel="load pH", ylabel="gradient length (CV)",
            title=f"(a) logistic: design space  (pseudo-$R^2$={logit.pseudo_r2:.2f})")
    ax1.legend(fontsize=8, loc="upper right", framealpha=0.9)

    # (b) Poisson expected number of co-eluting variants
    cs2 = ax2.contourf(Xc, Yc, M, levels=12, cmap="OrRd")
    fig.colorbar(cs2, ax=ax2, fraction=0.046, pad=0.04, label="$E[\\#\\,\\mathrm{co\\text{-}eluting}]$")
    sc = ax2.scatter(df.pH, df.gradient_cv, c=df.contam, cmap="OrRd", edgecolor="k",
                     s=30, zorder=3, vmin=0, vmax=df.contam.max())
    ax2.set(xlabel="load pH", ylabel="gradient length (CV)",
            title=f"(b) Poisson: co-eluting variants  (pseudo-$R^2$={pois.pseudo_r2:.2f})")
    save(fig, "chrom_glm.png")


# ── 9. Capture-step breakthrough: a logistic design space for recovery ────────
@figure
def fig_capture_glm():
    """Breakthrough curves and a logistic recovery design space for a CEX capture.

    A single-component frontal-loading study: vary the load density and the load
    conductivity (salt), measure the recovery (one minus the flow-through loss),
    and ask---per QbD---for the probability of meeting a recovery spec.  The
    boundary is genuinely probabilistic because replicate runs carry assay noise.
    """
    import pandas as pd

    from vlab_doe.config import make_rng
    from vlab_doe.doe.analysis import fit_glm_response, predict_glm_grid
    from vlab_doe.doe.factorial import Factor
    from vlab_doe.doe.lhs import latin_hypercube
    from vlab_doe.models.chromatography import (
        ColumnGeometry,
        ColumnSetup,
        ElutionProgram,
        Injection,
        Segment,
        cation_exchange,
        run_column,
    )
    from vlab_doe.perturbation import NoiseModel, add_measurement_noise

    L, dia, eps, feed = 0.1, 0.01, 0.4, 5.0
    geom = ColumnGeometry(length=L, diameter=dia, porosity=eps)
    u = L / (eps * 3.0 * 60.0)              # fixed 3-minute residence time

    def run(load_density, salt):
        iso = cation_exchange(beta=[0.004], nu=[2.5], ionic_capacity=1000.0,
                              q_max=[55.0], nu_ph=0.0, ph_ref=7.0, linear=False)
        inj = Injection.from_load_density(load_density, feed=feed, porosity=eps)
        lcv = inj.duration_cv
        segs = [Segment("equilibrate", 1.0, salt), Segment("load", lcv, salt),
                Segment("wash", 6.0, salt)]
        prog = ElutionProgram(segments=segs, injection=Injection(
            feed=np.array([feed]), start_cv=1.0, duration_cv=lcv))
        setup = ColumnSetup(geometry=geom, velocity=u, dispersion=2e-5, isotherm=iso,
                            program=prog, ph=7.0, mass_transfer=0.08, n_cells=40)
        res = run_column(setup)
        t_cv = L / (u * eps)
        loss = np.trapezoid(res.c_outlet[0], res.t) / (feed * lcv * t_cv)
        cv_since_load = res.t / t_cv - 1.0          # column volumes since load start
        return res, cv_since_load, float(np.clip(1.0 - loss, 0.0, 1.0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # (a) representative breakthrough curves
    cases = [(30.0, 50.0, C["cex"], "clean capture (low salt, low load)"),
             (55.0, 50.0, C["c"], "overload (low salt, high load)"),
             (30.0, 120.0, C["hic"], "weak binding (high salt)")]
    for ld, salt, col, label in cases:
        res, cv, _ = run(ld, salt)
        ax1.plot(cv, res.c_outlet[0] / feed, color=col, lw=2, label=label)
    ax1.axhline(0.1, color=C["mod"], ls=":", lw=1.3)
    ax1.text(0.2, 0.12, "10% breakthrough", fontsize=8, color=C["mod"])
    ax1.set(xlabel="throughput since load start (CV)",
            ylabel="$c_{\\rm out}/c_{\\rm feed}$",
            title="(a) breakthrough curves", xlim=(0, 12), ylim=(-0.02, 1.05))
    ax1.legend(fontsize=8, loc="center right")

    # (b) logistic recovery design space (replicate noisy assays => probabilistic)
    factors = [Factor("load_density", 15.0, 55.0), Factor("load_salt", 40.0, 140.0)]
    base = latin_hypercube(factors, 48, seed=5)
    yt = np.array([run(float(r.load_density), float(r.load_salt))[2]
                   for _, r in base.iterrows()])
    rng = make_rng(11)
    noise = NoiseModel(additive_sd=0.04)
    R, SPEC = 5, 0.90
    rows, pass_frac = [], []
    for (_, r), y in zip(base.iterrows(), yt):
        obs = add_measurement_noise(np.arange(R), np.full(R, y), noise, rng)
        verdicts = (obs >= SPEC).astype(int)
        pass_frac.append(verdicts.mean())
        for v in verdicts:
            rows.append(dict(load_density=r.load_density, load_salt=r.load_salt, passed=int(v)))
    df = pd.DataFrame(rows)

    glm = fit_glm_response(df, "passed", ["load_density", "load_salt"], family="binomial")
    X, Y, P = predict_glm_grid(glm, "load_density", "load_salt",
                               (15.0, 55.0), (40.0, 140.0), n=80)
    cs = ax2.contourf(X, Y, P, levels=np.linspace(0, 1, 11), cmap="Greens")
    fig.colorbar(cs, ax=ax2, fraction=0.046, pad=0.04, label="$P(\\mathrm{recovery}\\geq 90\\%)$")
    ax2.contour(X, Y, P, levels=[0.9], colors="k", linewidths=2, linestyles="--")
    ax2.scatter(base.load_density, base.load_salt, c=pass_frac, cmap="Greens",
                edgecolor="k", s=34, vmin=0, vmax=1, zorder=3)
    ax2.set(xlabel="load density (g/L resin)", ylabel="load conductivity / salt (mM)",
            title=f"(b) logistic recovery design space  (pseudo-$R^2$={glm.pseudo_r2:.2f})")
    save(fig, "capture_glm.png")


def main() -> int:
    for fn in (
        fig_affinity_laws,
        fig_henry_surface,
        fig_cex_gradient,
        fig_gradient_slope,
        fig_lhs_vs_random,
        fig_factorial_cube,
        fig_bo_convergence,
        fig_chrom_glm,
        fig_capture_glm,
    ):
        fn()
    # General rate model comparison figures (separate module; runs simulations).
    try:
        import make_grm_figures
        make_grm_figures.main()
    except Exception as exc:  # noqa: BLE001
        print(f"  !! GRM figures failed: {exc!r}")
    # Fermentation, covering-array, and screening figures (Chapter 3).
    try:
        import make_fermentation_figures
        make_fermentation_figures.main()
    except Exception as exc:  # noqa: BLE001
        print(f"  !! fermentation figures failed: {exc!r}")
    # Benchtop physical/chemical systems (Chapter 4).
    try:
        import make_benchtop_figures
        make_benchtop_figures.main()
    except Exception as exc:  # noqa: BLE001
        print(f"  !! benchtop figures failed: {exc!r}")
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
