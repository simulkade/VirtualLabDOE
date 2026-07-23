"""Figures for Chapter 4 (benchtop physical & chemical systems for DoE practice).

Self-contained like ``make_fermentation_figures``: sets up its own output
directory and matplotlib style and exposes ``main()`` so ``make_figures.py`` can
call it.  Every curve is produced from the *actual* benchtop model code.

Run directly::

    python doc/scripts/make_benchtop_figures.py
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
    "pipe": "#1f4e79",
    "lam": "#1f77b4",
    "turb": "#d62728",
    "ball": "#2ca02c",
    "bias": "#d62728",
    "yog": "#7e4794",
    "warm": "#ff7f0e",
    "est": "#1f4e79",
    "cat": "#d62728",
    "eq": "#7f7f7f",
}


def save(fig, name: str) -> None:
    path = FIGDIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(HERE.parent.parent)}")


def figure(fn):
    def wrapper():
        print(f"[fig] {fn.__name__}")
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"  !! {fn.__name__} failed: {exc!r}")

    return wrapper


# ── 1. Pipe: the pressure-drop characteristic and its two regimes ─────────────
@figure
def fig_pipe_characteristic():
    from vlab_doe.models.benchtop import pipe_flow as pf

    q = np.linspace(2e-7, 6e-4, 400)
    curve = pf.flow_curve(q, diameter=4e-3, length=1.0, temperature=20.0)
    hp = np.array([
        pf.hagen_poiseuille(pf.PipeFlowConfig(flow_rate=float(qq), diameter=4e-3, length=1.0))
        for qq in q
    ])
    lam = curve["reynolds"] <= 2300.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.1))
    q_lpm = q * 6.0e4  # m³/s → L/min
    ax1.plot(q_lpm, curve["pressure_drop"] / 1e3, color=C["pipe"], lw=2.3, label="model $\\Delta P$")
    ax1.plot(q_lpm, hp / 1e3, ls="--", color=C["lam"], lw=1.6,
             label="Hagen--Poiseuille (laminar)")
    ax1.axvline(q_lpm[np.argmax(~lam)], ls=":", color="0.5", lw=1)
    ax1.text(q_lpm[np.argmax(~lam)] * 1.02, ax1.get_ylim()[1] * 0.6,
             "transition\n(Re$\\approx$2300)", fontsize=8, color="0.4")
    ax1.set(xlabel="flow rate $Q$ (L/min)", ylabel="pressure drop $\\Delta P$ (kPa)",
            title="Pressure drop vs. flow rate")
    ax1.legend(fontsize=8, loc="upper left")

    ax2.loglog(curve["reynolds"], curve["friction_factor"], color=C["pipe"], lw=2.3)
    ax2.loglog(curve["reynolds"][lam], 64.0 / curve["reynolds"][lam], ls="--",
               color=C["lam"], lw=1.6, label="$64/\\mathrm{Re}$")
    ax2.loglog(curve["reynolds"], 0.316 * curve["reynolds"] ** -0.25, ls=":",
               color=C["turb"], lw=1.6, label="Blasius $0.316\\,\\mathrm{Re}^{-1/4}$")
    ax2.axvspan(2300, 4000, color="0.85", alpha=0.6)
    ax2.set(xlabel="Reynolds number", ylabel="friction factor $f$",
            title="The friction-factor law")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "benchtop_pipe.png")


# ── 2. Pipe: a factorial reveals power-law exponents as log-linear effects ────
@figure
def fig_pipe_effects():
    from vlab_doe.doe.analysis import fit_response_model
    from vlab_doe.doe.factorial import Factor, full_factorial, run_design
    from vlab_doe.models.benchtop import pipe_flow as pf

    factors = [
        Factor("logQ", np.log(1e-6), np.log(4e-6)),
        Factor("logL", np.log(0.5), np.log(2.0)),
        Factor("logd", np.log(3e-3), np.log(5e-3)),
    ]
    design = full_factorial(factors, levels=2, center_points=3)

    def evaluate(point):
        cfg = pf.PipeFlowConfig(flow_rate=np.exp(point["logQ"]),
                                length=np.exp(point["logL"]),
                                diameter=np.exp(point["logd"]), temperature=20.0)
        return {"logdP": np.log(pf.pressure_drop(cfg))}

    data = run_design(design, evaluate)
    res = fit_response_model(data, "logdP", ["logQ", "logL", "logd"], interactions=False)
    coefs = {"log Q\n(exact +1)": res.effects["logQ"],
             "log L\n(exact +1)": res.effects["logL"],
             "log d\n(exact -4)": res.effects["logd"]}
    truth = [1.0, 1.0, -4.0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.1))
    names = list(coefs)
    vals = list(coefs.values())
    ax1.bar(names, vals, color=[C["lam"], C["ball"], C["turb"]], alpha=0.85)
    ax1.scatter(names, truth, color="k", zorder=5, marker="D", s=40, label="mechanistic exponent")
    ax1.axhline(0, color="0.6", lw=0.8)
    ax1.set(ylabel="fitted log--log slope", title="Factorial recovers the power-law exponents")
    ax1.legend(fontsize=8)

    pred = res.model.fittedvalues
    ax2.scatter(data["logdP"], pred, color=C["pipe"], s=28)
    lo, hi = data["logdP"].min(), data["logdP"].max()
    ax2.plot([lo, hi], [lo, hi], ls="--", color="0.5")
    ax2.set(xlabel="observed $\\log\\Delta P$", ylabel="predicted",
            title=f"Fit quality ($R^2={res.model.rsquared:.4f}$)")
    fig.tight_layout()
    save(fig, "benchtop_pipe_effects.png")


# ── 3. Falling ball: calibration curve and the inertial bias ──────────────────
@figure
def fig_falling_ball():
    from vlab_doe.models.benchtop import falling_ball as fb

    mus = np.logspace(-1, 1.0, 40)  # 0.1 .. 10 Pa·s
    def cfg(mu, d=2e-3):
        return fb.FallingBallConfig(ball_diameter=d, ball_density=7800.0,
                                    fluid_density=1260.0, fluid_viscosity=mu,
                                    tube_diameter=0.03, fall_distance=0.1)
    times = np.array([fb.fall_time(cfg(m)) for m in mus])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.1))
    ax1.loglog(mus, times, "o-", color=C["ball"], lw=2, ms=4)
    ax1.set(xlabel="true viscosity $\\mu$ (Pa·s)", ylabel="fall time over 0.1 m (s)",
            title="The calibration curve (2 mm steel ball)")

    # Inertial bias: naive Stokes reading vs true, across ball sizes.
    diams = np.linspace(0.5e-3, 4e-3, 30)
    est = np.array([
        fb.infer_viscosity(fb.fall_time(cfg(1.0, d)), cfg(1.0, d), stokes_only=True)
        for d in diams
    ])
    re = np.array([fb.reynolds_number(cfg(1.0, d)) for d in diams])
    ax2.plot(re, 100.0 * (est - 1.0) / 1.0, color=C["bias"], lw=2.3)
    ax2.axhline(0, color="0.6", lw=0.8)
    ax2.set(xlabel="particle Reynolds number", ylabel="Stokes-reading error (%)",
            title="Inertial bias of the naive reading")
    ax2.text(0.05, 0.9, "true $\\mu=1$ Pa·s;\nStokes ignores the\n$0.15\\,\\mathrm{Re}^{0.687}$ drag term",
             transform=ax2.transAxes, fontsize=8, va="top", color="0.35")
    fig.tight_layout()
    save(fig, "benchtop_falling_ball.png")


# ── 4. Back extrusion: yogurt texture — speed, yield stress, temperature ──────
@figure
def fig_back_extrusion():
    from vlab_doe.models.benchtop import back_extrusion as be

    hb = be.HerschelBulkley(consistency=20.0, flow_index=0.4, yield_stress=30.0)
    speeds = np.linspace(0.2e-3, 5e-3, 40)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.1))
    for tau0, col, lab in [(0.0, C["lam"], "$\\tau_0=0$ (drinkable)"),
                           (30.0, C["yog"], "$\\tau_0=30$ Pa (set)"),
                           (80.0, C["turb"], "$\\tau_0=80$ Pa (firm)")]:
        rh = be.HerschelBulkley(consistency=20.0, flow_index=0.4, yield_stress=tau0)
        f = np.array([
            be.peak_force(be.BackExtrusionConfig(rheology=rh, probe_speed=float(v),
                                                 temperature=10.0, density=1040.0))
            for v in speeds
        ])
        ax1.plot(speeds * 1e3, f, color=col, lw=2.2, label=lab)
    ax1.set(xlabel="probe speed (mm/s)", ylabel="peak force (N)",
            title="Shear-thinning force curves ($n=0.4$)")
    ax1.legend(fontsize=8)

    temps = np.linspace(5.0, 30.0, 30)
    f_t = np.array([
        be.peak_force(be.BackExtrusionConfig(rheology=hb, probe_speed=1e-3,
                                             temperature=float(t), density=1040.0))
        for t in temps
    ])
    ax2.plot(temps, f_t, color=C["warm"], lw=2.3)
    ax2.set(xlabel="product temperature (°C)", ylabel="peak force at 1 mm/s (N)",
            title="Firmness softens as the gel warms")
    fig.tight_layout()
    save(fig, "benchtop_back_extrusion.png")


# ── 5. Ester hydrolysis: kinetics, catalyst vs. equilibrium, van 't Hoff ──────
@figure
def fig_ester_kinetics():
    from vlab_doe.models.benchtop import ester_hydrolysis as eh

    t = np.linspace(0, 8 * 3600, 400)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.1))

    for cH, col in [(0.05, "#9ecae1"), (0.1, C["est"]), (0.4, C["cat"])]:
        cfg = eh.EsterHydrolysisConfig(temperature=25.0, catalyst_conc=cH)
        out = eh.simulate(cfg, t)
        ax1.plot(t / 3600.0, out["conversion"], color=col, lw=2.2,
                 label=f"$c_{{H^+}}={cH}$ M")
    xeq = eh.equilibrium_conversion(eh.EsterHydrolysisConfig(temperature=25.0))
    ax1.axhline(xeq, ls=":", color=C["eq"], lw=1.6, label="equilibrium (same for all)")
    ax1.set(xlabel="time (h)", ylabel="ester conversion", ylim=(0, xeq * 1.25),
            title="Catalyst speeds the rate, not the equilibrium")
    ax1.legend(fontsize=8, loc="lower right")

    temps = np.linspace(15.0, 55.0, 40)
    xeq_t = np.array([eh.equilibrium_conversion(eh.EsterHydrolysisConfig(temperature=float(T)))
                      for T in temps])
    ax2.plot(temps, xeq_t, color=C["est"], lw=2.3)
    ax2.axvline(eh.METHYL_ACETATE_BP_C, ls="--", color=C["cat"], lw=1.4)
    ax2.text(eh.METHYL_ACETATE_BP_C - 1.0, xeq_t.min() + 0.2 * np.ptp(xeq_t),
             "boiling point\n56.9 °C", ha="right", fontsize=8, color=C["cat"])
    ax2.set(xlabel="temperature (°C)", ylabel="equilibrium conversion",
            title="Endothermic: warmer converts more")
    fig.tight_layout()
    save(fig, "benchtop_ester.png")


# ── 6. Ester DoE: a response surface of conversion in (T, time) ───────────────
@figure
def fig_ester_surface():
    from vlab_doe.models.benchtop import ester_hydrolysis as eh

    temps = np.linspace(20.0, 55.0, 45)
    hours = np.linspace(0.5, 8.0, 45)
    T, H = np.meshgrid(temps, hours)
    Z = np.empty_like(T)
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            cfg = eh.EsterHydrolysisConfig(temperature=float(T[i, j]), catalyst_conc=0.1)
            Z[i, j] = eh.conversion_at(cfg, float(H[i, j]) * 3600.0)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    cf = ax.contourf(T, H, Z, levels=14, cmap="viridis")
    cs = ax.contour(T, H, Z, levels=[0.10, 0.15, 0.20, 0.25], colors="w", linewidths=0.8)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f")
    fig.colorbar(cf, ax=ax, label="ester conversion")
    ax.set(xlabel="temperature (°C)", ylabel="reaction time (h)",
           title="Conversion surface at $c_{H^+}=0.1$ M")
    ax.grid(False)
    fig.tight_layout()
    save(fig, "benchtop_ester_surface.png")


# ── 7. Student capstone: screen → diagnose curvature → RSM → design space ──────
@figure
def fig_student_workflow():
    """The full design--run--analyse cycle a student would follow on the ester
    system, using the package's own factorial, perturbation, and analysis code."""
    import pandas as pd  # noqa: F401
    from vlab_doe.config import make_rng
    from vlab_doe import perturbation as pert
    from vlab_doe.doe.analysis import (fit_response_model,
                                             proven_acceptable_ranges)
    from vlab_doe.doe.factorial import Factor, full_factorial, run_design
    from vlab_doe.models.benchtop import ester_hydrolysis as eh

    rng = make_rng(7)
    factors = [Factor("temperature", 25.0, 50.0),
               Factor("catalyst", 0.05, 0.40),
               Factor("time_h", 0.5, 4.0)]
    noise = pert.NoiseModel(additive_sd=0.004, proportional_cv=0.02)

    def run_ester(pt):
        cfg = eh.EsterHydrolysisConfig(temperature=pt["temperature"], catalyst_conc=pt["catalyst"])
        return {"conversion": eh.conversion_at(cfg, pt["time_h"] * 3600.0)}

    # Stage 1 — 2-level screen with centre points
    screen = full_factorial(factors, levels=2, center_points=4)
    sdata = run_design(screen, run_ester)
    sdata["conversion"] = pert.add_measurement_noise(
        sdata["time_h"].to_numpy(), sdata["conversion"].to_numpy(), noise, rng)
    sfit = fit_response_model(sdata, "conversion", [f.name for f in factors], interactions=True)

    # Stage 2 — 3-level response-surface design
    rsm = full_factorial(factors, levels=3, center_points=3)
    rdata = run_design(rsm, run_ester)
    rdata["conversion"] = pert.add_measurement_noise(
        rdata["time_h"].to_numpy(), rdata["conversion"].to_numpy(), noise, rng)
    rfit = fit_response_model(rdata, "conversion", [f.name for f in factors],
                              interactions=True, quadratic=True)
    par = proven_acceptable_ranges(rfit, response_spec=(0.18, 0.30))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

    # Left: the escalation, seen as predicted-vs-observed for both models.
    is_cp = (sdata["run_type"] == "center_point").to_numpy()
    ax1.scatter(sdata["conversion"][~is_cp], sfit.model.fittedvalues[~is_cp],
                color="#9ecae1", s=45, label="2-level: factorial corners")
    ax1.scatter(sdata["conversion"][is_cp], sfit.model.fittedvalues[is_cp],
                color="#08519c", marker="s", s=60, label="2-level: centre points")
    ax1.scatter(rdata["conversion"], rfit.model.fittedvalues, color="#2ca02c",
                s=22, alpha=0.8, label=f"3-level RSM ($R^2={rfit.model.rsquared:.2f}$)")
    lims = [0.0, max(rdata["conversion"].max(), sdata["conversion"].max()) * 1.05]
    ax1.plot(lims, lims, ls="--", color="0.5")
    ax1.set(xlabel="observed conversion", ylabel="model prediction",
            title=f"Screen misses the curve ($R^2={sfit.model.rsquared:.2f}$); RSM fits it")
    ax1.legend(fontsize=8, loc="upper left")

    # Right: fitted RSM surface over the two dominant factors, with the spec window.
    cats = np.linspace(0.05, 0.40, 60)
    hours = np.linspace(0.5, 4.0, 60)
    Cx, Hy = np.meshgrid(cats, hours)
    grid = pd.DataFrame({"temperature": np.full(Cx.size, 37.5),
                         "catalyst": Cx.ravel(), "time_h": Hy.ravel()})
    Z = np.asarray(rfit.model.predict(grid)).reshape(Cx.shape)
    cf = ax2.contourf(Cx, Hy, Z, levels=14, cmap="viridis")
    ax2.contourf(Cx, Hy, np.ma.masked_outside(Z, 0.18, 0.30), levels=[0.18, 0.30],
                 colors="none", hatches=["////"])
    ax2.contour(Cx, Hy, Z, levels=[0.18, 0.30], colors="w", linewidths=1.4)
    fig.colorbar(cf, ax=ax2, label="fitted conversion")
    ax2.scatter(rdata["catalyst"], rdata["time_h"], c="0.9", s=10, edgecolors="k",
                linewidths=0.3, label="design points")
    ax2.set(xlabel="catalyst $c_{H^+}$ (mol/L)", ylabel="reaction time (h)",
            title="Design space at $37.5^\\circ$C (hatched = spec [0.18, 0.30])")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(False)
    fig.tight_layout()
    save(fig, "benchtop_student_workflow.png")

    # Echo the numbers the chapter quotes, so prose and figure stay in sync.
    cp_mean = sdata.loc[is_cp, "conversion"].mean()
    fc_mean = sdata.loc[~is_cp, "conversion"].mean()
    print(f"    [workflow] screen R2={sfit.model.rsquared:.3f}, RSM R2={rfit.model.rsquared:.3f}, "
          f"curvature={cp_mean - fc_mean:+.3f}")
    print("    [workflow] PAR:\n" + par.round(3).to_string(index=False))


def main() -> int:
    for fn in (
        fig_pipe_characteristic,
        fig_pipe_effects,
        fig_falling_ball,
        fig_back_extrusion,
        fig_ester_kinetics,
        fig_ester_surface,
        fig_student_workflow,
    ):
        fn()
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
