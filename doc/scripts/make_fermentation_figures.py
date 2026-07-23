"""Figures for Chapter 3 (milk fermentation, strain screening, data analysis).

Self-contained like ``make_grm_figures``: it sets up its own output directory and
matplotlib style and exposes ``main()`` so ``make_figures.py`` can call it.  Every figure is
produced from the *actual* package code, so the curves are real model output, not schematics.

Run directly::

    python doc/scripts/make_fermentation_figures.py
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
    "ph": "#1f4e79",
    "st": "#1f77b4",
    "lb": "#d62728",
    "coop": "#2ca02c",
    "indep": "#7f7f7f",
    "acid": "#7e4794",
    "gel": "#ff7f0e",
    "set": "#d62728",
}
T_GRID = np.linspace(0, 12, 241)


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


# ── 1. The acidification curve and its hidden states ──────────────────────────
@figure
def fig_ferm_acidification():
    from vlab_doe.models import fermentation as ferm

    setup = ferm.FermentationSetup(consortium=ferm.yogurt_blend(0.5, 0.5), temperature=43.0)
    r = ferm.run_fermentation(setup, T_GRID)
    fp = ferm.fingerprint(r)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))
    ax1.plot(r.t, r.ph, color=C["ph"], lw=2.2)
    ax1.axhline(ferm.PH_GEL, ls=":", c=C["gel"], label="gelation (pH 5.2)")
    ax1.axhline(ferm.PH_SET, ls=":", c=C["set"], label="set point (pH 4.6)")
    if np.isfinite(fp["t_set"]):
        ax1.axvline(fp["t_set"], ls="--", c="0.5", lw=1)
    ax1.set(xlabel="time (h)", ylabel="pH", title="The observable: pH(t)")
    ax1.legend(loc="upper right", fontsize=9)

    for name, x, col in zip(r.strain_names, r.biomass, (C["st"], C["lb"])):
        ax2.plot(r.t, x, color=col, lw=2, label=f"{name} biomass")
    ax2.plot(r.t, r.substrate / r.substrate[0], "--", color="0.4", label="lactose (norm.)")
    ax2.plot(r.t, r.lactic_acid / max(r.lactic_acid.max(), 1e-9), "-.", color=C["acid"],
             label="lactic acid (norm.)")
    ax2.set(xlabel="time (h)", ylabel="level", title="The hidden latent states")
    ax2.legend(loc="center right", fontsize=9)
    save(fig, "ferm_acidification.png")


# ── 2. The three sub-models: temperature, lag, buffering ──────────────────────
@figure
def fig_ferm_submodels():
    from vlab_doe.models import fermentation as ferm
    from vlab_doe.models.fermentation.kinetics import baranyi_alpha
    from vlab_doe.models.fermentation.milk import Milk, ph_from_acid

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    # (a) Rosso cardinal-temperature factor
    T = np.linspace(10, 55, 300)
    st = ferm.streptococcus_thermophilus()
    lb = ferm.lactobacillus_bulgaricus()
    for s, col, lab in [(st, C["st"], "S. thermophilus"), (lb, C["lb"], "L. bulgaricus")]:
        g = [ferm.cardinal_temperature_factor(t, s) for t in T]
        axes[0].plot(T, g, color=col, lw=2, label=lab)
        axes[0].axvline(s.t_opt, color=col, ls=":", lw=1)
    axes[0].set(xlabel="temperature (°C)", ylabel=r"$\gamma_T$",
                title="Rosso cardinal-temperature law")
    axes[0].legend(fontsize=8)

    # (b) Baranyi lag gate and resulting biomass
    t = T_GRID
    for q0, col, lab in [(0.1, C["lb"], "$Q_0=0.1$ (long lag)"),
                         (0.5, C["st"], "$Q_0=0.5$ (short lag)")]:
        mu = 1.2
        Q = q0 * np.exp(mu * t)
        alpha = baranyi_alpha(Q)
        axes[1].plot(t, alpha, color=col, lw=2, label=lab)
    axes[1].set(xlabel="time (h)", ylabel=r"$\alpha(t)=Q/(1+Q)$",
                title="Baranyi lag adjustment")
    axes[1].legend(fontsize=8)

    # (c) Milk titration curve pH(L)
    L = np.linspace(0, 200, 300)
    for l50, col, lab in [(40, C["st"], "$L_{50}=40$ (low buffer)"),
                          (48, C["ph"], "$L_{50}=48$ (default)"),
                          (60, C["lb"], "$L_{50}=60$ (high buffer)")]:
        ph = ph_from_acid(L, Milk(l50=l50))
        axes[2].plot(L, ph, color=col, lw=2, label=lab)
    axes[2].axhline(4.6, ls=":", c="0.5")
    axes[2].set(xlabel="lactic acid $L$ (mmol/L)", ylabel="pH",
                title="Milk buffering titration")
    axes[2].legend(fontsize=8)
    save(fig, "ferm_submodels.png")


# ── 3. Strain personalities and proto-cooperation ─────────────────────────────
@figure
def fig_ferm_cooperation():
    from vlab_doe.models import fermentation as ferm

    cases = [
        ("S. thermophilus alone", ferm.single_strain(ferm.streptococcus_thermophilus()), C["st"], "-"),
        ("L. bulgaricus alone", ferm.single_strain(ferm.lactobacillus_bulgaricus()), C["lb"], "-"),
        ("ST + LB (cooperation)", ferm.yogurt_blend(0.5, 0.5, cooperation=1.5), C["coop"], "-"),
        ("ST + LB (independent)", ferm.yogurt_blend(0.5, 0.5, cooperation=0.0), C["indep"], "--"),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for label, cons, col, ls in cases:
        r = ferm.run_fermentation(ferm.FermentationSetup(consortium=cons, temperature=43.0), T_GRID)
        ax.plot(r.t, r.ph, color=col, lw=2.2, ls=ls, label=label)
    ax.axhline(ferm.PH_SET, ls=":", c="0.5")
    ax.set(xlabel="time (h)", ylabel="pH",
           title="Strain personalities: the yogurt symbiosis")
    ax.legend(loc="upper right", fontsize=9)
    save(fig, "ferm_cooperation.png")


# ── 4. Uncertainty vs variability vs process noise ────────────────────────────
@figure
def fig_ferm_uncertainty():
    from vlab_doe.config import make_rng
    from vlab_doe.models import fermentation as ferm

    rng = make_rng(7)
    setup = ferm.FermentationSetup(consortium=ferm.yogurt_blend(), temperature=43.0)
    r = ferm.run_fermentation(setup, T_GRID)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)

    sample_t = np.linspace(0, 12, 25)
    obs = ferm.observe_ph(r, sample_t, rng=rng)
    axes[0].plot(r.t, r.ph, color=C["ph"], lw=2, label="truth")
    axes[0].plot(obs["t"], obs["ph"], "o", ms=4, color=C["set"], label="pH probe")
    axes[0].set(title="1. Measurement uncertainty", xlabel="time (h)", ylabel="pH")
    axes[0].legend(fontsize=8)

    batches = ferm.run_batches(setup, ferm.BatchVariability(), 40, T_GRID, rng)
    B = np.array([b.ph for b in batches])
    axes[1].fill_between(T_GRID, B.min(0), B.max(0), alpha=0.2, color=C["st"], label="range")
    axes[1].plot(T_GRID, B.mean(0), color=C["st"], lw=2, label="mean")
    axes[1].set(title="2. Batch variability (40 batches)", xlabel="time (h)")
    axes[1].legend(fontsize=8)

    sde = ferm.FermentationSetup(consortium=ferm.yogurt_blend(), temperature=43.0,
                                 process_noise_sd=0.2)
    for _ in range(8):
        rs = ferm.run_fermentation(sde, T_GRID, rng=rng)
        axes[2].plot(rs.t, rs.ph, lw=1, alpha=0.7)
    axes[2].set(title="3. Process noise (SDE, 8 runs)", xlabel="time (h)")
    save(fig, "ferm_uncertainty.png")


# ── 5. Covering-array coverage growth and block-size effect ───────────────────
@figure
def fig_ferm_covering():
    from vlab_doe.doe.covering import covering_array

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))

    # Coverage growth vs number of runs, for fixed max block size.
    for mx, col in [(2, C["indep"]), (3, C["st"]), (5, C["coop"])]:
        d = covering_array(50, 200, min_size=min(2, mx), max_size=mx, seed=3)
        fracs = []
        for n in range(1, 201):
            partial = type(d)(runs=d.runs[:n], n_items=50)
            fracs.append(partial.coverage(2)["coverage_fraction"])
        ax1.plot(range(1, 201), fracs, color=col, lw=2, label=f"blocks of ≤{mx}")
    ax1.axhline(1.0, ls=":", c="0.5")
    ax1.set(xlabel="number of experiments", ylabel="fraction of strain pairs co-tested",
            title="Pair coverage grows with runs")
    ax1.legend(loc="lower right", fontsize=9)

    # The design we actually use: block-size distribution + appearance balance.
    d = covering_array(50, 200, min_size=2, max_size=5, seed=11)
    sizes = np.array([len(r) for r in d.runs])
    ax2.bar(range(50), d.appearances(), color=C["coop"], alpha=0.85)
    ax2.set(xlabel="strain index", ylabel="experiments used in",
            title=f"Balanced strain usage (50 strains, 200 runs)")
    txt = "  ".join(f"{k}:{int((sizes == k).sum())}" for k in (2, 3, 4, 5))
    ax2.text(0.02, 0.95, f"block sizes — {txt}", transform=ax2.transAxes,
             fontsize=8, va="top")
    save(fig, "ferm_covering.png")


# ── 6. Tree-model importance recovers the strong acidifiers ───────────────────
@figure
def fig_ferm_importance():
    from vlab_doe.config import make_rng
    from vlab_doe.doe.covering import covering_array
    from vlab_doe.doe.importance import (
        gradient_boosting_importance,
        random_forest_importance,
    )
    from vlab_doe.models import fermentation as ferm

    rng = make_rng(2026)
    lib = ferm.random_strain_library(50, rng)
    design = covering_array(50, 200, min_size=2, max_size=5, seed=11)

    var = ferm.BatchVariability()
    final_ph = []
    for members in design.runs:
        setup = ferm.FermentationSetup(consortium=lib.consortium(members), temperature=43.0)
        batch = ferm.sample_batch(setup, var, rng)
        res = ferm.run_fermentation(batch, T_GRID, rng=rng)
        obs = ferm.observe_ph(res, T_GRID, rng=rng)
        final_ph.append(float(obs["ph"][-1]))
    final_ph = np.array(final_ph)
    X = design.matrix()
    names = lib.names

    solo = np.array([
        ferm.run_fermentation(
            ferm.FermentationSetup(consortium=lib.consortium([i]), temperature=43.0), T_GRID
        ).ph[-1]
        for i in range(50)
    ])

    rf = random_forest_importance(X, final_ph, feature_names=names, seed=1, n_repeats=6)
    xgb = gradient_boosting_importance(X, final_ph, feature_names=names, seed=1, n_repeats=6)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    top = rf.importances.head(12)
    ypos = np.arange(len(top))[::-1]
    ax1.barh(ypos, top.values, color=C["st"], alpha=0.85, label=f"RF (R²={rf.cv_score:.2f})")
    ax1.barh(ypos, xgb.importances[top.index].values, color=C["gel"], alpha=0.5,
             label=f"XGB (R²={xgb.cv_score:.2f})")
    ax1.set_yticks(ypos)
    ax1.set_yticklabels(top.index, fontsize=8)
    ax1.set(xlabel=r"permutation importance ($\Delta R^2$)",
            title="Top strains driving final pH")
    ax1.legend(fontsize=9)

    imp_full = rf.importances.reindex(names).values
    ax2.scatter(-solo, imp_full, s=28, color=C["coop"], edgecolor="k", lw=0.3)
    corr = np.corrcoef(imp_full, -solo)[0, 1]
    ax2.set(xlabel="true acidifying power  (− solo final pH)",
            ylabel="RF permutation importance",
            title=f"Recovery from noisy mixtures (r = {corr:.2f})")
    save(fig, "ferm_importance.png")


# ── 7. Regularized linear + logistic (GLM) recover the same strains ───────────
@figure
def fig_ferm_regularized():
    from vlab_doe.config import make_rng
    from vlab_doe.doe.covering import covering_array
    from vlab_doe.doe.importance import (
        logistic_screening,
        regularized_importance,
    )
    from vlab_doe.models import fermentation as ferm

    # Identical screen to fig_ferm_importance (same seeds), so the chapters agree.
    rng = make_rng(2026)
    lib = ferm.random_strain_library(50, rng)
    design = covering_array(50, 200, min_size=2, max_size=5, seed=11)
    var = ferm.BatchVariability()

    final_ph, did_set = [], []
    for members in design.runs:
        setup = ferm.FermentationSetup(consortium=lib.consortium(members), temperature=43.0)
        batch = ferm.sample_batch(setup, var, rng)
        res = ferm.run_fermentation(batch, T_GRID, rng=rng)
        obs = ferm.observe_ph(res, T_GRID, rng=rng)
        final_ph.append(float(obs["ph"][-1]))
        did_set.append(1 if np.isfinite(ferm.time_to_ph(T_GRID, res.ph, ferm.PH_SET)) else 0)
    final_ph = np.array(final_ph)
    did_set = np.array(did_set)
    X = design.matrix()
    names = lib.names

    solo = np.array([
        ferm.run_fermentation(
            ferm.FermentationSetup(consortium=lib.consortium([i]), temperature=43.0), T_GRID
        ).ph[-1]
        for i in range(50)
    ])

    net = regularized_importance(X, final_ph, feature_names=names, seed=1, n_repeats=6)
    log = logistic_screening(X, did_set, feature_names=names, seed=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # (a) elastic-net recovered acidifying effect vs the true (unseen) solo power.
    coef = net.model[-1].coef_                      # signed: negative -> lowers pH
    recovered = -coef                               # acidifying strength
    true_power = -solo
    r = np.corrcoef(recovered, true_power)[0, 1]
    ax1.scatter(true_power, recovered, s=28, color=C["coop"], edgecolor="k", lw=0.3)
    ax1.axhline(0, color=C["indep"], lw=0.8, ls=":")
    ax1.set(xlabel="true acidifying power  (− solo final pH)",
            ylabel="elastic-net effect  (− coefficient)",
            title=f"Regularized linear fit (CV $R^2$={net.cv_score:.2f},  r = {r:.2f})")

    # (b) logistic GLM: which strains most raise P(set)?
    top = log.coefficients.head(10)[::-1]
    cols = [C["coop"] if v > 0 else C["lb"] for v in top.values]
    ax2.barh(np.arange(len(top)), top.values, color=cols, alpha=0.85)
    ax2.axvline(0, color="k", lw=0.8)
    ax2.set_yticks(np.arange(len(top)))
    ax2.set_yticklabels(top.index, fontsize=8)
    ax2.set(xlabel="logistic coefficient  (+ raises $P(\\mathrm{set})$)",
            title=f"GLM for “did it set?” (CV AUC={log.cv_auc:.2f}, acc={log.cv_accuracy:.2f})")
    save(fig, "ferm_regularized.png")


def main() -> int:
    for fn in (
        fig_ferm_acidification,
        fig_ferm_submodels,
        fig_ferm_cooperation,
        fig_ferm_uncertainty,
        fig_ferm_covering,
        fig_ferm_importance,
        fig_ferm_regularized,
    ):
        fn()
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
