"""Generate the figures for the two *Foundations* chapters (Part I).

Like :mod:`make_figures`, every figure is produced from the *actual* package
code -- here the teaching modules in :mod:`vlab_doe.foundations` -- inside
its own ``try/except`` so one failure never aborts the build.

Run from anywhere::

    python doc/scripts/make_foundations_figures.py

Output: ``doc/figures/foundations_*.png`` (150 dpi, tight bounding boxes).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

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
    "data": "#1f77b4",
    "fit": "#d62728",
    "accent": "#1F4E79",
    "green": "#2C7A39",
    "amber": "#8A6D1B",
    "grey": "#7f7f7f",
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


# ══════════════════════════════════════════════════════════════════════════════
#  CHAPTER 1 — STATISTICAL FOUNDATIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Distributions and sampling ─────────────────────────────────────────────
@figure
def fig_distributions():
    rng = np.random.default_rng(3)
    mu, sigma = 4.6, 0.4         # e.g. the set-point pH of Chapter 5
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 400)
    pdf = stats.norm.pdf(x, mu, sigma)
    cdf = stats.norm.cdf(x, mu, sigma)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))

    # (a) the PDF, with the ±1σ and ±2σ mass shaded
    ax = axes[0]
    ax.plot(x, pdf, color=C["accent"], lw=2)
    for k, a in [(2, 0.12), (1, 0.20)]:
        m = np.abs(x - mu) <= k * sigma
        ax.fill_between(x[m], pdf[m], color=C["accent"], alpha=a)
    ax.axvline(mu, color=C["fit"], ls="--", lw=1)
    ax.set_title("(a) the density  $f(x)$")
    ax.set_xlabel("$x$")
    ax.set_ylabel("probability density")
    ax.annotate("$\\mu$", (mu, 0), textcoords="offset points", xytext=(4, 4),
                color=C["fit"])
    ax.annotate("68% within $\\pm\\sigma$", (mu, pdf.max() * 0.45),
                ha="center", fontsize=9)

    # (b) the CDF, reading off a probability
    ax = axes[1]
    ax.plot(x, cdf, color=C["accent"], lw=2)
    xq = mu + 0.5 * sigma
    pq = stats.norm.cdf(xq, mu, sigma)
    ax.plot([xq, xq, x[0]], [0, pq, pq], color=C["fit"], ls=":", lw=1.3)
    ax.set_title("(b) the distribution  $F(x)=P(X\\leq x)$")
    ax.set_xlabel("$x$")
    ax.set_ylabel("cumulative probability")
    ax.annotate(f"$P(X\\leq {xq:.1f})={pq:.2f}$", (x[0], pq),
                textcoords="offset points", xytext=(6, 6), fontsize=9,
                color=C["fit"])

    # (c) a finite sample converging to the law of large numbers
    ax = axes[2]
    sample = rng.normal(mu, sigma, 400)
    ax.hist(sample, bins=22, density=True, color=C["data"], alpha=0.45,
            edgecolor="white", label="400 draws")
    ax.plot(x, pdf, color=C["accent"], lw=2, label="true $f(x)$")
    ax.axvline(sample.mean(), color=C["fit"], ls="--", lw=1.3,
               label=f"$\\bar x={sample.mean():.2f}$")
    ax.set_title("(c) a sample, and its mean")
    ax.set_xlabel("$x$")
    ax.set_ylabel("density")
    ax.legend(fontsize=8, loc="upper right")

    save(fig, "foundations_distributions.png")


# ── 2. Ordinary least squares ─────────────────────────────────────────────────
@figure
def fig_ols():
    from vlab_doe.foundations.stats_demo import ols_fit

    rng = np.random.default_rng(7)
    x = np.linspace(0, 10, 24)
    beta_true = np.array([1.0, 0.8])
    y = beta_true[0] + beta_true[1] * x + rng.normal(0, 1.1, x.size)

    X = np.column_stack([np.ones_like(x), x])
    fit = ols_fit(X, y)

    xs = np.linspace(0, 10, 200)
    Xs = np.column_stack([np.ones_like(xs), xs])
    yhat = Xs @ fit.beta
    # 95% confidence band on the mean response: sqrt(diag(Xs cov Xs^T))
    se_mean = np.sqrt(np.einsum("ij,jk,ik->i", Xs, fit.cov, Xs))
    t = stats.t.ppf(0.975, df=x.size - 2)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.scatter(x, y, color=C["data"], zorder=3, label="observations $y_i$")
    ax.plot(xs, yhat, color=C["fit"], lw=2,
            label=f"fit $\\hat y={fit.beta[0]:.2f}+{fit.beta[1]:.2f}\\,x$")
    ax.fill_between(xs, yhat - t * se_mean, yhat + t * se_mean,
                    color=C["fit"], alpha=0.15, label="95% confidence band")
    # residual stems
    fitted = X @ fit.beta
    for xi, yi, fi in zip(x, y, fitted):
        ax.plot([xi, xi], [fi, yi], color=C["grey"], lw=0.9, alpha=0.7)
    ax.set_title("Least squares: the line that minimises $\\sum$ residual$^2$")
    ax.set_xlabel("predictor $x$")
    ax.set_ylabel("response $y$")
    ax.legend(fontsize=9, loc="upper left")
    ax.annotate(f"$R^2={fit.r_squared:.2f}$", (0.97, 0.04),
                xycoords="axes fraction", ha="right", fontsize=10)
    save(fig, "foundations_ols.png")


# ── 3. The bootstrap ──────────────────────────────────────────────────────────
@figure
def fig_bootstrap():
    from vlab_doe.foundations.stats_demo import bootstrap_ci

    rng = np.random.default_rng(11)
    # A small, skewed sample — exactly where a textbook normal CI is shaky.
    sample = rng.lognormal(mean=1.4, sigma=0.5, size=18)
    boot = bootstrap_ci(sample, statistic=np.mean, n_boot=20_000, seed=0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8),
                             gridspec_kw={"width_ratios": [1, 1.4]})

    ax = axes[0]
    ax.scatter(sample, rng.uniform(0, 1, sample.size), color=C["data"],
               alpha=0.8, zorder=3)
    ax.axvline(sample.mean(), color=C["fit"], ls="--",
               label=f"$\\bar x={sample.mean():.2f}$")
    ax.set_yticks([])
    ax.set_title("(a) one small, skewed sample\n($n=18$)")
    ax.set_xlabel("value")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.hist(boot.replicates, bins=50, density=True, color=C["data"],
            alpha=0.5, edgecolor="white")
    ax.axvline(boot.estimate, color=C["fit"], lw=2, label="estimate $\\bar x$")
    ax.axvspan(boot.ci_low, boot.ci_high, color=C["green"], alpha=0.18,
               label=f"95% CI [{boot.ci_low:.2f}, {boot.ci_high:.2f}]")
    ax.set_title("(b) resample 20{,}000 times $\\Rightarrow$\n"
                 "sampling distribution of the mean")
    ax.set_xlabel("bootstrap replicate of $\\bar x$")
    ax.set_ylabel("density")
    ax.legend(fontsize=9)
    save(fig, "foundations_bootstrap.png")


# ── 4. Variability versus uncertainty (variance components) ───────────────────
@figure
def fig_variance():
    from vlab_doe.foundations.stats_demo import variance_components

    rng = np.random.default_rng(5)
    k, m = 8, 6
    sigma_between, sigma_within = 0.30, 0.08
    batch_means = 4.6 + rng.normal(0, sigma_between, k)
    groups = batch_means[:, None] + rng.normal(0, sigma_within, (k, m))
    vc = variance_components(groups)

    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    for i in range(k):
        xs = np.full(m, i + 1) + rng.uniform(-0.12, 0.12, m)
        ax.scatter(xs, groups[i], color=C["data"], alpha=0.75, zorder=3, s=22)
        ax.plot([i + 1 - 0.25, i + 1 + 0.25], [vc.group_means[i]] * 2,
                color=C["fit"], lw=2.2, zorder=4)
    ax.axhline(vc.grand_mean, color=C["grey"], ls="--", label="grand mean")
    ax.set_xlabel("batch")
    ax.set_ylabel("measured pH at set")
    ax.set_title("Variability vs uncertainty: spread of batch means "
                 "(red) above replicate noise")
    ax.set_xticks(range(1, k + 1))

    txt = (f"within-batch  $\\sigma_{{\\rm meas}}\\approx{np.sqrt(vc.within):.3f}$\n"
           f"between-batch $\\sigma_{{\\rm batch}}\\approx{np.sqrt(vc.between):.3f}$")
    ax.annotate(txt, (0.985, 0.04), xycoords="axes fraction", ha="right",
                fontsize=9.5,
                bbox=dict(boxstyle="round", fc="white", ec=C["grey"], alpha=0.9))
    ax.legend(fontsize=9, loc="upper left")
    save(fig, "foundations_variance.png")


# ── 5. Generalized linear models: logistic regression ─────────────────────────
@figure
def fig_glm():
    from vlab_doe.foundations.stats_demo import logistic_fit, ols_fit, sigmoid

    rng = np.random.default_rng(4)
    # A binary outcome (e.g. "did the blend set?") driven by one predictor.
    x = rng.uniform(-3, 3, 60)
    p_true = sigmoid(1.1 * x - 0.3)
    y = (rng.uniform(size=x.size) < p_true).astype(float)

    X = np.column_stack([np.ones_like(x), x])
    glm = logistic_fit(X, y)
    lin = ols_fit(X, y)                       # the wrong tool, for contrast

    xs = np.linspace(-3.4, 3.4, 300)
    Xs = np.column_stack([np.ones_like(xs), xs])

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.scatter(x, y, color=C["data"], alpha=0.7, zorder=3, label="outcomes $y\\in\\{0,1\\}$")
    ax.plot(xs, sigmoid(Xs @ glm.beta), color=C["accent"], lw=2.4,
            label="logistic GLM  $P(y{=}1)=\\sigma(X\\beta)$")
    ax.plot(xs, Xs @ lin.beta, color=C["fit"], lw=1.6, ls="--",
            label="linear fit (leaves $[0,1]$)")
    ax.axhline(0, color=C["grey"], lw=0.8); ax.axhline(1, color=C["grey"], lw=0.8)
    ax.axhspan(1, 1.25, color=C["fit"], alpha=0.07)
    ax.axhspan(-0.25, 0, color=C["fit"], alpha=0.07)
    ax.set_ylim(-0.25, 1.25)
    ax.set_xlabel("predictor $x$")
    ax.set_ylabel("response / probability")
    ax.set_title("Generalized linear model: the logit link keeps a probability in $[0,1]$")
    ax.legend(fontsize=8.5, loc="center right")
    save(fig, "foundations_glm.png")


# ── 6. Regularization: ridge shrinks, the lasso selects ───────────────────────
@figure
def fig_regularization():
    from vlab_doe.foundations.stats_demo import coefficient_path

    rng = np.random.default_rng(8)
    n, p = 50, 8
    X = rng.normal(0, 1, (n, p))
    X[:, 1] = 0.8 * X[:, 0] + 0.6 * X[:, 1]      # induce some collinearity
    X = (X - X.mean(0)) / X.std(0)
    beta_true = np.zeros(p)
    beta_true[:3] = [3.0, -2.0, 1.5]             # only three features matter
    y = X @ beta_true + rng.normal(0, 1.0, n)
    y = y - y.mean()

    lam_ridge = np.logspace(2.3, -1.5, 60)
    lam_lasso = np.logspace(1.7, -2.0, 60)
    path_r = coefficient_path(X, y, lam_ridge, "ridge")
    path_l = coefficient_path(X, y, lam_lasso, "lasso")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    for ax, path, lam, title in [
        (axes[0], path_r, lam_ridge, "(a) ridge ($L_2$): everything shrinks"),
        (axes[1], path_l, lam_lasso, "(b) lasso ($L_1$): inactive terms hit zero"),
    ]:
        for j in range(p):
            relevant = j < 3
            ax.plot(lam, path[:, j],
                    color=(C["fit"] if relevant else C["grey"]),
                    lw=(2.2 if relevant else 1.0),
                    alpha=(1.0 if relevant else 0.6),
                    label=(f"$\\beta_{j+1}$ (true)" if relevant else None))
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xscale("log")
        ax.invert_xaxis()                        # strong penalty on the left
        ax.set_xlabel("penalty $\\lambda$  (strong $\\to$ weak)")
        ax.set_title(title, fontsize=11)
    axes[0].set_ylabel("coefficient $\\hat\\beta_j$")
    axes[1].legend(fontsize=8.5, loc="upper right")
    save(fig, "foundations_regularization.png")


# ── 7. Why design experiments: OFAT versus a factorial ────────────────────────
@figure
def fig_doe():
    # A response with a genuine interaction: the optimum is a diagonal ridge that
    # one-factor-at-a-time cannot climb.
    def response(x1, x2):
        return 10.0 - (x1 - x2) ** 2 - 0.35 * (x1 + x2 - 1.0) ** 2

    g = np.linspace(-2, 2, 200)
    G1, G2 = np.meshgrid(g, g)
    Z = response(G1, G2)

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    cs = ax.contourf(G1, G2, Z, levels=18, cmap="Blues")
    fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04, label="response (higher better)")

    # OFAT: optimise x1 at x2=-1.5, then x2 at that best x1 — and get stuck.
    x2_fix = -1.5
    x1_best = g[np.argmax(response(g, x2_fix))]
    x2_best = g[np.argmax(response(x1_best, g))]
    ax.plot([-1.5, x1_best], [x2_fix, x2_fix], "-o", color=C["fit"], lw=2, ms=5)
    ax.plot([x1_best, x1_best], [x2_fix, x2_best], "-o", color=C["fit"], lw=2, ms=5,
            label="one-factor-at-a-time")
    ax.scatter([x1_best], [x2_best], color=C["fit"], s=120, marker="X",
               edgecolor="k", zorder=5)

    # Factorial: four corners + centre span the interaction.
    corners = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1], [0, 0]], float)
    ax.scatter(corners[:, 0], corners[:, 1], color=C["green"], s=70, zorder=5,
               edgecolor="k", label="$2^2$ factorial + centre")
    ax.scatter([0.5], [0.5], color="white", edgecolor="k", marker="*", s=240,
               zorder=6, label="true optimum")
    ax.set_xlabel("factor $x_1$"); ax.set_ylabel("factor $x_2$")
    ax.set_title("Why design: OFAT (red) stalls; a factorial sees the interaction")
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.9)
    save(fig, "foundations_doe.png")


# ══════════════════════════════════════════════════════════════════════════════
#  CHAPTER 2 — FOUNDATIONS OF DOWNSTREAM SEPARATION
# ══════════════════════════════════════════════════════════════════════════════

# ── 5. The bioprocess train (schematic, but drawn from code) ──────────────────
@figure
def fig_bioprocess():
    fig, ax = plt.subplots(figsize=(11.5, 3.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis("off")

    stages = [
        ("Fermentation\n/ cell culture", C["amber"], "UPSTREAM"),
        ("Harvest &\nclarification", C["grey"], "DOWNSTREAM"),
        ("Capture\n(e.g. Protein A)", C["accent"], "DOWNSTREAM"),
        ("Polish\n(IEX, HIC)", C["accent"], "DOWNSTREAM"),
        ("UF / DF\n& formulate", C["green"], "DOWNSTREAM"),
    ]
    w, gap, y0, h = 1.9, 0.35, 0.7, 1.3
    for i, (label, col, _) in enumerate(stages):
        x = 0.3 + i * (w + gap)
        box = mpatches.FancyBboxPatch(
            (x, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.6, edgecolor=col, facecolor=col, alpha=0.16)
        ax.add_patch(box)
        ax.text(x + w / 2, y0 + h / 2, label, ha="center", va="center",
                fontsize=9.5)
        if i < len(stages) - 1:
            ax.annotate("", (x + w + gap, y0 + h / 2), (x + w, y0 + h / 2),
                        arrowprops=dict(arrowstyle="-|>", color=C["grey"], lw=1.8))

    # span bars labelling upstream vs downstream
    x_up_end = 0.3 + w
    ax.plot([0.3, x_up_end], [y0 + h + 0.35] * 2, color=C["amber"], lw=3)
    ax.text((0.3 + x_up_end) / 2, y0 + h + 0.55, "UPSTREAM",
            ha="center", fontsize=9, color=C["amber"], weight="bold")
    x_dn_start = 0.3 + (w + gap)
    x_dn_end = 0.3 + 4 * (w + gap) + w
    ax.plot([x_dn_start, x_dn_end], [y0 + h + 0.35] * 2, color=C["accent"], lw=3)
    ax.text((x_dn_start + x_dn_end) / 2, y0 + h + 0.55,
            "DOWNSTREAM  (this book)", ha="center", fontsize=9,
            color=C["accent"], weight="bold")

    ax.text(6, 0.25, "increasing purity  $\\longrightarrow$", ha="center",
            fontsize=9, style="italic", color=C["grey"])
    save(fig, "foundations_bioprocess.png")


# ── 6. The adsorption isotherm ────────────────────────────────────────────────
@figure
def fig_isotherm():
    from vlab_doe.foundations.separation_demo import henry, langmuir

    q_max, b = 50.0, 0.04
    H = q_max * b                       # initial slope = Henry constant
    c = np.linspace(0, 200, 300)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(c, langmuir(c, q_max, b), color=C["accent"], lw=2.2,
            label="Langmuir  $q^*=q_{\\max}bc/(1+bc)$")
    ax.plot(c, henry(c, H), color=C["fit"], lw=1.8, ls="--",
            label=f"linear (Henry)  $q^*=H c,\\ H={H:.1f}$")
    ax.axhline(q_max, color=C["grey"], ls=":", lw=1.4)
    ax.annotate("saturation capacity $q_{\\max}$", (c[-1], q_max),
                textcoords="offset points", xytext=(-8, 6), ha="right",
                fontsize=9, color=C["grey"])
    ax.annotate("dilute limit:\nslope $=H=q_{\\max}b$", (28, henry(28, H)),
                textcoords="offset points", xytext=(20, -22), fontsize=9,
                color=C["fit"],
                arrowprops=dict(arrowstyle="->", color=C["fit"], lw=1))
    ax.set_xlabel("mobile-phase concentration $c$  (g/L)")
    ax.set_ylabel("adsorbed loading $q^*$  (g/L resin)")
    ax.set_title("The adsorption isotherm: linear when dilute, saturating when loaded")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(0, q_max * 1.15)
    save(fig, "foundations_isotherm.png")


# ── 7. Plates: breakthrough sharpness and peak resolution ─────────────────────
@figure
def fig_column():
    from vlab_doe.foundations.separation_demo import (
        gaussian_peak,
        resolution,
        tanks_in_series_breakthrough,
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))

    # (a) breakthrough sharpens as the plate count grows
    ax = axes[0]
    for n, col in [(3, C["amber"]), (15, C["data"]), (80, C["accent"])]:
        bt = tanks_in_series_breakthrough(n_stages=n, retention_factor=4.0)
        ax.plot(bt.cv, bt.c_out_ratio, color=col, lw=2, label=f"$N={n}$ plates")
    ax.axhline(1.0, color=C["grey"], ls=":", lw=1)
    ax.set_title("(a) breakthrough: efficiency $=$ plate count")
    ax.set_xlabel("throughput  (column volumes)")
    ax.set_ylabel("$c_{\\rm out}/c_{\\rm feed}$")
    ax.legend(fontsize=9, loc="lower right")

    # (b) two bands and their resolution
    ax = axes[1]
    t = np.linspace(6, 16, 600)
    n_plates = 1500
    for tr, col, lab in [(10.0, C["data"], "A"), (11.6, C["green"], "B")]:
        ax.plot(t, gaussian_peak(t, tr, n_plates), color=col, lw=2,
                label=f"peak {lab}  ($t_R={tr}$)")
        ax.fill_between(t, gaussian_peak(t, tr, n_plates), color=col, alpha=0.15)
    rs = resolution(10.0, 11.6, n_plates)
    ax.set_title(f"(b) resolution of two peaks  $R_s={rs:.2f}$")
    ax.set_xlabel("retention time  (min)")
    ax.set_ylabel("detector signal")
    ax.legend(fontsize=9, loc="upper right")
    save(fig, "foundations_column.png")


# ── 8. Batch microbial growth ─────────────────────────────────────────────────
@figure
def fig_growth():
    from vlab_doe.foundations.separation_demo import monod_batch

    gc = monod_batch()

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(gc.t, gc.biomass, color=C["green"], lw=2.2, label="biomass $X$")
    ax.set_xlabel("time  (h)")
    ax.set_ylabel("biomass $X$  (a.u.)", color=C["green"])
    ax.tick_params(axis="y", labelcolor=C["green"])

    ax2 = ax.twinx()
    ax2.plot(gc.t, gc.substrate, color=C["amber"], lw=2.0, ls="--",
             label="substrate $S$")
    ax2.set_ylabel("substrate $S$  (g/L)", color=C["amber"])
    ax2.tick_params(axis="y", labelcolor=C["amber"])
    ax2.grid(False)

    # annotate the four phases
    phases = [(0.7, "lag"), (4.0, "exponential"), (7.2, "deceleration"),
              (12.5, "stationary")]
    ymax = gc.biomass.max()
    for x, name in phases:
        ax.axvline(x if name == "lag" else None) if False else None
        ax.annotate(name, (x, ymax * 0.93), ha="center", fontsize=8.5,
                    color=C["grey"], style="italic")
    for xb in (1.5, 6.0, 9.5):
        ax.axvline(xb, color=C["grey"], lw=0.7, ls=":", alpha=0.7)
    ax.set_title("Batch growth: Monod kinetics couple biomass to substrate")
    ax.set_xlim(0, gc.t[-1])
    save(fig, "foundations_growth.png")


# ══════════════════════════════════════════════════════════════════════════════
def main():
    for fn in (
        fig_distributions,
        fig_ols,
        fig_glm,
        fig_regularization,
        fig_doe,
        fig_bootstrap,
        fig_variance,
        fig_bioprocess,
        fig_isotherm,
        fig_column,
        fig_growth,
    ):
        fn()


if __name__ == "__main__":
    main()
