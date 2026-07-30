"""Frequentist analysis of covering-array screens, and why they strain under real interactions.

Covering arrays earn their keep in enzyme and formulation screening, where the working
assumption is **effect sparsity with weak interactions**: a handful of ingredients matter, they
mostly act additively, and co-testing every pair once is enough to notice the rare exception.
This module analyses the opposite regime — a strain panel whose *whole point* is strong,
signed, asymmetric pairwise interaction — using ordinary frequentist tools, so that the failure
modes show up as things a practitioner would actually see in their output: a lack-of-fit test
that rejects, standard errors that will not shrink, and a false-discovery rate that refuses to
come down.

The central obstacle
--------------------
The Scheffé quadratic over 16 strains has :math:`16 + 120 = 136` coefficients.  A 60-run design
cannot estimate them: the model is **supersaturated** (:math:`p \\gg n`), so there is no OLS fit
of the full model, no residual degrees of freedom, and no t-tests — regardless of how elegant
the design is.  **Pair coverage is not pair estimability.**  A covering array guarantees you
*observed* every pair; it says nothing about whether you can separate the pairs you observed.

What the module therefore does, in the order a practitioner would:

:func:`design_diagnostics`
    Look at the design before spending anything: coverage, replication λ, main-effect balance,
    the rank of the model matrix, and — the number that matters most — how strongly the 120
    interaction columns are correlated with one another.
:func:`fit_main_effects`
    Fit what *is* estimable (the 16 main effects) by OLS, with the usual t-tests.
:func:`lack_of_fit_test`
    Where the design has genuine replicates, split the residual into **pure error** and **lack
    of fit** and test the additive model formally.  This is the honest way to discover that
    interactions matter without being able to say which ones.
:func:`pair_contrasts`
    The two-stage screen: take the main-effects residuals and, for each pair, contrast the runs
    containing it against the runs that do not.  Correct the 120 p-values for multiplicity
    (Bonferroni and Benjamini–Hochberg).  Works under supersaturation because each pair is
    tested *marginally* rather than jointly — which is exactly where its bias comes from.
:func:`forward_selection`
    The other standard escape from supersaturation: enter interaction terms one at a time by
    F-to-enter.  Greedy, and its selection event invalidates the very p-values it reports —
    reported here because it is what people do, and its failure mode is worth seeing.
:func:`lenth_effects`
    Lenth's pseudo standard error: a margin of error from the effects themselves, for when
    there are no degrees of freedom left to estimate error with.
:func:`score_screen`
    Grade the outcome against the planted truth: power, false discovery rate, and where the
    real champion landed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .combination_search import Members, scheffe_features

__all__ = [
    "design_diagnostics",
    "fit_main_effects",
    "lack_of_fit_test",
    "pair_contrasts",
    "forward_selection",
    "lenth_effects",
    "score_screen",
    "true_pair_excess",
]


def _pairs(n_strains: int) -> list[tuple[int, int]]:
    return list(combinations(range(n_strains), 2))


def _pair_labels(names: Sequence[str]) -> list[str]:
    return [f"{names[i]}+{names[j]}" for i, j in _pairs(len(names))]


# ── 1. Look at the design before you spend anything ─────────────────────────────

def design_diagnostics(runs: Sequence[Members], n_strains: int) -> dict:
    """Estimability and balance diagnostics for a design, computed without running it.

    The headline entries are the ones that decide whether any analysis can succeed:

    ``n_parameters`` vs ``n_runs``
        136 against ~60 — the model is supersaturated and the full OLS fit does not exist.
    ``rank_full`` / ``rank_deficiency``
        Rank of the full Scheffé matrix.  Even the *achievable* rank is below the number of
        columns, so some coefficients are unidentifiable no matter what.
    ``max_abs_pair_corr`` and ``n_pairs_aliased``
        The largest correlation between two interaction columns, and how many pairs are
        **perfectly aliased** with another pair — the two carry identical information, so no
        amount of data separates them.  This is what a covering array does *not* control.
    ``dilution``
        Mean :math:`x_i x_j` over the runs in which a pair appears — how faintly the design
        looks at a typical pair, and thus what it costs in signal-to-noise.
    """
    pairs = _pairs(n_strains)
    x = scheffe_features(list(runs), n_strains)
    main, inter = x[:, :n_strains], x[:, n_strains:]

    lam = inter.sum(axis=0) > 0                      # which pairs are co-tested at all
    counts = np.array([sum(1 for r in runs if i in r and j in r) for i, j in pairs])
    appearances = np.array([sum(1 for r in runs if i in r) for i in range(n_strains)])

    # Correlations between the interaction columns that actually carry information.
    seen = inter[:, lam]
    centred = seen - seen.mean(axis=0)
    norms = np.linalg.norm(centred, axis=0)
    keep = norms > 1e-12
    corr = np.abs((centred[:, keep] / norms[keep]).T @ (centred[:, keep] / norms[keep]))
    np.fill_diagonal(corr, 0.0)

    nz = inter[inter > 0]
    return {
        "n_runs": len(runs),
        "n_parameters": n_strains + len(pairs),
        "block_sizes": sorted({len(r) for r in runs}),
        "mean_block_size": float(np.mean([len(r) for r in runs])),
        "n_distinct_runs": len({tuple(sorted(r)) for r in runs}),
        "n_replicated_runs": len(runs) - len({tuple(sorted(r)) for r in runs}),
        "pair_coverage": float(lam.mean()),
        "pairs_uncovered": int((~lam).sum()),
        "lambda_min": int(counts.min()),
        "lambda_max": int(counts.max()),
        "lambda_mean": float(counts.mean()),
        "lambda_sd": float(counts.std()),
        "appearance_sd": float(appearances.std()),
        "rank_full": int(np.linalg.matrix_rank(x)),
        "rank_deficiency": int(x.shape[1] - np.linalg.matrix_rank(x)),
        "rank_main": int(np.linalg.matrix_rank(main)),
        "max_abs_pair_corr": float(corr.max()) if corr.size else 0.0,
        "mean_abs_pair_corr": float(corr[np.triu_indices_from(corr, 1)].mean()) if corr.size else 0.0,
        "n_pairs_aliased": int((corr > 1 - 1e-9).any(axis=1).sum()),
        "dilution": float(nz.mean()) if nz.size else 0.0,
    }


# ── 2. Fit what is estimable ────────────────────────────────────────────────────

@dataclass
class MainEffectFit:
    """OLS fit of the additive (main-effects-only) Scheffé model."""

    result: object                       # statsmodels RegressionResults
    coefs: np.ndarray
    residuals: np.ndarray
    fitted: np.ndarray
    r_squared: float
    table: pd.DataFrame = field(repr=False)


def fit_main_effects(
    runs: Sequence[Members], y: np.ndarray, n_strains: int, *, names: Sequence[str] | None = None
) -> MainEffectFit:
    """OLS of the response on strain fractions only — the model a covering array can support.

    No intercept: the mixture fractions already sum to one, so each :math:`\\beta_i` is strain
    ``i``'s own contribution and the fitted value of a blend is the *average* of its members'
    contributions.  This is the additive world covering arrays are designed for, and fitting it
    is the right first move — the interesting question is how badly it fails.
    """
    names = list(names) if names is not None else [f"S{i:02d}" for i in range(n_strains)]
    x = scheffe_features(list(runs), n_strains)[:, :n_strains]
    model = sm.OLS(np.asarray(y, float), x).fit()
    table = pd.DataFrame({
        "strain": names,
        "coef": model.params,
        "std_err": model.bse,
        "t": model.tvalues,
        "p_value": model.pvalues,
    })
    return MainEffectFit(
        result=model, coefs=model.params, residuals=model.resid, fitted=model.fittedvalues,
        r_squared=float(model.rsquared), table=table,
    )


def lack_of_fit_test(runs: Sequence[Members], y: np.ndarray, n_strains: int) -> dict:
    """Formal F-test of the additive model, splitting residual SS into pure error and lack of fit.

    Only possible where the design **repeats whole blends**, because pure error must be measured
    without reference to any model: it is the spread *within* groups of identical runs.  The
    lack-of-fit mean square is then everything the additive model leaves behind that replication
    cannot explain, and their ratio is an F statistic.

    A design with no repeated blend has no pure-error degrees of freedom, and this test simply
    cannot be run — which is one concrete thing replication buys and coverage does not.
    Returns ``available: False`` in that case rather than inventing an answer.
    """
    y = np.asarray(y, float)
    fit = fit_main_effects(runs, y, n_strains)
    groups: dict[tuple[int, ...], list[int]] = {}
    for idx, r in enumerate(runs):
        groups.setdefault(tuple(sorted(r)), []).append(idx)

    ss_pure, df_pure = 0.0, 0
    for members, idx in groups.items():
        if len(idx) > 1:
            vals = y[idx]
            ss_pure += float(((vals - vals.mean()) ** 2).sum())
            df_pure += len(idx) - 1
    if df_pure == 0:
        return {"available": False, "reason": "no replicated blends — pure error is unestimable",
                "df_pure_error": 0, "r_squared": fit.r_squared}

    ss_resid = float((fit.residuals ** 2).sum())
    df_resid = int(len(y) - np.linalg.matrix_rank(scheffe_features(list(runs), n_strains)[:, :n_strains]))
    ss_lof, df_lof = ss_resid - ss_pure, df_resid - df_pure
    if df_lof <= 0:
        return {"available": False, "reason": "no degrees of freedom left for lack of fit",
                "df_pure_error": df_pure, "r_squared": fit.r_squared}

    f_stat = (ss_lof / df_lof) / (ss_pure / df_pure)
    return {
        "available": True,
        "f_stat": float(f_stat),
        "df_lack_of_fit": int(df_lof),
        "df_pure_error": int(df_pure),
        "p_value": float(stats.f.sf(f_stat, df_lof, df_pure)),
        "ms_lack_of_fit": float(ss_lof / df_lof),
        "ms_pure_error": float(ss_pure / df_pure),
        "pure_error_sd": float(np.sqrt(ss_pure / df_pure)),
        "r_squared": fit.r_squared,
    }


# ── 3. Screening the 120 pairs under supersaturation ────────────────────────────

def pair_contrasts(
    runs: Sequence[Members],
    y: np.ndarray,
    n_strains: int,
    *,
    names: Sequence[str] | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Two-stage screen: main-effects residuals, then one marginal contrast per pair.

    For each of the 120 pairs, compare the additive model's residuals on the runs containing
    **both** strains against the residuals on the runs that do not, by Welch's two-sample
    t-test.  A pair that genuinely accelerates the blend leaves systematically negative
    residuals in the runs it appears in.

    This sidesteps supersaturation because each pair is tested on its own rather than jointly —
    and that is precisely its weakness.  The contrast for pair :math:`(i,j)` absorbs the effect
    of **every other pair sharing those runs**, so in a design of four-strain blocks each test
    carries five other pairs' worth of contamination.  Strong interactions therefore do not just
    add noise, they add *bias*, and the bias is largest exactly around the strains that matter
    most — a genuinely strong pair drags its block-mates' contrasts along with it.

    Returns one row per pair with the contrast, Welch t, raw p, Bonferroni and
    Benjamini–Hochberg adjusted p, and the two significance flags.
    """
    names = list(names) if names is not None else [f"S{i:02d}" for i in range(n_strains)]
    y = np.asarray(y, float)
    resid = fit_main_effects(runs, y, n_strains, names=names).residuals
    members = [set(r) for r in runs]

    rows = []
    for i, j in _pairs(n_strains):
        inside = np.array([k for k, m in enumerate(members) if i in m and j in m])
        outside = np.array([k for k in range(len(runs)) if k not in set(inside.tolist())])
        n_in = len(inside)
        if n_in == 0 or len(outside) < 2:
            rows.append((i, j, n_in, np.nan, np.nan, np.nan))
            continue
        a, b = resid[inside], resid[outside]
        contrast = float(a.mean() - b.mean())
        if n_in < 2:
            # One run only: no within-pair variance, so borrow the outside spread for the SE.
            se = float(b.std(ddof=1) * np.sqrt(1 + 1 / len(b)))
            t = contrast / se if se > 0 else np.nan
            p = float(2 * stats.t.sf(abs(t), len(b) - 1)) if np.isfinite(t) else np.nan
        else:
            t_res = stats.ttest_ind(a, b, equal_var=False)
            t, p = float(t_res.statistic), float(t_res.pvalue)
        rows.append((i, j, n_in, contrast, t, p))

    df = pd.DataFrame(rows, columns=["i", "j", "n_runs_with_pair", "contrast", "t", "p_value"])
    df.insert(0, "pair", [f"{names[i]}+{names[j]}" for i, j in zip(df["i"], df["j"])])

    ok = df["p_value"].notna()
    df["p_bonferroni"] = np.nan
    df["p_bh"] = np.nan
    if ok.any():
        p = df.loc[ok, "p_value"].to_numpy()
        df.loc[ok, "p_bonferroni"] = multipletests(p, alpha=alpha, method="bonferroni")[1]
        df.loc[ok, "p_bh"] = multipletests(p, alpha=alpha, method="fdr_bh")[1]
    df["sig_bonferroni"] = df["p_bonferroni"] < alpha
    df["sig_bh"] = df["p_bh"] < alpha
    return df.sort_values("p_value").reset_index(drop=True)


def forward_selection(
    runs: Sequence[Members],
    y: np.ndarray,
    n_strains: int,
    *,
    names: Sequence[str] | None = None,
    max_terms: int = 10,
    p_enter: float = 0.01,
) -> pd.DataFrame:
    """Stepwise entry of interaction terms by F-to-enter, on top of all main effects.

    The other textbook route out of supersaturation.  Start from the additive model, and at each
    step add whichever single interaction column most reduces the residual sum of squares,
    stopping when the partial F-test no longer clears ``p_enter``.

    Two caveats travel with the output and both matter here.  The reported p-values are
    **post-selection**: the term was chosen *because* it looked good, so its nominal p-value is
    optimistic and the usual guarantees do not hold.  And because interaction columns in a
    blocked design are strongly correlated, an early wrong entry can lock the path — the
    procedure has no way back.
    """
    names = list(names) if names is not None else [f"S{i:02d}" for i in range(n_strains)]
    y = np.asarray(y, float)
    x = scheffe_features(list(runs), n_strains)
    pairs = _pairs(n_strains)
    base = x[:, :n_strains]
    usable = [k for k in range(len(pairs)) if x[:, n_strains + k].std() > 1e-12]

    chosen: list[int] = []
    rows = []
    current = base
    rss = float(sm.OLS(y, current).fit().ssr)
    for _ in range(max_terms):
        best = None
        for k in usable:
            if k in chosen:
                continue
            trial = np.column_stack([current, x[:, n_strains + k]])
            if np.linalg.matrix_rank(trial) < trial.shape[1]:
                continue                                  # aliased with what is already in
            fit = sm.OLS(y, trial).fit()
            df_res = len(y) - trial.shape[1]
            if df_res <= 0:
                continue
            f = (rss - fit.ssr) / (fit.ssr / df_res)
            if best is None or f > best[1]:
                best = (k, float(f), fit, df_res)
        if best is None:
            break
        k, f, fit, df_res = best
        p = float(stats.f.sf(f, 1, df_res))
        if p > p_enter:
            break
        chosen.append(k)
        current = np.column_stack([current, x[:, n_strains + k]])
        rss = float(fit.ssr)
        i, j = pairs[k]
        rows.append({"step": len(chosen), "pair": f"{names[i]}+{names[j]}", "i": i, "j": j,
                     "coef": float(fit.params[-1]), "f_to_enter": f, "p_value": p,
                     "r_squared": float(fit.rsquared)})
    return pd.DataFrame(rows)


def lenth_effects(contrasts: np.ndarray, *, alpha: float = 0.05) -> dict:
    """Lenth's pseudo standard error and margin of error for a set of contrasts.

    When a design leaves no degrees of freedom to estimate error, Lenth's method estimates it
    **from the effects themselves**, assuming most are null: take 1.5 × the median absolute
    contrast, discard anything beyond 2.5 of that, and re-take the median.  Effects exceeding
    the resulting margin of error are declared active.

    It is the standard tool for unreplicated screening designs, and it inherits one assumption
    that the strain panel violates by construction — that *most effects are inert*.  With a
    dense baseline cooperation between two whole culture groups the inert majority is not there,
    the PSE is inflated by the real effects, and the method loses power precisely when the
    system is most interactive.
    """
    c = np.asarray(contrasts, float)
    c = c[np.isfinite(c)]
    if c.size == 0:
        return {"pse": np.nan, "margin_of_error": np.nan, "n_active": 0, "active": np.array([])}
    s0 = 1.5 * np.median(np.abs(c))
    kept = np.abs(c)[np.abs(c) < 2.5 * s0] if s0 > 0 else np.abs(c)
    pse = 1.5 * np.median(kept) if kept.size else s0
    df = max(1, c.size // 3)
    me = float(stats.t.ppf(1 - alpha / 2, df) * pse)
    return {"pse": float(pse), "margin_of_error": me, "df": df,
            "n_active": int((np.abs(c) > me).sum()), "active": np.where(np.abs(c) > me)[0]}


# ── 4. Grading against the planted truth ────────────────────────────────────────

def true_pair_excess(lab, *, n_strains: int | None = None) -> pd.DataFrame:
    """How much each pair beats (or misses) what an additive model predicts — the truth to recover.

    Screening does not try to recover the simulator's internal :math:`k_{ij}`; it tries to
    recover the part of the *response* that additivity cannot explain.  So the target is defined
    the same way the analysis defines its estimate: fit the noise-free pair responses on strain
    fractions alone, and call the residual the pair's **excess**.  Negative excess = the pair
    acidifies faster than its members' average predicts, i.e. real positive synergy.

    Working on the acidification-rate scale keeps censored pairs finite and comparable.
    """
    n = n_strains or lab.n_strains
    pairs = _pairs(n)
    truth = np.array([lab.truth(p) for p in pairs])
    rate = 1.0 / np.maximum(truth, 1e-3)
    x = scheffe_features(pairs, n)[:, :n]
    fit = sm.OLS(rate, x).fit()
    return pd.DataFrame({
        "pair": [f"{lab.library.names[i]}+{lab.library.names[j]}" for i, j in pairs],
        "i": [i for i, _ in pairs], "j": [j for _, j in pairs],
        "t_set": truth, "rate": rate,
        "additive_rate": fit.fittedvalues,
        "excess_rate": fit.resid,
    })


def score_screen(
    contrasts: pd.DataFrame,
    excess: pd.DataFrame,
    *,
    champion: tuple[int, int],
    n_active: int = 10,
    flag: str = "sig_bh",
) -> dict:
    """Grade a screen: did it find the champion, and how many of its calls were real?

    The "active" set is defined from the simulator as the ``n_active`` pairs with the strongest
    true excess, so power and false-discovery rate are measured against a ground truth on the
    same footing as the estimate.  ``champion_rank`` — where the true best pair sits when the
    screen's own pairs are ordered by evidence — is the number that matters operationally: a
    screen that ranks it 3rd is useful, one that ranks it 60th is not.
    """
    truth = excess.set_index(["i", "j"])
    active = set(map(tuple, excess.reindex(
        excess["excess_rate"].sort_values(ascending=False).index).head(n_active)[["i", "j"]].values))

    df = contrasts.dropna(subset=["p_value"]).copy()
    # Responses are on the acidification-*rate* scale, so a synergy *raises* the response: the
    # evidence that a pair is good is a positive contrast with a small p-value.
    df["evidence"] = np.sign(df["contrast"]) * -np.log10(df["p_value"].clip(lower=1e-300))
    ranked = df.sort_values("evidence", ascending=False).reset_index(drop=True)
    keys = list(zip(ranked["i"], ranked["j"]))
    champ_rank = keys.index(tuple(sorted(champion))) + 1 if tuple(sorted(champion)) in keys else None

    called = {(int(r.i), int(r.j)) for r in df[df[flag] & (df["contrast"] > 0)].itertuples()}
    tp = len(called & active)
    corr = (np.corrcoef(df["contrast"], truth.loc[list(zip(df["i"], df["j"])), "excess_rate"])[0, 1]
            if len(df) > 2 else np.nan)
    return {
        "n_tested": int(len(df)),
        "n_called": len(called),
        "true_positives": tp,
        "power": tp / len(active) if active else np.nan,
        "fdr": 1 - tp / len(called) if called else np.nan,
        "champion_rank": champ_rank,
        "champion_found_top5": champ_rank is not None and champ_rank <= 5,
        "champion_significant": bool(
            df[(df["i"] == champion[0]) & (df["j"] == champion[1])][flag].any()),
        "contrast_truth_corr": float(corr) if np.isfinite(corr) else np.nan,
    }
