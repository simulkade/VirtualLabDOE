"""Tests for the Foundations teaching module and the regularized/GLM screening.

Fast, synthetic checks that the from-scratch statistics of
:mod:`vlab_doe.foundations.stats_demo` are correct, and that the
regularized-linear and logistic (GLM) screening analyses added to
:mod:`vlab_doe.doe.importance` behave as advertised.  These also pin the
scikit-learn API choices (e.g. the warning-free L1 logistic configuration)
against future deprecation churn.
"""

import numpy as np
import pytest

from vlab_doe.doe.importance import (
    ImportanceResult,
    LogisticScreenResult,
    logistic_screening,
    regularized_importance,
)
from vlab_doe.foundations import separation_demo as sep
from vlab_doe.foundations import stats_demo as st


# ── stats_demo: least squares, ANOVA, bootstrap, variance components ──────────

def test_ols_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 60)
    X = np.column_stack([np.ones_like(x), x])
    y = 2.0 + 3.0 * x + rng.normal(0, 0.5, x.size)
    fit = st.ols_fit(X, y)
    assert fit.beta == pytest.approx([2.0, 3.0], abs=0.2)
    assert fit.r_squared > 0.99
    assert np.all(fit.stderr > 0)


def test_regression_anova_flags_real_effect():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 1, 50)
    X = np.column_stack([np.ones_like(x), x])
    y = 1.0 + 4.0 * x + rng.normal(0, 0.3, x.size)
    res = st.regression_anova(X, y)
    assert res.f_stat > 50
    assert res.p_value < 1e-6


def test_bootstrap_ci_brackets_the_mean():
    rng = np.random.default_rng(2)
    sample = rng.normal(5.0, 1.0, 80)
    boot = st.bootstrap_ci(sample, n_boot=2000, seed=0)
    assert boot.ci_low < boot.estimate < boot.ci_high
    assert boot.ci_low < 5.0 < boot.ci_high


def test_variance_components_separates_scales():
    rng = np.random.default_rng(3)
    means = rng.normal(0, 1.0, 12)          # large between-batch spread
    groups = means[:, None] + rng.normal(0, 0.1, (12, 8))  # small within
    vc = st.variance_components(groups)
    assert vc.between > vc.within


# ── stats_demo: GLM and regularization ────────────────────────────────────────

def test_sigmoid_bounds_and_monotonic():
    z = np.linspace(-10, 10, 50)
    s = st.sigmoid(z)
    assert np.all((s >= 0) & (s <= 1))
    assert np.all(np.diff(s) > 0)


def test_logistic_fit_recovers_sign():
    rng = np.random.default_rng(4)
    x = rng.uniform(-3, 3, 400)
    X = np.column_stack([np.ones_like(x), x])
    p = st.sigmoid(1.5 * x - 0.5)
    y = (rng.uniform(size=x.size) < p).astype(float)
    fit = st.logistic_fit(X, y)
    assert fit.beta[1] > 0.8                 # positive slope recovered
    assert np.all((fit.probabilities >= 0) & (fit.probabilities <= 1))


def test_ridge_shrinks_relative_to_ols():
    rng = np.random.default_rng(5)
    X = rng.normal(0, 1, (40, 5))
    beta = np.array([2.0, -1.0, 0.0, 0.0, 0.5])
    y = X @ beta + rng.normal(0, 0.5, 40)
    ols = np.linalg.solve(X.T @ X, X.T @ y)
    ridged = st.ridge_fit(X, y, lam=20.0)
    assert np.linalg.norm(ridged) < np.linalg.norm(ols)


def test_lasso_selects_a_sparse_subset():
    rng = np.random.default_rng(6)
    X = rng.normal(0, 1, (60, 10))
    X = (X - X.mean(0)) / X.std(0)
    beta = np.zeros(10)
    beta[:3] = [3.0, -2.0, 1.5]
    y = X @ beta + rng.normal(0, 0.5, 60)
    y = y - y.mean()
    coef = st.lasso_fit(X, y, lam=10.0)
    nonzero = np.flatnonzero(np.abs(coef) > 1e-6)
    assert 0 < len(nonzero) < 10            # genuinely sparse
    assert set([0, 1, 2]).issubset(set(nonzero.tolist()))  # keeps the true three


def test_coefficient_path_grows_with_relaxing_penalty():
    rng = np.random.default_rng(7)
    X = rng.normal(0, 1, (50, 6))
    y = X @ np.array([2.0, 0, 0, -1.0, 0, 0]) + rng.normal(0, 0.4, 50)
    lambdas = np.logspace(1.5, -2, 8)       # strong -> weak
    path = st.coefficient_path(X, y, lambdas, "lasso")
    nnz = (np.abs(path) > 1e-6).sum(axis=1)
    assert nnz[0] <= nnz[-1]                 # more terms enter as penalty relaxes


# ── separation_demo sanity (used by the same chapters) ────────────────────────

def test_langmuir_saturates_at_qmax():
    assert sep.langmuir(1e9, q_max=50.0, b=0.1) == pytest.approx(50.0, rel=1e-6)
    # dilute limit slope is the Henry constant H = q_max * b
    small = sep.langmuir(1e-4, q_max=50.0, b=0.1)
    assert small == pytest.approx(50.0 * 0.1 * 1e-4, rel=1e-3)


# ── importance: regularized linear + logistic GLM screening ───────────────────

def _sparse_presence(rng, n, p, density=0.15):
    return (rng.uniform(size=(n, p)) < density).astype(float)


def test_regularized_importance_recovers_signal():
    rng = np.random.default_rng(8)
    n, p = 200, 30
    X = _sparse_presence(rng, n, p)
    effect = np.zeros(p)
    effect[[2, 5, 11]] = [2.0, -1.5, 1.0]
    y = X @ effect + rng.normal(0, 0.4, n)
    res = regularized_importance(X, y, seed=0, n_repeats=3)
    assert isinstance(res, ImportanceResult)
    assert res.cv_score > 0.4
    # the three real features should be among the most important
    top = list(res.importances.head(6).index)
    assert sum(name in top for name in ("f02", "f05", "f11")) >= 2


def test_logistic_screening_runs_and_predicts():
    rng = np.random.default_rng(9)
    n, p = 150, 20
    X = _sparse_presence(rng, n, p, density=0.25)
    eta = 2.5 * X[:, 3] - 2.0 * X[:, 7] + 0.2
    y = (rng.uniform(size=n) < st.sigmoid(eta)).astype(int)
    res = logistic_screening(X, y, seed=0)
    assert isinstance(res, LogisticScreenResult)
    assert 0.0 <= res.cv_auc <= 1.0
    assert res.cv_auc > 0.6                  # better than chance on a real signal
    assert len(res.coefficients) == p
