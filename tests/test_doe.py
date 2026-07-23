"""Phase 2 & 3 design tests."""

import numpy as np
import pandas as pd
import pytest

from vlab_doe.doe.factorial import Factor, full_factorial, run_design
from vlab_doe.doe.lhs import latin_hypercube, coverage_metrics

FACTORS = [
    Factor("pH", 5.0, 7.0),
    Factor("salt", 50.0, 200.0),
    Factor("load_density", 5.0, 30.0),
]


# ── Full factorial ────────────────────────────────────────────────────────────

def test_full_factorial_2level_dimensions():
    """A 2-level full factorial of k factors has 2**k rows."""
    design = full_factorial(FACTORS, levels=2)
    factorial_rows = design[design["run_type"] == "factorial"]
    assert len(factorial_rows) == 2 ** len(FACTORS)


def test_full_factorial_3level_dimensions():
    """A 3-level full factorial of k factors has 3**k rows."""
    design = full_factorial(FACTORS, levels=3)
    factorial_rows = design[design["run_type"] == "factorial"]
    assert len(factorial_rows) == 3 ** len(FACTORS)


def test_full_factorial_with_center_points():
    design = full_factorial(FACTORS, levels=2, center_points=5)
    assert (design["run_type"] == "center_point").sum() == 5


def test_full_factorial_column_names():
    design = full_factorial(FACTORS, levels=2)
    for f in FACTORS:
        assert f.name in design.columns


def test_full_factorial_values_within_bounds():
    design = full_factorial(FACTORS, levels=2)
    for f in FACTORS:
        assert design[f.name].min() >= f.low - 1e-9
        assert design[f.name].max() <= f.high + 1e-9


def test_full_factorial_center_points_at_midpoint():
    design = full_factorial(FACTORS, levels=2, center_points=3)
    cp = design[design["run_type"] == "center_point"]
    for f in FACTORS:
        assert np.allclose(cp[f.name].values, f.center)


def test_run_design_appends_responses():
    design = full_factorial([Factor("x", 0.0, 1.0)], levels=2)
    results = run_design(design, lambda pt: {"response": pt["x"] ** 2})
    assert "response" in results.columns
    assert len(results) == len(design)


# ── Latin Hypercube Sampling ──────────────────────────────────────────────────

def test_lhs_sample_count():
    design = latin_hypercube(FACTORS, n_samples=50, seed=0)
    assert len(design) == 50


def test_lhs_column_names():
    design = latin_hypercube(FACTORS, n_samples=20, seed=0)
    assert list(design.columns) == [f.name for f in FACTORS]


def test_lhs_values_within_bounds():
    design = latin_hypercube(FACTORS, n_samples=40, seed=42)
    for f in FACTORS:
        assert design[f.name].min() >= f.low - 1e-9
        assert design[f.name].max() <= f.high + 1e-9


def test_lhs_reproducible_with_same_seed():
    a = latin_hypercube(FACTORS, n_samples=30, seed=77)
    b = latin_hypercube(FACTORS, n_samples=30, seed=77)
    pd.testing.assert_frame_equal(a, b)


def test_lhs_different_seeds_differ():
    a = latin_hypercube(FACTORS, n_samples=20, seed=1)
    b = latin_hypercube(FACTORS, n_samples=20, seed=2)
    assert not a.equals(b)


def test_coverage_metrics_returns_expected_keys():
    design = latin_hypercube(FACTORS, n_samples=30, seed=5)
    m = coverage_metrics(design)
    assert "discrepancy" in m
    assert "min_pairwise_dist" in m
    assert "mean_pairwise_dist" in m


def test_lhs_lower_discrepancy_than_random():
    """Optimised LHS should have lower discrepancy than pure random sampling."""
    from scipy.stats import qmc
    rng = np.random.default_rng(0)
    n, k = 30, len(FACTORS)
    random_unit = rng.random((n, k))
    lhs_design = latin_hypercube(FACTORS, n_samples=n, seed=0)

    lows = np.array([f.low for f in FACTORS])
    highs = np.array([f.high for f in FACTORS])
    lhs_unit = (lhs_design.values - lows) / (highs - lows)

    disc_random = qmc.discrepancy(random_unit)
    disc_lhs = qmc.discrepancy(lhs_unit)
    assert disc_lhs < disc_random


# ── Generalized linear response models (Chapter 4 GLM application) ─────────────

def _glm_synthetic(seed=0, n=120):
    """A 2-CPP table with a logistic pass/fail and a Poisson count outcome."""
    rng = np.random.default_rng(seed)
    pH = rng.uniform(4.5, 6.0, n)
    grad = rng.uniform(8.0, 40.0, n)
    # pass is likely at low pH; count of co-eluters grows with pH
    eta = 6.0 * (5.0 - pH) + 0.05 * (grad - 24.0)
    passed = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(int)
    lam = np.exp(-3.0 + 1.0 * (pH - 4.5))
    count = rng.poisson(lam)
    return pd.DataFrame({"pH": pH, "gradient_cv": grad, "passed": passed, "contam": count})


def test_fit_glm_logistic_predicts_probabilities():
    from vlab_doe.doe.analysis import fit_glm_response, predict_glm_grid

    df = _glm_synthetic()
    res = fit_glm_response(df, "passed", ["pH", "gradient_cv"], family="binomial")
    assert res.family == "binomial"
    assert 0.0 <= res.pseudo_r2 <= 1.0
    # pH should drive the pass probability (negative logit coefficient)
    assert res.effects["pH"] < 0

    X, Y, P = predict_glm_grid(res, "pH", "gradient_cv", (4.5, 6.0), (8.0, 40.0), n=25)
    assert P.shape == (25, 25)
    assert np.all((P >= 0.0) & (P <= 1.0))            # genuine probabilities
    # low pH should be more likely to pass than high pH
    assert P[:, 0].mean() > P[:, -1].mean()


def test_fit_glm_poisson_predicts_nonnegative_rates():
    from vlab_doe.doe.analysis import fit_glm_response, predict_glm_grid

    df = _glm_synthetic()
    res = fit_glm_response(df, "contam", ["pH", "gradient_cv"], family="poisson")
    assert res.family == "poisson"
    _, _, M = predict_glm_grid(res, "pH", "gradient_cv", (4.5, 6.0), (8.0, 40.0), n=20)
    assert np.all(M >= 0)                              # expected counts are non-negative


def test_fit_glm_rejects_unknown_family():
    from vlab_doe.doe.analysis import fit_glm_response

    df = _glm_synthetic()
    with pytest.raises(ValueError):
        fit_glm_response(df, "passed", ["pH", "gradient_cv"], family="gaussian")
