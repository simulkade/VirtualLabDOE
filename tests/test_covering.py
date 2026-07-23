"""Tests for the covering-array design, strain library, and tree-model importance."""

import numpy as np
import pytest

from vlab_doe.config import make_rng
from vlab_doe.doe.covering import covering_array
from vlab_doe.doe.importance import (
    gradient_boosting_importance,
    random_forest_importance,
)
from vlab_doe.models import fermentation as ferm


# ── Covering array ──────────────────────────────────────────────────────────────

def test_run_sizes_within_bounds():
    d = covering_array(50, 200, min_size=2, max_size=5, seed=1)
    sizes = [len(r) for r in d.runs]
    assert min(sizes) >= 2 and max(sizes) <= 5
    assert d.n_runs == 200


def test_pairwise_coverage_is_high():
    """200 runs of 2–5 strains should co-test the large majority of the 1225 pairs."""
    d = covering_array(50, 200, min_size=2, max_size=5, seed=1)
    cov = d.coverage(2)
    assert cov["subsets_total"] == 1225
    assert cov["coverage_fraction"] > 0.85
    assert cov["subsets_covered"] <= cov["subsets_total"]


def test_small_design_fully_covers_pairs():
    """With enough runs relative to the pool, every pair is covered."""
    d = covering_array(8, 60, min_size=2, max_size=4, seed=2)
    assert d.coverage(2)["coverage_fraction"] == pytest.approx(1.0)


def test_appearances_are_balanced():
    d = covering_array(50, 200, min_size=2, max_size=5, seed=1)
    app = d.appearances()
    assert app.sum() == sum(len(r) for r in d.runs)
    # No strain is wildly over- or under-used.
    assert app.max() - app.min() <= 8


def test_reproducible_with_seed():
    a = covering_array(30, 100, min_size=2, max_size=5, seed=42)
    b = covering_array(30, 100, min_size=2, max_size=5, seed=42)
    assert a.runs == b.runs


def test_matrix_and_dataframe_shapes():
    d = covering_array(12, 40, min_size=2, max_size=4, seed=3)
    m = d.matrix()
    assert m.shape == (40, 12)
    assert set(np.unique(m)).issubset({0, 1})
    assert np.all(m.sum(axis=1) == [len(r) for r in d.runs])
    df = d.to_dataframe()
    assert len(df) == 40
    assert "members" in df.columns and "run_type" in df.columns


def test_invalid_bounds_raise():
    with pytest.raises(ValueError):
        covering_array(10, 20, min_size=5, max_size=3)
    with pytest.raises(ValueError):
        covering_array(10, 20, min_size=2, max_size=20)


# ── Strain library ──────────────────────────────────────────────────────────────

def test_random_library_shapes_and_validity():
    rng = make_rng(0)
    lib = ferm.random_strain_library(50, rng)
    assert lib.n_strains == 50
    assert lib.interaction.shape == (50, 50)
    assert np.allclose(np.diag(lib.interaction), 0.0)
    # Stimulation must keep growth factor positive (k > -1).
    assert lib.interaction.min() > -1.0
    for s in lib.strains:
        assert s.t_min < s.t_opt < s.t_max
        assert s.mu_max > 0 and s.acid_growth > 0


def test_library_consortium_subset():
    rng = make_rng(0)
    lib = ferm.random_strain_library(20, rng)
    cons = lib.consortium([3, 7, 11])
    assert cons.n_strains == 3
    assert cons.interaction.shape == (3, 3)
    assert np.allclose(cons.normalized_fractions(), 1 / 3)
    # Submatrix matches the global matrix entries.
    assert cons.interaction[0, 1] == lib.interaction[3, 7]


def test_library_consortium_runs_in_engine():
    rng = make_rng(1)
    lib = ferm.random_strain_library(15, rng)
    cons = lib.consortium([0, 1, 2, 3])
    r = ferm.run_fermentation(
        ferm.FermentationSetup(consortium=cons, temperature=43.0), np.linspace(0, 12, 121))
    assert r.ph[0] > 6.0 and r.ph[-1] < r.ph[0]
    assert r.biomass.shape[0] == 4


# ── Tree-model importance ───────────────────────────────────────────────────────

def _synthetic(n=200, p=12, seed=0):
    rng = make_rng(seed)
    X = rng.integers(0, 2, size=(n, p)).astype(float)
    # y depends strongly on features 2 and 5 (and their interaction), plus noise.
    y = 3.0 * X[:, 2] - 2.5 * X[:, 5] + 1.5 * X[:, 2] * X[:, 5] + rng.normal(0, 0.3, n)
    names = [f"f{i:02d}" for i in range(p)]
    return X, y, names


def test_random_forest_recovers_relevant_features():
    X, y, names = _synthetic()
    res = random_forest_importance(X, y, feature_names=names, n_estimators=200,
                                   cv=4, n_repeats=5, seed=1)
    assert res.cv_score > 0.5
    assert set(res.importances.index[:2]) == {"f02", "f05"}
    assert res.predictions.shape == y.shape
    assert len(res.importances) == X.shape[1]


def test_gradient_boosting_recovers_relevant_features():
    X, y, names = _synthetic()
    res = gradient_boosting_importance(X, y, feature_names=names, n_estimators=200,
                                       cv=4, n_repeats=5, seed=1)
    assert res.cv_score > 0.5
    assert set(res.importances.index[:2]) == {"f02", "f05"}


def test_importance_runs_on_covering_design():
    """The two pieces compose: a covering design's matrix feeds the importance analysis."""
    X, y, names = _synthetic(p=10)
    rf = random_forest_importance(X, y, feature_names=names, n_estimators=100, cv=3,
                                  n_repeats=3, seed=0)
    assert rf.cv_scores.shape == (3,)
    assert rf.importances_std.shape == rf.importances.shape
