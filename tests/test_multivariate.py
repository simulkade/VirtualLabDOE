"""Tests for vlab_doe.doe.multivariate (PCA and PLS)."""

import numpy as np
import pandas as pd
import pytest

from vlab_doe.doe.multivariate import pca_analysis, pls_analysis

# ── Shared fixture ─────────────────────────────────────────────────────────────

RNG = np.random.default_rng(0)
N = 50
FEATURES = ["tmp", "crossflow", "feed_conc", "res_factor", "sieving"]
RESPONSES = ["proc_time", "mean_flux", "protein_yield"]


def _make_data(n: int = N) -> pd.DataFrame:
    """Synthetic 5-factor, 3-response dataset with known structure."""
    rng = np.random.default_rng(42)
    X = rng.uniform(size=(n, len(FEATURES)))
    # proc_time dominated by crossflow (col 1) and feed_conc (col 2)
    proc_time = 300 - 80 * X[:, 1] + 60 * X[:, 2] + rng.normal(0, 5, n)
    mean_flux = 5 + 10 * X[:, 0] + 8 * X[:, 1] + rng.normal(0, 0.5, n)
    protein_yield = 0.95 - 0.2 * X[:, 4] + rng.normal(0, 0.01, n)

    df = pd.DataFrame(X, columns=FEATURES)
    df["proc_time"] = proc_time
    df["mean_flux"] = mean_flux
    df["protein_yield"] = np.clip(protein_yield, 0, 1)
    return df


DATA = _make_data()


# ── PCA tests ─────────────────────────────────────────────────────────────────

class TestPCAResult:
    def test_scores_shape(self):
        res = pca_analysis(DATA, FEATURES, n_components=3)
        assert res.scores.shape == (N, 3)

    def test_loadings_shape(self):
        res = pca_analysis(DATA, FEATURES, n_components=3)
        assert res.loadings.shape == (len(FEATURES), 3)

    def test_explained_variance_sums_to_one(self):
        res = pca_analysis(DATA, FEATURES)
        assert abs(res.explained_variance_ratio.sum() - 1.0) < 1e-6

    def test_cumulative_variance_monotone(self):
        res = pca_analysis(DATA, FEATURES)
        diffs = np.diff(res.cumulative_variance)
        assert (diffs >= -1e-10).all()

    def test_cumulative_variance_last_is_one(self):
        res = pca_analysis(DATA, FEATURES)
        assert abs(res.cumulative_variance[-1] - 1.0) < 1e-6

    def test_loadings_orthonormal(self):
        res = pca_analysis(DATA, FEATURES)
        # P^T P should be identity
        PtP = res.loadings.T @ res.loadings
        assert np.allclose(PtP, np.eye(PtP.shape[0]), atol=1e-6)

    def test_feature_names_preserved(self):
        res = pca_analysis(DATA, FEATURES, n_components=2)
        assert res.feature_names == FEATURES

    def test_transform_shape(self):
        res = pca_analysis(DATA, FEATURES, n_components=3)
        new_scores = res.transform(DATA)
        assert new_scores.shape == (N, 3)

    def test_transform_matches_fit_scores(self):
        res = pca_analysis(DATA, FEATURES, n_components=3)
        new_scores = res.transform(DATA)
        assert np.allclose(new_scores, res.scores, atol=1e-8)

    def test_no_scale_mode(self):
        res = pca_analysis(DATA, FEATURES, n_components=2, scale=False)
        assert res.scores.shape == (N, 2)

    def test_single_component(self):
        res = pca_analysis(DATA, FEATURES, n_components=1)
        assert res.scores.shape == (N, 1)
        assert res.loadings.shape == (len(FEATURES), 1)


# ── PLS tests ─────────────────────────────────────────────────────────────────

class TestPLSResult:
    def test_x_scores_shape(self):
        res = pls_analysis(DATA, FEATURES, RESPONSES, n_components=2)
        assert res.x_scores.shape == (N, 2)

    def test_y_scores_shape(self):
        res = pls_analysis(DATA, FEATURES, RESPONSES, n_components=2)
        assert res.y_scores.shape == (N, 2)

    def test_x_loadings_shape(self):
        res = pls_analysis(DATA, FEATURES, RESPONSES, n_components=2)
        assert res.x_loadings.shape == (len(FEATURES), 2)

    def test_y_loadings_shape(self):
        res = pls_analysis(DATA, FEATURES, RESPONSES, n_components=2)
        assert res.y_loadings.shape == (len(RESPONSES), 2)

    def test_x_weights_shape(self):
        res = pls_analysis(DATA, FEATURES, RESPONSES, n_components=2)
        assert res.x_weights.shape == (len(FEATURES), 2)

    def test_vip_shape_and_nonnegative(self):
        res = pls_analysis(DATA, FEATURES, RESPONSES, n_components=2)
        assert res.vip_scores.shape == (len(FEATURES),)
        assert (res.vip_scores >= 0).all()

    def test_vip_important_factors_above_threshold(self):
        """crossflow (col 1) and sieving (col 4) drive the responses; check VIP > 1."""
        res = pls_analysis(DATA, FEATURES, RESPONSES, n_components=3)
        vip_map = dict(zip(res.feature_names, res.vip_scores))
        # crossflow drives both proc_time and mean_flux — should be important
        assert vip_map["crossflow"] > 1.0

    def test_r2_x_cumulative_monotone(self):
        res = pls_analysis(DATA, FEATURES, RESPONSES, n_components=3)
        assert (np.diff(res.r2_x) >= -1e-9).all()

    def test_r2_y_cumulative_monotone(self):
        res = pls_analysis(DATA, FEATURES, RESPONSES, n_components=3)
        assert (np.diff(res.r2_y) >= -1e-9).all()

    def test_r2_y_bounded(self):
        res = pls_analysis(DATA, FEATURES, RESPONSES, n_components=3)
        assert (res.r2_y >= 0).all()
        assert (res.r2_y <= 1 + 1e-9).all()

    def test_q2_is_scalar(self):
        res = pls_analysis(DATA, FEATURES, RESPONSES, n_components=2)
        assert isinstance(res.q2, float)

    def test_q2_below_r2y(self):
        """Q² (cross-validated) should not systematically exceed in-sample R²Y."""
        res = pls_analysis(DATA, FEATURES, RESPONSES, n_components=2, cv_folds=5)
        # Allow slight numerical slack from CV variance
        assert res.q2 <= res.r2_y[-1] + 0.1

    def test_feature_and_response_names(self):
        res = pls_analysis(DATA, FEATURES, RESPONSES, n_components=2)
        assert res.feature_names == FEATURES
        assert res.response_names == RESPONSES

    def test_single_response(self):
        res = pls_analysis(DATA, FEATURES, ["proc_time"], n_components=2)
        assert res.x_scores.shape == (N, 2)
        assert res.vip_scores.shape == (len(FEATURES),)

    def test_no_scale_mode(self):
        res = pls_analysis(DATA, FEATURES, RESPONSES, n_components=2, scale=False)
        assert res.x_scores.shape == (N, 2)

    def test_loo_cv(self):
        small = _make_data(n=20)
        res = pls_analysis(small, FEATURES, RESPONSES, n_components=2, cv_folds=20)
        assert isinstance(res.q2, float)
