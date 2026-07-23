"""Multivariate analysis tools for design space data.

* :func:`pca_analysis` — principal component analysis with scores, loadings,
  and explained-variance diagnostics.
* :func:`pls_analysis` — PLS regression with VIP scores and cross-validated Q².

Both functions auto-scale inputs to zero mean and unit variance by default —
essential when factors and responses carry different physical units.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


# ── PCA ───────────────────────────────────────────────────────────────────────

@dataclass
class PCAResult:
    """Container returned by :func:`pca_analysis`."""

    scores: np.ndarray                    # (n_samples, n_components) — T
    loadings: np.ndarray                  # (n_features, n_components) — P
    explained_variance_ratio: np.ndarray  # (n_components,)
    cumulative_variance: np.ndarray       # (n_components,)
    feature_names: list[str]
    _pca: object = field(repr=False)
    _scaler: object = field(repr=False)

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Project new samples onto the fitted principal components."""
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names].values
        return self._pca.transform(self._scaler.transform(X.astype(float)))


def pca_analysis(
    data: pd.DataFrame,
    features: list[str],
    *,
    n_components: int | None = None,
    scale: bool = True,
) -> PCAResult:
    """Principal component analysis on a subset of DataFrame columns.

    The X matrix is decomposed as **X_s = T P^T + E** where T are the scores
    (sample coordinates in PC space), P the loadings (variable directions), and
    E the residual.  Columns of P are the eigenvectors of the sample covariance
    matrix, ordered by decreasing eigenvalue.

    Parameters
    ----------
    data:
        Input DataFrame.
    features:
        Column names to include in the X block.
    n_components:
        Number of PCs to retain.  Defaults to ``min(n_samples, n_features)``.
    scale:
        If True (default), standardise each column to zero mean and unit
        variance before fitting.  Set to False only when columns already share
        the same scale and zero mean.

    Returns
    -------
    PCAResult
        Scores, loadings, explained-variance ratio, cumulative variance, and
        a ``.transform()`` method for projecting new data.
    """
    X = data[features].values.astype(float)
    scaler = StandardScaler(with_std=scale)
    X_s = scaler.fit_transform(X)

    n_comp = n_components if n_components is not None else min(X.shape)
    pca = PCA(n_components=n_comp)
    scores = pca.fit_transform(X_s)

    evr = pca.explained_variance_ratio_
    return PCAResult(
        scores=scores,
        loadings=pca.components_.T,  # (n_features, n_components)
        explained_variance_ratio=evr,
        cumulative_variance=np.cumsum(evr),
        feature_names=list(features),
        _pca=pca,
        _scaler=scaler,
    )


# ── PLS ───────────────────────────────────────────────────────────────────────

@dataclass
class PLSResult:
    """Container returned by :func:`pls_analysis`."""

    x_scores: np.ndarray     # (n_samples, n_components) — T
    y_scores: np.ndarray     # (n_samples, n_components) — U
    x_loadings: np.ndarray   # (n_features, n_components) — P
    y_loadings: np.ndarray   # (n_targets, n_components) — Q
    x_weights: np.ndarray    # (n_features, n_components) — W
    vip_scores: np.ndarray   # (n_features,)
    r2_x: np.ndarray         # (n_components,) cumulative R²X
    r2_y: np.ndarray         # (n_components,) cumulative R²Y
    q2: float                # cross-validated Q²
    coefficients: np.ndarray # (n_features, n_targets) in auto-scaled space
    feature_names: list[str]
    response_names: list[str]
    _model: object = field(repr=False)
    _x_scaler: object = field(repr=False)
    _y_scaler: object = field(repr=False)


def _compute_vip(model: PLSRegression) -> np.ndarray:
    """Variable Importance in Projection (VIP) scores.

    VIP_j = sqrt( p * sum_a [ (SS_a / SS_total) * (w_ja / ||w_a||)^2 ] )

    where SS_a = t_a^T t_a * q_a^T q_a is the Y-variance captured by LV a.
    """
    T = model.x_scores_    # (n, A)
    W = model.x_weights_   # (p, A)
    Q = model.y_loadings_  # (q, A)
    p, A = W.shape

    ss = np.array([(T[:, a] @ T[:, a]) * (Q[:, a] @ Q[:, a]) for a in range(A)])
    ss_total = float(ss.sum()) or 1.0

    W_norm = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)
    return np.sqrt(p * (W_norm**2 @ (ss / ss_total)))


def pls_analysis(
    data: pd.DataFrame,
    features: list[str],
    responses: list[str],
    *,
    n_components: int = 2,
    cv_folds: int = 5,
    scale: bool = True,
) -> PLSResult:
    """PLS regression mapping process factors (X) to CQA responses (Y).

    Both X and Y are auto-scaled to zero mean and unit variance before fitting
    so that variables with different physical units contribute equally.

    The decomposition is **X_s = T P^T + E** and **Y_s = T Q^T + F**, where
    the shared score matrix T maximises Cov(Xw, Yc) at each step.

    Parameters
    ----------
    data:
        DataFrame containing both factor and response columns.
    features:
        X-block column names (CPPs / process factors).
    responses:
        Y-block column names (CQAs / responses).
    n_components:
        Number of PLS latent variables to extract.
    cv_folds:
        Number of k-fold cross-validation folds for Q² estimation.
        Use ``cv_folds=len(data)`` for leave-one-out (recommended for n<50).
    scale:
        Standardise X and Y to unit variance before fitting.

    Returns
    -------
    PLSResult
        Scores, loadings, weights, VIP scores, R²X, R²Y, Q², and
        regression coefficients (all in auto-scaled space).
    """
    X = data[features].values.astype(float)
    Y = data[responses].values.astype(float)
    if Y.ndim == 1:
        Y = Y[:, None]

    x_scaler = StandardScaler(with_std=scale)
    y_scaler = StandardScaler(with_std=scale)
    X_s = x_scaler.fit_transform(X)
    Y_s = y_scaler.fit_transform(Y)

    model = PLSRegression(n_components=n_components, scale=False)
    model.fit(X_s, Y_s)

    # Cumulative R²X and R²Y (Frobenius-norm based)
    ss_x = float(np.sum(X_s**2)) or 1.0
    ss_y = float(np.sum(Y_s**2)) or 1.0
    r2_x = np.array([
        1.0 - float(np.sum(
            (X_s - model.x_scores_[:, :a + 1] @ model.x_loadings_[:, :a + 1].T) ** 2
        )) / ss_x
        for a in range(n_components)
    ])
    r2_y = np.array([
        1.0 - float(np.sum(
            (Y_s - model.x_scores_[:, :a + 1] @ model.y_loadings_[:, :a + 1].T) ** 2
        )) / ss_y
        for a in range(n_components)
    ])

    # Q²: cross-validated R²Y — PLSRegression(scale=True) handles per-fold scaling
    q2_scores = cross_val_score(
        PLSRegression(n_components=n_components, scale=scale),
        X, Y,
        cv=min(cv_folds, len(X)),
        scoring="r2",
    )
    q2 = float(np.mean(q2_scores))

    return PLSResult(
        x_scores=model.x_scores_,
        y_scores=model.y_scores_,
        x_loadings=model.x_loadings_,
        y_loadings=model.y_loadings_,
        x_weights=model.x_weights_,
        vip_scores=_compute_vip(model),
        r2_x=r2_x,
        r2_y=r2_y,
        q2=q2,
        coefficients=model.coef_,
        feature_names=list(features),
        response_names=list(responses),
        _model=model,
        _x_scaler=x_scaler,
        _y_scaler=y_scaler,
    )
