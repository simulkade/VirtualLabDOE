"""The statistics of Chapter 1, implemented from first principles.

Four ideas recur throughout the monograph and are built here in the smallest
honest way:

#. **Least squares** -- the normal equations that fit every response surface
   (Chapter 4) and every calibration curve.  See :func:`ols_fit`.
#. **The analysis of variance** -- partitioning a response's scatter into a part
   a model explains and an unexplained residual, and testing it with an
   :math:`F`-ratio.  See :func:`regression_anova`.
#. **The bootstrap** -- turning one sample into a sampling distribution by
   resampling it, so that any statistic gets an error bar without a formula.
   See :func:`bootstrap_ci`.
#. **Variance components** -- separating *variability* (real spread between
   batches) from *uncertainty* (measurement noise), the distinction that
   organises the stochastic fermentation model of Chapter 5.  See
   :func:`variance_components`.
#. **Generalized linear models** -- extending the linear predictor through a
   *link* so the response need not be normal; logistic regression for a yes/no
   outcome, fitted by the same weighted normal equations.  See
   :func:`logistic_fit`.
#. **Regularization** -- penalizing coefficient size to tame collinearity and
   the "more features than runs" regime, with ridge shrinking smoothly and the
   lasso selecting.  See :func:`ridge_fit`, :func:`lasso_fit`.

Everything is NumPy; nothing here is clever.  The point is legibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


# ── 1. Ordinary least squares ─────────────────────────────────────────────────

@dataclass(frozen=True)
class OLSFit:
    """The result of an ordinary-least-squares fit ``y ≈ X @ beta``.

    Attributes
    ----------
    beta:
        Estimated coefficients, :math:`\\hat\\beta = (X^\\mathsf{T}X)^{-1}X^\\mathsf{T}y`.
    cov:
        Covariance of the estimate, :math:`\\sigma^2 (X^\\mathsf{T}X)^{-1}`.
    residuals:
        The vector ``y - X @ beta``.
    sigma2:
        Unbiased residual variance, ``RSS / (n - p)``.
    r_squared:
        Fraction of the response's variance the model explains.
    """

    beta: np.ndarray
    cov: np.ndarray
    residuals: np.ndarray
    sigma2: float
    r_squared: float

    @property
    def stderr(self) -> np.ndarray:
        """Standard error of each coefficient (the square root of ``diag(cov)``)."""
        return np.sqrt(np.diag(self.cov))


def ols_fit(X: np.ndarray, y: np.ndarray) -> OLSFit:
    """Fit ``y ≈ X @ beta`` by ordinary least squares via the normal equations.

    The algorithm is exactly the textbook one, and it is short enough to read in
    full:

    #. Form the *normal matrix* :math:`X^\\mathsf{T}X` and the right-hand side
       :math:`X^\\mathsf{T}y`.
    #. Solve the linear system :math:`(X^\\mathsf{T}X)\\,\\beta = X^\\mathsf{T}y`.
       (We use :func:`numpy.linalg.solve`, not an explicit inverse, because it is
       more accurate; the inverse is formed only to report the covariance.)
    #. The residual variance ``sigma2`` and the coefficient covariance
       ``sigma2 * inv(XtX)`` follow.

    Parameters
    ----------
    X:
        Design matrix of shape ``(n, p)``.  Include a column of ones for an
        intercept if you want one -- this routine adds nothing implicitly.
    y:
        Response vector of length ``n``.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape

    XtX = X.T @ X
    Xty = X.T @ y
    beta = np.linalg.solve(XtX, Xty)

    residuals = y - X @ beta
    rss = float(residuals @ residuals)
    dof = n - p
    sigma2 = rss / dof if dof > 0 else np.nan
    cov = sigma2 * np.linalg.inv(XtX)

    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - rss / tss if tss > 0 else np.nan
    return OLSFit(beta=beta, cov=cov, residuals=residuals, sigma2=sigma2, r_squared=r2)


# ── 2. The analysis of variance for a regression ──────────────────────────────

@dataclass(frozen=True)
class ANOVAResult:
    """Overall :math:`F`-test of a regression: does the model beat the mean?"""

    ss_model: float
    ss_residual: float
    df_model: int
    df_residual: int
    f_stat: float
    p_value: float


def regression_anova(X: np.ndarray, y: np.ndarray) -> ANOVAResult:
    """Partition the total variation of ``y`` and test the fit with an F-ratio.

    The total sum of squares splits cleanly into a part the model explains and a
    residual,

    .. math::

        \\underbrace{\\sum (y_i-\\bar y)^2}_{\\mathrm{SS_{tot}}}
        = \\underbrace{\\sum (\\hat y_i-\\bar y)^2}_{\\mathrm{SS_{model}}}
        + \\underbrace{\\sum (y_i-\\hat y_i)^2}_{\\mathrm{SS_{res}}} ,

    and the ratio of the two *mean squares* (sum of squares over degrees of
    freedom) is, under the null hypothesis that the model adds nothing, an
    :math:`F` random variable.  A small ``p_value`` says the explained variation
    is too large to be chance.  ``X`` is assumed to contain an intercept column.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape

    fit = ols_fit(X, y)
    yhat = X @ fit.beta
    ss_res = float(((y - yhat) ** 2).sum())
    ss_model = float(((yhat - y.mean()) ** 2).sum())

    df_model = p - 1                # minus one for the intercept
    df_res = n - p
    ms_model = ss_model / df_model
    ms_res = ss_res / df_res
    f_stat = ms_model / ms_res
    p_value = float(stats.f.sf(f_stat, df_model, df_res))
    return ANOVAResult(ss_model, ss_res, df_model, df_res, f_stat, p_value)


# ── 3. The bootstrap ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BootstrapResult:
    """A bootstrap sampling distribution and the percentile interval it implies."""

    estimate: float
    replicates: np.ndarray
    ci_low: float
    ci_high: float
    alpha: float


def bootstrap_ci(
    sample: np.ndarray,
    statistic=np.mean,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int | None = 0,
) -> BootstrapResult:
    """Bootstrap a confidence interval for ``statistic`` of ``sample``.

    The idea, due to Efron, needs no distributional assumption.  The one sample
    we have is our best picture of the population, so we *resample it, with
    replacement, many times*, recompute the statistic on each resample, and read
    the spread of those replicates as the sampling distribution of the
    statistic.  The percentile interval is then just the empirical
    :math:`[\\alpha/2,\\,1-\\alpha/2]` quantiles of the replicates.

    Parameters
    ----------
    sample:
        The observed data, length ``n``.
    statistic:
        Any callable mapping a 1-D array to a scalar (default the mean).
    n_boot:
        Number of bootstrap resamples.
    alpha:
        Significance level; ``alpha=0.05`` gives a 95% interval.
    seed:
        Seed for reproducibility.
    """
    sample = np.asarray(sample, dtype=float)
    n = sample.size
    rng = np.random.default_rng(seed)

    # Each row is one resample of size n drawn with replacement.
    idx = rng.integers(0, n, size=(n_boot, n))
    replicates = np.array([statistic(sample[row]) for row in idx])

    lo, hi = np.quantile(replicates, [alpha / 2, 1 - alpha / 2])
    return BootstrapResult(
        estimate=float(statistic(sample)),
        replicates=replicates,
        ci_low=float(lo),
        ci_high=float(hi),
        alpha=alpha,
    )


# ── 4. Variance components: variability versus uncertainty ────────────────────

@dataclass(frozen=True)
class VarianceComponents:
    """A one-way decomposition of scatter into between- and within-group parts.

    In the language of Chapter 5, ``between`` is *variability* (real,
    batch-to-batch biology that better instruments do not remove) and ``within``
    is *uncertainty* (measurement noise that replicate readings average away).
    """

    within: float          # σ²_within  (measurement / repeat noise)
    between: float          # σ²_between (true group-to-group spread)
    grand_mean: float
    group_means: np.ndarray


def variance_components(groups: np.ndarray) -> VarianceComponents:
    """Split scatter into within-group and between-group variance.

    Given ``groups`` of shape ``(k, m)`` -- ``k`` batches each measured ``m``
    times -- the classic balanced random-effects estimators are

    .. math::

        \\hat\\sigma^2_{\\text{within}} = \\mathrm{MS_{within}}, \\qquad
        \\hat\\sigma^2_{\\text{between}}
            = \\frac{\\mathrm{MS_{between}} - \\mathrm{MS_{within}}}{m},

    where the mean squares come from the same sum-of-squares partition as the
    ANOVA of :func:`regression_anova`.  The between-batch estimate is floored at
    zero (a negative estimate just means the data cannot resolve any real
    between-batch spread above the noise).
    """
    groups = np.asarray(groups, dtype=float)
    k, m = groups.shape

    group_means = groups.mean(axis=1)
    grand_mean = groups.mean()

    ss_within = float(((groups - group_means[:, None]) ** 2).sum())
    ss_between = float(m * ((group_means - grand_mean) ** 2).sum())

    ms_within = ss_within / (k * (m - 1))
    ms_between = ss_between / (k - 1)

    sigma2_within = ms_within
    sigma2_between = max((ms_between - ms_within) / m, 0.0)
    return VarianceComponents(
        within=sigma2_within,
        between=sigma2_between,
        grand_mean=float(grand_mean),
        group_means=group_means,
    )


# ── 5. Generalized linear models: logistic regression by IRLS ─────────────────

@dataclass(frozen=True)
class LogisticFit:
    """A fitted logistic regression ``P(y=1) = 1/(1 + e^{-Xβ})``."""

    beta: np.ndarray
    probabilities: np.ndarray
    n_iter: int


def sigmoid(z: np.ndarray) -> np.ndarray:
    """The logistic link's inverse, ``1/(1+e^{-z})``, written to avoid overflow."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos, neg = z >= 0, z < 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def logistic_fit(
    X: np.ndarray, y: np.ndarray, max_iter: int = 50, tol: float = 1e-9
) -> LogisticFit:
    """Fit a logistic regression by iteratively reweighted least squares (IRLS).

    A *generalized linear model* keeps the linear predictor :math:`\\eta = X\\beta`
    of ordinary regression but passes it through a *link*.  For a binary response
    the link is the logit, so the mean is :math:`\\mu = \\mathrm{sigmoid}(\\eta)`
    and the response is Bernoulli.  There is no closed form for the
    maximum-likelihood :math:`\\hat\\beta`, but Newton's method on the
    log-likelihood turns out to be a *weighted* least-squares problem solved
    repeatedly -- the same normal equations as :func:`ols_fit`, now with weights
    :math:`w_i = \\mu_i(1-\\mu_i)` that say how informative each observation is:

    .. math::

        \\beta \\leftarrow \\beta
          + (X^\\mathsf{T} W X)^{-1} X^\\mathsf{T}(y - \\mu).

    This is the whole of GLM fitting; swapping the link and the weights gives
    Poisson regression, probit, and the rest.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    p = X.shape[1]
    beta = np.zeros(p)

    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        mu = sigmoid(X @ beta)
        w = np.clip(mu * (1.0 - mu), 1e-9, None)     # IRLS weights
        XtWX = X.T @ (w[:, None] * X)
        grad = X.T @ (y - mu)
        # A whisker of ridge keeps the Newton step well-posed under separation.
        delta = np.linalg.solve(XtWX + 1e-8 * np.eye(p), grad)
        beta = beta + delta
        if np.max(np.abs(delta)) < tol:
            break
    return LogisticFit(beta=beta, probabilities=sigmoid(X @ beta), n_iter=n_iter)


# ── 6. Regularization: ridge and the lasso ────────────────────────────────────

def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Ridge (L2-penalized) regression: ``min ||y-Xβ||² + λ||β||²``.

    The penalty adds ``λ`` to the diagonal of the normal matrix before solving,

    .. math::  \\hat\\beta = (X^\\mathsf{T}X + \\lambda I)^{-1} X^\\mathsf{T}y,

    which is always invertible even when :math:`X^\\mathsf{T}X` is singular (more
    features than runs, or collinear columns).  Ridge *shrinks* every coefficient
    toward zero but sets none exactly to zero.  Standardize the columns of ``X``
    and center ``y`` first, so the single ``λ`` penalizes comparable quantities.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    p = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ y)


def _soft_threshold(z: float, gamma: float) -> float:
    """The soft-thresholding operator ``sign(z)·max(|z|-γ, 0)``."""
    return np.sign(z) * max(abs(z) - gamma, 0.0)


def lasso_fit(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    max_iter: int = 1000,
    tol: float = 1e-7,
) -> np.ndarray:
    """Lasso (L1-penalized) regression by coordinate descent.

    The lasso minimizes :math:`\\tfrac12\\|y-X\\beta\\|^2 + \\lambda\\|\\beta\\|_1`.
    The L1 penalty has a corner at zero, so it does not merely shrink
    coefficients -- it drives the unhelpful ones *exactly* to zero, performing
    variable selection.  Cycling over coordinates, each one-dimensional update is
    a soft-threshold of the partial residual,

    .. math::

        \\beta_j \\leftarrow
          \\frac{S\\!\\bigl(X_j^\\mathsf{T} r_{(-j)},\\ \\lambda\\bigr)}
               {\\|X_j\\|^2},

    where :math:`r_{(-j)}` is the residual holding all but the ``j``-th
    coefficient fixed and :math:`S` is :func:`_soft_threshold`.  Columns of ``X``
    should be standardized and ``y`` centered.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    beta = np.zeros(p)
    col_norm2 = (X ** 2).sum(axis=0)
    col_norm2[col_norm2 == 0] = 1.0

    for _ in range(max_iter):
        beta_old = beta.copy()
        for j in range(p):
            # Partial residual with coordinate j removed.
            r_j = y - X @ beta + X[:, j] * beta[j]
            rho = X[:, j] @ r_j
            beta[j] = _soft_threshold(rho, lam) / col_norm2[j]
        if np.max(np.abs(beta - beta_old)) < tol:
            break
    return beta


def coefficient_path(X: np.ndarray, y: np.ndarray, lambdas, method: str = "lasso"):
    """Trace coefficients across a grid of penalties ``λ`` (a regularization path).

    Returns an array of shape ``(len(lambdas), n_features)``.  As ``λ`` falls the
    path runs from the all-zero (ridge: all-shrunk) solution toward the
    unpenalized least-squares fit; the lasso path is piecewise where features
    enter one by one.
    """
    fit = lasso_fit if method == "lasso" else ridge_fit
    return np.array([fit(X, y, lam) for lam in lambdas])


__all__ = [
    "OLSFit",
    "ols_fit",
    "ANOVAResult",
    "regression_anova",
    "BootstrapResult",
    "bootstrap_ci",
    "VarianceComponents",
    "variance_components",
    "LogisticFit",
    "sigmoid",
    "logistic_fit",
    "ridge_fit",
    "lasso_fit",
    "coefficient_path",
]
