"""Phase 2 — Statistical analysis of factorial designs.

Fits an OLS response-surface model via ``statsmodels``, produces a Type-II
ANOVA table, and derives **proven acceptable ranges (PAR)** for each factor.

Design:
- The formula is constructed programmatically: main effects + optional
  two-way interactions + optional quadratic terms (Response Surface Model).
- PAR derivation inverts the linear model: for each factor, while holding
  all others at their centre points, find the factor range where the predicted
  response stays within the specification bounds.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.anova as sa


@dataclass
class AnovaResult:
    """Container for a fitted response model and its ANOVA table."""

    model: object           # statsmodels OLSResults object
    anova_table: pd.DataFrame
    effects: pd.Series      # regression coefficients (main effects & interactions)
    factor_names: list[str] = None   # plain factor names (no interactions/quadratics)


def fit_response_model(
    data: pd.DataFrame,
    response: str,
    factors: Sequence[str],
    *,
    interactions: bool = True,
    quadratic: bool = False,
) -> AnovaResult:
    """Fit an OLS response model and compute its Type-II ANOVA table.

    The formula follows Wilkinson-Rogers notation::

        yield ~ pH + salt + load_density
              + pH:salt + pH:load_density + salt:load_density   (if interactions)
              + I(pH**2) + I(salt**2) + ...                      (if quadratic)

    Parameters
    ----------
    data:
        Design + response table (physical units).
    response:
        Name of the response column.
    factors:
        Factor column names (must exist in *data*).
    interactions:
        Include all two-way interaction terms.
    quadratic:
        Include quadratic (second-order) terms for a response-surface model.
        Requires data from a 3-level (or CCD) design.
    """
    terms: list[str] = list(factors)

    if quadratic:
        for f in factors:
            terms.append(f"I({f}**2)")

    if interactions:
        for f1, f2 in itertools.combinations(factors, 2):
            terms.append(f"{f1}:{f2}")

    # Patsy can't parse Python reserved words (e.g. "yield") as column names.
    # Alias the response to a safe internal name before building the formula.
    _RESPONSE_ALIAS = "__response__"
    data_fit = data.copy()
    data_fit[_RESPONSE_ALIAS] = data_fit[response]

    formula = f"{_RESPONSE_ALIAS} ~ " + " + ".join(terms)

    model = smf.ols(formula, data=data_fit).fit()
    anova_table = sa.anova_lm(model, typ=2)

    return AnovaResult(
        model=model,
        anova_table=anova_table,
        effects=model.params,
        factor_names=list(factors),
    )


@dataclass
class GLMResult:
    """A fitted generalized linear response model and its goodness-of-fit.

    Attributes
    ----------
    model:
        The fitted ``statsmodels`` GLM results object.
    family:
        ``"binomial"`` (logistic) or ``"poisson"`` (log-linear).
    effects:
        Estimated coefficients on the linear-predictor (logit / log) scale.
    deviance, null_deviance:
        Residual and null deviance (the GLM analogue of RSS and total SS).
    pseudo_r2:
        McFadden's pseudo-:math:`R^2`, ``1 - llf/llnull`` -- 0 for a model no
        better than the intercept, approaching 1 for a perfect fit.
    factor_names:
        Plain factor names (no interaction / quadratic terms).
    """

    model: object
    family: str
    effects: pd.Series
    deviance: float
    null_deviance: float
    pseudo_r2: float
    factor_names: list[str] = None


def fit_glm_response(
    data: pd.DataFrame,
    response: str,
    factors: Sequence[str],
    *,
    family: str = "binomial",
    interactions: bool = True,
    quadratic: bool = False,
) -> GLMResult:
    """Fit a generalized linear response model over the process factors.

    Where :func:`fit_response_model` fits a continuous response by OLS, this fits
    a response that is *not* continuous-and-normal through the appropriate link
    (Chapter 1):

    * ``family="binomial"`` -- logistic regression of a 0/1 outcome (e.g. "did the
      run meet its purity/yield specification?"), modelling :math:`P(\\text{pass})`;
    * ``family="poisson"`` -- log-linear regression of a count (e.g. "how many
      impurity peaks contaminate the pool?"), modelling the expected count.

    The formula is built programmatically in the same Wilkinson--Rogers notation
    as :func:`fit_response_model` (main effects, optional two-way interactions and
    quadratics), so a probabilistic *design space* may carry the same interaction
    and curvature structure as a classical response surface.
    """
    fam = family.lower()
    families = {
        "binomial": sm.families.Binomial(),
        "poisson": sm.families.Poisson(),
    }
    if fam not in families:
        raise ValueError(f"family must be one of {sorted(families)}, got {family!r}")

    terms: list[str] = list(factors)
    if quadratic:
        terms += [f"I({f}**2)" for f in factors]
    if interactions:
        terms += [f"{f1}:{f2}" for f1, f2 in itertools.combinations(factors, 2)]

    # Patsy can't parse Python reserved words (e.g. "yield") as column names.
    _RESPONSE_ALIAS = "__response__"
    data_fit = data.copy()
    data_fit[_RESPONSE_ALIAS] = data_fit[response]
    formula = f"{_RESPONSE_ALIAS} ~ " + " + ".join(terms)

    model = smf.glm(formula, data=data_fit, family=families[fam]).fit()
    pseudo_r2 = float(1.0 - model.llf / model.llnull) if model.llnull != 0 else np.nan

    return GLMResult(
        model=model,
        family=fam,
        effects=model.params,
        deviance=float(model.deviance),
        null_deviance=float(model.null_deviance),
        pseudo_r2=pseudo_r2,
        factor_names=list(factors),
    )


def predict_glm_grid(
    result: GLMResult,
    x_factor: str,
    y_factor: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    *,
    n: int = 60,
    fixed: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a fitted GLM on a 2-D grid of two factors.

    Returns ``(X, Y, M)`` mesh arrays where ``M`` is the predicted *mean* on the
    natural scale -- a probability for a binomial model, an expected count for a
    Poisson one. Any factors other than ``x_factor`` / ``y_factor`` are held at
    the values in ``fixed`` (default: their training-data means), so the surface
    is a 2-D slice through the design space.  Thresholding ``M`` (e.g.
    :math:`M \\ge 0.95`) yields the *probabilistic design space*.
    """
    model = result.model
    orig = model.model.data.orig_exog
    factor_cols = list(result.factor_names)

    centres = {c: float(orig[c].mean()) for c in factor_cols}
    if fixed:
        centres.update(fixed)

    xs = np.linspace(*x_range, n)
    ys = np.linspace(*y_range, n)
    X, Y = np.meshgrid(xs, ys)

    grid = pd.DataFrame({c: np.full(X.size, centres[c]) for c in factor_cols})
    grid[x_factor] = X.ravel()
    grid[y_factor] = Y.ravel()
    M = np.asarray(model.predict(grid)).reshape(X.shape)
    return X, Y, M


def proven_acceptable_ranges(
    result: AnovaResult,
    *,
    response_spec: tuple[float, float],
) -> pd.DataFrame:
    """Derive per-factor proven acceptable ranges (PAR) from the fitted model.

    For each factor *x_i*, all other factors are held at their training-data
    mean (the "centre point"), and the factor range where the model prediction
    stays within *response_spec = (lo, hi)* is reported.

    This is a univariate PAR analysis.  For a multivariate design space see
    the full contour / RSM plots.

    Parameters
    ----------
    result:
        Fitted model from :func:`fit_response_model`.
    response_spec:
        Acceptable response bounds ``(lo, hi)``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``factor``, ``par_low``, ``par_high``, ``factor_range_low``,
        ``factor_range_high``.
    """
    model = result.model
    lo_spec, hi_spec = response_spec

    # Use stored factor names; fall back to model frame columns if not available
    if result.factor_names:
        factor_cols = list(result.factor_names)
    else:
        orig_data = model.model.data.orig_exog
        factor_cols = [c for c in orig_data.columns if c != "__response__"]

    # Build a centre-point DataFrame using the training data means
    orig_data = model.model.data.orig_exog
    centre_df = pd.DataFrame(
        {col: [float(orig_data[col].mean())] for col in factor_cols}
    )

    records = []
    for var in factor_cols:
        # Range of this factor in training data
        x_lo = float(orig_data[var].min())
        x_hi = float(orig_data[var].max())

        # Scan over the factor range with all others at their centres
        x_scan = np.linspace(x_lo, x_hi, 200)
        preds = []
        for xv in x_scan:
            row = centre_df.copy()
            row[var] = xv
            preds.append(float(model.predict(row).iloc[0]))

        preds = np.array(preds)
        in_spec = (preds >= lo_spec) & (preds <= hi_spec)

        if in_spec.any():
            par_lo = float(x_scan[in_spec].min())
            par_hi = float(x_scan[in_spec].max())
        else:
            par_lo = par_hi = float("nan")

        records.append(
            {
                "factor": var,
                "par_low": par_lo,
                "par_high": par_hi,
                "factor_range_low": x_lo,
                "factor_range_high": x_hi,
            }
        )

    return pd.DataFrame(records)
