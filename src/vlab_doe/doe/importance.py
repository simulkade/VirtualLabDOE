"""Tree-model screening analysis: which features drive the response.

A covering-array screen produces a wide, sparse, binary design (which strains were in each
run) against a noisy response.  Linear ANOVA struggles here — the interesting structure is
nonlinear and interaction-heavy — so we analyse it with **tree ensembles**:

* :func:`random_forest_importance` — ``sklearn`` :class:`~sklearn.ensemble.RandomForestRegressor`,
* :func:`gradient_boosting_importance` — ``xgboost`` :class:`~xgboost.XGBRegressor`.

Both report the same things through :class:`ImportanceResult` so they can be compared head to
head: a cross-validated R² (how much of the response the model actually explains out-of-sample)
and **permutation importance** (how much CV-honest accuracy is lost when each feature is
shuffled).  Permutation importance is used for both models because impurity / gain importances
are not comparable across the two libraries and are biased toward high-cardinality features.

For the classical, interpretable counterpart -- the regularized linear models of Chapter 1 --
two more analyses are provided:

* :func:`regularized_importance` — an **elastic-net** linear fit for a continuous response,
  returning the same :class:`ImportanceResult` (so its CV R² and permutation importance sit
  beside the trees') plus *signed* coefficients that name and direct each strain's effect;
* :func:`logistic_screening` — a **regularized logistic regression** (a generalized linear
  model) for a *binary* outcome such as "did the blend reach the set point?", graded by
  cross-validated ROC-AUC and accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNetCV, LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ImportanceResult:
    """Fitted screening model plus its cross-validated quality and feature ranking.

    Attributes
    ----------
    model:
        The estimator fitted on the full data.
    cv_score:
        Mean cross-validated R².
    cv_scores:
        Per-fold R² scores.
    importances:
        Permutation importance per feature (mean), sorted descending.
    importances_std:
        Standard deviation of the permutation importance per feature (same index as
        ``importances``).
    predictions:
        Out-of-fold cross-validated predictions, aligned with the input rows.
    feature_names:
        Feature names in input order.
    """

    model: object
    cv_score: float
    cv_scores: np.ndarray
    importances: pd.Series
    importances_std: pd.Series
    predictions: np.ndarray
    feature_names: list[str]


def _analyse(
    estimator,
    X,
    y,
    feature_names: Sequence[str] | None,
    cv: int,
    seed: int | None,
    n_repeats: int,
) -> ImportanceResult:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if feature_names is None:
        feature_names = [f"f{i:02d}" for i in range(X.shape[1])]
    feature_names = list(feature_names)

    splitter = KFold(n_splits=cv, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(estimator, X, y, cv=splitter, scoring="r2")
    predictions = cross_val_predict(estimator, X, y, cv=splitter)

    estimator.fit(X, y)
    perm = permutation_importance(
        estimator, X, y, n_repeats=n_repeats, random_state=seed, scoring="r2"
    )
    order = np.argsort(perm.importances_mean)[::-1]
    importances = pd.Series(
        perm.importances_mean[order], index=[feature_names[i] for i in order]
    )
    importances_std = pd.Series(
        perm.importances_std[order], index=[feature_names[i] for i in order]
    )

    return ImportanceResult(
        model=estimator,
        cv_score=float(np.mean(cv_scores)),
        cv_scores=cv_scores,
        importances=importances,
        importances_std=importances_std,
        predictions=predictions,
        feature_names=feature_names,
    )


def random_forest_importance(
    X,
    y,
    *,
    feature_names: Sequence[str] | None = None,
    n_estimators: int = 400,
    max_depth: int | None = None,
    cv: int = 5,
    n_repeats: int = 10,
    seed: int | None = None,
) -> ImportanceResult:
    """Random-forest screening analysis (CV R² + permutation importance)."""
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1,
    )
    return _analyse(rf, X, y, feature_names, cv, seed, n_repeats)


def gradient_boosting_importance(
    X,
    y,
    *,
    feature_names: Sequence[str] | None = None,
    n_estimators: int = 400,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    cv: int = 5,
    n_repeats: int = 10,
    seed: int | None = None,
) -> ImportanceResult:
    """XGBoost gradient-boosting screening analysis (CV R² + permutation importance)."""
    from xgboost import XGBRegressor

    xgb = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )
    return _analyse(xgb, X, y, feature_names, cv, seed, n_repeats)


def regularized_importance(
    X,
    y,
    *,
    feature_names: Sequence[str] | None = None,
    l1_ratio: Sequence[float] | float = (0.5, 0.7, 0.9, 0.95, 1.0),
    cv: int = 5,
    n_repeats: int = 10,
    seed: int | None = None,
) -> ImportanceResult:
    """Regularized *linear* screening analysis (elastic net) for a continuous response.

    The classical counterpart to the tree ensembles: a penalized linear model that
    survives the "more candidate effects than runs" regime by shrinkage and
    selection (Chapter 1, regularization).  The elastic net blends the lasso's
    sparsity (it zeroes inactive strains, giving an interpretable short list) with
    the ridge's stability under the correlated columns a covering array produces;
    the penalty strength and the L1/L2 mix are chosen by internal
    cross-validation (:class:`~sklearn.linear_model.ElasticNetCV`).

    Returns the same :class:`ImportanceResult` as the tree analyses -- so the CV R²
    and permutation importance are directly comparable -- with the fitted model's
    *signed* coefficients available as ``result.model[-1].coef_``.
    """
    net = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "net",
                ElasticNetCV(
                    l1_ratio=list(l1_ratio) if np.ndim(l1_ratio) else l1_ratio,
                    cv=cv,
                    random_state=seed,
                    max_iter=20_000,
                ),
            ),
        ]
    )
    return _analyse(net, X, y, feature_names, cv, seed, n_repeats)


@dataclass
class LogisticScreenResult:
    """A regularized logistic-regression (GLM) screen of a binary outcome.

    Attributes
    ----------
    model:
        The fitted scaler+logistic pipeline.
    cv_auc:
        Mean cross-validated area under the ROC curve.
    cv_accuracy:
        Mean cross-validated classification accuracy.
    coefficients:
        Signed logit coefficients per feature, sorted by descending magnitude;
        a positive coefficient raises the modelled probability of the event.
    selected:
        Names of the features the L1 penalty kept (non-zero coefficient).
    feature_names:
        Feature names in input order.
    """

    model: object
    cv_auc: float
    cv_accuracy: float
    coefficients: pd.Series
    selected: list[str]
    feature_names: list[str]


def logistic_screening(
    X,
    y,
    *,
    feature_names: Sequence[str] | None = None,
    cv: int = 5,
    seed: int | None = None,
) -> LogisticScreenResult:
    """Regularized logistic regression of a **binary** screening outcome (a GLM).

    Where :func:`regularized_importance` models a continuous response, this models
    a yes/no one -- the canonical example being "did the blend reach the set
    point?".  It is a *generalized linear model*: the same linear predictor
    :math:`X\\beta`, now mapped through the logit link to a probability and fitted
    by penalized maximum likelihood.  An L1 penalty (chosen by internal CV) keeps
    the model sparse, so the surviving coefficients name the strains that make a
    blend likely to set; the fit is graded out-of-sample by ROC-AUC and accuracy.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    if feature_names is None:
        feature_names = [f"f{i:02d}" for i in range(X.shape[1])]
    feature_names = list(feature_names)

    # An L1-penalised logistic regression whose penalty strength C is tuned by an
    # inner grid search; the StandardScaler keeps the single penalty comparable
    # across the presence columns.
    tuned = GridSearchCV(
        LogisticRegression(solver="saga", l1_ratio=1.0, max_iter=10_000),
        {"C": np.logspace(-2.0, 1.5, 10)},
        cv=cv,
        scoring="roc_auc",
    )
    clf = Pipeline([("scale", StandardScaler()), ("logit", tuned)])

    # Honest out-of-sample grading by an outer (nested) cross-validation.
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    scores = cross_validate(
        clf, X, y, cv=splitter, scoring=("roc_auc", "accuracy")
    )
    cv_auc = scores["test_roc_auc"]
    cv_acc = scores["test_accuracy"]

    clf.fit(X, y)
    coef = clf[-1].best_estimator_.coef_.ravel()
    order = np.argsort(np.abs(coef))[::-1]
    coefficients = pd.Series(coef[order], index=[feature_names[i] for i in order])
    selected = [feature_names[i] for i in range(len(coef)) if abs(coef[i]) > 1e-8]

    return LogisticScreenResult(
        model=clf,
        cv_auc=float(np.mean(cv_auc)),
        cv_accuracy=float(np.mean(cv_acc)),
        coefficients=coefficients,
        selected=selected,
        feature_names=feature_names,
    )
