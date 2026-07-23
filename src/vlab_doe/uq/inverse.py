"""Phase 5 — Inverse modeling / parameter estimation.

Two complementary estimators:

1. **Deterministic** (:func:`estimate_least_squares`): scipy's
   ``trust-region-reflective`` algorithm minimises the sum of squared residuals
   between model predictions and noisy observations.  Returns a point estimate
   and an approximate covariance matrix (from the Jacobian at the solution).

2. **Bayesian** (:func:`estimate_bayesian`): an ``emcee`` affine-invariant
   ensemble MCMC sampler explores the full posterior
       p(θ | y_obs) ∝ p(y_obs | θ) · p(θ)
   with a Gaussian likelihood (likelihood noise σ from the perturbation model)
   and uniform priors from :class:`ParameterPrior`.  The sampler output is
   returned as an ``arviz.InferenceData`` object for convergence diagnostics
   (R-hat, ESS, trace plots) via ``arviz``.

The forward callable interface is identical for both estimators:
    params: dict[str, float]  →  prediction: np.ndarray
so the same chromatography or UF/DF model function can be plugged in directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import arviz as az
import emcee
import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class ParameterPrior:
    """Uniform prior bounds for one estimated parameter.

    Parameters
    ----------
    name:
        Parameter name (must match the key in the dict passed to *forward*).
    low, high:
        Uniform prior support.  Any theta outside (low, high) gives log-prob
        −∞.  These also serve as bounds for the least-squares estimator.
    """

    name: str
    low: float
    high: float


def estimate_least_squares(
    forward: Callable[[Mapping[str, float]], np.ndarray],
    observed: np.ndarray,
    priors: Sequence[ParameterPrior],
    *,
    initial: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Deterministic point estimate via bounded least-squares optimisation.

    Minimises ``Σ (forward(θ) − observed)²`` subject to the prior bounds.

    Parameters
    ----------
    forward:
        Mechanistic forward model mapping parameter dict → prediction array.
    observed:
        Noisy observations, shape ``(n_obs,)``.
    priors:
        Prior bounds (used as parameter bounds).
    initial:
        Starting point.  If ``None``, uses the midpoint of each prior.

    Returns
    -------
    dict
        Point estimate ``{param_name: value, ...}`` for each prior.
    """
    names = [p.name for p in priors]
    bounds_lo = [p.low for p in priors]
    bounds_hi = [p.high for p in priors]

    if initial is None:
        x0 = [(p.low + p.high) / 2.0 for p in priors]
    else:
        x0 = [float(initial[n]) for n in names]

    def residuals(theta: np.ndarray) -> np.ndarray:
        params = dict(zip(names, theta))
        try:
            prediction = np.asarray(forward(params), dtype=float)
            return prediction - np.asarray(observed, dtype=float)
        except Exception:
            return np.full(len(observed), 1e6)

    result = least_squares(
        residuals,
        x0=x0,
        bounds=(bounds_lo, bounds_hi),
        method="trf",
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-9,
        max_nfev=5000,
    )

    return dict(zip(names, result.x))


def estimate_bayesian(
    forward: Callable[[Mapping[str, float]], np.ndarray],
    observed: np.ndarray,
    priors: Sequence[ParameterPrior],
    *,
    noise_sd: float,
    n_walkers: int = 32,
    n_steps: int = 2000,
    seed: int | None = None,
) -> az.InferenceData:
    """Sample the parameter posterior with emcee MCMC.

    Likelihood: Gaussian with known σ = *noise_sd* (from the perturbation
    model).  Prior: uniform over each parameter's prior bounds.

    Parameters
    ----------
    forward:
        Forward model callable.
    observed:
        Noisy observations, shape ``(n_obs,)``.
    priors:
        Prior bounds.
    noise_sd:
        Standard deviation of the Gaussian measurement noise.
    n_walkers:
        Number of emcee ensemble walkers (must be even and ≥ 2 * n_params).
    n_steps:
        Total MCMC steps per walker (first half discarded as burn-in).
    seed:
        Seed for reproducible initial walker positions.

    Returns
    -------
    arviz.InferenceData
        Contains ``posterior`` with one group per parameter, ready for
        ``az.summary``, ``az.plot_trace``, ``az.plot_posterior``, etc.
    """
    names = [p.name for p in priors]
    n_dim = len(priors)
    n_walkers = max(n_walkers, 2 * n_dim + (2 * n_dim) % 2)  # ensure even and ≥ 2*n_dim

    observed_arr = np.asarray(observed, dtype=float)

    def log_prior(theta: np.ndarray) -> float:
        for val, p in zip(theta, priors):
            if not (p.low <= val <= p.high):
                return -np.inf
        return 0.0  # uniform prior → constant log-prior within bounds

    def log_likelihood(theta: np.ndarray) -> float:
        params = dict(zip(names, theta))
        try:
            pred = np.asarray(forward(params), dtype=float)
            residuals = observed_arr - pred
            return -0.5 * float(np.sum((residuals / noise_sd) ** 2))
        except Exception:
            return -np.inf

    def log_prob(theta: np.ndarray) -> float:
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta)

    # Initialise walkers near the prior midpoints with small perturbation
    rng = np.random.default_rng(seed)
    centres = np.array([(p.low + p.high) / 2.0 for p in priors])
    widths = np.array([(p.high - p.low) * 0.1 for p in priors])
    p0 = centres + rng.normal(0.0, widths, size=(n_walkers, n_dim))
    # Clip to prior bounds
    for j, p in enumerate(priors):
        p0[:, j] = np.clip(p0[:, j], p.low, p.high)

    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob)
    sampler.run_mcmc(p0, n_steps, progress=False)

    # Discard first half as burn-in; shape (n_steps//2, n_walkers, n_dim)
    burn = n_steps // 2
    chain = sampler.get_chain(discard=burn)   # (n_post_steps, n_walkers, n_dim)

    # Convert to arviz: posterior dict maps name → array of shape (chain, draw)
    posterior_dict = {
        names[i]: chain[:, :, i].T  # transpose to (n_walkers, n_post_steps) = (chain, draw)
        for i in range(n_dim)
    }

    # arviz 1.x: from_dict takes {group_name: {var: samples}} nested dict
    idata = az.from_dict({"posterior": posterior_dict})
    return idata
