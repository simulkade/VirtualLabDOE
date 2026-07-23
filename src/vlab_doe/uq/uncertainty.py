"""Phase 5 — Uncertainty quantification.

Separates and propagates the two kinds of uncertainty using the
**law of total variance** (Eve's law)::

    Var[Y] = E[Var[Y|θ]] + Var[E[Y|θ]]
           = aleatoric    + epistemic

* **Aleatoric** — irreducible process/measurement noise.  For a Gaussian
  measurement model with known σ this is simply σ² at every point.

* **Epistemic** — reducible parameter/model uncertainty.  Estimated as the
  variance of the posterior-predictive *mean* over parameter draws from the
  MCMC posterior.

The posterior predictive propagation uses Monte-Carlo:

1. Draw *n_draws* parameter vectors from the posterior.
2. Evaluate the forward model at each draw.
3. Compute the mean and variance of the resulting ensemble.

The ``posterior`` argument accepts any ``arviz.InferenceData`` object with a
``posterior`` group, as returned by :func:`vlab_doe.uq.inverse.estimate_bayesian`.
"""

from __future__ import annotations

from typing import Callable, Mapping

import arviz as az
import numpy as np


def decompose_uncertainty(
    posterior: az.InferenceData,
    forward: Callable[[Mapping[str, float]], np.ndarray],
    *,
    noise_sd: float,
    n_draws: int = 500,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Decompose predictive variance into aleatoric and epistemic components.

    Parameters
    ----------
    posterior:
        Posterior samples (``arviz.InferenceData`` from
        :func:`vlab_doe.uq.inverse.estimate_bayesian`).
    forward:
        Mechanistic forward model: ``params: dict[str, float]`` → ``np.ndarray``.
    noise_sd:
        Aleatoric measurement noise standard deviation (from the perturbation
        model — the *known* noise level used in the likelihood).
    n_draws:
        Number of posterior samples to draw for the MC propagation.
    seed:
        RNG seed.

    Returns
    -------
    dict with numpy arrays, all shape ``(n_obs,)``:

    * ``"mean"`` — posterior predictive mean E[Y].
    * ``"epistemic_var"`` — Var[E[Y|θ]] — variance of conditional means.
    * ``"aleatoric_var"`` — E[Var[Y|θ]] = σ² — irreducible noise.
    * ``"total_var"`` — sum of the above.
    * ``"epistemic_sd"`` — sqrt(epistemic_var).
    * ``"aleatoric_sd"`` — sqrt(aleatoric_var) = noise_sd (constant).
    """
    # Extract parameter names and posterior samples
    param_names = list(posterior.posterior.data_vars)
    n_chains = posterior.posterior.dims["chain"]
    n_post_steps = posterior.posterior.dims["draw"]
    n_total = n_chains * n_post_steps

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n_total, size=min(n_draws, n_total))

    predictions = []
    for idx in indices:
        chain_idx = int(idx // n_post_steps)
        draw_idx = int(idx % n_post_steps)
        params = {
            name: float(posterior.posterior[name].values[chain_idx, draw_idx])
            for name in param_names
        }
        try:
            pred = np.asarray(forward(params), dtype=float)
            predictions.append(pred)
        except Exception:
            continue  # skip failed evaluations

    if not predictions:
        raise RuntimeError("All forward evaluations failed during MC propagation.")

    predictions = np.array(predictions)  # (n_valid_draws, n_obs)

    mean = predictions.mean(axis=0)
    epistemic_var = predictions.var(axis=0)
    aleatoric_var = np.full_like(mean, noise_sd**2)
    total_var = epistemic_var + aleatoric_var

    return {
        "mean": mean,
        "epistemic_var": epistemic_var,
        "aleatoric_var": aleatoric_var,
        "total_var": total_var,
        "epistemic_sd": np.sqrt(epistemic_var),
        "aleatoric_sd": np.full_like(mean, noise_sd),
    }
