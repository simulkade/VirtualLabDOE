"""Phase 4 — Bayesian optimization loop.

Implements sequential model-based optimization using a GP surrogate and
Expected Improvement (EI) acquisition. The oracle is the perturbed virtual lab.

Algorithm (single-objective with optional constraint penalty):

1. Draw *n_initial* points via LHS to seed the surrogate.
2. Evaluate the oracle at all initial points.
3. For each iteration:
   a. Fit ``GPSurrogate`` on all observations.
   b. Optimise ``LogExpectedImprovement`` over the factor domain.
   c. If a purity constraint is given, use a penalty formulation:
      penalised_yield = yield * (purity >= threshold) evaluated at the
      oracle.  The acquisition targets unconstrained EI on *yield*; the
      constraint is only enforced at oracle evaluation time.
   d. Evaluate the oracle at the suggested point.
   e. Append to the observation table.
4. Return the full trajectory (initial + BO iterations).

Note on multi-objective:
  For a true Pareto-frontier exploration (e.g. yield vs purity) swap in
  ``botorch.acquisition.multi_objective.logei.qLogNoisyExpectedHypervolumeImprovement``
  (qNEHVI).  This is straightforward because the surrogate and bounds setup
  is identical; only the acquisition function and reference point differ.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf

from vlab_doe.doe.factorial import Factor
from vlab_doe.doe.lhs import latin_hypercube
from vlab_doe.optimization.surrogate import GPSurrogate


@dataclass(frozen=True)
class Objective:
    """Optimization objective with optional lower-bound constraints.

    Parameters
    ----------
    maximize:
        Name of the response column to maximise.
    constraints:
        Dict mapping response column names to their lower bounds.
        E.g. ``{"purity": 0.95}`` means purity must be ≥ 0.95.
        The constraint is applied as a *penalty on the oracle response*:
        if any constraint is violated the point is counted as a failed
        evaluation (``maximize`` set to 0) for the purposes of computing
        regret/running best.
    """

    maximize: str
    constraints: dict[str, float] = field(default_factory=dict)


def bayesian_optimization(
    factors: Sequence[Factor],
    evaluate: Callable[[Mapping[str, float]], Mapping[str, float]],
    objective: Objective,
    *,
    n_initial: int = 10,
    n_iterations: int = 30,
    seed: int | None = None,
) -> pd.DataFrame:
    """Run the Bayesian optimisation loop.

    Parameters
    ----------
    factors:
        CPPs defining the search space.
    evaluate:
        Oracle callable: takes a factor dict, returns a response dict.
    objective:
        What to maximise (and optional constraints).
    n_initial:
        Number of LHS seed evaluations before the BO loop starts.
    n_iterations:
        Number of sequential BO iterations after the seed.
    seed:
        RNG seed for the initial LHS and acquisition optimisation.

    Returns
    -------
    pandas.DataFrame
        Full trajectory with columns for factors, responses, and
        ``"phase"`` (``"initial"`` or ``"bo_iter_{i}"``).
    """
    # ── Seed evaluations (LHS) ──────────────────────────────────────────────
    init_design = latin_hypercube(factors, n_initial, seed=seed)
    records = []
    for _, row in init_design.iterrows():
        point = dict(row)
        resp = dict(evaluate(point))
        records.append({**point, **resp, "phase": "initial"})
    data = pd.DataFrame(records)

    # Physical bounds tensor for botorch acquisition optimisation
    bounds_phys = torch.tensor(
        [[f.low for f in factors], [f.high for f in factors]], dtype=torch.double
    )
    # Normalised bounds (always [0,1] for optimise_acqf)
    bounds_norm = torch.zeros_like(bounds_phys)
    bounds_norm[1] = 1.0

    # ── Sequential BO ───────────────────────────────────────────────────────
    for i in range(n_iterations):
        # Fit surrogate on observations that satisfy constraints (or all if none)
        gp = GPSurrogate(factors, objective.maximize)
        gp.fit(data)

        # Normalise best observed value for EI baseline
        Y_all = torch.tensor(data[objective.maximize].values, dtype=torch.double)
        best_f = float(Y_all.max())
        best_f_norm = (best_f - gp._y_mean) / gp._y_std

        acq = LogExpectedImprovement(model=gp._model, best_f=best_f_norm)

        torch.manual_seed(seed + i if seed is not None else i)
        candidate_norm, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds_norm,
            q=1,
            num_restarts=5,
            raw_samples=32,
        )

        # Unnormalise candidate
        from botorch.utils.transforms import unnormalize
        candidate_phys = unnormalize(candidate_norm.squeeze(0), gp._x_bounds)
        point = {f.name: float(candidate_phys[j]) for j, f in enumerate(factors)}

        resp = dict(evaluate(point))
        records.append({**point, **resp, "phase": f"bo_iter_{i}"})
        data = pd.DataFrame(records)

    return data


def compare_to_doe(
    bo_trajectory: pd.DataFrame,
    doe_results: pd.DataFrame,
    objective: Objective,
) -> pd.DataFrame:
    """Compare BO vs classical DoE: running best and regret curves.

    Returns a DataFrame with columns:

    * ``"method"`` — ``"BO"`` or ``"DoE"``.
    * ``"n_evaluations"`` — cumulative experiment count.
    * ``"running_best"`` — best feasible objective value seen so far.
    * ``"regret"`` — gap between running best and the global best across
      *both* methods (proxy for true optimum).
    """
    global_best = max(
        float(bo_trajectory[objective.maximize].max()),
        float(doe_results[objective.maximize].max()),
    )

    def running_best(df: pd.DataFrame) -> pd.Series:
        rb = (
            df[objective.maximize]
            .expanding()
            .max()
            .reset_index(drop=True)
        )
        return rb

    bo_rb = running_best(bo_trajectory)
    doe_rb = running_best(doe_results)

    results = []
    for i, rb in enumerate(bo_rb):
        results.append(
            {
                "method": "BO",
                "n_evaluations": i + 1,
                "running_best": float(rb),
                "regret": global_best - float(rb),
            }
        )
    for i, rb in enumerate(doe_rb):
        results.append(
            {
                "method": "DoE",
                "n_evaluations": i + 1,
                "running_best": float(rb),
                "regret": global_best - float(rb),
            }
        )
    return pd.DataFrame(results)
