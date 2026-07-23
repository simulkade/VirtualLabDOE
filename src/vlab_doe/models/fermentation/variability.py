"""Batch-to-batch biological variability (aleatoric).

This is the layer the user cares most about: real fermentations of the "same" recipe drift
because the inoculum viability, the cells' physiological state (lag), the strains' intrinsic
rates, and — crucially — the milk lot's buffering and lactose all change from batch to batch.
That is *variability* (irreducible spread across batches), distinct from the *measurement
uncertainty* added in :mod:`.observe` and the within-batch *process noise* of the SDE engine.

:func:`sample_batch` draws one perturbed :class:`~.engine.FermentationSetup` from a population
described by :class:`BatchVariability`.  Multiplicative factors are drawn lognormally
(``value · exp(N(0, sd))``), exactly as :func:`vlab_doe.perturbation.jitter_parameters`
does, so for small ``sd`` the spread is ≈ a fractional CV; the temperature offset is additive.
:func:`run_batches` simulates an ensemble of such draws.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .engine import FermentationResult, FermentationSetup, run_fermentation


@dataclass(frozen=True)
class BatchVariability:
    """Population spread of the batch parameters.

    Each field is the standard deviation of a lognormal multiplier (≈ fractional CV), except
    ``temperature_sd`` which is the SD of an additive temperature offset in °C.

    Parameters
    ----------
    inoculum_cv:
        Spread in total inoculum (pitching / viability variability).
    lag_cv:
        Spread in each strain's Baranyi lag state ``Q0`` (the most stochastic trait of real
        starters).
    mu_max_cv:
        Spread in each strain's maximum growth rate.
    buffering_cv:
        Spread in the milk titration midpoint ``L50`` (lot buffering capacity).
    lactose_cv:
        Spread in milk lactose ``S0``.
    temperature_sd:
        SD of the additive incubator-temperature offset (°C).
    """

    inoculum_cv: float = 0.20
    lag_cv: float = 0.40
    mu_max_cv: float = 0.08
    buffering_cv: float = 0.10
    lactose_cv: float = 0.05
    temperature_sd: float = 0.4


def _lognormal(value: float, sd: float, rng: np.random.Generator) -> float:
    if sd <= 0.0:
        return float(value)
    return float(value) * float(np.exp(rng.normal(0.0, sd)))


def sample_batch(
    setup: FermentationSetup,
    variability: BatchVariability,
    rng: np.random.Generator,
) -> FermentationSetup:
    """Draw one batch-perturbed copy of ``setup`` from the variability population."""
    v = variability

    # Per-strain: jitter mu_max and lag_state independently.
    new_strains = [
        replace(
            s,
            mu_max=_lognormal(s.mu_max, v.mu_max_cv, rng),
            lag_state=_lognormal(s.lag_state, v.lag_cv, rng),
        )
        for s in setup.consortium.strains
    ]
    new_consortium = replace(setup.consortium, strains=new_strains)

    # Milk lot: jitter buffering midpoint and lactose.
    new_milk = replace(
        setup.milk,
        l50=_lognormal(setup.milk.l50, v.buffering_cv, rng),
        lactose=_lognormal(setup.milk.lactose, v.lactose_cv, rng),
    )

    return replace(
        setup,
        consortium=new_consortium,
        milk=new_milk,
        total_inoculum=_lognormal(setup.total_inoculum, v.inoculum_cv, rng),
        temperature=setup.temperature + (rng.normal(0.0, v.temperature_sd) if v.temperature_sd > 0 else 0.0),
    )


def run_batches(
    setup: FermentationSetup,
    variability: BatchVariability,
    n_batches: int,
    t_eval: np.ndarray,
    rng: np.random.Generator,
) -> list[FermentationResult]:
    """Simulate an ensemble of ``n_batches`` batch-variable replicates.

    Each replicate draws its own parameters via :func:`sample_batch` and is then simulated
    with :func:`~.engine.run_fermentation` (which also adds within-batch process noise if the
    setup's ``process_noise_sd`` is non-zero).  All randomness flows from the single ``rng`` for
    reproducibility.
    """
    results = []
    for _ in range(int(n_batches)):
        batch_setup = sample_batch(setup, variability, rng)
        results.append(run_fermentation(batch_setup, t_eval, rng=rng))
    return results
