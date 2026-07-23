"""Phase 1.3 — Perturbation module.

Converts the *Mechanistic Truth* produced by the forward models into a
*Virtual Experiment* by layering realistic laboratory variability:

* **Measurement noise** — proportional + additive Gaussian noise.
* **Systematic effects** — linear baseline drift and constant calibration bias.
* **Batch-to-batch effects** — lognormal jitter on true model parameters.

Everything is seedable via :func:`vlab_doe.config.make_rng` so noisy replicates
are reproducible across runs with the same seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class NoiseModel:
    """Configuration for measurement and systematic noise.

    Parameters
    ----------
    additive_sd:
        Standard deviation of additive Gaussian noise (same units as signal).
    proportional_cv:
        Coefficient of variation for proportional (multiplicative) noise.
        E.g. 0.02 ≈ 2% CV.
    drift_slope:
        Linear baseline drift per unit of the independent axis *x*
        (signal units per x-unit).
    bias:
        Constant calibration offset (signal units).
    """

    additive_sd: float = 0.0
    proportional_cv: float = 0.0
    drift_slope: float = 0.0
    bias: float = 0.0


def add_measurement_noise(
    x: np.ndarray,
    signal: np.ndarray,
    noise: NoiseModel,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply measurement noise and systematic effects to a clean signal.

    The model applied in order is::

        noisy = signal
              * (1 + proportional_cv * N(0,1))   # multiplicative
              + additive_sd * N(0,1)              # additive
              + drift_slope * x                   # baseline drift
              + bias                              # calibration offset

    Parameters
    ----------
    x:
        Independent axis (used for the drift term); same shape as *signal*.
    signal:
        Clean model output (the "Mechanistic Truth").
    noise:
        Noise configuration.
    rng:
        Seeded NumPy generator (from :func:`vlab_doe.config.make_rng`).
    """
    x = np.asarray(x, dtype=float)
    result = np.asarray(signal, dtype=float).copy()

    if noise.proportional_cv != 0.0:
        result = result * (1.0 + rng.normal(0.0, noise.proportional_cv, size=result.shape))

    if noise.additive_sd != 0.0:
        result = result + rng.normal(0.0, noise.additive_sd, size=result.shape)

    if noise.drift_slope != 0.0:
        result = result + noise.drift_slope * x

    if noise.bias != 0.0:
        result = result + noise.bias

    return result


def jitter_parameters(
    params: Mapping[str, float],
    relative_sd: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Return a batch-perturbed copy of ``params`` with lognormal jitter.

    Each parameter is multiplied by ``exp(N(0, relative_sd))``.  For small
    ``relative_sd`` this is approximately ``1 ± relative_sd`` (multiplicative
    noise preserving sign), which models batch-to-batch variability in the true
    underlying parameters (e.g. resin lot-to-lot differences).

    Parameters
    ----------
    params:
        True parameter values keyed by name.
    relative_sd:
        Standard deviation of the lognormal multiplier (≈ fractional CV for
        small values, e.g. 0.05 = 5% batch-to-batch variability).
    rng:
        Seeded NumPy generator.
    """
    return {
        name: float(value) * float(np.exp(rng.normal(0.0, relative_sd)))
        for name, value in params.items()
    }
