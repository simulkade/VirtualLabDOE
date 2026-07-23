"""Turn the clean pH truth into a measured pH series.

The fermentation model's only observable is pH, sampled by a probe at discrete times.  This
module is a thin adapter onto the project-wide :mod:`vlab_doe.perturbation` layer, so the
pH measurement gets the same additive-noise / calibration-bias treatment as every other
virtual measurement in the lab.  Batch-to-batch *biological* variability is handled separately
in :mod:`.variability`; here we add only the *measurement* (epistemic) noise.
"""

from __future__ import annotations

import numpy as np

from ...perturbation import NoiseModel, add_measurement_noise
from .engine import FermentationResult

#: A typical benchtop pH probe: ~0.02 pH additive noise, small calibration offset modelled
#: per batch by drawing the bias outside this module.
DEFAULT_PH_NOISE = NoiseModel(additive_sd=0.02)


def observe_ph(
    result: FermentationResult,
    sample_times: np.ndarray,
    noise: NoiseModel = DEFAULT_PH_NOISE,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Sample the pH curve at ``sample_times`` and add measurement noise.

    The clean pH is linearly interpolated onto the (typically coarser) probe sampling grid,
    then :func:`vlab_doe.perturbation.add_measurement_noise` applies additive noise,
    drift and calibration bias.

    Parameters
    ----------
    result:
        A clean batch result from :func:`~.engine.run_fermentation`.
    sample_times:
        Probe sampling times (h); need not coincide with the simulation grid.
    noise:
        Measurement-noise model; defaults to :data:`DEFAULT_PH_NOISE`.
    rng:
        Seeded generator.  Required if the noise model has any non-zero stochastic term.

    Returns
    -------
    dict with ``"t"`` (sample times), ``"ph"`` (noisy measured pH) and ``"ph_true"``
    (the noise-free pH at those times).
    """
    sample_times = np.asarray(sample_times, dtype=float)
    ph_true = np.interp(sample_times, result.t, result.ph)

    needs_rng = noise.additive_sd != 0.0 or noise.proportional_cv != 0.0
    if needs_rng and rng is None:
        raise ValueError("rng is required for a stochastic noise model")
    if rng is None:
        rng = np.random.default_rng(0)  # only reached when noise is fully deterministic

    ph_obs = add_measurement_noise(sample_times, ph_true, noise, rng)
    return {"t": sample_times, "ph": ph_obs, "ph_true": ph_true}
