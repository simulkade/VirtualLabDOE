"""Product-quality metrics derived from a fermentation curve.

The pH series is an indirect indicator of several outcomes at once, so we summarise a batch by
a small **fingerprint** of features a dairy technologist would actually care about:

* ``t_gel`` — time to gelation (pH 5.2), when the casein network starts to set.
* ``t_set`` — time to the target set pH (4.6), the usual "fermentation done" milestone and the
  classic strain-screening readout.
* ``final_ph`` — pH at the end of incubation.
* ``post_acidification`` — how far the pH keeps dropping after the set point (over-acidification
  during cold storage is a common defect).
* ``max_rate`` / ``t_max_rate`` — steepest acidification rate and when it happens.
* ``aroma`` — final aroma-proxy level.
* ``frac_*`` — community composition (biomass fraction) at the end.

:func:`fingerprint_distance` turns a fingerprint into a single number: the weighted, scaled
distance to a *reference* fingerprint.  That is the objective for the replacement use-case —
find the strain blend whose product profile best matches the strain being phased out.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .engine import FermentationResult

#: Default pH milestones (gelation, set point).
PH_GEL = 5.2
PH_SET = 4.6


def time_to_ph(t: np.ndarray, ph: np.ndarray, target: float) -> float:
    """First time the pH curve falls to ``target`` (linear interpolation).

    Returns ``nan`` if the curve never reaches the target within the window.
    """
    t = np.asarray(t, dtype=float)
    ph = np.asarray(ph, dtype=float)
    below = ph <= target
    if not below.any():
        return float("nan")
    k = int(np.argmax(below))          # first index at/under target
    if k == 0:
        return float(t[0])
    # Linear interpolation between k-1 (above) and k (at/under).
    p0, p1 = ph[k - 1], ph[k]
    t0, t1 = t[k - 1], t[k]
    if p0 == p1:
        return float(t1)
    frac = (p0 - target) / (p0 - p1)
    return float(t0 + frac * (t1 - t0))


def fingerprint(
    result: FermentationResult,
    *,
    ph_gel: float = PH_GEL,
    ph_set: float = PH_SET,
) -> dict[str, float]:
    """Summarise a batch into its product fingerprint (see module docstring)."""
    t, ph = result.t, result.ph
    t_set = time_to_ph(t, ph, ph_set)
    final_ph = float(ph[-1])

    # Acidification rate (negative slope); report the steepest drop as a positive number.
    drate = -np.gradient(ph, t)
    apex = int(np.argmax(drate))

    fp = {
        "t_gel": time_to_ph(t, ph, ph_gel),
        "t_set": t_set,
        "final_ph": final_ph,
        "post_acidification": float(ph_set - final_ph) if np.isfinite(t_set) else 0.0,
        "max_rate": float(drate[apex]),
        "t_max_rate": float(t[apex]),
        "aroma": float(result.aroma[-1]),
    }

    final_biomass = np.clip(result.biomass[:, -1], 0.0, None)
    total = float(final_biomass.sum())
    for name, x in zip(result.strain_names, final_biomass):
        key = "frac_" + name.split(".")[-1].strip().lower().replace(" ", "_")
        fp[key] = float(x / total) if total > 0 else 0.0
    return fp


#: Characteristic scales used to non-dimensionalise fingerprint features before comparison.
_DEFAULT_SCALES = {
    "t_gel": 1.0,            # h
    "t_set": 1.0,            # h
    "final_ph": 0.1,         # pH units
    "post_acidification": 0.1,
    "max_rate": 0.2,         # pH/h
    "t_max_rate": 1.0,
    "aroma": 0.05,
}


def fingerprint_distance(
    fp: Mapping[str, float],
    reference: Mapping[str, float],
    *,
    weights: Mapping[str, float] | None = None,
    scales: Mapping[str, float] | None = None,
) -> float:
    """Weighted, scaled Euclidean distance between two fingerprints.

    Only the keys present in *both* fingerprints (and in ``scales``) are compared, so passing a
    subset of weights/scales focuses the objective on the attributes that matter for the
    product being matched.  ``nan`` features (e.g. a set point never reached) are penalised by
    treating the difference as one full characteristic scale.

    Parameters
    ----------
    fp:
        Candidate fingerprint (from :func:`fingerprint`).
    reference:
        Target fingerprint to match (e.g. the strain being replaced).
    weights:
        Optional per-feature weights; default 1.0 for every compared feature.
    scales:
        Optional per-feature characteristic scales; default :data:`_DEFAULT_SCALES`.
    """
    scales = dict(_DEFAULT_SCALES if scales is None else scales)
    keys = [k for k in scales if k in fp and k in reference]
    total = 0.0
    for k in keys:
        w = 1.0 if weights is None else float(weights.get(k, 0.0))
        if w == 0.0:
            continue
        a, b = fp[k], reference[k]
        if not (np.isfinite(a) and np.isfinite(b)):
            diff = 1.0                      # one full scale-unit penalty for a missing milestone
        else:
            diff = (a - b) / scales[k]
        total += w * diff * diff
    return float(np.sqrt(total))
