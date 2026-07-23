"""Phase 1.4 — Milk fermentation (yogurt) model.

A stochastic, mechanism-flavoured virtual lab for batch acidification of milk by lactic-acid
bacteria.  Lactose is fermented to lactic acid, which titrates the milk down a sigmoidal pH
curve; gelation and the target "set" pH are read off that curve.  The only thing the lab
measures is the **pH time series** — an indirect indicator that lumps together biomass,
substrate, acid and aroma — which makes it a realistic testbed for experimental design,
separating measurement uncertainty from batch-to-batch variability, and optimising a strain
blend to match a target product.

Three layers of randomness are modelled explicitly:

* **batch variability** (:mod:`.variability`) — biological lot-to-lot spread (inoculum, lag,
  rates, milk buffering, incubator temperature);
* **process noise** (:mod:`.engine`) — an optional within-batch SDE on biomass;
* **measurement noise** (:mod:`.observe`) — the pH probe.

Quick start::

    import numpy as np
    from vlab_doe.models import fermentation as ferm

    setup = ferm.FermentationSetup(
        consortium=ferm.yogurt_blend(fraction_st=0.6, fraction_lb=0.4),
        temperature=43.0,
    )
    t = np.linspace(0, 12, 240)                 # hours
    result = ferm.run_fermentation(setup, t)
    fp = ferm.fingerprint(result)
    print("set point reached at", fp["t_set"], "h; final pH", fp["final_ph"])
"""

from __future__ import annotations

# ── Strains & consortia ──
from .strains import (
    Consortium,
    Strain,
    StrainLibrary,
    bifidobacterium,
    cardinal_temperature_factor,
    lactobacillus_acidophilus,
    lactobacillus_bulgaricus,
    random_strain_library,
    single_strain,
    streptococcus_thermophilus,
    yogurt_blend,
)

# ── Milk substrate ──
from .milk import Milk, ph_from_acid

# ── Kinetics ──
from .kinetics import FermentationKinetics, make_kinetics

# ── Engine ──
from .engine import FermentationResult, FermentationSetup, run_fermentation

# ── Batch variability ──
from .variability import BatchVariability, run_batches, sample_batch

# ── Measurement ──
from .observe import DEFAULT_PH_NOISE, observe_ph

# ── Metrics ──
from .metrics import PH_GEL, PH_SET, fingerprint, fingerprint_distance, time_to_ph

__all__ = [
    # strains
    "Strain",
    "Consortium",
    "StrainLibrary",
    "random_strain_library",
    "cardinal_temperature_factor",
    "streptococcus_thermophilus",
    "lactobacillus_bulgaricus",
    "lactobacillus_acidophilus",
    "bifidobacterium",
    "single_strain",
    "yogurt_blend",
    # milk
    "Milk",
    "ph_from_acid",
    # kinetics
    "FermentationKinetics",
    "make_kinetics",
    # engine
    "FermentationSetup",
    "FermentationResult",
    "run_fermentation",
    # variability
    "BatchVariability",
    "sample_batch",
    "run_batches",
    # observe
    "observe_ph",
    "DEFAULT_PH_NOISE",
    # metrics
    "time_to_ph",
    "fingerprint",
    "fingerprint_distance",
    "PH_GEL",
    "PH_SET",
]
