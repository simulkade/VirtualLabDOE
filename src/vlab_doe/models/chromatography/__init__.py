"""Phase 1.1 — Mechanistic chromatography models.

Two layers share one isotherm vocabulary:

* **Legacy** (:func:`simulate` + :class:`ChromatographyConfig`) — the original
  single-component, isocratic, linearised equilibrium-dispersive model.  Unchanged, so
  existing notebooks/tests and the DoE/optimization/UQ layers keep working.
* **Multi-mode engine** (:func:`run_column` + :class:`ColumnSetup`) — a transport-dispersive
  model with a linear-driving-force (LDF) mass-transfer term, supporting multiple
  components, every chromatography mode (CEX/AEX, HIC salting-out, RP-HPLC, and nonlinear
  high-resolution ion exchange), and **gradient elution** via an :class:`ElutionProgram`
  whose inlet modulator can change linearly in time.

Quick start::

    from vlab_doe.models import chromatography as chrom

    iso = chrom.cation_exchange(beta=[5e-3, 8e-3], nu=[4.0, 5.0], q_max=120.0)
    inj = chrom.Injection.from_load_density(load_density=15.0, feed=[0.5, 0.5],
                                            porosity=0.4)
    program = chrom.ElutionProgram.linear_gradient(
        inj, m_start=50.0, m_end=500.0, gradient_cv=20.0)
    setup = chrom.ColumnSetup(
        geometry=chrom.ColumnGeometry(0.1, 0.01, 0.4),
        velocity=1e-3, dispersion=1e-7, isotherm=iso, program=program)
    result = chrom.run_column(setup)
"""

from __future__ import annotations

# ── Geometry ──
from .geometry import ColumnGeometry

# ── Isotherms (legacy helpers + new mode laws) ──
from .isotherms import (
    Isotherm,
    LinearSolventStrengthLaw,
    ModulatorLaw,
    SaltingOutLaw,
    SMALaw,
    SMAParameters,
    langmuir_isotherm,
    sma_henry_constant,
    sma_isotherm,
)

# ── Elution program ──
from .program import CompiledProgram, ElutionProgram, Injection, Segment

# ── Engine ──
from .engine import ChromatogramResult, ColumnSetup, run_column

# ── General Rate Model (PyFVTool finite-volume solver) ──
from .grm import (
    GRMResult,
    GRMSetup,
    ParticleProperties,
    film_coefficient,
    run_grm,
)

# ── Mode presets ──
from .modes import (
    anion_exchange,
    cation_exchange,
    high_resolution_iex,
    hic,
    reversed_phase,
)

# ── Metrics ──
from .metrics import peak_moments, plate_count, pool_metrics, resolution

# ── Legacy single-component isocratic model ──
from .legacy import ChromatographyConfig, simulate

__all__ = [
    # geometry
    "ColumnGeometry",
    # isotherms / laws
    "Isotherm",
    "ModulatorLaw",
    "SMALaw",
    "SaltingOutLaw",
    "LinearSolventStrengthLaw",
    "SMAParameters",
    "langmuir_isotherm",
    "sma_henry_constant",
    "sma_isotherm",
    # program
    "Segment",
    "Injection",
    "ElutionProgram",
    "CompiledProgram",
    # engine
    "ColumnSetup",
    "ChromatogramResult",
    "run_column",
    # general rate model (PyFVTool)
    "GRMSetup",
    "ParticleProperties",
    "GRMResult",
    "run_grm",
    "film_coefficient",
    # modes
    "cation_exchange",
    "anion_exchange",
    "hic",
    "reversed_phase",
    "high_resolution_iex",
    # metrics
    "pool_metrics",
    "peak_moments",
    "plate_count",
    "resolution",
    # legacy
    "ChromatographyConfig",
    "simulate",
]
