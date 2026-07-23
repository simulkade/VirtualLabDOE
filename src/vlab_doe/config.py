"""Global configuration: reproducible RNG and canonical filesystem paths.

Every phase pulls its randomness through :func:`make_rng` and writes artifacts
under :data:`DATA_DIR` / :data:`RESULTS_DIR` so that synthetic datasets and
figures are fully regenerable from a seed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Repo root = two levels up from this file (src/vlab_doe/config.py -> repo/).
ROOT_DIR: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
RESULTS_DIR: Path = ROOT_DIR / "results"
REPORTS_DIR: Path = ROOT_DIR / "reports"

#: Default master seed used across the project unless overridden.
DEFAULT_SEED: int = 20260531


def make_rng(seed: int | None = None) -> np.random.Generator:
    """Return a NumPy :class:`~numpy.random.Generator` seeded for reproducibility.

    Parameters
    ----------
    seed:
        Seed to use. ``None`` falls back to :data:`DEFAULT_SEED`.
    """
    return np.random.default_rng(DEFAULT_SEED if seed is None else seed)


def ensure_dirs() -> None:
    """Create the standard data/results/reports directories if missing."""
    for directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
