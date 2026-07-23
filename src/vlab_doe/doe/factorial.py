"""Phase 2 — Full factorial Design of Experiments.

Generates 2-level / 3-level full factorial designs (with optional replicated
center points) over a set of critical process parameters (CPPs), then evaluates
the virtual lab at each design point.

Implementation notes:
- The coded levels (−1, 0, +1) are mapped to physical units via linear scaling.
- Two-level designs are constructed with :func:`itertools.product`.
- Three-level designs add the midpoint (0) to each factor's level set.
- Center points are added as unencoded (physical) rows at the factor midpoints.
  They allow estimation of pure experimental error and quadratic curvature.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Factor:
    """A continuous design factor with a name and physical low/high bounds."""

    name: str
    low: float
    high: float

    @property
    def center(self) -> float:
        return (self.low + self.high) / 2.0

    @property
    def half_range(self) -> float:
        return (self.high - self.low) / 2.0

    def decode(self, coded: float) -> float:
        """Convert a coded level (−1, 0, +1) to physical units."""
        return self.center + coded * self.half_range


def full_factorial(
    factors: Sequence[Factor],
    *,
    levels: int = 2,
    center_points: int = 0,
) -> pd.DataFrame:
    """Build a full factorial design in physical units.

    Parameters
    ----------
    factors:
        The CPPs to vary.
    levels:
        2 or 3.  A 2-level design uses coded values ±1; a 3-level design
        also includes 0.  The number of runs is ``levels^k`` (plus center
        points, which are always at the factor centers regardless of *levels*).
    center_points:
        Number of replicated center-point runs appended after the factorial.
        These allow estimation of pure experimental error and curvature.

    Returns
    -------
    pandas.DataFrame
        One row per design point, columns named after the factors, values in
        physical units.  Column ``"run_type"`` labels each row as
        ``"factorial"`` or ``"center_point"``.
    """
    if levels not in (2, 3):
        raise ValueError(f"levels must be 2 or 3, got {levels}")

    coded_levels = [-1.0, 1.0] if levels == 2 else [-1.0, 0.0, 1.0]

    rows = []
    for combo in itertools.product(coded_levels, repeat=len(factors)):
        row = {f.name: f.decode(c) for f, c in zip(factors, combo)}
        row["run_type"] = "factorial"
        rows.append(row)

    for _ in range(center_points):
        row = {f.name: f.center for f in factors}
        row["run_type"] = "center_point"
        rows.append(row)

    df = pd.DataFrame(rows)
    # Randomise run order to protect against time trends
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df


def run_design(
    design: pd.DataFrame,
    evaluate: Callable[[Mapping[str, float]], Mapping[str, float]],
) -> pd.DataFrame:
    """Evaluate the virtual lab at every design point.

    Parameters
    ----------
    design:
        Output of :func:`full_factorial` (or any factor table with numeric
        columns).  The ``"run_type"`` column, if present, is preserved but not
        passed to *evaluate*.
    evaluate:
        Callable that takes a dict of factor values and returns a dict of
        response values.  This is the bridge to the perturbed mechanistic
        model.

    Returns
    -------
    pandas.DataFrame
        *design* joined with the response columns returned by *evaluate*.
    """
    factor_cols = [c for c in design.columns if c != "run_type"]
    records = []
    for _, row in design.iterrows():
        point = {col: float(row[col]) for col in factor_cols}
        responses = dict(evaluate(point))
        records.append({**point, **responses})

    result = pd.DataFrame(records)
    if "run_type" in design.columns:
        result.insert(len(factor_cols), "run_type", design["run_type"].values)
    return result
