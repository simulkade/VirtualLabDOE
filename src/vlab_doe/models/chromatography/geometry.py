"""Column geometry — physical dimensions and packing."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ColumnGeometry:
    """Physical column dimensions and packing."""

    length: float       # m
    diameter: float     # m
    porosity: float     # bed void fraction ε (−)

    @property
    def area(self) -> float:
        """Cross-sectional area (m²)."""
        return math.pi / 4.0 * self.diameter**2

    @property
    def volume(self) -> float:
        """Total (empty) column volume (m³)."""
        return self.area * self.length

    @property
    def resin_volume(self) -> float:
        """Stationary-phase (resin) volume (m³) = V_column·(1−ε)."""
        return self.volume * (1.0 - self.porosity)
