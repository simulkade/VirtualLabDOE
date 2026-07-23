"""Adsorption isotherms for every chromatography mode.

The engine works with a single explicit, multi-component equilibrium isotherm,

    q*_i = q_max,i · b_i(m, pH) · c_i / (1 + Σ_j b_j(m, pH) · c_j)        (nonlinear)
    q*_i = H_i(m, pH) · c_i,        H_i = q_max,i · b_i                   (linear / dilute)

where ``m`` is the *modulator* (salt concentration for IEX/HIC, organic fraction for
RP-HPLC).  Only the **modulator law** ``b_i(m, pH)`` differs between modes:

* **Ion exchange (CEX/AEX)** — Steric-Mass-Action salt/pH dependence
  ``b_i = β_i · (Λ/m)^ν_i · exp(ν_pH,i·(pH − pH_ref))``.  Binding falls as salt rises.
* **HIC (salting-out)** — ``b_i = β_i · exp(K_s,i · m)``.  Binding *grows* with salt.
* **RP-HPLC (linear solvent strength)** — ``b_i = β_i · exp(−S_i · φ)``.  Binding falls
  as the organic fraction φ rises.

The dilute limit ``H_i = q_max,i · b_i`` recovers an analytic Henry's constant, so the
linear regime matches the original linearised model exactly.

The legacy helpers (:class:`SMAParameters`, :func:`sma_henry_constant`,
:func:`sma_isotherm`, :func:`langmuir_isotherm`) are retained unchanged for backward
compatibility.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np


# ── Legacy isotherm helpers (unchanged public API) ────────────────────────────

@dataclass(frozen=True)
class SMAParameters:
    """Linearised Steric Mass Action isotherm parameters.

    The linearised Henry's constant is:
        H = K · exp(ν_pH · (pH − pH_ref)) · (Λ / salt)^ν

    Parameters
    ----------
    equilibrium_constant:
        Dimensionless equilibrium constant K for each component.
    characteristic_charge:
        Characteristic charge ν (determines salt sensitivity).
    steric_factor:
        Steric exclusion factor σ (used in full non-linear SMA).
    ionic_capacity:
        Resin ionic capacity Λ (mM).
    ph_ref:
        Reference pH for the pH-shift exponent (default 7.0).
    nu_ph:
        Sensitivity of ln(H) to pH (default 0 = pH-independent).
    """

    equilibrium_constant: Sequence[float]
    characteristic_charge: Sequence[float]
    steric_factor: Sequence[float]
    ionic_capacity: float
    ph_ref: float = 7.0
    nu_ph: float = 0.0


def langmuir_isotherm(
    c: np.ndarray,
    q_max: np.ndarray,
    k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Competitive multi-component Langmuir isotherm.

    Parameters
    ----------
    c:
        Mobile-phase concentrations, shape ``(n_components,)``.
    q_max:
        Maximum binding capacities, same shape.
    k:
        Adsorption equilibrium constants, same shape.

    Returns
    -------
    q:
        Bound concentrations.
    dq_dc:
        Diagonal of the Jacobian ∂q_i/∂c_i (off-diagonals neglected for
        the retardation factor in the ED model).
    """
    c = np.asarray(c, dtype=float)
    q_max = np.asarray(q_max, dtype=float)
    k = np.asarray(k, dtype=float)

    denom = 1.0 + float(np.dot(k, c))
    q = q_max * c / denom

    # ∂q_i/∂c_i = q_max_i · (1 + Σ_{j≠i} k_j c_j) / denom²
    cross_sum = np.dot(k, c)  # Σ_j k_j c_j
    dq_dc = q_max * (1.0 + cross_sum - k * c) / denom**2

    return q, dq_dc


def sma_henry_constant(salt: float, ph: float, params: SMAParameters, index: int = 0) -> float:
    """Linearised SMA Henry's constant for component *index*.

    H = K · exp(ν_pH · (pH − pH_ref)) · (Λ / salt)^ν

    At low protein concentration q ≈ H · c, giving an analytically tractable
    retardation factor R = 1 + (1−ε)/ε · H.
    """
    K = float(np.asarray(params.equilibrium_constant)[index])
    nu = float(np.asarray(params.characteristic_charge)[index])
    H = (
        K
        * math.exp(params.nu_ph * (ph - params.ph_ref))
        * (params.ionic_capacity / salt) ** nu
    )
    return H


def sma_isotherm(
    c: np.ndarray,
    salt: float,
    ph: float,
    params: SMAParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearised SMA bound concentration and isotherm slope.

    Returns ``(q, dq_dc)`` for all components (shape ``(n_components,)``).
    """
    n = len(params.equilibrium_constant)
    H = np.array([sma_henry_constant(salt, ph, params, i) for i in range(n)])
    c_arr = np.asarray(c, dtype=float)
    return H * c_arr, H


# ── Modulator laws b_i(m, pH) ─────────────────────────────────────────────────

def _col(arr: np.ndarray, ndim: int) -> np.ndarray:
    """Reshape a ``(n,)`` parameter vector to broadcast against ``b`` of *ndim* dims."""
    return arr.reshape((-1,) + (1,) * (ndim - 1))


class ModulatorLaw(ABC):
    """Maps the modulator ``m`` (and pH) to per-component affinity factors ``b_i``."""

    n_components: int

    @abstractmethod
    def b(self, m, ph: float) -> np.ndarray:
        """Affinity factors ``b_i``.

        ``m`` may be a scalar (→ shape ``(n,)``) or a 1-D array of cell values
        (→ shape ``(n, len(m))``).
        """


@dataclass(frozen=True)
class SMALaw(ModulatorLaw):
    """Ion-exchange salt/pH law: ``b_i = β_i·(Λ/m)^ν_i·exp(ν_pH,i·(pH−pH_ref))``.

    ``ν_pH > 0`` (cation exchange) increases binding as pH rises above ``pH_ref``;
    ``ν_pH < 0`` (anion exchange) increases binding as pH falls below it.
    """

    beta: np.ndarray            # affinity prefactor β_i
    nu: np.ndarray              # characteristic charge ν_i (salt sensitivity)
    ionic_capacity: float       # resin ionic capacity Λ (mM)
    nu_ph: np.ndarray           # pH sensitivity ν_pH,i
    ph_ref: float = 7.0

    @property
    def n_components(self) -> int:  # type: ignore[override]
        return len(self.beta)

    def b(self, m, ph: float) -> np.ndarray:
        m_arr = np.atleast_1d(np.asarray(m, dtype=float))
        out = (
            _col(self.beta, 2)
            * (self.ionic_capacity / m_arr[None, :]) ** _col(self.nu, 2)
            * np.exp(_col(self.nu_ph, 2) * (ph - self.ph_ref))
        )
        return out[:, 0] if np.ndim(m) == 0 else out


@dataclass(frozen=True)
class SaltingOutLaw(ModulatorLaw):
    """HIC salting-out law: ``b_i = β_i·exp(K_s,i·m)`` — binding grows with salt."""

    beta: np.ndarray            # affinity prefactor at zero salt
    ks: np.ndarray              # salting-out coefficient K_s,i (1/mM)

    @property
    def n_components(self) -> int:  # type: ignore[override]
        return len(self.beta)

    def b(self, m, ph: float) -> np.ndarray:
        m_arr = np.atleast_1d(np.asarray(m, dtype=float))
        out = _col(self.beta, 2) * np.exp(_col(self.ks, 2) * m_arr[None, :])
        return out[:, 0] if np.ndim(m) == 0 else out


@dataclass(frozen=True)
class LinearSolventStrengthLaw(ModulatorLaw):
    """RP-HPLC law: ``b_i = β_i·exp(−S_i·φ)`` — binding falls as organic φ rises."""

    beta: np.ndarray            # affinity prefactor in pure aqueous (φ = 0)
    s: np.ndarray               # solvent-strength slope S_i

    @property
    def n_components(self) -> int:  # type: ignore[override]
        return len(self.beta)

    def b(self, m, ph: float) -> np.ndarray:
        m_arr = np.atleast_1d(np.asarray(m, dtype=float))
        out = _col(self.beta, 2) * np.exp(-_col(self.s, 2) * m_arr[None, :])
        return out[:, 0] if np.ndim(m) == 0 else out


# ── Competitive-Langmuir isotherm wrapping a modulator law ─────────────────────

@dataclass(frozen=True)
class Isotherm:
    """Multi-component competitive Langmuir with a modulator-dependent affinity.

    Parameters
    ----------
    law:
        The mode-specific :class:`ModulatorLaw` giving ``b_i(m, pH)``.
    q_max:
        Per-component saturation capacity ``q_max,i`` (g/L resin).  Sets the
        overload behaviour; ignored (as a saturation term) when ``linear=True``.
    linear:
        If ``True`` use the dilute Henry form ``q* = H·c`` (no competition /
        no capacity limit).  If ``False`` use the full competitive Langmuir.
    """

    law: ModulatorLaw
    q_max: np.ndarray
    linear: bool = False

    @property
    def n_components(self) -> int:
        return self.law.n_components

    def henry(self, m, ph: float) -> np.ndarray:
        """Dilute-limit Henry's constant ``H_i = q_max,i · b_i(m, pH)``."""
        b = self.law.b(m, ph)
        return _col(np.asarray(self.q_max, float), b.ndim) * b

    def q_star(self, c, m, ph: float) -> np.ndarray:
        """Equilibrium bound concentration ``q*_i`` for mobile concentrations ``c``.

        ``c`` and the returned array share the shape of ``law.b(m, pH)``:
        ``(n,)`` for scalar ``m`` or ``(n, n_cells)`` for a per-cell modulator.
        """
        c = np.asarray(c, dtype=float)
        b = self.law.b(m, ph)
        qmax = _col(np.asarray(self.q_max, float), b.ndim)
        numerator = qmax * b * c
        if self.linear:
            return numerator
        denom = 1.0 + np.sum(b * c, axis=0)
        return numerator / denom
