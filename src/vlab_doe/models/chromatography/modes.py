"""Preset isotherm builders for the common chromatography modes.

Each helper assembles a ready :class:`~.isotherms.Isotherm` from physically meaningful
parameters, so callers don't hand-wire the modulator law.  All parameters accept either a
scalar (broadcast to every component) or a per-component sequence.

* :func:`cation_exchange` / :func:`anion_exchange` — Steric-Mass-Action ion exchange.
  They differ only in the sign of the pH sensitivity ``nu_ph``: a cation exchanger binds
  more strongly as pH falls (protein more positive), an anion exchanger as pH rises.
* :func:`hic` — hydrophobic-interaction salting-out (binding grows with salt).
* :func:`reversed_phase` — RP-HPLC linear-solvent-strength (binding falls with organic %).
* :func:`high_resolution_iex` — a nonlinear (overloaded, competitive) ion-exchange isotherm
  for multi-component resolution studies.
"""

from __future__ import annotations

import numpy as np

from .isotherms import (
    Isotherm,
    LinearSolventStrengthLaw,
    SaltingOutLaw,
    SMALaw,
)


def _vec(x, n: int) -> np.ndarray:
    """Broadcast *x* to a length-*n* float array."""
    return np.broadcast_to(np.asarray(x, dtype=float), (n,)).astype(float)


def cation_exchange(
    *,
    beta,
    nu,
    ionic_capacity: float = 1000.0,
    q_max=100.0,
    nu_ph=-1.0,
    ph_ref: float = 7.0,
    linear: bool = False,
) -> Isotherm:
    """Cation-exchange (CEX) Steric-Mass-Action isotherm.

    ``nu_ph`` defaults negative: binding strengthens as pH drops below ``ph_ref``
    (the protein becomes more positively charged).
    """
    beta = np.atleast_1d(np.asarray(beta, dtype=float))
    n = len(beta)
    law = SMALaw(
        beta=beta,
        nu=_vec(nu, n),
        ionic_capacity=ionic_capacity,
        nu_ph=_vec(nu_ph, n),
        ph_ref=ph_ref,
    )
    return Isotherm(law=law, q_max=_vec(q_max, n), linear=linear)


def anion_exchange(
    *,
    beta,
    nu,
    ionic_capacity: float = 1000.0,
    q_max=100.0,
    nu_ph=1.0,
    ph_ref: float = 7.0,
    linear: bool = False,
) -> Isotherm:
    """Anion-exchange (AEX) Steric-Mass-Action isotherm.

    ``nu_ph`` defaults positive: binding strengthens as pH rises above ``ph_ref``
    (the protein becomes more negatively charged).
    """
    return cation_exchange(
        beta=beta,
        nu=nu,
        ionic_capacity=ionic_capacity,
        q_max=q_max,
        nu_ph=nu_ph,
        ph_ref=ph_ref,
        linear=linear,
    )


def hic(
    *,
    beta,
    ks,
    q_max=100.0,
    linear: bool = False,
) -> Isotherm:
    """Hydrophobic-interaction (HIC) salting-out isotherm.

    ``ks`` is the salting-out coefficient (1/mM); binding grows exponentially with salt,
    so HIC is eluted by a *decreasing* salt gradient.
    """
    beta = np.atleast_1d(np.asarray(beta, dtype=float))
    n = len(beta)
    law = SaltingOutLaw(beta=beta, ks=_vec(ks, n))
    return Isotherm(law=law, q_max=_vec(q_max, n), linear=linear)


def reversed_phase(
    *,
    beta,
    s,
    q_max=100.0,
    linear: bool = False,
) -> Isotherm:
    """RP-HPLC linear-solvent-strength isotherm.

    The modulator is the organic fraction φ ∈ [0, 1]; ``s`` is the solvent-strength
    slope ``S``.  Binding falls as φ rises, so RP is eluted by an *increasing* organic
    gradient.
    """
    beta = np.atleast_1d(np.asarray(beta, dtype=float))
    n = len(beta)
    law = LinearSolventStrengthLaw(beta=beta, s=_vec(s, n))
    return Isotherm(law=law, q_max=_vec(q_max, n), linear=linear)


def high_resolution_iex(
    *,
    beta,
    nu,
    ionic_capacity: float = 1000.0,
    q_max=100.0,
    nu_ph=-1.0,
    ph_ref: float = 7.0,
) -> Isotherm:
    """Nonlinear (overloaded, competitive) cation-exchange isotherm.

    Identical SMA salt/pH law to :func:`cation_exchange`, but always nonlinear so that
    finite capacity and inter-component competition shape the peaks — the regime where a
    shallow gradient resolves closely-spaced species.
    """
    return cation_exchange(
        beta=beta,
        nu=nu,
        ionic_capacity=ionic_capacity,
        q_max=q_max,
        nu_ph=nu_ph,
        ph_ref=ph_ref,
        linear=False,
    )
