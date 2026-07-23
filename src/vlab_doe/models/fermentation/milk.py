"""Milk substrate and its acid → pH titration curve.

The only thing the lab actually measures is pH, but the bacteria respond to (and produce)
*lactic acid*.  The link between the two is milk's **buffering capacity**: casein, phosphate
and citrate resist the pH change, so the pH-versus-acid curve is a shallow sigmoid, not a
straight Henderson–Hasselbalch line.  We model it empirically::

    pH(L) = pH_inf + (pH0 - pH_inf) / (1 + (L / L50)^n_buf)

* ``pH0`` — pH of fresh milk (~6.6).
* ``pH_inf`` — the floor the pH asymptotes toward as acid accumulates.
* ``L50`` — titratable acid at the curve's midpoint; this is the practical handle on buffering
  capacity and is a major lot-to-lot variability source.
* ``n_buf`` — steepness of the drop through the gelation region.

Because ``L50`` and the initial lactose ``S0`` vary between milk lots, :class:`Milk` is the
natural carrier for that batch-to-batch variability (see :mod:`.variability`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Milk:
    """Composition and buffering of the milk base.

    Parameters
    ----------
    lactose:
        Initial lactose concentration ``S0`` (g/L); whole cow's milk is ~48 g/L.
    ph0:
        pH of the fresh, un-fermented milk.
    ph_inf:
        Asymptotic pH floor as lactic acid accumulates.
    l50:
        Titratable lactic acid (mmol/L) at the midpoint of the titration curve — the
        buffering handle.
    n_buf:
        Hill steepness of the titration curve.
    """

    lactose: float = 48.0
    ph0: float = 6.6
    ph_inf: float = 3.9
    l50: float = 48.0
    n_buf: float = 2.2


def ph_from_acid(lactic_acid, milk: Milk):
    """Milk pH for a given lactic-acid concentration (empirical titration curve).

    Accepts a scalar or array of lactic-acid concentrations (mmol/L) and returns the pH in the
    same shape.

    Parameters
    ----------
    lactic_acid:
        Lactic-acid concentration ``L`` (mmol/L); negatives are clamped to 0.
    milk:
        Milk buffering parameters.
    """
    L = np.clip(np.asarray(lactic_acid, dtype=float), 0.0, None)
    drop = (milk.ph0 - milk.ph_inf) / (1.0 + (L / milk.l50) ** milk.n_buf)
    ph = milk.ph_inf + drop
    return float(ph) if np.ndim(lactic_acid) == 0 else ph
