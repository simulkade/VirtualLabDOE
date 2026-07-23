"""Right-hand side of the fermentation dynamics.

The state vector packs, for ``n`` strains::

    y = [ X_0 … X_{n-1}, Q_0 … Q_{n-1}, S, L, A ]          (length 2n + 3)

where ``X`` is biomass, ``Q`` is the Baranyi physiological state (the lag clock), ``S`` is
lactose, ``L`` is lactic acid and ``A`` is the aroma proxy.  pH is *not* a state — it is read
back from ``L`` through the milk titration curve at every evaluation, so the acid that the
bacteria produce feeds straight back into how much they are inhibited.

Per strain *i* the specific growth rate is::

    mu_i = mu_max_i · gamma_T,i · S/(K_S,i + S) · I_acid,i(pH) · interaction_i

and biomass follows a logistic, lag-gated Baranyi–Roberts law::

    alpha_i = Q_i / (1 + Q_i)
    dX_i/dt = alpha_i · mu_i · X_i · (1 - X_tot / X_max,i)
    dQ_i/dt = mu_max_i · gamma_T,i · Q_i

Substrate, acid (Luedeking–Piret) and aroma are summed over strains.  The whole bundle is
assembled once by :func:`make_kinetics` so the engine can reuse the same drift for both the
deterministic (`solve_ivp`) and stochastic (Euler–Maruyama) integrators.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .milk import Milk, ph_from_acid
from .strains import Consortium, cardinal_temperature_factor

#: Reference pH at which the acid-inhibition factor reaches 1 (fresh-milk pH).
PH_REFERENCE = 6.6


def monod(s: np.ndarray, k_s: np.ndarray) -> np.ndarray:
    """Monod substrate-limitation factor ``S / (K_S + S)`` (clamped to S ≥ 0)."""
    s = np.clip(s, 0.0, None)
    return s / (k_s + s)


def baranyi_alpha(q: np.ndarray) -> np.ndarray:
    """Baranyi adjustment function ``alpha = Q / (1 + Q)`` ∈ [0, 1] (the lag gate)."""
    q = np.clip(q, 0.0, None)
    return q / (1.0 + q)


def acid_inhibition(ph: float, ph_min: np.ndarray, ph_ref: float = PH_REFERENCE) -> np.ndarray:
    """Growth attenuation by accumulated acid.

    Linear ramp ``(pH - ph_min) / (ph_ref - ph_min)`` clamped to [0, 1]: full growth at the
    reference pH, zero once the pH falls to a strain's ``ph_min``.
    """
    return np.clip((ph - ph_min) / (ph_ref - ph_min), 0.0, 1.0)


def interaction_factor(x: np.ndarray, interaction: np.ndarray, half: float) -> np.ndarray:
    """Per-strain growth multiplier from pairwise stimulation.

    ``factor_i = Π_j (1 + k[i,j]·X_j/(X_j + half))`` for ``j ≠ i``.  With a zero interaction
    matrix this is all ones (independent strains).
    """
    x = np.clip(x, 0.0, None)
    sat = x / (x + half)                       # shape (n,), saturating presence of each strain
    # contribution[i, j] = stimulation of i by j ; diagonal forced to 1
    contribution = 1.0 + interaction * sat[np.newaxis, :]
    np.fill_diagonal(contribution, 1.0)
    return np.prod(contribution, axis=1)


@dataclass
class FermentationKinetics:
    """Pre-compiled dynamics for one consortium / milk / temperature combination.

    Build with :func:`make_kinetics`.  Holds the per-strain parameter arrays and the
    temperature-corrected ``mu_max·gamma_T`` so the hot loop is pure NumPy.
    """

    n: int
    milk: Milk
    # per-strain arrays (length n)
    mu_eff: np.ndarray          # mu_max · gamma_T
    k_s: np.ndarray
    inv_yield: np.ndarray       # 1 / yield_biomass
    ph_min: np.ndarray
    acid_growth: np.ndarray
    acid_maintenance: np.ndarray
    aroma_yield: np.ndarray
    x_max: np.ndarray
    interaction: np.ndarray     # (n, n)
    interaction_half: float
    fractions: np.ndarray       # normalised inoculum fractions
    aroma_decay: float = 0.05

    # ── state-vector helpers ────────────────────────────────────────────────────
    def unpack(self, y: np.ndarray):
        """Split a state vector into ``(X, Q, S, L, A)``."""
        n = self.n
        return y[:n], y[n:2 * n], y[2 * n], y[2 * n + 1], y[2 * n + 2]

    def initial_state(self, total_inoculum: float, lag_state: np.ndarray) -> np.ndarray:
        """Assemble ``y0`` from a total inoculum split by inoculum fractions."""
        n = self.n
        y0 = np.zeros(2 * n + 3)
        y0[:n] = total_inoculum * self.fractions
        y0[n:2 * n] = lag_state
        y0[2 * n] = self.milk.lactose
        y0[2 * n + 1] = 0.0          # no acid at inoculation
        y0[2 * n + 2] = 0.0          # no aroma
        return y0

    def ph(self, y: np.ndarray) -> float:
        """pH read back from the lactic-acid component of a state vector."""
        return ph_from_acid(y[2 * self.n + 1], self.milk)

    # ── drift ───────────────────────────────────────────────────────────────────
    def growth_rates(self, y: np.ndarray):
        """Return ``(dX_growth, dQ, ph)`` — the per-strain biomass growth and lag updates."""
        n = self.n
        X, Q, S, L, _A = self.unpack(y)
        X = np.clip(X, 0.0, None)
        ph = ph_from_acid(L, self.milk)

        alpha = baranyi_alpha(Q)
        mu = (
            self.mu_eff
            * monod(S, self.k_s)
            * acid_inhibition(ph, self.ph_min)
            * interaction_factor(X, self.interaction, self.interaction_half)
        )
        x_tot = X.sum()
        logistic = np.clip(1.0 - x_tot / self.x_max, 0.0, None)
        dX = alpha * mu * X * logistic
        dQ = self.mu_eff * Q
        return dX, dQ, ph

    def drift(self, t: float, y: np.ndarray) -> np.ndarray:
        """Deterministic time-derivative of the full state vector."""
        n = self.n
        X, Q, S, L, A = self.unpack(y)
        dX, dQ, _ph = self.growth_rates(y)

        substrate_gate = 1.0 if S > 0.0 else 0.0
        dS = -substrate_gate * float((self.inv_yield * dX).sum())
        # Luedeking–Piret: growth-associated + maintenance.  Maintenance acid is gated by the
        # same acid tolerance as growth, so a strain that has hit its ph_min stops acidifying
        # (otherwise an acid-sensitive strain like ST would over-acidify past its stall pH).
        ph = ph_from_acid(L, self.milk)
        i_acid = acid_inhibition(ph, self.ph_min)
        dL = float((self.acid_growth * dX).sum()) + substrate_gate * float(
            (self.acid_maintenance * i_acid * np.clip(X, 0.0, None)).sum()
        )
        dA = float((self.aroma_yield * dX).sum()) - self.aroma_decay * A

        dy = np.empty_like(y)
        dy[:n] = dX
        dy[n:2 * n] = dQ
        dy[2 * n] = dS
        dy[2 * n + 1] = dL
        dy[2 * n + 2] = dA
        return dy

    def diffusion(self, y: np.ndarray, sigma: float) -> np.ndarray:
        """Per-state noise amplitude for Euler–Maruyama (multiplicative on biomass only)."""
        n = self.n
        g = np.zeros_like(y)
        g[:n] = sigma * np.clip(y[:n], 0.0, None)
        return g


def make_kinetics(
    consortium: Consortium,
    milk: Milk,
    temperature: float,
    *,
    aroma_decay: float = 0.05,
) -> FermentationKinetics:
    """Compile a :class:`FermentationKinetics` for a consortium at a fixed temperature."""
    strains = consortium.strains
    gamma_T = np.array([cardinal_temperature_factor(temperature, s) for s in strains])
    mu_max = np.array([s.mu_max for s in strains])
    return FermentationKinetics(
        n=consortium.n_strains,
        milk=milk,
        mu_eff=mu_max * gamma_T,
        k_s=np.array([s.k_s for s in strains]),
        inv_yield=np.array([1.0 / s.yield_biomass for s in strains]),
        ph_min=np.array([s.ph_min for s in strains]),
        acid_growth=np.array([s.acid_growth for s in strains]),
        acid_maintenance=np.array([s.acid_maintenance for s in strains]),
        aroma_yield=np.array([s.aroma_yield for s in strains]),
        x_max=np.array([s.x_max for s in strains]),
        interaction=consortium.interaction,
        interaction_half=consortium.interaction_half,
        fractions=consortium.normalized_fractions(),
        aroma_decay=aroma_decay,
    )
