"""Benchtop model 4 — Acid-catalysed hydrolysis of methyl acetate.

A batch of methyl acetate is mixed with water and a strong-acid catalyst and
left to react:

.. math::

    \\underbrace{\\mathrm{CH_3COOCH_3}}_{E}
    + \\underbrace{\\mathrm{H_2O}}_{W}
    \\;\\underset{k_r}{\\overset{k_f}{\\rightleftharpoons}}\\;
    \\underbrace{\\mathrm{CH_3COOH}}_{A}
    + \\underbrace{\\mathrm{CH_3OH}}_{M}.

The reaction is **reversible** and reaches an equilibrium short of complete
conversion, and it is slow enough (hours) at room temperature that the *time* of
sampling is itself a design factor.

Kinetics
--------
The hydrogen ion is a catalyst: it multiplies the forward *and* reverse rates
equally, so it speeds the approach to equilibrium **without moving the
equilibrium itself**.  With concentrations in mol/L and the reaction extent
:math:`x` (also mol/L),

.. math::

    \\frac{\\dd x}{\\dd t}
      = c_{\\mathrm{H^+}}\\big[\\,k_f\\,(E_0-x)(W_0-x)
        - k_r\\,(A_0+x)(M_0+x)\\,\\big],

with Arrhenius rate constants :math:`k_f = A_f e^{-E_{a,f}/RT}` and
:math:`k_r = A_r e^{-E_{a,r}/RT}`.  Their ratio is the equilibrium constant,

.. math::

    K_{\\mathrm{eq}}(T) = \\frac{k_f}{k_r}
      = \\frac{A_f}{A_r}\\,e^{-(E_{a,f}-E_{a,r})/RT},

so temperature moves *both* the rate and the equilibrium (van 't Hoff), whereas
the catalyst moves only the rate --- a clean separation the DoE exercises are
built to reveal.  The default activation energies make the hydrolysis mildly
endothermic, so warmer runs convert a little more ester at equilibrium.

Operating constraint
--------------------
Methyl acetate boils at ``METHYL_ACETATE_BP_C`` ≈ 56.9 °C, so the temperature
factor must stay below it to keep the ester in the liquid phase; the design
helpers clip to this ceiling and record it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

_R_GAS = 8.314462618  # J/(mol·K)

#: Normal boiling point of methyl acetate (°C) — the upper temperature limit.
METHYL_ACETATE_BP_C = 56.9


# ── Kinetic parameters ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class KineticParameters:
    """Arrhenius parameters for the forward and reverse rate constants.

    Units are lumped: ``pre_exp_*`` carry whatever units make
    ``rate = c_H · k · [·][·]`` come out in mol/(L·s).  The defaults are tuned so
    that a stoichiometric mixture with ~0.1 M acid at 25 °C approaches
    equilibrium over a few hours, with :math:`K_\\text{eq}(25^\\circ\\mathrm{C})
    \\approx 0.18`.
    """

    pre_exp_forward: float = 4.6e6
    pre_exp_reverse: float = 2.0e5
    activation_forward: float = 62_000.0
    activation_reverse: float = 50_000.0

    def k_forward(self, temp_c: float) -> float:
        t = float(temp_c) + 273.15
        return self.pre_exp_forward * np.exp(-self.activation_forward / (_R_GAS * t))

    def k_reverse(self, temp_c: float) -> float:
        t = float(temp_c) + 273.15
        return self.pre_exp_reverse * np.exp(-self.activation_reverse / (_R_GAS * t))

    def equilibrium_constant(self, temp_c: float) -> float:
        """:math:`K_\\text{eq}=k_f/k_r` at ``temp_c``."""
        return self.k_forward(temp_c) / self.k_reverse(temp_c)


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EsterHydrolysisConfig:
    """A single hydrolysis batch.

    Parameters
    ----------
    temperature:
        Reaction temperature (°C); must stay below ``METHYL_ACETATE_BP_C``.
    catalyst_conc:
        Acid (H⁺) concentration :math:`c_{H^+}` (mol/L).  ``pH`` may be supplied
        instead via :meth:`from_ph`.
    ester0, water0:
        Initial methyl-acetate and water concentrations (mol/L).
    acid0, methanol0:
        Initial product concentrations (mol/L); usually zero.
    kinetics:
        :class:`KineticParameters` (defaults tuned for a few-hour experiment).
    """

    temperature: float = 25.0
    catalyst_conc: float = 0.1
    ester0: float = 4.0
    water0: float = 4.0
    acid0: float = 0.0
    methanol0: float = 0.0
    kinetics: KineticParameters = field(default_factory=KineticParameters)

    @classmethod
    def from_ph(cls, ph: float, **kwargs) -> "EsterHydrolysisConfig":
        """Construct with the catalyst set from a pH (``c_H⁺ = 10**-pH``)."""
        return cls(catalyst_conc=10.0 ** (-float(ph)), **kwargs)

    def __post_init__(self) -> None:
        if self.temperature >= METHYL_ACETATE_BP_C:
            raise ValueError(
                f"temperature {self.temperature} °C is at/above the methyl-acetate "
                f"boiling point ({METHYL_ACETATE_BP_C} °C); keep it lower."
            )


# ── Equilibrium ───────────────────────────────────────────────────────────────

def equilibrium_extent(config: EsterHydrolysisConfig) -> float:
    """Equilibrium reaction extent :math:`x_\\text{eq}` (mol/L).

    Solves the quadratic :math:`K_\\text{eq}(E_0-x)(W_0-x)=(A_0+x)(M_0+x)` for the
    physically admissible root (:math:`0 \\le x \\le \\min(E_0,W_0)`).
    """
    keq = config.kinetics.equilibrium_constant(config.temperature)
    e0, w0, a0, m0 = config.ester0, config.water0, config.acid0, config.methanol0
    # K(e0-x)(w0-x) = (a0+x)(m0+x)  →  (K-1)x² - [K(e0+w0)+(a0+m0)]x + [K e0 w0 - a0 m0] = 0
    a = keq - 1.0
    b = -(keq * (e0 + w0) + (a0 + m0))
    c = keq * e0 * w0 - a0 * m0
    x_max = min(e0, w0)
    if abs(a) < 1e-12:  # K ≈ 1, linear
        x = -c / b
    else:
        disc = max(b * b - 4.0 * a * c, 0.0)
        roots = [(-b - np.sqrt(disc)) / (2.0 * a), (-b + np.sqrt(disc)) / (2.0 * a)]
        admissible = [r for r in roots if -1e-9 <= r <= x_max + 1e-9]
        x = min(admissible) if admissible else float(np.clip(roots[0], 0.0, x_max))
    return float(np.clip(x, 0.0, x_max))


def equilibrium_conversion(config: EsterHydrolysisConfig) -> float:
    """Fractional conversion of ester at equilibrium, :math:`x_\\text{eq}/E_0`."""
    return equilibrium_extent(config) / config.ester0


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate(config: EsterHydrolysisConfig, t_eval: np.ndarray) -> dict[str, np.ndarray]:
    """Integrate the batch kinetics over time.

    Parameters
    ----------
    config:
        Batch configuration.
    t_eval:
        Output time grid (seconds).

    Returns
    -------
    dict with keys ``"t"`` (s), ``"ester"``, ``"water"``, ``"acetic_acid"``,
    ``"methanol"`` (all mol/L) and ``"conversion"`` (–).
    """
    t_eval = np.asarray(t_eval, dtype=float)
    kin = config.kinetics
    kf = kin.k_forward(config.temperature) * config.catalyst_conc
    kr = kin.k_reverse(config.temperature) * config.catalyst_conc
    e0, w0, a0, m0 = config.ester0, config.water0, config.acid0, config.methanol0

    def rhs(_t, y):
        x = y[0]
        return [kf * (e0 - x) * (w0 - x) - kr * (a0 + x) * (m0 + x)]

    sol = solve_ivp(
        rhs, (t_eval[0], t_eval[-1]), [0.0], method="LSODA",
        t_eval=t_eval, rtol=1e-8, atol=1e-10,
    )
    x = sol.y[0]
    return {
        "t": sol.t,
        "ester": e0 - x,
        "water": w0 - x,
        "acetic_acid": a0 + x,
        "methanol": m0 + x,
        "conversion": x / e0,
    }


def conversion_at(config: EsterHydrolysisConfig, time_s: float) -> float:
    """Fractional ester conversion at a single sampling time (s)."""
    if time_s <= 0.0:
        return config.acid0 / config.ester0  # initial conversion (usually 0)
    out = simulate(config, np.array([0.0, float(time_s)]))
    return float(out["conversion"][-1])


def time_to_conversion(config: EsterHydrolysisConfig, target: float) -> float:
    """Time (s) to reach ``target`` fractional conversion.

    Returns ``inf`` if the target exceeds the equilibrium conversion (it can
    never be reached).
    """
    x_eq = equilibrium_conversion(config)
    if target >= x_eq:
        return float("inf")
    # Bracket by expanding the horizon until the target is passed, then bisect.
    from scipy.optimize import brentq

    def g(t: float) -> float:
        return conversion_at(config, t) - target

    hi = 60.0
    while g(hi) < 0.0 and hi < 1e8:
        hi *= 4.0
    if g(hi) < 0.0:
        return float("inf")
    return float(brentq(g, 0.0, hi, xtol=1.0, rtol=1e-6))
