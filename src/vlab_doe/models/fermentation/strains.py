"""Lactic-acid-bacteria strains and consortia for milk fermentation.

A :class:`Strain` bundles the kinetic personality of one organism — how fast it grows
(``mu_max``), its temperature window (cardinal temperatures), how much acid it tolerates
before stalling (``ph_min``), how efficiently it turns lactose into biomass and lactic acid,
and how much aroma it throws off.  Strains differ mostly in these few numbers, and those
differences are exactly what a strain-selection DoE is trying to resolve.

The yogurt presets follow the textbook division of labour:

* :func:`streptococcus_thermophilus` (ST) — the fast early acidifier.  Grows quickly, but is
  acid-sensitive (``ph_min`` ~ 4.7) so on its own it stalls before a full set.
* :func:`lactobacillus_bulgaricus` (LB) — slower to start, acid-tolerant (``ph_min`` ~ 3.8),
  so it drives the final acidification and the post-acidification overshoot, and it makes most
  of the acetaldehyde that gives yogurt its flavour.
* :func:`lactobacillus_acidophilus` / :func:`bifidobacterium` — probiotic adjuncts that grow
  slowly and acidify weakly on their own.

In a real yogurt culture ST and LB stimulate each other (*proto-cooperation*): ST releases
formate / CO₂ that LB needs, and LB's proteolysis frees peptides that ST needs.  A
:class:`Consortium` carries this as an interaction matrix; :func:`yogurt_blend` wires up the
canonical ST↔LB pair with that mutual stimulation switched on.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Strain:
    """Kinetic parameters for a single lactic-acid-bacterium.

    Parameters
    ----------
    name:
        Human-readable label (used in results and plots).
    mu_max:
        Maximum specific growth rate at the optimum temperature (1/h).
    t_min, t_opt, t_max:
        Cardinal temperatures (°C) of the Rosso growth-rate model.  Growth is zero outside
        ``[t_min, t_max]`` and peaks at ``t_opt``.
    k_s:
        Monod half-saturation constant for lactose (g/L).
    yield_biomass:
        Biomass produced per gram of lactose consumed (gX/gS).
    ph_min:
        pH at which growth stops; lower means more acid-tolerant.  Acidification continues
        (via maintenance) only while the strain is still growing, so ``ph_min`` sets how far
        a strain can drive the pH on its own.
    acid_growth, acid_maintenance:
        Luedeking–Piret lactic-acid coefficients: growth-associated (mmol acid per unit
        biomass formed) and non-growth maintenance (mmol acid per unit biomass per hour).
    aroma_yield:
        Acetaldehyde-proxy produced per unit biomass formed (arbitrary flavour units).
    x_max:
        Carrying capacity for this strain's biomass (same units as inoculum, dimensionless
        scaled CFU).
    lag_state:
        Baranyi physiological state ``Q0`` at inoculation.  Small ``Q0`` → long lag; the lag
        time is ``ln(1 + 1/Q0) / (mu_max·gamma_T)``.
    """

    name: str
    mu_max: float
    t_min: float
    t_opt: float
    t_max: float
    k_s: float = 5.0
    yield_biomass: float = 0.10
    ph_min: float = 4.2
    acid_growth: float = 90.0
    acid_maintenance: float = 1.5
    aroma_yield: float = 0.2
    x_max: float = 1.0
    lag_state: float = 0.05


def cardinal_temperature_factor(temperature: float, strain: Strain) -> float:
    """Rosso cardinal-temperature growth factor ``gamma_T`` ∈ [0, 1].

    Implements the Rosso (1993) cardinal-temperature model with inflection::

        gamma = (T-Tmin)²(T-Tmax)
              / [ (Topt-Tmin)·( (Topt-Tmin)(T-Topt)
                                - (Topt-Tmax)(Topt+Tmin-2T) ) ]

    The factor is 1 at ``t_opt`` and 0 outside ``[t_min, t_max]``; multiply ``mu_max`` by it
    to get the temperature-corrected growth rate.

    Parameters
    ----------
    temperature:
        Incubation temperature (°C).
    strain:
        Strain whose cardinal temperatures are used.
    """
    T = float(temperature)
    Tmin, Topt, Tmax = strain.t_min, strain.t_opt, strain.t_max
    if T <= Tmin or T >= Tmax:
        return 0.0
    numerator = (T - Tmin) ** 2 * (T - Tmax)
    denominator = (Topt - Tmin) * (
        (Topt - Tmin) * (T - Topt) - (Topt - Tmax) * (Topt + Tmin - 2.0 * T)
    )
    if denominator == 0.0:
        return 0.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))


# ── Yogurt strain presets ──────────────────────────────────────────────────────

def streptococcus_thermophilus(**overrides) -> Strain:
    """*Streptococcus thermophilus* — fast, acid-sensitive early acidifier."""
    base = dict(
        name="S. thermophilus",
        mu_max=1.4,
        t_min=18.0, t_opt=42.0, t_max=50.0,
        k_s=4.0,
        yield_biomass=0.12,
        ph_min=4.7,          # stalls relatively early
        acid_growth=125.0,
        acid_maintenance=1.0,
        aroma_yield=0.10,
        x_max=1.0,
        lag_state=0.5,       # short lag
    )
    base.update(overrides)
    return Strain(**base)


def lactobacillus_bulgaricus(**overrides) -> Strain:
    """*Lactobacillus delbrueckii* subsp. *bulgaricus* — slow, acid-tolerant finisher."""
    base = dict(
        name="L. bulgaricus",
        mu_max=1.1,
        t_min=20.0, t_opt=45.0, t_max=52.0,
        k_s=6.0,
        yield_biomass=0.09,
        ph_min=3.8,          # drives the final / post-acidification
        acid_growth=135.0,
        acid_maintenance=2.5,
        aroma_yield=0.45,    # most of the acetaldehyde
        x_max=1.0,
        lag_state=0.3,       # longer lag
    )
    base.update(overrides)
    return Strain(**base)


def lactobacillus_acidophilus(**overrides) -> Strain:
    """*Lactobacillus acidophilus* — slow probiotic adjunct, weak acidifier."""
    base = dict(
        name="L. acidophilus",
        mu_max=0.45,
        t_min=20.0, t_opt=38.0, t_max=48.0,
        k_s=7.0,
        yield_biomass=0.08,
        ph_min=4.0,
        acid_growth=70.0,
        acid_maintenance=1.0,
        aroma_yield=0.15,
        x_max=1.0,
        lag_state=0.03,
    )
    base.update(overrides)
    return Strain(**base)


def bifidobacterium(**overrides) -> Strain:
    """*Bifidobacterium* spp. — slow probiotic adjunct."""
    base = dict(
        name="Bifidobacterium",
        mu_max=0.40,
        t_min=20.0, t_opt=39.0, t_max=47.0,
        k_s=8.0,
        yield_biomass=0.07,
        ph_min=4.3,
        acid_growth=60.0,
        acid_maintenance=0.8,
        aroma_yield=0.10,
        x_max=1.0,
        lag_state=0.03,
    )
    base.update(overrides)
    return Strain(**base)


# ── Consortium ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Consortium:
    """A community of strains inoculated together.

    Parameters
    ----------
    strains:
        Ordered list of :class:`Strain` members.
    fractions:
        Inoculum fraction of each strain (relative composition of the starter); summed to the
        total inoculum at simulation time.  Need not be normalised — it is normalised on use.
    interaction:
        Square stimulation matrix ``k`` (shape ``(n, n)``).  Entry ``k[i, j]`` is how strongly
        strain *j* stimulates strain *i*: its growth rate is multiplied by
        ``1 + k[i, j]·X_j/(X_j + interaction_half)``.  Zero ⇒ no effect (the default for
        independent strains); positive ⇒ proto-cooperation.
    interaction_half:
        Half-saturation biomass for the stimulation term (same units as biomass).
    """

    strains: list[Strain]
    fractions: np.ndarray
    interaction: np.ndarray
    interaction_half: float = 0.2

    def __post_init__(self) -> None:
        n = len(self.strains)
        frac = np.atleast_1d(np.asarray(self.fractions, dtype=float))
        inter = np.asarray(self.interaction, dtype=float).reshape(n, n)
        if frac.shape != (n,):
            raise ValueError(f"fractions must have length {n}, got {frac.shape}")
        # Frozen dataclass: assign normalised arrays via object.__setattr__.
        object.__setattr__(self, "fractions", frac)
        object.__setattr__(self, "interaction", inter)

    @property
    def n_strains(self) -> int:
        return len(self.strains)

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.strains]

    def normalized_fractions(self) -> np.ndarray:
        """Inoculum fractions normalised to sum to 1."""
        total = float(self.fractions.sum())
        if total <= 0.0:
            raise ValueError("inoculum fractions must sum to a positive value")
        return self.fractions / total

    def with_fractions(self, fractions) -> "Consortium":
        """Return a copy with new (un-normalised) inoculum fractions."""
        return replace(self, fractions=np.asarray(fractions, dtype=float))


def single_strain(strain: Strain) -> Consortium:
    """A consortium of one strain (no interactions)."""
    return Consortium(strains=[strain], fractions=np.array([1.0]), interaction=np.zeros((1, 1)))


def yogurt_blend(
    fraction_st: float = 0.5,
    fraction_lb: float = 0.5,
    *,
    cooperation: float = 1.2,
    st: Strain | None = None,
    lb: Strain | None = None,
) -> Consortium:
    """Canonical *S. thermophilus* + *L. bulgaricus* yogurt culture.

    The two members are wired with symmetric proto-cooperation of strength ``cooperation``:
    each one's growth rate is boosted by the presence of the other.  Set ``cooperation=0`` to
    recover two independent strains (useful as a control to isolate the symbiosis effect).

    Parameters
    ----------
    fraction_st, fraction_lb:
        Relative inoculum fractions of ST and LB.
    cooperation:
        Mutual stimulation coefficient ``k`` for both off-diagonal entries.
    st, lb:
        Optional custom strains; default to the standard presets.
    """
    st = st if st is not None else streptococcus_thermophilus()
    lb = lb if lb is not None else lactobacillus_bulgaricus()
    inter = np.array([[0.0, cooperation], [cooperation, 0.0]])
    return Consortium(
        strains=[st, lb],
        fractions=np.array([fraction_st, fraction_lb], dtype=float),
        interaction=inter,
    )


# ── Strain library (large combinatorial screens) ────────────────────────────────

@dataclass(frozen=True)
class StrainLibrary:
    """A pool of candidate strains plus the global pairwise interaction matrix.

    A screening campaign draws small combinations from a large pool; this holds the whole pool
    once and hands out a :class:`Consortium` for any subset (carrying the matching submatrix of
    interactions).  Build a synthetic one with :func:`random_strain_library`.

    Attributes
    ----------
    strains:
        The candidate strains (length ``n``).
    interaction:
        Global ``(n, n)`` stimulation matrix with the same meaning as
        :attr:`Consortium.interaction`; entry ``[i, j]`` is how strongly *j* affects *i*
        (positive = cooperation, negative = antagonism).  The diagonal is zero.
    interaction_half:
        Half-saturation biomass passed through to every consortium.
    """

    strains: list[Strain]
    interaction: np.ndarray
    interaction_half: float = 0.2

    def __post_init__(self) -> None:
        n = len(self.strains)
        inter = np.asarray(self.interaction, dtype=float).reshape(n, n)
        object.__setattr__(self, "interaction", inter)

    @property
    def n_strains(self) -> int:
        return len(self.strains)

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.strains]

    def consortium(self, indices, fractions=None) -> Consortium:
        """Build a :class:`Consortium` from a subset of the library.

        Parameters
        ----------
        indices:
            Iterable of library indices selecting the strains in this combination.
        fractions:
            Optional inoculum fractions (defaults to an equal split).
        """
        idx = list(indices)
        if len(idx) == 0:
            raise ValueError("need at least one strain in the consortium")
        strains = [self.strains[i] for i in idx]
        sub = self.interaction[np.ix_(idx, idx)]
        if fractions is None:
            fractions = np.full(len(idx), 1.0 / len(idx))
        return Consortium(
            strains=strains,
            fractions=np.asarray(fractions, dtype=float),
            interaction=sub,
            interaction_half=self.interaction_half,
        )


def random_strain_library(
    n: int,
    rng: np.random.Generator,
    *,
    n_cooperative_pairs: int | None = None,
    n_antagonistic_pairs: int | None = None,
) -> StrainLibrary:
    """Generate a synthetic library of ``n`` candidate strains with planted interactions.

    Each strain's kinetics are drawn from realistic thermophilic-LAB ranges, so the library
    spans strong fast acidifiers through weak slow ones — exactly the heterogeneity a screen is
    meant to resolve.  A sparse set of cooperative (positive) and antagonistic (negative)
    interaction pairs is planted as ground truth that downstream analysis should recover.

    Parameters
    ----------
    n:
        Number of strains.
    rng:
        Seeded generator (from :func:`vlab_doe.config.make_rng`).
    n_cooperative_pairs, n_antagonistic_pairs:
        How many positive / negative interaction pairs to plant.  Default ≈ ``n//5`` and
        ``n//10``.
    """
    if n_cooperative_pairs is None:
        n_cooperative_pairs = max(1, n // 5)
    if n_antagonistic_pairs is None:
        n_antagonistic_pairs = max(1, n // 10)

    strains: list[Strain] = []
    for i in range(n):
        t_opt = float(rng.uniform(39.0, 46.0))
        strains.append(
            Strain(
                name=f"strain_{i:02d}",
                mu_max=float(rng.uniform(0.6, 1.5)),
                t_min=t_opt - float(rng.uniform(18.0, 26.0)),
                t_opt=t_opt,
                t_max=t_opt + float(rng.uniform(4.0, 8.0)),
                k_s=float(rng.uniform(3.0, 8.0)),
                yield_biomass=float(rng.uniform(0.07, 0.13)),
                ph_min=float(rng.uniform(3.7, 4.9)),          # acid tolerance spread
                acid_growth=float(rng.uniform(80.0, 145.0)),  # acidifying power spread
                acid_maintenance=float(rng.uniform(0.5, 3.0)),
                aroma_yield=float(rng.uniform(0.05, 0.5)),
                x_max=1.0,
                lag_state=float(rng.uniform(0.1, 0.6)),
            )
        )

    interaction = np.zeros((n, n))
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(all_pairs)
    cursor = 0
    for (i, j) in all_pairs[cursor:cursor + n_cooperative_pairs]:
        k = float(rng.uniform(0.8, 1.8))                      # mutual cooperation
        interaction[i, j] = interaction[j, i] = k
    cursor += n_cooperative_pairs
    for (i, j) in all_pairs[cursor:cursor + n_antagonistic_pairs]:
        k = float(rng.uniform(-0.8, -0.4))                    # mutual antagonism (> -1)
        interaction[i, j] = interaction[j, i] = k

    return StrainLibrary(strains=strains, interaction=interaction)
