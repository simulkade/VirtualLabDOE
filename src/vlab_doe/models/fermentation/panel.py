"""A 16-strain yogurt culture panel with hand-planted pairwise interactions.

:func:`random_strain_library` draws an anonymous pool from uniform ranges — fine for a large
screening demo, but useless as a *ground truth* for a strain-selection study, because nothing
about it is interpretable and the planted interactions are arbitrary.  This module is the
opposite: a small, fixed, named panel of **16 candidate cultures** whose kinetics were written
by hand to look like a real dairy-culture collection, and whose interaction matrix encodes the
mechanisms a dairy microbiologist would actually name:

* **ST — *Streptococcus thermophilus*** (5 isolates): fast, short lag, acid-*sensitive*
  (``ph_min`` 4.6–4.8), so on their own they stall around pH 4.6–4.8 and only just reach (or
  miss) the set point.
* **LB — *Lactobacillus delbrueckii* subsp. *bulgaricus*** (4 isolates): slower and longer lag,
  but acid-tolerant (``ph_min`` 3.85–4.05) and strongly proteolytic — the finishers.
* **LH — *Lactobacillus helveticus*** (2 isolates): very acid-tolerant, highly proteolytic;
  they liberate peptides that ST needs, so they *stimulate ST one-directionally*.
* **LA / BB — probiotic adjuncts** (*L. acidophilus*, *Bifidobacterium*, 3 isolates): slow, weak
  acidifiers that cannot set milk alone in the incubation window.
* **LL — *Lactococcus lactis*** (2 isolates): mesophilic. At the 43 °C incubation temperature
  their Rosso factor is **zero** — they are realistic "dud" candidates that a screen must not be
  fooled by.

Interactions (:data:`PANEL_INTERACTIONS`) are of three kinds, all of them real yogurt
microbiology:

1. **Proto-cooperation (ST ↔ LB/LH).**  ST releases formate/CO₂ and pyruvate that the
   lactobacilli need; the lactobacilli's proteases free the peptides ST needs.  Every ST×LB pair
   gets a modest baseline boost, and a few specific isolate pairs are much better matched than
   the rest — exactly the strain-specificity that makes pair screening worth doing.
2. **Asymmetric stimulation (LH → ST).**  Proteolysis is a one-way favour: LH feeds ST, ST does
   little for LH.  The interaction matrix is therefore *not symmetric*.
3. **Antagonism.**  LB-04 is a bacteriocin producer that inhibits ST-05 and the probiotic
   adjuncts; two closely related ST isolates compete for the same niche.

The panel is calibrated so that the champion pair (:data:`CHAMPION_PAIR`) is **not** the pair of
the two individually best strains: it wins through synergy.  Any design that only looks at
single strains, or that fits main effects without interactions, is therefore guaranteed to miss
it — which is the whole point of the accompanying pair-screening study
(:mod:`vlab_doe.doe.combination_search`).

Quick start::

    from vlab_doe.models.fermentation.panel import yogurt_strain_panel, CHAMPION_PAIR
    library = yogurt_strain_panel()
    print(library.names, CHAMPION_PAIR)
"""

from __future__ import annotations

import numpy as np

from .strains import Strain, StrainLibrary

#: The pair the panel is calibrated to make the fastest acidifier (ground truth).
CHAMPION_PAIR: tuple[str, str] = ("ST-03", "LB-02")

#: Baseline mutual proto-cooperation between *any* ST isolate and *any* LB/LH isolate.
BASELINE_PROTO_COOPERATION: float = 0.30

# ── The 16 candidate cultures ──────────────────────────────────────────────────
# Every entry is a Strain kwargs dict.  Values are in the ranges used by the
# textbook presets in strains.py, spread to look like a real culture collection:
# five ST isolates that differ mostly in rate and acid tolerance, four LB isolates
# that differ in lag and acidifying power, and a tail of adjuncts and duds.
_PANEL: tuple[dict, ...] = (
    # ── S. thermophilus: fast, short lag, acid-sensitive ──
    dict(name="ST-01", mu_max=1.45, t_min=18.0, t_opt=42.0, t_max=50.0, k_s=4.0,
         yield_biomass=0.125, ph_min=4.75, acid_growth=130.0, acid_maintenance=1.0,
         aroma_yield=0.10, lag_state=0.55),
    dict(name="ST-02", mu_max=1.30, t_min=19.0, t_opt=43.0, t_max=50.5, k_s=4.2,
         yield_biomass=0.120, ph_min=4.65, acid_growth=121.0, acid_maintenance=1.1,
         aroma_yield=0.12, lag_state=0.45),
    dict(name="ST-03", mu_max=1.18, t_min=18.5, t_opt=42.5, t_max=49.5, k_s=4.5,
         yield_biomass=0.115, ph_min=4.70, acid_growth=116.0, acid_maintenance=0.9,
         aroma_yield=0.11, lag_state=0.40),
    dict(name="ST-04", mu_max=1.08, t_min=17.5, t_opt=41.0, t_max=48.5, k_s=5.0,
         yield_biomass=0.110, ph_min=4.80, acid_growth=108.0, acid_maintenance=0.8,
         aroma_yield=0.09, lag_state=0.30),
    dict(name="ST-05", mu_max=1.36, t_min=19.5, t_opt=44.0, t_max=51.0, k_s=3.8,
         yield_biomass=0.122, ph_min=4.60, acid_growth=126.0, acid_maintenance=1.2,
         aroma_yield=0.13, lag_state=0.50),
    # ── L. bulgaricus: slower, longer lag, acid-tolerant, aromatic ──
    dict(name="LB-01", mu_max=1.05, t_min=20.0, t_opt=45.0, t_max=52.0, k_s=6.0,
         yield_biomass=0.092, ph_min=3.85, acid_growth=138.0, acid_maintenance=2.5,
         aroma_yield=0.45, lag_state=0.25),
    dict(name="LB-02", mu_max=0.96, t_min=20.5, t_opt=44.5, t_max=51.5, k_s=6.4,
         yield_biomass=0.088, ph_min=3.90, acid_growth=132.0, acid_maintenance=2.2,
         aroma_yield=0.42, lag_state=0.20),
    dict(name="LB-03", mu_max=1.14, t_min=21.0, t_opt=45.5, t_max=52.5, k_s=5.6,
         yield_biomass=0.095, ph_min=3.95, acid_growth=142.0, acid_maintenance=2.8,
         aroma_yield=0.50, lag_state=0.30),
    dict(name="LB-04", mu_max=0.86, t_min=20.0, t_opt=44.0, t_max=51.0, k_s=6.8,
         yield_biomass=0.085, ph_min=4.05, acid_growth=128.0, acid_maintenance=2.0,
         aroma_yield=0.38, lag_state=0.15),
    # ── L. helveticus: very acid-tolerant, strongly proteolytic ──
    dict(name="LH-01", mu_max=1.00, t_min=21.0, t_opt=44.0, t_max=51.0, k_s=6.2,
         yield_biomass=0.090, ph_min=3.75, acid_growth=140.0, acid_maintenance=2.6,
         aroma_yield=0.30, lag_state=0.20),
    dict(name="LH-02", mu_max=0.90, t_min=20.5, t_opt=43.0, t_max=50.0, k_s=6.6,
         yield_biomass=0.087, ph_min=3.80, acid_growth=134.0, acid_maintenance=2.4,
         aroma_yield=0.28, lag_state=0.18),
    # ── Probiotic adjuncts: slow, weak acidifiers ──
    dict(name="LA-01", mu_max=0.50, t_min=20.0, t_opt=38.0, t_max=48.0, k_s=7.0,
         yield_biomass=0.080, ph_min=4.05, acid_growth=75.0, acid_maintenance=1.0,
         aroma_yield=0.15, lag_state=0.05),
    dict(name="LA-02", mu_max=0.45, t_min=20.0, t_opt=37.5, t_max=47.0, k_s=7.4,
         yield_biomass=0.078, ph_min=4.10, acid_growth=70.0, acid_maintenance=0.9,
         aroma_yield=0.14, lag_state=0.04),
    dict(name="BB-01", mu_max=0.42, t_min=20.0, t_opt=39.0, t_max=47.0, k_s=8.0,
         yield_biomass=0.072, ph_min=4.30, acid_growth=62.0, acid_maintenance=0.8,
         aroma_yield=0.10, lag_state=0.04),
    # ── Mesophilic Lactococcus: no growth at all at 43 °C (realistic duds) ──
    dict(name="LL-01", mu_max=0.90, t_min=8.0, t_opt=30.0, t_max=40.0, k_s=5.0,
         yield_biomass=0.100, ph_min=4.40, acid_growth=110.0, acid_maintenance=1.0,
         aroma_yield=0.20, lag_state=0.30),
    dict(name="LL-02", mu_max=0.85, t_min=6.0, t_opt=28.0, t_max=39.0, k_s=5.4,
         yield_biomass=0.098, ph_min=4.45, acid_growth=105.0, acid_maintenance=1.0,
         aroma_yield=0.22, lag_state=0.28),
)

#: Named interactions on top of the ST×LB/LH baseline.  Each entry is
#: ``(receiver, donor, k, symmetric)``: the *receiver*'s growth rate is multiplied by
#: ``1 + k·X_donor/(X_donor + half)``.  ``symmetric`` mirrors the entry back.
PANEL_INTERACTIONS: tuple[tuple[str, str, float, bool], ...] = (
    # 1. Strain-specific proto-cooperation on top of the ST×LB baseline.
    #    The champion pair is exceptionally well matched: ST-03's formate output happens to suit
    #    LB-02, whose proteolysis suits ST-03.
    ("ST-03", "LB-02", 3.00, True),
    #    A decoy: two individually strong isolates that also cooperate, but less well.
    ("ST-01", "LB-03", 0.85, True),
    ("ST-02", "LB-01", 0.55, True),
    ("ST-05", "LB-03", 0.45, True),
    # 2. Asymmetric proteolytic stimulation: L. helveticus feeds every ST isolate; ST gives
    #    little back.
    ("ST-01", "LH-01", 0.80, False),
    ("ST-02", "LH-01", 0.80, False),
    ("ST-03", "LH-01", 0.75, False),
    ("ST-04", "LH-01", 0.85, False),
    ("ST-05", "LH-01", 0.70, False),
    ("ST-01", "LH-02", 0.55, False),
    ("ST-02", "LH-02", 0.55, False),
    ("ST-03", "LH-02", 0.50, False),
    ("ST-04", "LH-02", 0.60, False),
    ("ST-05", "LH-02", 0.50, False),
    ("LH-01", "ST-02", 0.10, False),
    ("LH-02", "ST-02", 0.10, False),
    # 3. Antagonism: LB-04 is a bacteriocin producer; ST-02/ST-05 compete for the same niche.
    ("ST-05", "LB-04", -0.70, False),
    ("ST-01", "LB-04", -0.35, False),
    ("LA-01", "LB-04", -0.60, False),
    ("BB-01", "LB-01", -0.45, False),
    ("ST-05", "ST-02", -0.25, True),
    # 4. A probiotic adjunct that ST-02 happens to support (relevant for adjunct blends).
    ("LA-01", "ST-02", 0.50, False),
)


def _strain_group(name: str) -> str:
    """Culture group prefix (``"ST"``, ``"LB"``, ``"LH"``, ``"LA"``, ``"BB"``, ``"LL"``)."""
    return name.split("-")[0]


def panel_interaction_matrix(
    names: list[str],
    *,
    baseline_cooperation: float = BASELINE_PROTO_COOPERATION,
    champion_synergy: float | None = None,
    interaction_scale: float = 1.0,
) -> np.ndarray:
    """Assemble the ``(16, 16)`` interaction matrix from the baseline plus named entries.

    Parameters
    ----------
    names:
        Strain names in library order.
    baseline_cooperation:
        Mutual boost applied to *every* ST × (LB or LH) pair before the named entries — the
        generic, non-strain-specific part of proto-cooperation.
    champion_synergy:
        Optional override for the strength of the :data:`CHAMPION_PAIR` entry, used to tune how
        far ahead of the field the ground-truth pair sits.
    interaction_scale:
        Multiplies the *whole* finished matrix.  ``1.0`` is the calibrated panel; ``0.0`` makes
        the strains behave purely additively.  Sweeping it is how the covering-array study asks
        the question those designs are normally asked in practice — *how strong may the pairwise
        interactions get before a pair-coverage design stops working?*
    """
    index = {name: i for i, name in enumerate(names)}
    n = len(names)
    inter = np.zeros((n, n))

    # Generic ST <-> LB / LH proto-cooperation.
    for a in names:
        for b in names:
            if a == b:
                continue
            ga, gb = _strain_group(a), _strain_group(b)
            if (ga == "ST" and gb in ("LB", "LH")) or (gb == "ST" and ga in ("LB", "LH")):
                inter[index[a], index[b]] = baseline_cooperation

    # Named, strain-specific entries (they replace the baseline for those cells).
    champion = tuple(CHAMPION_PAIR)
    for receiver, donor, k, symmetric in PANEL_INTERACTIONS:
        if champion_synergy is not None and tuple(sorted((receiver, donor))) == tuple(sorted(champion)):
            k = champion_synergy
        inter[index[receiver], index[donor]] = k
        if symmetric:
            inter[index[donor], index[receiver]] = k

    np.fill_diagonal(inter, 0.0)
    return inter * float(interaction_scale)


def yogurt_strain_panel(
    *,
    baseline_cooperation: float = BASELINE_PROTO_COOPERATION,
    champion_synergy: float | None = None,
    interaction_scale: float = 1.0,
    interaction_half: float = 0.08,
) -> StrainLibrary:
    """Build the fixed 16-strain candidate panel with its planted interactions.

    Parameters
    ----------
    baseline_cooperation, champion_synergy, interaction_scale:
        Passed to :func:`panel_interaction_matrix`.
    interaction_half:
        Half-saturation biomass of the stimulation term (see :class:`~.strains.Consortium`).
        The default (0.08, well below the carrying capacity) makes cooperation switch on early
        in the batch, while the strains are still growing — which is when it can actually change
        the time to set point.
    """
    strains = [Strain(**spec) for spec in _PANEL]
    names = [s.name for s in strains]
    inter = panel_interaction_matrix(
        names,
        baseline_cooperation=baseline_cooperation,
        champion_synergy=champion_synergy,
        interaction_scale=interaction_scale,
    )
    return StrainLibrary(strains=strains, interaction=inter, interaction_half=interaction_half)


def panel_groups() -> dict[str, str]:
    """Map each panel strain name to its culture group (for plotting / grouping)."""
    return {spec["name"]: _strain_group(str(spec["name"])) for spec in _PANEL}


def champion_indices(library: StrainLibrary) -> tuple[int, int]:
    """Library indices of the ground-truth :data:`CHAMPION_PAIR`, sorted."""
    names = library.names
    i, j = (names.index(CHAMPION_PAIR[0]), names.index(CHAMPION_PAIR[1]))
    return (i, j) if i < j else (j, i)
