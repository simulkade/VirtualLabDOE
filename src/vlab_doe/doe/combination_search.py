"""Finding the best *pair* out of a strain panel, under an experiment budget.

The question this module answers is the everyday one in strain selection: *given 16 candidate
cultures, which two, blended 50/50, give the best time to the set pH — and how few
fermentations do I have to run to find them?*  With 16 strains there are only
:math:`\\binom{16}{2} = 120` pairs, so brute force is *possible* — but each run is a real batch
(here ~6–12 h of incubation), the readout is noisy, and 120 batches is a month of lab work.  The
interesting question is which **design** finds the winner with a fraction of that.

Two things make it hard, and both are modelled explicitly:

* **Interactions.**  The best pair is not the pair of the two individually best strains — it
  wins through proto-cooperation (see :mod:`vlab_doe.models.fermentation.panel`).  Any design
  that only screens single strains, or fits a main-effects-only model, is structurally unable
  to find it.
* **Noise.**  One batch carries biological batch-to-batch variability *and* pH-probe error, so
  the observed set time of a given blend scatters by ~0.5–1 h.  A design that ranks 120 pairs
  from one replicate each can pick the wrong winner even after spending the full budget —
  which is why replication and racing designs are in the comparison.

Contents
--------
:class:`VirtualStrainLab`
    The bench: turns a strain combination into one noisy batch, counts experiments against the
    budget, caches draws (so repeated queries are cheap and *common random numbers* line up
    across strategies), and knows the noise-free truth for scoring.
Search strategies
    :func:`exhaustive_pair_scan`, :func:`random_pair_search`, :func:`single_strain_screen`,
    :func:`covering_scheffe_search`, :func:`d_optimal_scheffe_search`,
    :func:`thompson_scheffe_search`, :func:`successive_halving_search`,
    :func:`greedy_exchange_search` — all with the same signature
    ``(lab, budget, rng) -> SearchResult``, so they can be benchmarked head to head.
:func:`benchmark_strategies`
    Runs every strategy at every budget over repeated virtual campaigns and scores each one
    against the ground truth (did it find the champion, what is the true rank of its pick, how
    much slower is its pick than the true best).

The mixture-model strategies share one statistical idea worth naming: a blend of strains is a
**mixture experiment**, so the natural model is the *Scheffé quadratic*

.. math::  y = \\sum_i b_i x_i + \\sum_{i<j} b_{ij} x_i x_j

in the inoculum fractions :math:`x` (which sum to 1, hence no intercept).  The
:math:`b_{ij}` are exactly the pairwise synergies the study is hunting, and because a run with
``m`` strains has :math:`x_i = 1/m`, runs of different sizes all inform the same coefficients —
that is what lets a design of 2- and 3-strain blends say something about all 120 pairs.

Two details decide whether that model is any use, and both are easy to get wrong:

* **Fit the rate, not the time.**  About 40% of pairs never reach the set pH inside the horizon,
  so ``t_set`` is *censored* and piles up at a single value that drags the linear fit around.
  The model-based strategies therefore regress the acidification rate :math:`1/t_{set}`
  (:func:`acidification_rate`), where "never got there" is a smooth approach to zero, and
  convert predictions back with :func:`rate_to_objective`.
* **Scale the columns before penalising them.**  The interaction columns are
  :math:`O(1/m^2)` against :math:`O(1/m)` for the main effects, so an unscaled LASSO penalty
  shrinks precisely the synergy terms the study exists to find.  :func:`fit_scheffe`
  normalises the columns first.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV

from ..models.fermentation.engine import FermentationSetup, run_fermentation
from ..models.fermentation.metrics import time_to_ph
from ..models.fermentation.milk import Milk
from ..models.fermentation.observe import DEFAULT_PH_NOISE, observe_ph
from ..models.fermentation.strains import StrainLibrary
from ..models.fermentation.variability import BatchVariability, sample_batch
from ..perturbation import NoiseModel
from .covering import covering_array

Members = tuple[int, ...]


# ── The bench ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Experiment:
    """One virtual fermentation: what was inoculated and what the probe said.

    Attributes
    ----------
    members:
        Library indices of the strains in the blend (equal inoculum split).
    replicate:
        0-based replicate index of this exact blend (each replicate is an independent batch).
    t_set:
        Measured time to the target pH (h), **censored at the incubation horizon** when the
        blend never gets there.
    censored:
        Whether the set point was missed inside the horizon.
    final_ph:
        Measured pH at the end of incubation — free extra information from the same batch, used
        by strategies to rank blends that all failed to set.
    value:
        The objective being minimised (see :class:`VirtualStrainLab`): the set time itself, or
        its distance to a target set time.
    """

    members: Members
    replicate: int
    t_set: float
    censored: bool
    final_ph: float
    value: float


@dataclass
class VirtualStrainLab:
    """A budgeted virtual bench for strain-combination experiments.

    Every :meth:`run` is one batch: the setup is perturbed by biological batch-to-batch
    variability, integrated, and read out by a noisy pH probe on a realistic sampling grid.
    Draws are cached on ``(members, replicate)`` and seeded from the lab seed, so

    * re-reading an experiment is free (but still counts against the budget only when it is a
      *new* replicate),
    * two strategies that happen to run the same blend see the *same* batch — common random
      numbers, which makes budget-for-budget comparisons far less noisy.

    Parameters
    ----------
    library:
        The candidate strain panel.
    temperature, total_inoculum, milk:
        Fixed process conditions — this study varies only the strain combination.
    horizon:
        Incubation window (h).  Blends that never reach the target pH are censored here.
    target_ph:
        Set-point pH the time is measured to (4.6 for yogurt).
    target_time:
        ``None`` (default) → the objective is the set time itself, *lower is better*.  A number
        → the objective is ``|t_set - target_time|``, i.e. hit a target fermentation length.
    probe_interval:
        Spacing of the pH probe samples (h).
    variability, noise:
        The biological and measurement noise layers.  Set ``variability=None`` and
        ``noise=None`` for a noise-free lab (useful for teaching plots).
    seed:
        Master seed; every batch draws its own generator from ``(seed, members, replicate)``.
    """

    library: StrainLibrary
    temperature: float = 43.0
    total_inoculum: float = 0.05
    milk: Milk = field(default_factory=Milk)
    horizon: float = 12.0
    target_ph: float = 4.6
    target_time: float | None = None
    probe_interval: float = 0.25
    n_grid: int = 241
    variability: BatchVariability | None = field(default_factory=BatchVariability)
    noise: NoiseModel | None = DEFAULT_PH_NOISE
    seed: int = 0

    # ── bookkeeping (not user-set) ──
    history: list[Experiment] = field(default_factory=list, init=False)
    _cache: dict[tuple[Members, int], Experiment] = field(default_factory=dict, init=False)
    _reps: dict[Members, int] = field(default_factory=dict, init=False)
    _truth: dict[Members, float] = field(default_factory=dict, init=False)

    # ── grids ──
    @property
    def t_grid(self) -> np.ndarray:
        """Simulation output grid (h)."""
        return np.linspace(0.0, self.horizon, self.n_grid)

    @property
    def probe_times(self) -> np.ndarray:
        """pH-probe sampling times (h)."""
        return np.arange(0.0, self.horizon + 1e-9, self.probe_interval)

    @property
    def n_strains(self) -> int:
        return self.library.n_strains

    @property
    def n_experiments(self) -> int:
        """How many batches have been run (the budget spent)."""
        return len(self.history)

    def all_pairs(self) -> list[Members]:
        """Every unordered pair of strains, as sorted index tuples."""
        return [tuple(p) for p in combinations(range(self.n_strains), 2)]

    def label(self, members: Iterable[int]) -> str:
        """Human-readable blend label, e.g. ``"ST-03+LB-02"``."""
        names = self.library.names
        return "+".join(names[i] for i in members)

    # ── objective ──
    def _objective(self, t_set: float) -> float:
        if self.target_time is None:
            return float(t_set)
        return float(abs(t_set - self.target_time))

    def _setup(self, members: Members) -> FermentationSetup:
        return FermentationSetup(
            consortium=self.library.consortium(list(members)),
            milk=self.milk,
            temperature=self.temperature,
            total_inoculum=self.total_inoculum,
        )

    # ── running experiments ──
    def run(self, members: Iterable[int]) -> Experiment:
        """Run one fresh batch of ``members`` and charge it to the budget."""
        key = tuple(sorted(int(i) for i in members))
        if not key:
            raise ValueError("a blend needs at least one strain")
        rep = self._reps.get(key, 0)
        self._reps[key] = rep + 1
        exp = self._draw(key, rep)
        self.history.append(exp)
        return exp

    def run_mean(self, members: Iterable[int], n_replicates: int = 1) -> float:
        """Run ``n_replicates`` fresh batches of one blend and return the mean objective."""
        return float(np.mean([self.run(members).value for _ in range(int(n_replicates))]))

    def _draw(self, key: Members, replicate: int) -> Experiment:
        """Simulate (or recall) the ``replicate``-th batch of a blend."""
        cached = self._cache.get((key, replicate))
        if cached is not None:
            return cached

        rng = np.random.default_rng([self.seed, replicate, *key])
        setup = self._setup(key)
        if self.variability is not None:
            setup = sample_batch(setup, self.variability, rng)
        result = run_fermentation(setup, self.t_grid)

        if self.noise is not None:
            obs = observe_ph(result, self.probe_times, self.noise, rng)
            t, ph = obs["t"], obs["ph"]
        else:
            t, ph = result.t, result.ph

        t_set = time_to_ph(t, ph, self.target_ph)
        censored = not np.isfinite(t_set)
        t_set = self.horizon if censored else float(t_set)
        exp = Experiment(
            members=key,
            replicate=replicate,
            t_set=t_set,
            censored=censored,
            final_ph=float(ph[-1]),
            value=self._objective(t_set),
        )
        self._cache[(key, replicate)] = exp
        return exp

    # ── ground truth (free: never charged to the budget) ──
    def truth(self, members: Iterable[int]) -> float:
        """Noise-free objective of a blend — for scoring only, never for searching."""
        key = tuple(sorted(int(i) for i in members))
        if key in self._truth:
            return self._truth[key]
        result = run_fermentation(self._setup(key), self.t_grid)
        t_set = time_to_ph(result.t, result.ph, self.target_ph)
        if not np.isfinite(t_set):
            t_set = self.horizon
        value = self._objective(float(t_set))
        self._truth[key] = value
        return value

    def pair_truth_table(self) -> pd.DataFrame:
        """Noise-free objective for all 120 pairs, ranked best first.

        Columns: ``members``, ``label``, ``a``, ``b``, ``value``, ``rank`` (1 = best).
        """
        pairs = self.all_pairs()
        values = np.array([self.truth(p) for p in pairs])
        order = np.argsort(values, kind="stable")
        names = self.library.names
        df = pd.DataFrame(
            {
                "members": [pairs[k] for k in order],
                "label": [self.label(pairs[k]) for k in order],
                "a": [names[pairs[k][0]] for k in order],
                "b": [names[pairs[k][1]] for k in order],
                "value": values[order],
            }
        )
        df["rank"] = np.arange(1, len(df) + 1)
        return df

    def single_truth(self) -> pd.Series:
        """Noise-free objective of every strain on its own, indexed by strain name."""
        return pd.Series(
            {name: self.truth([i]) for i, name in enumerate(self.library.names)}
        )

    def best_pair(self) -> Members:
        """The ground-truth champion pair."""
        return tuple(self.pair_truth_table().iloc[0]["members"])

    def pair_rank(self, members: Iterable[int]) -> int:
        """True rank (1 = best) of a pair among all 120."""
        key = tuple(sorted(int(i) for i in members))
        table = self.pair_truth_table()
        hit = table.index[[m == key for m in table["members"]]]
        return int(table.loc[hit[0], "rank"])

    def regret(self, members: Iterable[int]) -> float:
        """True objective of a choice minus the true best (h).  0 = the champion."""
        return float(self.truth(members) - self.pair_truth_table().iloc[0]["value"])

    def reset(self) -> None:
        """Clear the budget counter and history, keeping the cached batches and truth."""
        self.history.clear()
        self._reps.clear()


# ── Strategy plumbing ───────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """What a strategy did and what it recommends.

    Attributes
    ----------
    strategy:
        Strategy name.
    choice:
        The recommended pair (library indices).
    n_experiments:
        Batches actually run.
    observed_value:
        The strategy's own estimate of the objective for its choice (noisy).
    trace:
        Per-experiment record: the blend run, its size and its observed value — enough to plot
        how each design spent its budget.
    notes:
        Free-form extras (e.g. model R², design diagnostics).
    """

    strategy: str
    choice: Members
    n_experiments: int
    observed_value: float
    trace: pd.DataFrame
    notes: dict = field(default_factory=dict)


class _Budget:
    """Charge experiments to a fixed budget and refuse to overspend."""

    def __init__(self, lab: VirtualStrainLab, budget: int):
        self.lab = lab
        self.budget = int(budget)
        self.spent = 0
        self.trace: list[dict] = []

    @property
    def left(self) -> int:
        return self.budget - self.spent

    def can(self, n: int = 1) -> bool:
        return self.left >= n

    def run(self, members: Iterable[int]) -> Experiment | None:
        """Run one batch, or return ``None`` if the budget is exhausted."""
        if not self.can():
            return None
        exp = self.lab.run(members)
        self.spent += 1
        self.trace.append(
            {
                "order": self.spent,
                "members": exp.members,
                "label": self.lab.label(exp.members),
                "size": len(exp.members),
                "t_set": exp.t_set,
                "censored": exp.censored,
                "final_ph": exp.final_ph,
                "value": exp.value,
            }
        )
        return exp

    def mean(self, members: Iterable[int], n: int) -> float | None:
        """Run up to ``n`` replicates; returns the mean, or ``None`` if none could be run."""
        vals = []
        for _ in range(int(n)):
            exp = self.run(members)
            if exp is None:
                break
            vals.append(exp.value)
        return float(np.mean(vals)) if vals else None

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.trace)


def _best_by_mean(records: pd.DataFrame) -> tuple[Members, float]:
    """Blend with the lowest mean observed value among 2-strain runs in a trace."""
    pairs = records[records["size"] == 2]
    if pairs.empty:
        pairs = records
    grouped = pairs.groupby("members")["value"].mean()
    key = grouped.idxmin()
    return tuple(key), float(grouped.min())


# ── Strategy 1: exhaustive pair scan (the brute-force reference) ────────────────

def exhaustive_pair_scan(lab: VirtualStrainLab, budget: int, rng: np.random.Generator) -> SearchResult:
    """Run every pair, then spend anything left over replicating the leaders.

    The reference design: no modelling, no adaptivity, 120 batches minimum.  With a budget below
    120 it degenerates to a random subset (which is exactly :func:`random_pair_search`), so the
    interesting comparison is at budget ≥ 120, where the surplus goes into replication — the
    honest way to beat measurement noise.
    """
    b = _Budget(lab, budget)
    pairs = lab.all_pairs()
    rng.shuffle(pairs)
    means: dict[Members, list[float]] = {}
    for p in pairs:
        exp = b.run(p)
        if exp is None:
            break
        means.setdefault(p, []).append(exp.value)

    # Surplus budget: replicate the current leaders (top decile), which is where the ranking is
    # actually decided.
    while b.can():
        ranked = sorted(means, key=lambda k: float(np.mean(means[k])))
        leaders = ranked[: max(2, len(ranked) // 10)]
        progressed = False
        for p in leaders:
            exp = b.run(p)
            if exp is None:
                break
            means[p].append(exp.value)
            progressed = True
        if not progressed:
            break

    choice = min(means, key=lambda k: float(np.mean(means[k])))
    return SearchResult(
        strategy="exhaustive pair scan",
        choice=choice,
        n_experiments=b.spent,
        observed_value=float(np.mean(means[choice])),
        trace=b.frame(),
        notes={"pairs_tested": len(means)},
    )


# ── Strategy 2: random pairs (the do-nothing baseline) ──────────────────────────

def random_pair_search(lab: VirtualStrainLab, budget: int, rng: np.random.Generator) -> SearchResult:
    """Test randomly chosen pairs and keep the best one seen.  The baseline to beat."""
    b = _Budget(lab, budget)
    pairs = lab.all_pairs()
    rng.shuffle(pairs)
    means: dict[Members, list[float]] = {}
    idx = 0
    while b.can():
        p = pairs[idx % len(pairs)]
        idx += 1
        exp = b.run(p)
        if exp is None:
            break
        means.setdefault(p, []).append(exp.value)
    choice = min(means, key=lambda k: float(np.mean(means[k])))
    return SearchResult(
        strategy="random pairs",
        choice=choice,
        n_experiments=b.spent,
        observed_value=float(np.mean(means[choice])),
        trace=b.frame(),
        notes={"pairs_tested": len(means)},
    )


# ── Strategy 3: one-factor-at-a-time — screen singles, then combine winners ─────

def single_strain_screen(
    lab: VirtualStrainLab,
    budget: int,
    rng: np.random.Generator,
    *,
    n_finalists: int | None = None,
) -> SearchResult:
    """Screen all 16 strains alone, then test pairs of the best ones.

    The classical, intuitive campaign — and the one this virtual lab is built to punish.  Solo
    runs measure only *intrinsic* acidifying power; the champion pair wins on synergy, so it is
    invisible here unless both of its members happen to look good alone.  Strains that never set
    on their own are ranked by their final pH, which is free from the same batch.
    """
    b = _Budget(lab, budget)
    singles: dict[int, Experiment] = {}
    for i in range(lab.n_strains):
        exp = b.run([i])
        if exp is None:
            break
        singles[i] = exp

    # Rank: uncensored (actually set) first by objective, then censored ones by how far down the
    # pH they managed to push the milk.
    def key(i: int) -> tuple[int, float, float]:
        e = singles[i]
        return (1 if e.censored else 0, e.value, e.final_ph)

    ranked = sorted(singles, key=key)

    # How many finalists can we afford to cross with each other?
    if n_finalists is None:
        n_finalists = 2
        while len(list(combinations(range(n_finalists + 1), 2))) <= b.left and n_finalists < len(ranked):
            n_finalists += 1
    finalists = ranked[:n_finalists]

    means: dict[Members, list[float]] = {}
    for p in combinations(sorted(finalists), 2):
        exp = b.run(p)
        if exp is None:
            break
        means.setdefault(tuple(p), []).append(exp.value)

    # Leftover budget → replicate the leading pairs.
    while b.can() and means:
        for p in sorted(means, key=lambda k: float(np.mean(means[k])))[:3]:
            exp = b.run(p)
            if exp is None:
                break
            means[p].append(exp.value)

    if means:
        choice = min(means, key=lambda k: float(np.mean(means[k])))
        observed = float(np.mean(means[choice]))
    else:  # budget ran out during the solo screen — best guess is the top two singles
        choice = tuple(sorted(ranked[:2])) if len(ranked) >= 2 else (0, 1)
        observed = float("nan")
    return SearchResult(
        strategy="singles then top pairs",
        choice=choice,
        n_experiments=b.spent,
        observed_value=observed,
        trace=b.frame(),
        notes={"n_finalists": len(finalists),
               "finalists": [lab.library.names[i] for i in finalists]},
    )


# ── Scheffé mixture model shared by the model-based strategies ──────────────────

def scheffe_features(blends: Sequence[Members], n_strains: int) -> np.ndarray:
    """Scheffé quadratic design matrix for a list of equal-split blends.

    Column block 1 is the linear term ``x_i`` (the inoculum fraction of strain *i*, i.e.
    ``1/len(blend)`` if present); block 2 is the binary term ``x_i·x_j`` for every strain pair in
    library order.  Shape ``(n_blends, n_strains + C(n_strains, 2))``.
    """
    pairs = list(combinations(range(n_strains), 2))
    pair_index = {p: k for k, p in enumerate(pairs)}
    X = np.zeros((len(blends), n_strains + len(pairs)))
    for r, members in enumerate(blends):
        m = len(members)
        frac = 1.0 / m
        for i in members:
            X[r, i] = frac
        for i, j in combinations(sorted(members), 2):
            X[r, n_strains + pair_index[(i, j)]] = frac * frac
    return X


def acidification_rate(experiments: Sequence[Experiment]) -> np.ndarray:
    """Model response: the acidification-rate proxy ``1 / t_set`` (1/h).

    Fitting a linear model to the set time itself is a specification error, because ~40 % of
    blends never reach the set point and pile up at the censoring horizon.  The reciprocal is
    the natural fix: it is the *rate* of the process the model is really describing, it is
    uncensored (a blend that never sets simply has a low rate), and it makes the response far
    closer to additive-plus-interaction in the blend fractions.
    """
    return np.array([1.0 / max(e.t_set, 1e-3) for e in experiments], dtype=float)


def fit_scheffe(
    blends: Sequence[Members],
    values: Sequence[float],
    n_strains: int,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float]:
    """Fit the Scheffé quadratic by LASSO (sparse synergies) and return ``(coefs, alpha)``.

    There are ``16 + 120 = 136`` coefficients and only a few dozen runs, so the fit is
    under-determined by construction; the L1 penalty resolves it the way the biology suggests —
    most strain pairs do *nothing* special, a handful matter.  ``alpha`` is chosen by
    cross-validation when there are enough runs, and falls back to a fixed small penalty
    otherwise.

    Columns are scaled to unit norm before the fit and the coefficients scaled back afterwards.
    Without that, the interaction columns (whose entries are ``1/m²``) would be penalised roughly
    ``m`` times harder than the linear ones and the L1 fit would shrink away exactly the synergy
    terms the study is looking for.
    """
    X = scheffe_features(blends, n_strains)
    y = np.asarray(values, dtype=float)
    scale = np.linalg.norm(X, axis=0)
    scale[scale == 0.0] = 1.0
    n = len(y)
    if n >= 12:
        model = LassoCV(cv=min(5, max(3, n // 4)), fit_intercept=False, max_iter=50000,
                        random_state=0 if rng is None else int(rng.integers(1 << 30)))
    else:
        model = Lasso(alpha=1e-3, fit_intercept=False, max_iter=50000)
    model.fit(X / scale, y)
    alpha = float(getattr(model, "alpha_", getattr(model, "alpha", np.nan)))
    return np.asarray(model.coef_, dtype=float) / scale, alpha


def predict_pairs(coefs: np.ndarray, n_strains: int) -> np.ndarray:
    """Model prediction for every pair, in :func:`itertools.combinations` order."""
    pairs = [tuple(p) for p in combinations(range(n_strains), 2)]
    return scheffe_features(pairs, n_strains) @ coefs


def rate_to_objective(lab: VirtualStrainLab, predicted_rate: np.ndarray) -> np.ndarray:
    """Convert predicted acidification rates into the lab's objective (lower = better)."""
    t_hat = 1.0 / np.clip(predicted_rate, 1.0 / (10.0 * lab.horizon), None)
    t_hat = np.clip(t_hat, 0.0, lab.horizon)
    if lab.target_time is None:
        return t_hat
    return np.abs(t_hat - lab.target_time)


# ── Strategy 4: covering-array mixture design + Scheffé model ───────────────────

def covering_scheffe_search(
    lab: VirtualStrainLab,
    budget: int,
    rng: np.random.Generator,
    *,
    confirm_fraction: float = 0.3,
    max_block: int = 3,
    n_confirm: int = 5,
) -> SearchResult:
    """Cover every strain pair in small blends, fit the Scheffé model, then confirm the top few.

    The design phase runs 2–``max_block``-strain blends chosen by
    :func:`~vlab_doe.doe.covering.covering_array` so that as many of the 120 strain pairs as
    possible are co-inoculated at least once — a strength-2 covering array with a block-size
    constraint.  Because a 3-strain blend informs three pairwise coefficients at once, this buys
    pairwise information faster than testing pairs one at a time.  The fitted model then ranks
    all 120 pairs, and the confirmation phase spends the reserved budget replicating the model's
    shortlist — a screen-then-confirm campaign, which is how real strain selection is run.
    """
    b = _Budget(lab, budget)
    n_design = max(4, int(round(budget * (1.0 - confirm_fraction))))
    design = covering_array(
        lab.n_strains, n_design, min_size=2, max_size=max_block, strength=2,
        seed=int(rng.integers(1 << 30)),
    )

    blends: list[Members] = []
    runs: list[Experiment] = []
    for members in design.runs:
        exp = b.run(members)
        if exp is None:
            break
        blends.append(exp.members)
        runs.append(exp)

    coefs, alpha = fit_scheffe(blends, acidification_rate(runs), lab.n_strains, rng=rng)
    pairs = lab.all_pairs()
    predicted = rate_to_objective(lab, predict_pairs(coefs, lab.n_strains))
    shortlist = [pairs[k] for k in np.argsort(predicted)[:n_confirm]]

    means = _confirm(b, shortlist)
    if means:
        choice, observed = _pick(means)
    else:
        choice, observed = shortlist[0], float(predicted.min())
    return SearchResult(
        strategy="covering array + Scheffé",
        choice=choice,
        n_experiments=b.spent,
        observed_value=observed,
        trace=b.frame(),
        notes={
            "n_design_runs": len(blends),
            "pair_coverage": design.coverage(2)["coverage_fraction"],
            "lasso_alpha": alpha,
            "shortlist": [lab.label(p) for p in shortlist],
            "coefs": coefs,
        },
    )


def _confirm(b: _Budget, shortlist: Sequence[Members]) -> dict[Members, list[float]]:
    """Spend the remaining budget replicating a shortlist, round-robin."""
    means: dict[Members, list[float]] = {}
    while b.can():
        progressed = False
        for p in shortlist:
            exp = b.run(p)
            if exp is None:
                break
            means.setdefault(p, []).append(exp.value)
            progressed = True
        if not progressed:
            break
    return means


def _pick(means: dict[Members, list[float]]) -> tuple[Members, float]:
    choice = min(means, key=lambda k: float(np.mean(means[k])))
    return choice, float(np.mean(means[choice]))


# ── Strategy 5: greedy Bayesian-D-optimal mixture design + Scheffé model ────────

def d_optimal_scheffe_search(
    lab: VirtualStrainLab,
    budget: int,
    rng: np.random.Generator,
    *,
    confirm_fraction: float = 0.3,
    max_block: int = 3,
    n_candidates: int = 400,
    n_confirm: int = 5,
) -> SearchResult:
    """Build the design by greedy Bayesian D-optimality, then fit and confirm as above.

    Same model and same confirmation step as :func:`covering_scheffe_search`; only the design
    differs.  Runs are chosen one at a time from a candidate pool of 2- and 3-strain blends to
    maximise ``log det(XᵀX + λI)`` — the classic D-optimality criterion, ridge-regularised
    because with 136 coefficients and a few dozen runs ``XᵀX`` is singular (that ridge is exactly
    a Bayesian prior on the coefficients, which is also what the LASSO fit assumes).  Where the
    covering array optimises *combinatorial* coverage, this optimises the *information matrix* of
    the model actually being fitted.

    Two details matter and are worth stating, because getting them wrong quietly wrecks the
    design.  The interaction columns are rescaled (``x_i·x_j`` is 4× smaller than ``x_i`` for a
    50/50 blend), since ridge-regularised D-optimality is *not* invariant to column scaling and
    would otherwise buy precision on the main effects that the study does not need.  And single
    strains are excluded from the candidate pool: they carry the most information per run about
    the linear terms and none at all about synergies, so an unconstrained criterion spends its
    first 16 runs on solo batches — the very mistake :func:`single_strain_screen` makes.
    """
    b = _Budget(lab, budget)
    n_design = max(4, int(round(budget * (1.0 - confirm_fraction))))

    candidates: list[Members] = list(lab.all_pairs())
    if max_block >= 3:
        triples = [tuple(sorted(rng.choice(lab.n_strains, size=3, replace=False)))
                   for _ in range(n_candidates)]
        candidates += sorted(set(triples))
    weights = np.ones(lab.n_strains + len(lab.all_pairs()))
    weights[lab.n_strains:] = 4.0                     # put synergies on the same scale
    cand_X = scheffe_features(candidates, lab.n_strains) * weights

    n_features = cand_X.shape[1]
    lam = 1e-3
    info = lam * np.eye(n_features)
    chosen: list[int] = []
    for _ in range(n_design):
        inv = np.linalg.inv(info)
        # Greedy D-optimality: the run maximising log det(I + x xᵀ) = log(1 + xᵀ inv x).
        gains = np.einsum("ij,jk,ik->i", cand_X, inv, cand_X)
        gains = gains - 1e6 * np.isin(np.arange(len(candidates)), chosen)   # no exact repeats
        pick = int(np.argmax(gains))
        chosen.append(pick)
        x = cand_X[pick]
        info += np.outer(x, x)

    blends: list[Members] = []
    runs: list[Experiment] = []
    for k in chosen:
        exp = b.run(candidates[k])
        if exp is None:
            break
        blends.append(exp.members)
        runs.append(exp)

    coefs, alpha = fit_scheffe(blends, acidification_rate(runs), lab.n_strains, rng=rng)
    pairs = lab.all_pairs()
    predicted = rate_to_objective(lab, predict_pairs(coefs, lab.n_strains))
    shortlist = [pairs[k] for k in np.argsort(predicted)[:n_confirm]]

    means = _confirm(b, shortlist)
    if means:
        choice, observed = _pick(means)
    else:
        choice, observed = shortlist[0], float(predicted.min())
    return SearchResult(
        strategy="D-optimal + Scheffé",
        choice=choice,
        n_experiments=b.spent,
        observed_value=observed,
        trace=b.frame(),
        notes={"n_design_runs": len(blends), "lasso_alpha": alpha,
               "shortlist": [lab.label(p) for p in shortlist], "coefs": coefs},
    )


# ── Strategy 6: sequential Bayesian search (Thompson sampling) ──────────────────

def thompson_scheffe_search(
    lab: VirtualStrainLab,
    budget: int,
    rng: np.random.Generator,
    *,
    n_initial: int = 12,
    prior_sd: float = 0.3,
    synergy_prior_sd: float | None = None,
    noise_sd: float = 0.03,
    confirm_fraction: float = 0.15,
    n_confirm: int = 3,
    n_triples: int = 240,
    include_triples: bool = True,
) -> SearchResult:
    """Adaptive search: Bayesian linear model on the Scheffé features + Thompson sampling.

    The one design here that *learns while it runs*.  After a small random start, each iteration

    1. draws one plausible coefficient vector from the model posterior,
    2. runs the pair that vector says is best (so exploration comes from the posterior's own
       uncertainty rather than from a hand-tuned rule),
    3. updates the posterior with the result.

    Because the model is linear in the Scheffé features, the posterior is a closed-form Gaussian
    — no GP libraries, no acquisition optimisation, and every experiment updates *all* 120 pair
    predictions at once through the shared main-effect terms.  The recommendation at the end is
    the pair with the best posterior mean, which is deliberately *not* the same as the best
    single observation: it is shrunk toward what the whole campaign supports.

    Like the other model-based designs it models the acidification rate ``1/t_set``
    (:func:`acidification_rate`), so the prior and noise widths are in 1/h.
    ``synergy_prior_sd`` (default: same as ``prior_sd``) sets how surprising a strain-pair
    synergy is *a priori*, and it is the parameter to think hardest about: tighten it and the
    sampler exploits the pairs it already likes; widen it and it keeps chasing untested pairs.
    The last ``confirm_fraction`` of the budget is held back to replicate the top ``n_confirm``
    pairs, so the final answer rests on repeat batches rather than on one lucky run.

    **Why the action set includes triples.**  In a Scheffé quadratic each pair owns a private
    coefficient, so a pair that has never shared a jar sits at its prior no matter how much data
    the campaign gathers — a sampler restricted to pairs can only ever "find" the champion by
    running it, which degenerates into a randomised exhaustive scan.  A three-strain blend
    constrains three synergy coefficients at once, so ``n_triples`` sampled triples join the 120
    pairs in the candidate pool.  A candidate blend is scored by the *best pair it contains*
    under the drawn coefficients, which is what makes a triple worth running: it is optimistic
    about the pairs it would reveal, and it reveals three of them per batch.
    """
    b = _Budget(lab, budget)
    pairs = lab.all_pairs()
    n_features = lab.n_strains + len(pairs)
    pair_X = scheffe_features(pairs, lab.n_strains)

    blends: list[Members] = []
    runs: list[Experiment] = []

    def observe(members: Members) -> bool:
        exp = b.run(members)
        if exp is None:
            return False
        blends.append(exp.members)
        runs.append(exp)
        return True

    # Seed: random pairs (plus optional triples, which inform three synergies per batch).
    seeds: list[Members] = [pairs[k] for k in rng.choice(len(pairs), size=min(n_initial, len(pairs)), replace=False)]
    if include_triples:
        seeds += [tuple(sorted(rng.choice(lab.n_strains, size=3, replace=False)))
                  for _ in range(max(0, n_initial // 3))]
    for members in seeds:
        if not observe(members):
            break

    prior_var = np.full(n_features, prior_sd ** 2)
    prior_var[lab.n_strains:] = (synergy_prior_sd or prior_sd) ** 2
    prior_prec = np.diag(1.0 / prior_var)
    n_explore = max(1, int(round(budget * (1.0 - confirm_fraction))))

    # Action set: the 120 pairs, plus sampled triples that probe three synergies per batch.
    pair_index = {p: k for k, p in enumerate(pairs)}
    actions: list[Members] = list(pairs)
    if include_triples and lab.n_strains >= 3:
        seen = set(actions)
        for _ in range(int(n_triples)):
            t = tuple(sorted(int(x) for x in rng.choice(lab.n_strains, size=3, replace=False)))
            if t not in seen:
                seen.add(t)
                actions.append(t)
    # Row k of `contains` lists the pair-feature columns that action k would inform.
    contains = [[pair_index[(a, c)] for i, a in enumerate(act) for c in act[i + 1:]]
                for act in actions]

    def posterior(mean_only: bool = False):
        """Gaussian posterior over the Scheffé coefficients given the runs so far.

        The prior is centred on "every strain acidifies at the average rate seen so far, no
        synergies" — i.e. the linear terms start at the campaign mean rather than at zero.  With
        a zero-centred prior an untested strain would look infinitely *slow*, and the search
        would never revisit it.
        """
        X = scheffe_features(blends, lab.n_strains)
        y = acidification_rate(runs)
        prior_mean = np.zeros(n_features)
        prior_mean[: lab.n_strains] = float(np.mean(y))
        prec = prior_prec + X.T @ X / noise_sd ** 2
        residual = X.T @ (y - X @ prior_mean) / noise_sd ** 2
        if mean_only:
            return prior_mean + np.linalg.solve(prec, residual), None
        cov = np.linalg.inv(prec)
        return prior_mean + cov @ residual, cov

    while b.spent < n_explore and b.can():
        mean, cov = posterior()
        draw = rng.multivariate_normal(mean, cov, method="cholesky")
        sampled_objective = rate_to_objective(lab, pair_X @ draw)
        # Score each action by the best pair it would put in the jar together.
        scores = [min(sampled_objective[k] for k in cols) for cols in contains]
        if not observe(actions[int(np.argmin(scores))]):
            break

    # Shortlist by posterior mean, then spend what is left confirming it.
    mean, _ = posterior(mean_only=True)
    posterior_pair = rate_to_objective(lab, pair_X @ mean)
    shortlist = [pairs[k] for k in np.argsort(posterior_pair)[:n_confirm]]
    means = _confirm(b, shortlist)
    if means:
        choice, observed = _pick(means)
    else:
        choice, observed = shortlist[0], float(posterior_pair.min())
    return SearchResult(
        strategy="Thompson sampling",
        choice=choice,
        n_experiments=b.spent,
        observed_value=observed,
        trace=b.frame(),
        notes={"n_unique_blends": len(set(blends)),
               "shortlist": [lab.label(p) for p in shortlist],
               "posterior_mean": mean},
    )


# ── Strategy 7: successive halving (a race, for the noise) ──────────────────────

def successive_halving_search(
    lab: VirtualStrainLab,
    budget: int,
    rng: np.random.Generator,
    *,
    survivor_fraction: float = 0.5,
) -> SearchResult:
    """Race all pairs against each other, halving the field between replication rounds.

    Built for the *noise* rather than the interactions: instead of spending one batch on each of
    120 pairs and trusting the winner, it gives every survivor another replicate each round and
    throws away the worst half.  Budget therefore concentrates on the contenders, and the final
    call rests on several batches, not one.  If the budget is too small for a full first round,
    it starts from a random subset.
    """
    b = _Budget(lab, budget)
    pairs = lab.all_pairs()
    rng.shuffle(pairs)
    # First round must be affordable: keep at most `budget/2` arms so at least two rounds run.
    arms = pairs[: max(2, min(len(pairs), budget // 2))]
    means: dict[Members, list[float]] = {p: [] for p in arms}

    while b.can() and len(arms) > 1:
        for p in list(arms):
            exp = b.run(p)
            if exp is None:
                break
            means[p].append(exp.value)
        arms = sorted(arms, key=lambda k: float(np.mean(means[k])) if means[k] else np.inf)
        keep = max(1, int(round(len(arms) * survivor_fraction)))
        arms = arms[:keep]
    # Any leftover budget goes to the finalists.
    while b.can() and arms:
        for p in arms:
            exp = b.run(p)
            if exp is None:
                break
            means[p].append(exp.value)

    tested = {p: v for p, v in means.items() if v}
    choice, observed = _pick(tested)
    return SearchResult(
        strategy="successive halving",
        choice=choice,
        n_experiments=b.spent,
        observed_value=observed,
        trace=b.frame(),
        notes={"pairs_tested": len(tested),
               "replicates_on_winner": len(tested[choice])},
    )


# ── Strategy 8: greedy exchange (what a bench scientist does) ───────────────────

def greedy_exchange_search(
    lab: VirtualStrainLab,
    budget: int,
    rng: np.random.Generator,
    *,
    n_neighbours: int = 5,
) -> SearchResult:
    """Local search: keep one partner, swap the other, keep whatever improves.

    The informal campaign a bench scientist runs: start from a pair, try a few variants that
    replace one member, move to the best variant if it beats the incumbent, restart from a fresh
    random pair when stuck.  Cheap and often decent — but it only ever sees pairs it has already
    decided to look at, so it can walk straight past a synergy that involves two strains that are
    both mediocre on their own.
    """
    b = _Budget(lab, budget)
    means: dict[Members, list[float]] = {}

    def evaluate(p: Members) -> float | None:
        if p in means:
            return float(np.mean(means[p]))
        exp = b.run(p)
        if exp is None:
            return None
        means[p] = [exp.value]
        return exp.value

    incumbent = tuple(sorted(rng.choice(lab.n_strains, size=2, replace=False)))
    best = evaluate(incumbent)
    while b.can():
        # ``evaluate`` is free for a pair already run, so a sweep that only revisits known pairs
        # costs nothing and changes nothing.  Once the whole space is cached that is *every*
        # sweep, and without this guard the loop spins forever on a budget it can no longer
        # spend.  Track the spend and stop when a full sweep buys no new experiment.
        spent_before = b.spent
        improved = False
        for slot in (0, 1):
            keep = incumbent[1 - slot]
            options = [i for i in range(lab.n_strains) if i not in incumbent]
            rng.shuffle(options)
            for cand in options[:n_neighbours]:
                trial = tuple(sorted((keep, cand)))
                v = evaluate(trial)
                if v is None:
                    break
                if best is None or v < best:
                    incumbent, best, improved = trial, v, True
        if not improved:                      # stuck → restart somewhere else
            incumbent = tuple(sorted(rng.choice(lab.n_strains, size=2, replace=False)))
            v = evaluate(incumbent)
            if v is None:
                break
            if best is None or v < best:
                best = v
        if b.spent == spent_before:           # nothing new left to try — the search is done
            break

    choice, observed = _pick(means)
    return SearchResult(
        strategy="greedy exchange",
        choice=choice,
        n_experiments=b.spent,
        observed_value=observed,
        trace=b.frame(),
        notes={"pairs_tested": len(means)},
    )


#: The strategies compared in the study, in reporting order.
STRATEGIES: dict[str, Callable[..., SearchResult]] = {
    "random pairs": random_pair_search,
    "singles then top pairs": single_strain_screen,
    "greedy exchange": greedy_exchange_search,
    "covering array + Scheffé": covering_scheffe_search,
    "D-optimal + Scheffé": d_optimal_scheffe_search,
    "Thompson sampling": thompson_scheffe_search,
    "successive halving": successive_halving_search,
    "exhaustive pair scan": exhaustive_pair_scan,
}


# ── Benchmark harness ───────────────────────────────────────────────────────────

def benchmark_strategies(
    library: StrainLibrary,
    budgets: Sequence[int],
    *,
    strategies: dict[str, Callable[..., SearchResult]] | None = None,
    n_campaigns: int = 10,
    base_seed: int = 2026,
    progress: bool = False,
    **lab_kwargs,
) -> pd.DataFrame:
    """Score every strategy at every budget over repeated virtual campaigns.

    Each *campaign* is a fresh lab seed — a different draw of batch variability and probe noise —
    and every strategy sees the same campaign, so the comparison is paired (common random
    numbers).  Scoring is against the noise-free truth, which the strategies never see.

    Parameters
    ----------
    library:
        The candidate panel (e.g. :func:`~vlab_doe.models.fermentation.panel.yogurt_strain_panel`).
    budgets:
        Experiment budgets to test, e.g. ``(20, 40, 60, 90, 120)``.
    strategies:
        Mapping name → strategy function; defaults to :data:`STRATEGIES`.
    n_campaigns:
        Repeats per (strategy, budget) cell.
    base_seed:
        Master seed; campaign *c* uses lab seed ``base_seed + c``.
    progress:
        Print a line per campaign.
    **lab_kwargs:
        Forwarded to :class:`VirtualStrainLab` (temperature, horizon, target_time, …).

    Returns
    -------
    DataFrame with one row per (campaign, budget, strategy): the chosen pair, whether it is the
    champion, the true rank of the choice among all 120 pairs, and the **regret** — how much
    worse the choice truly is than the champion, in hours.
    """
    strategies = dict(STRATEGIES if strategies is None else strategies)
    rows: list[dict] = []
    for campaign in range(int(n_campaigns)):
        lab = VirtualStrainLab(library=library, seed=base_seed + campaign, **lab_kwargs)
        truth_table = lab.pair_truth_table()          # cached, free
        rank_of = {m: r for m, r in zip(truth_table["members"], truth_table["rank"])}
        best_value = float(truth_table.iloc[0]["value"])
        champion = tuple(truth_table.iloc[0]["members"])

        for budget in budgets:
            # ``enumerate`` (not ``hash(name)``) keys the stream: Python's string hash is salted
            # per process, which would make the whole benchmark irreproducible.
            for strategy_id, (name, fn) in enumerate(strategies.items()):
                lab.reset()
                rng = np.random.default_rng([base_seed, campaign, int(budget), strategy_id])
                t0 = time.perf_counter()
                res = fn(lab, int(budget), rng)
                elapsed = time.perf_counter() - t0
                choice = tuple(sorted(res.choice))
                rows.append(
                    {
                        "campaign": campaign,
                        "budget": int(budget),
                        "strategy": name,
                        "choice": lab.label(choice),
                        "n_experiments": res.n_experiments,
                        "found_champion": choice == champion,
                        "true_rank": int(rank_of.get(choice, len(truth_table))),
                        "regret_h": float(lab.truth(choice) - best_value),
                        "observed_value": res.observed_value,
                        "seconds": elapsed,
                    }
                )
        if progress:
            print(f"campaign {campaign + 1}/{n_campaigns} done", flush=True)
    return pd.DataFrame(rows)


def summarize_benchmark(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate :func:`benchmark_strategies` output to one row per (strategy, budget).

    Reports the hit rate (fraction of campaigns that found the champion), mean and median regret
    in hours, the median true rank of the choice, and the mean budget actually spent.
    """
    agg = (
        results.groupby(["strategy", "budget"])
        .agg(
            hit_rate=("found_champion", "mean"),
            mean_regret_h=("regret_h", "mean"),
            median_regret_h=("regret_h", "median"),
            p90_regret_h=("regret_h", lambda s: float(np.quantile(s, 0.9))),
            median_rank=("true_rank", "median"),
            top5_rate=("true_rank", lambda s: float((s <= 5).mean())),
            mean_experiments=("n_experiments", "mean"),
        )
        .reset_index()
    )
    return agg.sort_values(["budget", "mean_regret_h"]).reset_index(drop=True)
