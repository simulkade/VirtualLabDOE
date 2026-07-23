"""Covering-array designs for combinatorial screening.

Full-factorial and LHS designs vary a handful of *continuous* factors.  Strain screening is a
different problem: there are many discrete "ingredients" (here, candidate strains) and each
experiment is a small **combination** of them.  Testing every combination is hopeless
(``C(50, 2..5)`` is astronomical), but we don't need to — to estimate which strains and which
*pairs* of strains drive the outcome it is enough that **every pair of strains is co-tested in
at least one run**.  That is a (strength-2) **covering array**.

The twist versus a textbook covering array over binary factors is a **block-size constraint**:
each run activates only ``min_size``–``max_size`` strains (a real fermentation mixes a few
strains, not 25).  This makes the design a block-size-constrained covering array — equivalently
a covering design over pairs.  We build it greedily: repeatedly add to the current run the
strain that newly covers the most still-uncovered pairs, breaking ties toward the least-used
strain so appearances stay balanced (good replication for downstream regression).

Strength 2 (pairs) is the default and the natural choice for screening; higher strength (e.g.
3, for three-way interactions) is supported but costs far more runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from ..config import make_rng


@dataclass(frozen=True)
class CoveringArrayDesign:
    """A combinatorial design: each run is a subset of the ``n_items`` items.

    Attributes
    ----------
    runs:
        One tuple of item indices per experiment.
    n_items:
        Total number of items (e.g. candidate strains) in the pool.
    item_names:
        Optional human-readable names, one per item.
    """

    runs: list[tuple[int, ...]]
    n_items: int
    item_names: list[str] | None = None

    @property
    def n_runs(self) -> int:
        return len(self.runs)

    def names(self) -> list[str]:
        return self.item_names or [f"item_{i:02d}" for i in range(self.n_items)]

    def matrix(self) -> np.ndarray:
        """Binary presence matrix, shape ``(n_runs, n_items)`` (1 = item used in that run)."""
        m = np.zeros((self.n_runs, self.n_items), dtype=int)
        for r, members in enumerate(self.runs):
            m[r, list(members)] = 1
        return m

    def to_dataframe(self, *, prefix: str = "x") -> pd.DataFrame:
        """Presence table: one boolean column per item plus a ``members`` summary column.

        Mirrors :func:`vlab_doe.doe.factorial.full_factorial` output (one row per run,
        a ``run_type`` column) so it flows through the same downstream tooling.
        """
        names = self.names() if self.item_names else [f"{prefix}{i:02d}" for i in range(self.n_items)]
        mat = self.matrix()
        df = pd.DataFrame(mat, columns=names)
        df.insert(0, "members", [tuple(self.names()[i] for i in r) for r in self.runs])
        df["run_type"] = "covering"
        return df

    def appearances(self) -> np.ndarray:
        """Number of runs each item appears in, shape ``(n_items,)``."""
        return self.matrix().sum(axis=0)

    def coverage(self, strength: int = 2) -> dict:
        """Combinatorial coverage diagnostics at the given strength.

        Returns the total number of ``strength``-subsets of items, how many are covered by at
        least one run, the coverage fraction, the min/mean/max redundancy (how many runs cover
        each covered subset), and the per-item appearance balance.
        """
        counts: dict[tuple[int, ...], int] = {}
        for members in self.runs:
            if len(members) < strength:
                continue
            for sub in combinations(sorted(members), strength):
                counts[sub] = counts.get(sub, 0) + 1

        total = _n_choose_k(self.n_items, strength)
        covered = len(counts)
        redundancy = np.array(list(counts.values())) if counts else np.array([0])
        app = self.appearances()
        return {
            "strength": strength,
            "subsets_total": total,
            "subsets_covered": covered,
            "coverage_fraction": covered / total if total else 1.0,
            "redundancy_min": int(redundancy.min()),
            "redundancy_mean": float(redundancy.mean()),
            "redundancy_max": int(redundancy.max()),
            "appearances_min": int(app.min()),
            "appearances_mean": float(app.mean()),
            "appearances_max": int(app.max()),
        }


def _n_choose_k(n: int, k: int) -> int:
    from math import comb

    return comb(n, k) if 0 <= k <= n else 0


def covering_array(
    n_items: int,
    n_runs: int,
    *,
    min_size: int = 2,
    max_size: int = 5,
    strength: int = 2,
    seed: int | None = None,
) -> CoveringArrayDesign:
    """Greedily build a block-size-constrained covering array.

    Each of ``n_runs`` runs activates between ``min_size`` and ``max_size`` items, chosen to
    cover as many still-uncovered ``strength``-subsets of items as possible.

    Parameters
    ----------
    n_items:
        Size of the item pool (e.g. 50 candidate strains).
    n_runs:
        Number of experiments (rows) in the design.
    min_size, max_size:
        Inclusive bounds on how many items each run may activate.
    strength:
        Coverage strength ``t``: every ``t``-subset of items is targeted for co-occurrence in
        at least one run.  ``2`` (pairs) is the default.
    seed:
        Seed for tie-breaking / size draws (via :func:`vlab_doe.config.make_rng`).

    Returns
    -------
    CoveringArrayDesign
        Use :meth:`CoveringArrayDesign.coverage` to see how complete the coverage is — with a
        block-size cap it may be below 1.0 if ``n_runs`` is too small, which the caller can
        inspect rather than the function silently failing.
    """
    if not (1 <= min_size <= max_size <= n_items):
        raise ValueError("require 1 <= min_size <= max_size <= n_items")
    if strength < 1:
        raise ValueError("strength must be >= 1")
    rng = make_rng(seed)

    # Set of uncovered t-subsets, stored as sorted tuples.
    uncovered: set[tuple[int, ...]] = set(combinations(range(n_items), strength))
    n_subsets_total = max(len(uncovered), 1)
    appearances = np.zeros(n_items, dtype=int)
    sizes = np.arange(min_size, max_size + 1)

    runs: list[tuple[int, ...]] = []
    for _ in range(n_runs):
        # Block size is drawn across the full [min_size, max_size] range, but biased toward
        # larger blocks while many pairs are still uncovered (larger blocks cover more pairs
        # per run), so coverage completes without giving up the requested size variety.
        frac_remaining = len(uncovered) / n_subsets_total
        weights = 1.0 + frac_remaining * (sizes - min_size)
        size = int(rng.choice(sizes, p=weights / weights.sum()))
        block = _greedy_block(n_items, size, strength, uncovered, appearances, rng)
        runs.append(tuple(sorted(block)))
        appearances[list(block)] += 1
        # Remove newly covered subsets.
        if len(block) >= strength:
            for sub in combinations(sorted(block), strength):
                uncovered.discard(sub)

    return CoveringArrayDesign(runs=runs, n_items=n_items)


def _greedy_block(
    n_items: int,
    size: int,
    strength: int,
    uncovered: set[tuple[int, ...]],
    appearances: np.ndarray,
    rng: np.random.Generator,
) -> list[int]:
    """Pick ``size`` items that cover many uncovered ``strength``-subsets, kept balanced.

    Greedy: seed the block with the least-used item that participates in an uncovered subset
    (or the globally least-used item if everything is already covered), then repeatedly add the
    item that maximises newly-covered subsets among the current block, tie-broken by fewest
    prior appearances and a little randomness.
    """
    block: list[int] = []
    # Seed: prefer an item that still has uncovered subsets; fall back to least-used.
    if uncovered:
        seed_counts = np.zeros(n_items)
        for sub in uncovered:
            for i in sub:
                seed_counts[i] += 1
        # least-used among items that touch the most uncovered subsets
        score = seed_counts - 0.001 * appearances
        first = int(_argmax_random(score, rng))
    else:
        first = int(_argmax_random(-appearances.astype(float), rng))
    block.append(first)

    while len(block) < size:
        best_item, best_gain, best_tie = -1, -1, None
        block_set = set(block)
        for cand in range(n_items):
            if cand in block_set:
                continue
            # newly covered subsets if we add `cand`: those t-subsets formed by cand + (t-1)
            # current block members, that are still uncovered.
            if strength == 1:
                gain = 1 if (cand,) in uncovered else 0
            else:
                gain = 0
                for combo in combinations(block, strength - 1):
                    sub = tuple(sorted(combo + (cand,)))
                    if sub in uncovered:
                        gain += 1
            tie = -appearances[cand] + rng.random() * 0.5   # prefer rare items, jitter ties
            if gain > best_gain or (gain == best_gain and (best_tie is None or tie > best_tie)):
                best_item, best_gain, best_tie = cand, gain, tie
        block.append(best_item)
    return block


def _argmax_random(values: np.ndarray, rng: np.random.Generator) -> int:
    """argmax with random tie-breaking."""
    values = np.asarray(values, dtype=float)
    top = np.flatnonzero(values >= values.max() - 1e-12)
    return int(rng.choice(top))
