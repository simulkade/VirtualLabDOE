"""Six ways to spend ~60–70 runs so that every pair of 16 strains is co-tested.

:mod:`vlab_doe.doe.covering` answers "how do I cover all the pairs at all?" greedily.  This
module asks the harder, more practical question: given that a *budget* of about 60–70 blends is
what the lab will actually run, and that there are :math:`\\binom{16}{2} = 120` pairs, **what
shape should those blends have?**  The choice is not free — it trades three things against each
other, and no design wins on all three:

**Coverage.**  A run of ``k`` strains co-tests :math:`\\binom{k}{2}` pairs, so block size is the
only lever that buys coverage: 65 two-strain runs can never see more than 65 of the 120 pairs,
while 65 four-strain runs sweep 390 pair-slots.

**Dilution.**  In the Scheffé parameterisation a ``k``-blend has :math:`x_i = 1/k`, so a pair's
own term enters at :math:`x_i x_j = 1/k^2` — 1/4 for a pair, 1/16 for a four-blend.  Larger
blocks see more pairs but see each of them **four times more faintly**, against unchanged batch
noise.  Coverage and signal-to-noise pull in opposite directions.

**Confounding.**  Every pair inside a block contributes to the *same* scalar reading.  A block
of four carries six pair terms at once, so what a run measures is a sum, and separating the six
requires other runs that break them apart.  Balance in the *design* is what makes that possible.

Designs built here
------------------
:func:`random_pairs_design`
    65 randomly chosen pairs.  The "no design" reference: every reading is a clean, undiluted,
    unconfounded measurement of exactly one pair — of the 54% of pairs it happens to pick.
:func:`greedy_covering_design`
    :func:`~vlab_doe.doe.covering.covering_array` with mixed 2–5-strain blocks.
:func:`triples_covering_design`
    The same greedy coverage rule, but every block is a triple — the mildest dilution that still
    covers three pairs per run.
:func:`bibd_replicated_design`
    The classical answer: the **affine plane AG(2,4)**, a balanced incomplete block design with
    ``v=16, k=4, r=5, b=20, λ=1`` — every pair co-tested *exactly once*, every strain in exactly
    five blocks — run three times over for pure-error degrees of freedom.
:func:`bibd_relabelled_design`
    The same 20 blocks, but the three repeats use independent strain **relabellings**, so λ is
    still 3 while the six-pair bundles differ every time.  Same balance, broken aliasing.
:func:`augmented_triples_design`
    16 single-strain runs plus 48 triples: pays a quarter of the budget for clean main effects
    so the interaction contrasts are not leaning on them.

All builders return a :class:`~vlab_doe.doe.covering.CoveringArrayDesign`, so they share the
coverage/appearance diagnostics, and :data:`COVERING_DESIGNS` maps a name to each builder for
sweeping them head to head.
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable

import numpy as np

from .covering import CoveringArrayDesign, covering_array

Block = tuple[int, ...]

#: Default run budget for the study — inside the 60–70 window, and divisible by both 20 (the
#: BIBD block count) and 3 (so the augmented design splits evenly).
DEFAULT_N_RUNS: int = 60


# ── 1. The reference: pairs only ────────────────────────────────────────────────

def random_pairs_design(
    n_items: int = 16, n_runs: int = DEFAULT_N_RUNS, *, seed: int | None = None
) -> CoveringArrayDesign:
    """``n_runs`` distinct pairs drawn at random — the design that makes no attempt to cover.

    Worth carrying through the whole comparison because it is the honest baseline: each run
    measures one pair with no dilution and no confounding, which is the best possible *per-pair*
    information.  Its weakness is equally stark — the :math:`120 - n_{runs}` pairs it never
    draws are not estimated poorly, they are not estimated at all.
    """
    rng = np.random.default_rng(seed)
    pairs = list(combinations(range(n_items), 2))
    picked = rng.choice(len(pairs), size=min(n_runs, len(pairs)), replace=False)
    runs = sorted(pairs[k] for k in picked)
    return CoveringArrayDesign(runs=runs, n_items=n_items)


# ── 2–3. Greedy covering arrays ─────────────────────────────────────────────────

def greedy_covering_design(
    n_items: int = 16, n_runs: int = DEFAULT_N_RUNS, *, seed: int | None = None
) -> CoveringArrayDesign:
    """Greedy strength-2 covering array with mixed 2–5-strain blocks.

    The general-purpose construction from :mod:`~vlab_doe.doe.covering`: at each step take the
    strain that newly covers the most still-uncovered pairs, breaking ties toward the
    least-used strain.  Block sizes vary, which is good for coverage speed but means the runs
    are *not* on a common dilution scale — a 2-blend and a 5-blend inform the same coefficients
    with very different weights.
    """
    return covering_array(n_items, n_runs, min_size=2, max_size=5, strength=2, seed=seed)


def triples_covering_design(
    n_items: int = 16, n_runs: int = DEFAULT_N_RUNS, *, seed: int | None = None
) -> CoveringArrayDesign:
    """Greedy strength-2 covering array in which **every** block is a triple.

    Three pairs per run at :math:`x_i x_j = 1/9` is the gentlest trade available that still
    covers faster than one pair per run: :math:`3 n_{runs}` pair-slots for 120 pairs, and every
    observation on the same dilution scale, so the runs are directly comparable.
    """
    return covering_array(n_items, n_runs, min_size=3, max_size=3, strength=2, seed=seed)


# ── 4–5. Balanced incomplete block designs ──────────────────────────────────────

def affine_plane_ag24() -> list[Block]:
    """The 20 blocks of the affine plane ``AG(2, 4)``: a ``(16, 4, 1)`` BIBD.

    Label the 16 strains by the cells of a 4×4 grid over :math:`GF(4)`.  The blocks are the
    lines of the plane: 4 rows, 4 columns, and 4 "slopes" of 4 parallel lines each — 20 blocks
    of 4 in five *parallel classes*, each class partitioning all 16 strains.  Two points lie on
    exactly one common line, so **every pair of strains is co-tested exactly once** and every
    strain appears in exactly five blocks.

    This is as balanced as a block design over pairs can be, and it is the natural reference
    point for the whole study: any imbalance in the other designs is a departure from this.
    """
    # GF(4) as {0,1,2,3} with 2 = x, 3 = x+1 under the AES-style tables.
    mul = np.array([
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2],
    ])
    add = np.array([
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0],
    ])
    point = lambda x, y: 4 * x + y            # noqa: E731  (grid cell -> strain index)

    blocks: list[Block] = []
    for c in range(4):                                    # the vertical class: x = c
        blocks.append(tuple(sorted(point(c, y) for y in range(4))))
    for m in range(4):                                    # 4 slopes x 4 intercepts: y = m x + b
        for b in range(4):
            blocks.append(tuple(sorted(point(x, add[mul[m, x], b]) for x in range(4))))
    return blocks


def _check_bibd(blocks: list[Block], n_items: int = 16) -> None:
    """Assert the (16, 4, 1) property: every pair exactly once, every point five times."""
    counts: dict[tuple[int, int], int] = {}
    for blk in blocks:
        for pair in combinations(sorted(blk), 2):
            counts[pair] = counts.get(pair, 0) + 1
    if len(counts) != n_items * (n_items - 1) // 2 or set(counts.values()) != {1}:
        raise AssertionError("affine_plane_ag24 did not produce a (16, 4, 1) BIBD")


def bibd_replicated_design(
    n_items: int = 16, n_runs: int = DEFAULT_N_RUNS, *, seed: int | None = None
) -> CoveringArrayDesign:
    """The AG(2,4) BIBD run ``n_runs / 20`` times over — the *replicated* balanced design.

    Perfect balance and, because the same 20 blocks recur, genuine **replicates**: the
    within-block spread is a model-free estimate of pure error, which is what makes a formal
    lack-of-fit F-test possible.  The cost is that the six-pair bundle carried by each block is
    identical every time, so whatever two pairs are confounded in one replicate stay confounded
    in all of them — replication buys precision, never resolution.
    """
    blocks = affine_plane_ag24()
    _check_bibd(blocks, n_items)
    reps = max(1, round(n_runs / len(blocks)))
    return CoveringArrayDesign(runs=sorted(blocks * reps), n_items=n_items)


def bibd_relabelled_design(
    n_items: int = 16, n_runs: int = DEFAULT_N_RUNS, *, seed: int | None = None
) -> CoveringArrayDesign:
    """AG(2,4) repeated under independent strain **relabellings**.

    Each repeat permutes which strain sits in which cell of the grid, so every repeat is still a
    perfect (16, 4, 1) BIBD — balance untouched, λ = 3 — but the six pairs bundled into a block
    change completely between repeats.  This is the direct counterpart to
    :func:`bibd_replicated_design`: the same balance and the same budget, spent on **breaking
    the aliasing** instead of on pure error.  Comparing the two isolates what balance alone can
    and cannot do.
    """
    rng = np.random.default_rng(seed)
    base = affine_plane_ag24()
    _check_bibd(base, n_items)
    reps = max(1, round(n_runs / len(base)))
    runs: list[Block] = []
    for r in range(reps):
        perm = np.arange(n_items) if r == 0 else rng.permutation(n_items)
        runs += [tuple(sorted(int(perm[i]) for i in blk)) for blk in base]
    return CoveringArrayDesign(runs=sorted(runs), n_items=n_items)


# ── 6. Augmented with single-strain runs ────────────────────────────────────────

def augmented_triples_design(
    n_items: int = 16, n_runs: int = DEFAULT_N_RUNS, *, seed: int | None = None
) -> CoveringArrayDesign:
    """``n_items`` single-strain runs plus a greedy triples covering array for the rest.

    In every blend-only design the main effects and the interactions are estimated from the same
    diluted readings, so a strain that merely *acidifies fast* and a strain that *synergises*
    look alike.  Spending 16 of 60 runs on single strains pins the main effects down directly
    and leaves the blends to say something about the interactions alone.  The bet is that the
    quarter of the budget it costs is repaid by cleaner interaction contrasts — the study is
    what decides whether that bet pays.
    """
    singles: list[Block] = [(i,) for i in range(n_items)]
    remaining = max(0, n_runs - n_items)
    blends = covering_array(n_items, remaining, min_size=3, max_size=3, strength=2, seed=seed)
    return CoveringArrayDesign(runs=singles + list(blends.runs), n_items=n_items)


#: Name -> builder, all with the signature ``(n_items, n_runs, *, seed)``.
COVERING_DESIGNS: dict[str, Callable[..., CoveringArrayDesign]] = {
    "random pairs": random_pairs_design,
    "greedy covering (2-5)": greedy_covering_design,
    "triples covering": triples_covering_design,
    "BIBD replicated": bibd_replicated_design,
    "BIBD relabelled": bibd_relabelled_design,
    "singles + triples": augmented_triples_design,
}


def build_designs(
    n_items: int = 16, n_runs: int = DEFAULT_N_RUNS, *, seed: int | None = 0
) -> dict[str, CoveringArrayDesign]:
    """Build every design in :data:`COVERING_DESIGNS` at the same budget."""
    return {name: fn(n_items, n_runs, seed=seed) for name, fn in COVERING_DESIGNS.items()}
