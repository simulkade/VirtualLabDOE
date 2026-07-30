"""Phase 2.x — covering-array design variants and their frequentist analysis."""

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from vlab_doe.doe.combination_search import VirtualStrainLab, scheffe_features
from vlab_doe.doe.covering_analysis import (
    design_diagnostics,
    fit_main_effects,
    forward_selection,
    lack_of_fit_test,
    lenth_effects,
    pair_contrasts,
    score_screen,
    true_pair_excess,
)
from vlab_doe.doe.covering_designs import (
    COVERING_DESIGNS,
    affine_plane_ag24,
    bibd_relabelled_design,
    bibd_replicated_design,
    build_designs,
    random_pairs_design,
)
from vlab_doe.models.fermentation.panel import champion_indices, yogurt_strain_panel

N_STRAINS, N_RUNS = 16, 60
ALL_PAIRS = list(combinations(range(N_STRAINS), 2))


@pytest.fixture(scope="module")
def library():
    return yogurt_strain_panel()


@pytest.fixture(scope="module")
def designs():
    return build_designs(N_STRAINS, N_RUNS, seed=7)


# ── The combinatorics ───────────────────────────────────────────────────────────

def test_affine_plane_is_a_16_4_1_bibd():
    """Every pair on exactly one line, every point on exactly five — the design's whole claim."""
    blocks = affine_plane_ag24()
    assert len(blocks) == 20 and {len(b) for b in blocks} == {4}
    counts = {p: 0 for p in ALL_PAIRS}
    for blk in blocks:
        for pair in combinations(sorted(blk), 2):
            counts[pair] += 1
    assert set(counts.values()) == {1}
    appearances = np.bincount([i for b in blocks for i in b], minlength=N_STRAINS)
    assert set(appearances.tolist()) == {5}


def test_affine_plane_has_five_parallel_classes():
    """The 20 lines fall into 5 classes of 4 disjoint blocks, each partitioning all 16 points."""
    blocks = affine_plane_ag24()
    classes = [blocks[i:i + 4] for i in range(0, 20, 4)]
    for cls in classes:
        assert sorted(i for b in cls for i in b) == list(range(N_STRAINS))


@pytest.mark.parametrize("name", sorted(COVERING_DESIGNS))
def test_designs_respect_the_budget(name):
    design = COVERING_DESIGNS[name](N_STRAINS, N_RUNS, seed=3)
    assert design.n_runs == N_RUNS
    assert all(len(set(r)) == len(r) for r in design.runs)          # no repeated strain in a jar
    assert all(set(r) <= set(range(N_STRAINS)) for r in design.runs)


def test_covering_designs_cover_every_pair_but_random_pairs_does_not(designs):
    """The point of a covering array — and the point of keeping the naive baseline."""
    for name in ("greedy covering (2-5)", "triples covering",
                 "BIBD replicated", "BIBD relabelled"):
        assert designs[name].coverage()["coverage_fraction"] == 1.0
    assert designs["random pairs"].coverage()["coverage_fraction"] == pytest.approx(0.5)


def test_both_bibd_variants_have_identical_balance(designs):
    """Same lambda, same appearances — they differ only in *which* strains meet."""
    rep, rel = designs["BIBD replicated"].coverage(), designs["BIBD relabelled"].coverage()
    for key in ("coverage_fraction", "redundancy_min", "redundancy_max",
                "appearances_min", "appearances_max"):
        assert rep[key] == rel[key]


def test_relabelling_breaks_the_aliasing_that_replication_preserves():
    """The study's central design finding, as an assertion.

    Replicating a (16, 4, 1) BIBD leaves each block's six pairs on identical model columns;
    relabelling between repeats splits them apart at no extra cost.
    """
    def distinct_columns(design):
        inter = scheffe_features(list(design.runs), N_STRAINS)[:, N_STRAINS:]
        used = inter[:, inter.sum(axis=0) > 0]
        return len({tuple(np.flatnonzero(used[:, k]).tolist()) for k in range(used.shape[1])})

    rep = distinct_columns(bibd_replicated_design(N_STRAINS, N_RUNS, seed=1))
    rel = distinct_columns(bibd_relabelled_design(N_STRAINS, N_RUNS, seed=1))
    assert rep == 20                    # one column per block: 6 pairs each, indistinguishable
    assert rel > 100                    # nearly every pair separable
    assert rel > 5 * rep


# ── Diagnostics ─────────────────────────────────────────────────────────────────

def test_every_design_is_supersaturated(designs):
    """136 Scheffe coefficients against 60 runs — no design here can fit the full model."""
    for name, design in designs.items():
        d = design_diagnostics(list(design.runs), N_STRAINS)
        assert d["n_parameters"] == N_STRAINS + len(ALL_PAIRS) == 136
        assert d["rank_full"] <= d["n_runs"] < d["n_parameters"]
        assert d["rank_deficiency"] > 0


def test_diagnostics_flag_the_replicated_bibd_as_fully_aliased(designs):
    rep = design_diagnostics(list(designs["BIBD replicated"].runs), N_STRAINS)
    rel = design_diagnostics(list(designs["BIBD relabelled"].runs), N_STRAINS)
    assert rep["n_pairs_aliased"] == 120                  # all of them
    assert rel["n_pairs_aliased"] < rep["n_pairs_aliased"]
    assert rep["rank_full"] == 20                         # 20 distinct blocks, nothing more


def test_dilution_falls_with_block_size(designs):
    """x_i x_j = 1/k^2: the tax bigger blocks pay for their coverage."""
    dil = {n: design_diagnostics(list(d.runs), N_STRAINS)["dilution"] for n, d in designs.items()}
    assert dil["random pairs"] == pytest.approx(0.25)
    assert dil["triples covering"] == pytest.approx(1 / 9)
    assert dil["BIBD replicated"] == pytest.approx(1 / 16)


# ── The frequentist analysis ────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def campaign(library, designs):
    """One run of the triples design, with its response — reused by the analysis tests."""
    lab = VirtualStrainLab(library=library, seed=99)
    runs = list(designs["triples covering"].runs)
    y = np.array([1.0 / max(lab.run(r).value, 1e-3) for r in runs])
    return lab, runs, y


def test_main_effects_fit_recovers_the_strong_soloists(campaign, library):
    lab, runs, y = campaign
    fit = fit_main_effects(runs, y, N_STRAINS, names=library.names)
    assert fit.coefs.shape == (N_STRAINS,)
    assert len(fit.table) == N_STRAINS
    assert 0.0 <= fit.r_squared <= 1.0
    # The acid-tolerant LB/LH isolates really are the fastest strains on their own, and the
    # additive model does find them, from blends alone.
    best = set(fit.table.nlargest(4, "coef")["strain"])
    assert {"LB-03", "LB-01"} <= best
    assert all(s.startswith(("LB", "LH")) for s in best)


def test_additive_main_effects_cannot_spot_a_passenger_strain(campaign, library):
    """A real weakness, asserted: a strain that does nothing does not look bad.

    LL-01/LL-02 cannot grow at 43 C, so they contribute nothing to any blend they join.  But an
    additive mixture model splits a blend's response evenly across its members, so a passenger
    is credited with its jar-mates' work and lands mid-table rather than last.  Nothing in the
    covering-array output flags it; only a single-strain run would.
    """
    lab, runs, y = campaign
    table = fit_main_effects(runs, y, N_STRAINS, names=library.names).table
    rank = {s: r for r, s in enumerate(table.sort_values("coef")["strain"])}
    assert rank["LL-01"] > 3 and rank["LL-02"] > 3      # neither is among the four worst


def test_lack_of_fit_needs_replicated_blends(library, designs):
    """Only the replicated design can test its own adequacy; the others cannot, and say so."""
    lab = VirtualStrainLab(library=library, seed=5)
    rep_runs = list(designs["BIBD replicated"].runs)
    y = np.array([1.0 / max(lab.run(r).value, 1e-3) for r in rep_runs])
    lof = lack_of_fit_test(rep_runs, y, N_STRAINS)
    assert lof["available"] and lof["df_pure_error"] > 0
    assert lof["p_value"] < 0.05                       # additivity is genuinely wrong here

    # A design with no repeated blend has no pure-error degrees of freedom and must say so
    # rather than invent an answer.
    solo_runs = list(designs["random pairs"].runs)
    assert len({tuple(sorted(r)) for r in solo_runs}) == len(solo_runs)
    y2 = np.array([1.0 / max(lab.run(r).value, 1e-3) for r in solo_runs])
    assert lack_of_fit_test(solo_runs, y2, N_STRAINS)["available"] is False


def test_pair_contrasts_cover_all_pairs_with_adjusted_p(campaign, library):
    lab, runs, y = campaign
    c = pair_contrasts(runs, y, N_STRAINS, names=library.names)
    assert len(c) == len(ALL_PAIRS)
    ok = c["p_value"].notna()
    assert (c.loc[ok, "p_bonferroni"] >= c.loc[ok, "p_value"] - 1e-12).all()
    assert (c.loc[ok, "p_bh"] >= c.loc[ok, "p_value"] - 1e-12).all()
    assert (c.loc[ok, "p_bonferroni"] >= c.loc[ok, "p_bh"] - 1e-12).all()


def test_aliased_pairs_receive_identical_statistics(library, designs):
    """Identical columns must produce identical inference — that is what aliasing *means*."""
    lab = VirtualStrainLab(library=library, seed=5)
    runs = list(designs["BIBD replicated"].runs)
    y = np.array([1.0 / max(lab.run(r).value, 1e-3) for r in runs])
    c = pair_contrasts(runs, y, N_STRAINS, names=library.names).set_index("pair")
    block = sorted(affine_plane_ag24()[0])
    labels = [f"{library.names[i]}+{library.names[j]}" for i, j in combinations(block, 2)]
    stats = c.loc[labels, ["contrast", "t", "p_value"]]
    assert np.allclose(stats.to_numpy(), stats.iloc[0].to_numpy())


def test_true_pair_excess_singles_out_the_champion(library):
    """The champion's departure from additivity should dominate — it is why it wins."""
    lab = VirtualStrainLab(library=library, seed=1)
    excess = true_pair_excess(lab)
    assert len(excess) == len(ALL_PAIRS)
    best = excess.sort_values("excess_rate", ascending=False).iloc[0]
    i, j = champion_indices(library)
    assert (int(best["i"]), int(best["j"])) == (i, j)
    assert best["excess_rate"] > 2 * excess["excess_rate"].nlargest(2).iloc[1]


def test_score_screen_reports_a_champion_rank(campaign, library):
    lab, runs, y = campaign
    c = pair_contrasts(runs, y, N_STRAINS, names=library.names)
    score = score_screen(c, true_pair_excess(lab), champion=champion_indices(library))
    assert 1 <= score["champion_rank"] <= len(ALL_PAIRS)
    assert 0.0 <= score["power"] <= 1.0
    assert -1.0 <= score["contrast_truth_corr"] <= 1.0


def test_forward_selection_enters_the_champion_first(campaign, library):
    """Conditioning beats marginal screening: the champion should be the first term in."""
    lab, runs, y = campaign
    fs = forward_selection(runs, y, N_STRAINS, names=library.names, max_terms=4)
    assert not fs.empty
    i, j = champion_indices(library)
    assert fs.iloc[0]["pair"] == f"{library.names[i]}+{library.names[j]}"
    assert fs["r_squared"].is_monotonic_increasing


def test_lenth_margin_grows_with_a_denser_effect_set():
    """Lenth assumes most effects are inert; when they are not, its margin inflates."""
    rng = np.random.default_rng(0)
    sparse = rng.normal(0, 0.01, 120); sparse[0] = 0.5
    dense = rng.normal(0, 0.01, 120) + rng.normal(0, 0.2, 120)
    assert lenth_effects(sparse)["margin_of_error"] < lenth_effects(dense)["margin_of_error"]
    assert lenth_effects(sparse)["n_active"] >= 1


def test_interaction_scale_dials_the_panel_between_regimes():
    """The sweep's control knob: scale 0 must be additive, scale 1 the calibrated panel."""
    flat = yogurt_strain_panel(interaction_scale=0.0)
    full = yogurt_strain_panel()
    assert np.allclose(flat.interaction, 0.0)
    assert not np.allclose(full.interaction, 0.0)
    lab_flat = VirtualStrainLab(library=flat, seed=1)
    lab_full = VirtualStrainLab(library=full, seed=1)
    # With no interactions the champion pair loses its advantage and the best pair changes.
    assert lab_flat.best_pair() != champion_indices(full)
    assert lab_full.best_pair() == champion_indices(full)
    assert (true_pair_excess(lab_flat)["excess_rate"].max()
            < true_pair_excess(lab_full)["excess_rate"].max())
