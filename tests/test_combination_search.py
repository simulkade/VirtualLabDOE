"""Phase 2.x — strain-pair screening: the panel's ground truth and the search designs."""

import numpy as np
import pytest

from vlab_doe.doe.combination_search import (
    STRATEGIES,
    VirtualStrainLab,
    acidification_rate,
    fit_scheffe,
    predict_pairs,
    scheffe_features,
    summarize_benchmark,
)
from vlab_doe.models.fermentation.panel import (
    CHAMPION_PAIR,
    PANEL_INTERACTIONS,
    panel_groups,
    yogurt_strain_panel,
)

BUDGET = 30


@pytest.fixture(scope="module")
def library():
    return yogurt_strain_panel()


@pytest.fixture(scope="module")
def lab(library):
    return VirtualStrainLab(library=library, seed=2026)


# ── The panel ───────────────────────────────────────────────────────────────────

def test_panel_shape_and_groups(library):
    assert len(library.strains) == 16
    groups = panel_groups()
    assert set(groups) == set(library.names)
    counts = {g: sum(1 for v in groups.values() if v == g) for g in set(groups.values())}
    assert counts["ST"] == 5 and counts["LB"] == 4 and counts["LH"] == 2


def test_interaction_matrix_is_signed_and_asymmetric(library):
    k = library.interaction
    assert k.shape == (16, 16)
    assert np.allclose(np.diag(k), 0.0)             # no self-interaction
    assert (k > 0).any() and (k < 0).any()          # both synergy and antagonism planted
    assert not np.allclose(k, k.T)                  # LH -> ST proteolysis is one-way


def test_planted_antagonism_reaches_the_matrix(library):
    idx = {n: i for i, n in enumerate(library.names)}
    for receiver, donor, k, _ in PANEL_INTERACTIONS:
        if k < 0:
            assert library.interaction[idx[receiver], idx[donor]] < 0


def test_mesophilic_duds_never_acidify(lab):
    """LL isolates have t_max < 43 C, so they must not reach the set point alone."""
    singles = lab.single_truth()
    for name in ("LL-01", "LL-02"):
        assert singles[name] >= lab.horizon - 1e-9


# ── The ground truth the study is graded against ────────────────────────────────

def test_champion_pair_is_the_true_optimum(lab, library):
    """The panel is calibrated so CHAMPION_PAIR wins outright; the study depends on it."""
    truth = lab.pair_truth_table()
    assert len(truth) == 120
    best = truth.iloc[0]
    assert {best["a"], best["b"]} == set(CHAMPION_PAIR)
    assert truth.iloc[1]["value"] - best["value"] > 0.5     # a real, not marginal, margin


def test_champion_is_not_the_pair_of_the_best_soloists(lab):
    """The whole point of the study: synergy, not raw single-strain speed, decides."""
    best_two = lab.single_truth().sort_values().index[:2]
    assert set(best_two) != set(CHAMPION_PAIR)


def test_pair_rank_and_regret_agree_with_the_table(lab, library):
    idx = {n: i for i, n in enumerate(library.names)}
    champion = tuple(sorted((idx[CHAMPION_PAIR[0]], idx[CHAMPION_PAIR[1]])))
    assert lab.best_pair() == champion
    assert lab.pair_rank(champion) == 1
    assert lab.regret(champion) == pytest.approx(0.0, abs=1e-9)
    worst = tuple(lab.pair_truth_table().iloc[-1]["members"])
    assert lab.regret(worst) > 0.0


# ── Noise and reproducibility ───────────────────────────────────────────────────

def test_replicates_differ_but_are_reproducible(library):
    lab_a = VirtualStrainLab(library=library, seed=11)
    lab_b = VirtualStrainLab(library=library, seed=11)
    members = (0, 5)
    runs_a = [lab_a.run(members).value for _ in range(4)]
    runs_b = [lab_b.run(members).value for _ in range(4)]
    assert runs_a == pytest.approx(runs_b)          # same seed -> same batches
    assert len(set(runs_a)) > 1                     # but batches are not identical


def test_noise_is_comparable_to_the_champion_margin(lab, library):
    """If a single jar were decisive the design question would be uninteresting."""
    idx = {n: i for i, n in enumerate(library.names)}
    champion = tuple(sorted((idx[CHAMPION_PAIR[0]], idx[CHAMPION_PAIR[1]])))
    vals = np.array([lab._draw(champion, r).value for r in range(20)])
    truth = lab.pair_truth_table()
    margin = truth.iloc[1]["value"] - truth.iloc[0]["value"]
    assert 0.1 * margin < vals.std(ddof=1) < 3.0 * margin


# ── The Scheffé mixture model ───────────────────────────────────────────────────

def test_scheffe_features_are_mixture_fractions():
    x = scheffe_features([(0, 1), (0, 1, 2)], n_strains=4)
    assert x.shape == (2, 4 + 6)
    assert x[0, :4].sum() == pytest.approx(1.0)     # inoculum fractions sum to one
    assert x[1, :4].sum() == pytest.approx(1.0)
    assert x[0, 4] == pytest.approx(0.25)           # x0*x1 = 0.5*0.5 for a pair
    assert x[0, 5] == pytest.approx(0.0)            # strain 2 absent -> no 0x2 term


def test_fit_scheffe_recovers_a_planted_synergy():
    rng = np.random.default_rng(0)
    n = 6
    blends = [(i, j) for i in range(n) for j in range(i + 1, n)]
    truth = np.zeros(n + len(blends))
    truth[:n] = 0.2
    truth[n + blends.index((1, 3))] = 4.0          # one strong pair
    x = scheffe_features(blends, n)
    y = x @ truth + rng.normal(0, 1e-3, len(blends))
    coefs, _ = fit_scheffe(blends, y, n, rng=rng)
    assert int(np.argmax(predict_pairs(coefs, n))) == blends.index((1, 3))


def test_acidification_rate_is_finite_for_censored_runs(lab):
    """Censoring is why the model fits 1/t_set: it stays finite and ordered."""
    runs = [lab.run(lab.best_pair()), lab.run((14, 15))]   # the champion and the two duds
    rates = acidification_rate(runs)
    assert np.all(np.isfinite(rates)) and np.all(rates > 0)
    assert rates[0] > rates[1]                              # censored duds rank last, not NaN


# ── The search designs ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_strategy_respects_budget_and_returns_a_pair(library, name):
    lab = VirtualStrainLab(library=library, seed=5)
    res = STRATEGIES[name](lab, BUDGET, np.random.default_rng(3))
    assert res.n_experiments <= BUDGET
    assert len(res.choice) == 2 and len(set(res.choice)) == 2
    assert res.choice == tuple(sorted(res.choice))
    assert len(res.trace) == res.n_experiments
    assert set(res.choice) <= set(range(lab.n_strains))


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_strategy_is_reproducible(library, name):
    def once():
        lab = VirtualStrainLab(library=library, seed=5)
        return STRATEGIES[name](lab, BUDGET, np.random.default_rng(3)).choice
    assert once() == once()


def test_thompson_sampling_proposes_multi_strain_blends(library):
    """Triples are what let one jar inform three synergy coefficients at once."""
    lab = VirtualStrainLab(library=library, seed=5)
    res = STRATEGIES["Thompson sampling"](lab, 60, np.random.default_rng(3))
    assert (res.trace["size"] > 2).any()


def test_exhaustive_scan_finds_the_champion_with_replication(library):
    """Given enough budget to replicate, brute force must land on the true best pair."""
    lab = VirtualStrainLab(library=library, seed=5)
    res = STRATEGIES["exhaustive pair scan"](lab, 360, np.random.default_rng(1))
    assert lab.pair_rank(res.choice) <= 3


# ── The benchmark harness ───────────────────────────────────────────────────────

def test_summarize_benchmark_aggregates_per_strategy_and_budget(library):
    from vlab_doe.doe.combination_search import benchmark_strategies

    results = benchmark_strategies(library, budgets=[BUDGET], n_campaigns=2, base_seed=1)
    assert len(results) == 2 * len(STRATEGIES)
    summary = summarize_benchmark(results)
    assert len(summary) == len(STRATEGIES)
    assert summary["hit_rate"].between(0, 1).all()
    assert (summary["mean_regret_h"] >= 0).all()
    assert (summary["mean_experiments"] <= BUDGET).all()
