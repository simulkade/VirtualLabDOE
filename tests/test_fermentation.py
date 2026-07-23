"""Phase 1.4 — milk fermentation model tests."""

import numpy as np
import pytest

from vlab_doe.config import make_rng
from vlab_doe.models import fermentation as ferm


def _grid(t_end: float = 12.0, n: int = 481) -> np.ndarray:
    return np.linspace(0.0, t_end, n)


# ── Basic forward model ─────────────────────────────────────────────────────────

def test_run_returns_expected_shapes():
    setup = ferm.FermentationSetup(consortium=ferm.yogurt_blend(), temperature=43.0)
    t = _grid()
    r = ferm.run_fermentation(setup, t)
    assert r.ph.shape == t.shape
    assert r.biomass.shape == (2, len(t))
    assert r.substrate.shape == t.shape
    assert len(r.strain_names) == 2


def test_ph_drops_monotonically_from_fresh_milk():
    """A yogurt batch starts near fresh-milk pH and acidifies past the set point."""
    setup = ferm.FermentationSetup(consortium=ferm.yogurt_blend(), temperature=43.0)
    r = ferm.run_fermentation(setup, _grid())
    assert r.ph[0] > 6.4                       # fresh milk
    assert r.ph[-1] < 4.7                       # set
    # Monotone non-increasing (acid only accumulates).
    assert np.all(np.diff(r.ph) <= 1e-9)


def test_states_nonnegative_and_substrate_consumed():
    setup = ferm.FermentationSetup(consortium=ferm.yogurt_blend(), temperature=43.0)
    r = ferm.run_fermentation(setup, _grid())
    assert r.biomass.min() >= -1e-9
    assert r.lactic_acid.min() >= -1e-9
    assert r.substrate[-1] < r.substrate[0]    # lactose consumed
    assert r.biomass[:, -1].sum() > r.biomass[:, 0].sum()  # growth happened


# ── Process-factor responses (the DoE handles) ──────────────────────────────────

def test_set_time_decreases_with_temperature_toward_optimum():
    blend = ferm.yogurt_blend()
    t = _grid()
    t_set_40 = ferm.time_to_ph(*_ph(blend, 40.0, t))
    t_set_45 = ferm.time_to_ph(*_ph(blend, 45.0, t))
    assert t_set_45 < t_set_40                  # 45 °C is closer to the LB/ST optimum


def test_set_time_decreases_with_inoculum():
    blend = ferm.yogurt_blend()
    t = _grid()
    low = ferm.run_fermentation(
        ferm.FermentationSetup(consortium=blend, temperature=43.0, total_inoculum=0.005), t)
    high = ferm.run_fermentation(
        ferm.FermentationSetup(consortium=blend, temperature=43.0, total_inoculum=0.05), t)
    assert ferm.time_to_ph(high.t, high.ph, 4.6) < ferm.time_to_ph(low.t, low.ph, 4.6)


def test_cold_incubation_fails_to_set():
    """Well below the cardinal minimum, growth is negligible and milk barely acidifies."""
    setup = ferm.FermentationSetup(consortium=ferm.yogurt_blend(), temperature=15.0)
    r = ferm.run_fermentation(setup, _grid())
    assert r.ph[-1] > 5.5


def _ph(consortium, temperature, t):
    r = ferm.run_fermentation(
        ferm.FermentationSetup(consortium=consortium, temperature=temperature), t)
    return r.t, r.ph, 4.6


# ── Strain personalities & interaction ──────────────────────────────────────────

def test_st_acidifies_faster_early_than_lb_alone():
    t = _grid()
    st = ferm.run_fermentation(
        ferm.FermentationSetup(consortium=ferm.single_strain(ferm.streptococcus_thermophilus())), t)
    lb = ferm.run_fermentation(
        ferm.FermentationSetup(consortium=ferm.single_strain(ferm.lactobacillus_bulgaricus())), t)
    # ST reaches gelation (pH 5.2) before LB does.
    assert ferm.time_to_ph(st.t, st.ph, 5.2) < ferm.time_to_ph(lb.t, lb.ph, 5.2)


def test_st_alone_stalls_above_set_point():
    """ST is acid-sensitive: on its own it cannot drive the milk to the set pH."""
    r = ferm.run_fermentation(
        ferm.FermentationSetup(consortium=ferm.single_strain(ferm.streptococcus_thermophilus())),
        _grid())
    assert r.ph[-1] > 4.6


def test_cooperation_reaches_set_point_faster():
    """ST↔LB proto-cooperation reaches the set point sooner than the independent mix."""
    t = _grid()
    coop = ferm.run_fermentation(
        ferm.FermentationSetup(consortium=ferm.yogurt_blend(cooperation=1.5)), t)
    indep = ferm.run_fermentation(
        ferm.FermentationSetup(consortium=ferm.yogurt_blend(cooperation=0.0)), t)
    t_coop = ferm.time_to_ph(coop.t, coop.ph, 4.6)
    t_indep = ferm.time_to_ph(indep.t, indep.ph, 4.6)
    # Either the independent mix never sets (nan) or it sets later.
    assert np.isfinite(t_coop)
    assert not np.isfinite(t_indep) or t_coop < t_indep


# ── Cardinal temperature & titration sub-models ─────────────────────────────────

def test_cardinal_factor_peaks_at_optimum_and_zero_outside():
    s = ferm.streptococcus_thermophilus()
    assert ferm.cardinal_temperature_factor(s.t_opt, s) == pytest.approx(1.0, abs=1e-6)
    assert ferm.cardinal_temperature_factor(s.t_min - 1, s) == 0.0
    assert ferm.cardinal_temperature_factor(s.t_max + 1, s) == 0.0


def test_ph_from_acid_monotone_decreasing():
    milk = ferm.Milk()
    L = np.linspace(0, 200, 50)
    ph = ferm.ph_from_acid(L, milk)
    assert ph[0] == pytest.approx(milk.ph0, abs=1e-6)
    assert np.all(np.diff(ph) <= 0)
    assert ph[-1] > milk.ph_inf


# ── Stochastic layers ───────────────────────────────────────────────────────────

def test_process_noise_keeps_biomass_nonnegative():
    rng = make_rng(0)
    setup = ferm.FermentationSetup(
        consortium=ferm.yogurt_blend(), temperature=43.0, process_noise_sd=0.2)
    r = ferm.run_fermentation(setup, _grid(n=241), rng=rng)
    assert r.biomass.min() >= 0.0
    assert r.ph[-1] < 5.0


def test_process_noise_requires_rng():
    setup = ferm.FermentationSetup(consortium=ferm.yogurt_blend(), process_noise_sd=0.1)
    with pytest.raises(ValueError):
        ferm.run_fermentation(setup, _grid(n=50))


def test_batch_variability_produces_spread():
    rng = make_rng(7)
    setup = ferm.FermentationSetup(consortium=ferm.yogurt_blend(), temperature=43.0)
    batches = ferm.run_batches(setup, ferm.BatchVariability(), 20, _grid(n=241), rng)
    t_sets = np.array([ferm.time_to_ph(b.t, b.ph, 4.6) for b in batches])
    finite = t_sets[np.isfinite(t_sets)]
    assert len(finite) >= 15                    # most batches set
    assert finite.std() > 0.1                   # genuine batch-to-batch spread


def test_observe_ph_adds_measurement_noise():
    rng = make_rng(3)
    r = ferm.run_fermentation(
        ferm.FermentationSetup(consortium=ferm.yogurt_blend(), temperature=43.0), _grid())
    sample_t = np.linspace(0, 12, 25)
    obs = ferm.observe_ph(r, sample_t, rng=rng)
    assert obs["ph"].shape == sample_t.shape
    assert not np.allclose(obs["ph"], obs["ph_true"])   # noise was added
    assert np.std(obs["ph"] - obs["ph_true"]) < 0.1     # but small


# ── Fingerprint / target matching ───────────────────────────────────────────────

def test_fingerprint_has_expected_keys():
    r = ferm.run_fermentation(
        ferm.FermentationSetup(consortium=ferm.yogurt_blend(), temperature=43.0), _grid())
    fp = ferm.fingerprint(r)
    for key in ("t_gel", "t_set", "final_ph", "post_acidification", "max_rate", "aroma"):
        assert key in fp


def test_fingerprint_distance_zero_to_self_and_positive_to_other():
    t = _grid()
    ref = ferm.fingerprint(ferm.run_fermentation(
        ferm.FermentationSetup(consortium=ferm.yogurt_blend(fraction_st=0.5, fraction_lb=0.5),
                               temperature=43.0), t))
    other = ferm.fingerprint(ferm.run_fermentation(
        ferm.FermentationSetup(consortium=ferm.yogurt_blend(fraction_st=0.9, fraction_lb=0.1),
                               temperature=39.0), t))
    assert ferm.fingerprint_distance(ref, ref) == 0.0
    assert ferm.fingerprint_distance(other, ref) > 0.0


def test_time_to_ph_returns_nan_when_target_not_reached():
    t = np.linspace(0, 1, 10)
    ph = np.full_like(t, 6.5)
    assert np.isnan(ferm.time_to_ph(t, ph, 4.6))
