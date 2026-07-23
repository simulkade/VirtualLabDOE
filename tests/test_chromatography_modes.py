"""Tests for the multi-mode chromatography engine (CEX/AEX, HIC, RP) and gradients."""

import numpy as np
import pytest

from vlab_doe.models import chromatography as chrom

GEOM = chrom.ColumnGeometry(length=0.1, diameter=0.01, porosity=0.4)


def _retention_time(result: chrom.ChromatogramResult, idx: int = 0) -> float:
    return chrom.peak_moments(result.t, result.c_outlet[idx])["retention_time"]


def _isocratic_setup(isotherm, modulator, *, load=2.0, feed=(1.0,), n_cells=60):
    inj = chrom.Injection.from_load_density(load, list(feed), GEOM.porosity)
    program = chrom.ElutionProgram.isocratic(modulator, inj, run_cv=12.0)
    return chrom.ColumnSetup(
        geometry=GEOM, velocity=1e-3, dispersion=1e-7,
        isotherm=isotherm, program=program, n_cells=n_cells,
    )


# ── Engine basics ─────────────────────────────────────────────────────────────

def test_run_column_shapes_and_nonnegative():
    iso = chrom.cation_exchange(beta=[5e-3], nu=[4.0], q_max=120.0, nu_ph=0.0, linear=True)
    res = chrom.run_column(_isocratic_setup(iso, 400.0))
    assert res.c_outlet.shape[0] == 1
    assert res.c_outlet.shape[1] == len(res.t)
    assert res.m_outlet.shape == res.t.shape
    assert float(res.c_outlet.min()) >= -1e-6


def test_two_component_outlet_has_two_rows():
    iso = chrom.high_resolution_iex(beta=[5e-3, 6e-3], nu=[4.0, 4.3], q_max=120.0, nu_ph=0.0)
    inj = chrom.Injection.from_load_density(4.0, [0.5, 0.5], GEOM.porosity)
    program = chrom.ElutionProgram.linear_gradient(
        inj, m_start=50.0, m_end=500.0, gradient_cv=12.0)
    res = chrom.run_column(
        chrom.ColumnSetup(GEOM, 1e-3, 1e-7, iso, program, mass_transfer=2.0, n_cells=80))
    assert res.c_outlet.shape[0] == 2


# ── Mode-specific elution physics ─────────────────────────────────────────────

def test_iex_higher_salt_elutes_earlier():
    """Ion exchange: more salt → weaker binding → earlier elution."""
    iso = chrom.cation_exchange(beta=[5e-3], nu=[4.0], q_max=120.0, nu_ph=0.0, linear=True)
    t_low = _retention_time(chrom.run_column(_isocratic_setup(iso, 300.0)))
    t_high = _retention_time(chrom.run_column(_isocratic_setup(iso, 500.0)))
    assert t_high < t_low


def test_hic_higher_salt_elutes_later():
    """HIC salting-out: more salt → STRONGER binding → later elution (opposite of IEX)."""
    iso = chrom.hic(beta=[1e-3], ks=[0.01], q_max=120.0, linear=True)
    t_low = _retention_time(chrom.run_column(_isocratic_setup(iso, 200.0)))
    t_high = _retention_time(chrom.run_column(_isocratic_setup(iso, 400.0)))
    assert t_high > t_low


def test_rp_higher_organic_elutes_earlier():
    """RP-HPLC: more organic modifier → weaker binding → earlier elution."""
    iso = chrom.reversed_phase(beta=[1e3], s=[20.0], q_max=120.0, linear=True)
    t_low = _retention_time(chrom.run_column(_isocratic_setup(iso, 0.30)))
    t_high = _retention_time(chrom.run_column(_isocratic_setup(iso, 0.45)))
    assert t_high < t_low


def test_iex_salt_gradient_elutes_protein():
    """A bound protein washed at low salt elutes when the salt gradient reaches it."""
    iso = chrom.cation_exchange(beta=[5e-3], nu=[4.0], q_max=120.0, nu_ph=0.0)
    inj = chrom.Injection.from_load_density(8.0, [1.0], GEOM.porosity)
    program = chrom.ElutionProgram.linear_gradient(
        inj, m_start=50.0, m_end=500.0, gradient_cv=15.0)
    res = chrom.run_column(chrom.ColumnSetup(GEOM, 1e-3, 1e-7, iso, program, n_cells=80))
    mom = chrom.peak_moments(res.t, res.c_outlet[0])
    # A real peak emerges, and it does so during the rising-salt portion of the run.
    assert mom["area"] > 0.0
    salt_at_apex = float(np.interp(mom["apex_time"], res.t, res.m_outlet))
    assert salt_at_apex > 50.0


# ── Resolution / high-resolution IEX ──────────────────────────────────────────

def test_shallow_gradient_improves_resolution():
    """Two closely-spaced species resolve better under a shallower gradient."""
    def resolution_for(gradient_cv: float) -> float:
        iso = chrom.high_resolution_iex(
            beta=[5e-3, 6e-3], nu=[4.0, 4.3], q_max=120.0, nu_ph=0.0)
        inj = chrom.Injection.from_load_density(5.0, [0.5, 0.5], GEOM.porosity)
        program = chrom.ElutionProgram.linear_gradient(
            inj, m_start=50.0, m_end=500.0, gradient_cv=gradient_cv)
        res = chrom.run_column(
            chrom.ColumnSetup(GEOM, 1e-3, 1e-7, iso, program, mass_transfer=2.0, n_cells=100))
        return chrom.resolution(res.t, res.c_outlet[0], res.c_outlet[1])

    assert resolution_for(30.0) > resolution_for(8.0)


# ── Isotherm-level invariants ─────────────────────────────────────────────────

def test_henry_limit_recovers_linear_isotherm():
    """At vanishing load the nonlinear competitive isotherm matches the Henry slope."""
    kwargs = dict(beta=[5e-3], nu=[4.0], q_max=120.0, nu_ph=0.0)
    iso_lin = chrom.cation_exchange(linear=True, **kwargs)
    iso_nl = chrom.cation_exchange(linear=False, **kwargs)
    c = np.array([1e-6])
    q_lin = iso_lin.q_star(c, 100.0, 7.0)
    q_nl = iso_nl.q_star(c, 100.0, 7.0)
    assert np.allclose(q_lin, q_nl, rtol=1e-3)
    # Linear branch is exactly Henry * c.
    assert np.allclose(q_lin, iso_lin.henry(100.0, 7.0) * c)


def test_overload_reduces_bound_fraction():
    """Nonlinear isotherm saturates: q*/c at high load is below the Henry slope."""
    iso = chrom.cation_exchange(beta=[5e-3], nu=[4.0], q_max=120.0, nu_ph=0.0)
    henry = float(iso.henry(100.0, 7.0)[0])
    q_high = float(iso.q_star(np.array([5.0]), 100.0, 7.0)[0])
    assert q_high / 5.0 < henry


def test_sma_law_henry_matches_legacy():
    """The IEX modulator law reduces to the legacy linearised SMA Henry constant."""
    iso = chrom.cation_exchange(beta=[1.0], nu=[3.0], ionic_capacity=1000.0,
                                q_max=1.0, nu_ph=0.0)
    params = chrom.SMAParameters(
        equilibrium_constant=[1.0], characteristic_charge=[3.0],
        steric_factor=[5.0], ionic_capacity=1000.0)
    h_new = float(iso.henry(250.0, 7.0)[0])
    h_legacy = chrom.sma_henry_constant(250.0, 7.0, params)
    assert np.isclose(h_new, h_legacy, rtol=1e-9)


def test_aex_pH_opposite_to_cex():
    """CEX binds harder as pH falls; AEX binds harder as pH rises (opposite signs)."""
    cex = chrom.cation_exchange(beta=[5e-3], nu=[4.0], q_max=120.0)
    aex = chrom.anion_exchange(beta=[5e-3], nu=[4.0], q_max=120.0)
    h_cex_low, h_cex_high = cex.henry(200.0, 6.0)[0], cex.henry(200.0, 8.0)[0]
    h_aex_low, h_aex_high = aex.henry(200.0, 6.0)[0], aex.henry(200.0, 8.0)[0]
    assert h_cex_low > h_cex_high   # CEX stronger at low pH
    assert h_aex_high > h_aex_low   # AEX stronger at high pH


def test_modulator_equilibrates_to_inlet():
    """The unretained modulator leaving the column approaches the inlet value."""
    iso = chrom.cation_exchange(beta=[5e-3], nu=[4.0], q_max=120.0, nu_ph=0.0, linear=True)
    res = chrom.run_column(_isocratic_setup(iso, 350.0))
    assert abs(float(res.m_outlet[-1]) - 350.0) < 5.0
