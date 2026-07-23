"""Tests for the PyFVTool general rate model (GRM) solver.

The defining property of the GRM solver, relative to the method-of-lines engine,
is that the fully-implicit coupled finite-volume discretisation is **mass
conservative** even for strongly-bound, multi-component, gradient-elution runs
(the regime where the MoL engine can lose mass).  These tests assert that.
"""

from __future__ import annotations

import numpy as np
import pytest

from vlab_doe.models.chromatography import (
    ColumnGeometry,
    ElutionProgram,
    Injection,
    GRMSetup,
    ParticleProperties,
    cation_exchange,
    film_coefficient,
    high_resolution_iex,
    run_grm,
)


def _geom():
    return ColumnGeometry(length=0.1, diameter=0.01, porosity=0.4)


def _breakthrough_injection(feed):
    # continuous feed (breakthrough): effectively-infinite load duration
    return Injection(feed=np.atleast_1d(np.asarray(feed, float)), start_cv=0.0,
                     duration_cv=1e6)


def test_film_coefficient_correlation_positive_and_scaling():
    """Wilson--Geankoplis film coefficient is positive and grows with velocity."""
    k_lo = film_coefficient(velocity=1e-3, porosity=0.4, particle_radius=4e-5,
                            molecular_diffusivity=1e-10)
    k_hi = film_coefficient(velocity=4e-3, porosity=0.4, particle_radius=4e-5,
                            molecular_diffusivity=1e-10)
    assert k_lo > 0.0
    assert k_hi > k_lo  # faster flow -> thinner film -> larger k_f


def test_single_component_breakthrough_conserves_mass():
    """Single-component isocratic breakthrough is mass-conservative."""
    iso = cation_exchange(beta=[1.0], nu=[0.0], ionic_capacity=1000.0,
                          q_max=[5.0], linear=True)
    prog = ElutionProgram.isocratic(100.0, _breakthrough_injection(1.0), run_cv=15.0)
    part = ParticleProperties(radius=4e-5, porosity=0.5, pore_diffusivity=5e-11,
                              film_coeff=3e-5, n_radial=10)
    setup = GRMSetup(geometry=_geom(), velocity=2e-3, dispersion=2e-7,
                     isotherm=iso, program=prog, particle=part, n_cells=30)
    res = run_grm(setup, n_steps=400)
    assert abs(res.mass_balance_error) < 1e-6
    # outlet rises from ~0 (retained) toward the feed (saturated)
    assert res.c_outlet[0, 5] < 0.1
    assert res.c_outlet[0, -1] > 0.9


def test_intraparticle_gradient_present():
    """The GRM resolves a radial concentration profile inside the bead."""
    iso = cation_exchange(beta=[1.0], nu=[0.0], ionic_capacity=1000.0,
                          q_max=[20.0], linear=True)
    prog = ElutionProgram.isocratic(100.0, _breakthrough_injection(1.0), run_cv=4.0)
    part = ParticleProperties(radius=4e-5, porosity=0.5, pore_diffusivity=5e-12,
                              film_coeff=3e-5, n_radial=12)
    setup = GRMSetup(geometry=_geom(), velocity=2e-3, dispersion=2e-7,
                     isotherm=iso, program=prog, particle=part, n_cells=30)
    res = run_grm(setup, n_steps=300)
    # a partially loaded bead: surface concentration exceeds the centre
    cp = res.cp_profile[0]                       # (n_cells, n_radial)
    j = res.c_outlet.shape[1]  # noqa: F841
    # pick an axial cell that is mid-loading (not fully saturated, not empty)
    surf = cp[:, -1]
    cen = cp[:, 0]
    mid = np.argmax((surf > 0.05) & (surf < 0.95))
    assert cp[mid, -1] >= cp[mid, 0] - 1e-9      # surface >= centre (uptake)
    assert cp[mid, -1] - cp[mid, 0] > 1e-3       # a genuine radial gradient


def test_strong_binding_isocratic_conserves_mass():
    """Strongly-bound multi-component isocratic load conserves mass (storage form)."""
    iso = high_resolution_iex(beta=[1.0, 1.0, 1.0], nu=[3.0, 3.5, 4.0],
                              ionic_capacity=1000.0, q_max=[10.0, 10.0, 10.0])
    inj = Injection.from_load_density(5.0, feed=[0.4, 0.4, 0.4], porosity=0.4)
    prog = ElutionProgram.isocratic(150.0, inj, run_cv=15.0)
    part = ParticleProperties(radius=4e-5, porosity=0.5, pore_diffusivity=1e-11,
                              film_coeff=2e-5, n_radial=8)
    setup = GRMSetup(geometry=_geom(), velocity=2e-3, dispersion=2e-7,
                     isotherm=iso, program=prog, particle=part, ph=5.0, n_cells=30)
    res = run_grm(setup, n_steps=300)
    assert abs(res.mass_balance_error) < 1e-8   # machine precision


def test_multicomponent_gradient_conserves_mass():
    """The multi-component nonlinear *gradient* run conserves mass to ~machine precision.

    This is the regime that motivated the GRM solver: strongly-bound, competitive,
    gradient elution.
    """
    iso = high_resolution_iex(beta=[1.0, 1.0, 1.0], nu=[3.0, 3.5, 4.0],
                              ionic_capacity=1000.0, q_max=[10.0, 10.0, 10.0])
    inj = Injection.from_load_density(3.0, feed=[0.3, 0.3, 0.3], porosity=0.4)
    prog = ElutionProgram.linear_gradient(inj, m_start=200.0, m_end=1000.0,
                                          gradient_cv=25.0, strip_cv=5.0)
    part = ParticleProperties(radius=4e-5, porosity=0.5, pore_diffusivity=2e-11,
                              film_coeff=3e-5, n_radial=10)
    setup = GRMSetup(geometry=_geom(), velocity=2e-3, dispersion=2e-7,
                     isotherm=iso, program=prog, particle=part, ph=5.0, n_cells=40)
    res = run_grm(setup, n_steps=400)
    assert abs(res.mass_balance_error) < 1e-8
    # species elute in order of increasing characteristic charge (binding strength)
    apex = [res.t[np.argmax(res.c_outlet[i])] for i in range(3)]
    assert apex[0] <= apex[1] <= apex[2]


def test_correlation_defaults_used_when_unspecified():
    """If film_coeff / pore_diffusivity are None they come from correlations."""
    iso = cation_exchange(beta=[1.0], nu=[0.0], ionic_capacity=1000.0,
                          q_max=[5.0], linear=True)
    prog = ElutionProgram.isocratic(100.0, _breakthrough_injection(1.0), run_cv=4.0)
    part = ParticleProperties(radius=4e-5, porosity=0.5, n_radial=8)  # nothing given
    setup = GRMSetup(geometry=_geom(), velocity=2e-3, dispersion=2e-7,
                     isotherm=iso, program=prog, particle=part,
                     molecular_diffusivity=1e-10, n_cells=20)
    res = run_grm(setup, n_steps=200)
    assert res.k_film[0] > 0.0
    # default pore diffusivity = eps_p * D_m / tortuosity
    assert res.pore_diffusivity[0] == pytest.approx(0.5 * 1e-10 / 2.0)
