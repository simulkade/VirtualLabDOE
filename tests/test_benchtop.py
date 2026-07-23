"""Tests for the benchtop physical/chemical models and their use in DoE.

Each model is checked against a closed-form limit, for physically correct
monotonicity, and (for two of them) for a Design-of-Experiments round trip: run
a factorial through the model and confirm the analysis recovers the known
structure or parameter.
"""

import numpy as np
import pytest

from vlab_doe.doe.analysis import fit_response_model
from vlab_doe.doe.factorial import Factor, full_factorial, run_design
from vlab_doe.models.benchtop import back_extrusion as be
from vlab_doe.models.benchtop import ester_hydrolysis as eh
from vlab_doe.models.benchtop import falling_ball as fb
from vlab_doe.models.benchtop import pipe_flow as pf


# ── Pipe flow ─────────────────────────────────────────────────────────────────

def test_pipe_laminar_equals_hagen_poiseuille():
    """In the laminar regime, Darcy--Weisbach must equal Hagen--Poiseuille."""
    cfg = pf.PipeFlowConfig(flow_rate=3e-6, diameter=4e-3, length=1.0, temperature=20.0)
    assert pf.reynolds_number(cfg) < 2300.0
    assert pf.pressure_drop(cfg) == pytest.approx(pf.hagen_poiseuille(cfg), rel=1e-9)


def test_pipe_water_properties():
    """Water correlations return the textbook 20 °C values."""
    assert pf.water_viscosity(20.0) == pytest.approx(1.0e-3, rel=0.02)
    assert pf.water_density(20.0) == pytest.approx(998.2, rel=1e-3)


def test_pipe_diameter_fourth_power_law():
    """Halving the diameter multiplies laminar ΔP by 16 (the d^-4 law)."""
    a = pf.PipeFlowConfig(flow_rate=2e-6, diameter=4e-3, length=1.0)
    b = pf.PipeFlowConfig(flow_rate=2e-6, diameter=2e-3, length=1.0)
    assert pf.pressure_drop(b) / pf.pressure_drop(a) == pytest.approx(16.0, rel=1e-6)


def test_pipe_friction_factor_continuous_across_transition():
    """The blended friction factor is continuous through Re = 2300 and 4000."""
    for re_edge in (2300.0, 4000.0):
        below = pf.friction_factor(re_edge * 0.999)
        above = pf.friction_factor(re_edge * 1.001)
        assert below == pytest.approx(above, rel=2e-3)


def test_pipe_monotonic_in_flow():
    curve = pf.flow_curve(np.linspace(1e-6, 3e-4, 40), diameter=4e-3, length=1.0)
    assert np.all(np.diff(curve["pressure_drop"]) > 0)


# ── Falling ball ──────────────────────────────────────────────────────────────

def _ball(mu: float, d: float = 1.5e-3) -> fb.FallingBallConfig:
    return fb.FallingBallConfig(
        ball_diameter=d, ball_density=7800.0, fluid_density=1260.0,
        fluid_viscosity=mu, tube_diameter=0.03, fall_distance=0.1,
    )


def test_falling_ball_wide_tube_recovers_stokes():
    """With a tiny ball in a very wide tube and high viscosity (Re→0),
    the terminal velocity approaches the unbounded Stokes value."""
    cfg = fb.FallingBallConfig(
        ball_diameter=1e-4, ball_density=7800.0, fluid_density=1000.0,
        fluid_viscosity=10.0, tube_diameter=1.0, fall_distance=0.1,
    )
    assert fb.reynolds_number(cfg) < 1e-3
    assert fb.terminal_velocity(cfg) == pytest.approx(fb.stokes_velocity(cfg), rel=1e-3)


def test_falling_ball_wall_slows_the_ball():
    """A narrower tube (larger d/D) gives a smaller wall factor and slower fall."""
    assert fb.wall_factor(2e-3, 0.05) > fb.wall_factor(2e-3, 0.01)
    wide = _ball(1.0)
    narrow = fb.FallingBallConfig(
        ball_diameter=1.5e-3, ball_density=7800.0, fluid_density=1260.0,
        fluid_viscosity=1.0, tube_diameter=0.006, fall_distance=0.1,
    )
    assert fb.terminal_velocity(narrow) < fb.terminal_velocity(wide)


def test_falling_ball_full_inversion_recovers_viscosity():
    """Inverting the full model recovers the exact viscosity used to make t."""
    cfg = _ball(0.6)
    t = fb.fall_time(cfg)
    assert fb.infer_viscosity(t, cfg, stokes_only=False) == pytest.approx(0.6, rel=1e-4)


def test_falling_ball_stokes_reading_is_biased_high():
    """The naive Stokes reading over-estimates μ because it ignores inertia,
    and the bias grows for a bigger, faster ball (higher Re)."""
    small = _ball(1.0, d=1.0e-3)
    big = _ball(1.0, d=3.0e-3)
    est_small = fb.infer_viscosity(fb.fall_time(small), small, stokes_only=True)
    est_big = fb.infer_viscosity(fb.fall_time(big), big, stokes_only=True)
    assert est_small > 1.0  # biased high
    assert est_big > est_small  # bias grows with Re
    assert fb.reynolds_number(big) > fb.reynolds_number(small)


# ── Back extrusion ────────────────────────────────────────────────────────────

def test_back_extrusion_newtonian_slit_limit():
    """With τ0=0, n=1, K=μ the pressure gradient equals the analytic
    Newtonian slit result ΔP/L = 2μ/b · (6 ū / b)."""
    mu = 2.5
    hb = be.HerschelBulkley(consistency=mu, flow_index=1.0, yield_stress=0.0,
                            activation_energy=0.0)
    cfg = be.BackExtrusionConfig(rheology=hb, probe_radius=0.010, cup_radius=0.0125,
                                 immersion_depth=0.02, probe_speed=1e-3, density=0.0)
    b, w, u_bar = be._gap_geometry(cfg)
    analytic = (2.0 * mu / b) * (6.0 * u_bar / b)  # (2n+1)/n = 3, ×(2ū/b)
    assert be.pressure_gradient(cfg) == pytest.approx(analytic, rel=1e-9)


def test_back_extrusion_yield_stress_offset():
    """Raising the yield stress adds a speed-independent force offset."""
    common = dict(flow_index=0.4, ref_temperature=10.0, activation_energy=0.0)
    soft = be.HerschelBulkley(consistency=20.0, yield_stress=0.0, **common)
    firm = be.HerschelBulkley(consistency=20.0, yield_stress=40.0, **common)
    kw = dict(probe_radius=0.01, cup_radius=0.0125, immersion_depth=0.02,
              probe_speed=1e-3, temperature=10.0, density=0.0)
    assert be.peak_force(be.BackExtrusionConfig(rheology=firm, **kw)) > \
        be.peak_force(be.BackExtrusionConfig(rheology=soft, **kw))


def test_back_extrusion_shear_thinning_and_temperature():
    """Force rises sublinearly with speed (n<1) and falls as product warms."""
    hb = be.HerschelBulkley(consistency=20.0, flow_index=0.4, yield_stress=30.0)
    slow = be.BackExtrusionConfig(rheology=hb, probe_speed=1e-3, temperature=10.0, density=0.0)
    fast = be.BackExtrusionConfig(rheology=hb, probe_speed=1e-2, temperature=10.0, density=0.0)
    warm = be.BackExtrusionConfig(rheology=hb, probe_speed=1e-3, temperature=25.0, density=0.0)
    # 10× speed but n=0.4 → force ratio well below 10×
    assert be.peak_force(fast) / be.peak_force(slow) < 10.0
    assert be.peak_force(warm) < be.peak_force(slow)


def test_back_extrusion_requires_gap():
    hb = be.HerschelBulkley(consistency=20.0, flow_index=0.5, yield_stress=10.0)
    with pytest.raises(ValueError):
        be.pressure_gradient(be.BackExtrusionConfig(rheology=hb, probe_radius=0.02,
                                                    cup_radius=0.02))


# ── Ester hydrolysis ──────────────────────────────────────────────────────────

def test_ester_simulation_reaches_equilibrium():
    """The long-time conversion matches the algebraic equilibrium extent."""
    cfg = eh.EsterHydrolysisConfig(temperature=25.0, catalyst_conc=0.5)
    out = eh.simulate(cfg, np.linspace(0, 20 * 3600, 400))
    assert out["conversion"][-1] == pytest.approx(eh.equilibrium_conversion(cfg), rel=1e-3)


def test_ester_catalyst_speeds_rate_not_equilibrium():
    """More acid reaches equilibrium sooner but at the same conversion."""
    low = eh.EsterHydrolysisConfig(temperature=25.0, catalyst_conc=0.1)
    high = eh.EsterHydrolysisConfig(temperature=25.0, catalyst_conc=0.5)
    target = 0.9 * eh.equilibrium_conversion(low)
    assert eh.time_to_conversion(high, target) < eh.time_to_conversion(low, target)
    assert eh.equilibrium_conversion(high) == pytest.approx(
        eh.equilibrium_conversion(low), rel=1e-9)


def test_ester_endothermic_equilibrium_shifts_with_temperature():
    """Warmer runs convert more ester at equilibrium (Keq rises with T)."""
    cold = eh.EsterHydrolysisConfig(temperature=15.0)
    hot = eh.EsterHydrolysisConfig(temperature=50.0)
    assert eh.equilibrium_conversion(hot) > eh.equilibrium_conversion(cold)


def test_ester_boiling_point_guard():
    with pytest.raises(ValueError):
        eh.EsterHydrolysisConfig(temperature=eh.METHYL_ACETATE_BP_C + 1.0)


def test_ester_mass_conservation():
    """Ester consumed equals product formed at every time."""
    cfg = eh.EsterHydrolysisConfig(temperature=30.0, catalyst_conc=0.3)
    out = eh.simulate(cfg, np.linspace(0, 5 * 3600, 50))
    consumed = cfg.ester0 - out["ester"]
    assert np.allclose(consumed, out["acetic_acid"] - cfg.acid0, atol=1e-9)
    assert np.allclose(consumed, out["methanol"] - cfg.methanol0, atol=1e-9)


# ── DoE round trips ───────────────────────────────────────────────────────────

def test_pipe_doe_recovers_log_linear_effects():
    """A 2-level factorial on log-transformed pipe factors recovers the
    Hagen--Poiseuille exponents: +1 on log Q, +1 on log L, -4 on log d."""
    factors = [
        Factor("logQ", np.log(1e-6), np.log(4e-6)),
        Factor("logL", np.log(0.5), np.log(2.0)),
        Factor("logd", np.log(3e-3), np.log(5e-3)),
    ]
    design = full_factorial(factors, levels=2)

    def evaluate(point):
        cfg = pf.PipeFlowConfig(
            flow_rate=np.exp(point["logQ"]), length=np.exp(point["logL"]),
            diameter=np.exp(point["logd"]), temperature=20.0,
        )
        return {"logdP": np.log(pf.pressure_drop(cfg))}

    data = run_design(design, evaluate)
    res = fit_response_model(data, "logdP", ["logQ", "logL", "logd"], interactions=False)
    assert res.effects["logQ"] == pytest.approx(1.0, abs=0.02)
    assert res.effects["logL"] == pytest.approx(1.0, abs=0.02)
    assert res.effects["logd"] == pytest.approx(-4.0, abs=0.05)


def test_falling_ball_doe_recovers_viscosity():
    """Running a factorial of ball choices through the instrument and inverting
    each with the (calibrated) full model recovers the true fluid viscosity."""
    mu_true = 0.8
    factors = [
        Factor("d", 1.0e-3, 2.0e-3),
        Factor("rho_s", 7000.0, 8000.0),
    ]
    design = full_factorial(factors, levels=2)

    def evaluate(point):
        cfg = fb.FallingBallConfig(
            ball_diameter=point["d"], ball_density=point["rho_s"],
            fluid_density=1260.0, fluid_viscosity=mu_true,
            tube_diameter=0.03, fall_distance=0.1,
        )
        t = fb.fall_time(cfg)
        return {"mu_hat": fb.infer_viscosity(t, cfg, stokes_only=False)}

    data = run_design(design, evaluate)
    assert np.allclose(data["mu_hat"].to_numpy(), mu_true, rtol=1e-3)
