"""Phase 1 model tests."""

import numpy as np
import pytest

from vlab_doe.models import chromatography as chrom
from vlab_doe.models import ufdf


# ── Chromatography ────────────────────────────────────────────────────────────

def _make_chrom_config(salt: float = 300.0, load_density: float = 2.0) -> chrom.ChromatographyConfig:
    """A fast-running test config: high salt → weak binding → peak emerges quickly."""
    return chrom.ChromatographyConfig(
        geometry=chrom.ColumnGeometry(length=0.1, diameter=0.01, porosity=0.4),
        velocity=1e-3,
        dispersion=1e-7,
        isotherm=chrom.SMAParameters(
            equilibrium_constant=[0.02],
            characteristic_charge=[2.0],
            steric_factor=[5.0],
            ionic_capacity=1000.0,
        ),
        salt=salt,
        ph=7.0,
        load_density=load_density,
    )


def test_chromatography_returns_expected_keys():
    config = _make_chrom_config()
    t_eval = np.linspace(0, 200, 100)
    result = chrom.simulate(config, t_eval)
    assert "c_outlet" in result
    assert "t" in result
    assert "henry" in result
    assert result["c_outlet"].shape[0] == 1  # single component


def test_chromatography_outlet_shape_matches_t_eval():
    config = _make_chrom_config()
    t_eval = np.linspace(0, 200, 150)
    result = chrom.simulate(config, t_eval)
    assert result["c_outlet"].shape[1] == len(t_eval)


def test_chromatography_outlet_nonnegative():
    config = _make_chrom_config()
    t_eval = np.linspace(0, 300, 100)
    result = chrom.simulate(config, t_eval)
    assert float(result["c_outlet"].min()) >= -1e-6  # allow tiny numerical noise


def test_chromatography_outlet_concentration_eventually_rises():
    """Protein injected at the inlet must emerge at the outlet (low binding case)."""
    config = _make_chrom_config(salt=500.0, load_density=2.0)
    H = chrom.sma_henry_constant(500.0, 7.0, config.isotherm)
    R = 1 + 0.6 / 0.4 * H  # retardation factor
    # Time for protein front to traverse column: t_front = L * R / u
    t_front = config.geometry.length * R / config.velocity
    t_eval = np.linspace(0, t_front * 3, 200)
    result = chrom.simulate(config, t_eval)
    # At late time, outlet should see significant concentration (> 0.1 * c_feed)
    assert float(result["c_outlet"][0, -1]) > 0.1


def test_chromatography_higher_salt_gives_lower_henry():
    """More salt → weaker binding → smaller Henry constant (SMA salt dependence)."""
    iso = chrom.SMAParameters(
        equilibrium_constant=[1.0],
        characteristic_charge=[3.0],
        steric_factor=[5.0],
        ionic_capacity=1000.0,
    )
    H_low = chrom.sma_henry_constant(100.0, 7.0, iso)
    H_high = chrom.sma_henry_constant(300.0, 7.0, iso)
    assert H_high < H_low


def test_langmuir_isotherm_positive():
    q, dq_dc = chrom.langmuir_isotherm(
        np.array([0.5, 0.2]), np.array([10.0, 8.0]), np.array([0.5, 0.3])
    )
    assert np.all(q >= 0)
    assert np.all(dq_dc >= 0)


def test_pool_metrics_full_pool():
    """Pool over the entire chromatogram should have yield = 1 and purity = 1."""
    t = np.linspace(0, 10, 100)
    c_outlet = np.exp(-((t - 5) ** 2) / 2.0)[np.newaxis, :]  # single Gaussian peak
    metrics = chrom.pool_metrics(t, c_outlet, cut_start=0, cut_end=10, target_index=0)
    assert abs(metrics["yield"] - 1.0) < 0.01
    assert abs(metrics["purity"] - 1.0) < 1e-9


# ── UF / DF ───────────────────────────────────────────────────────────────────

def _make_ufdf_config(tmp: float = 1.0, crossflow: float = 1.0) -> ufdf.UFDFConfig:
    return ufdf.UFDFConfig(
        membrane=ufdf.MembraneProperties(
            area=0.1,
            hydraulic_resistance=1e12,
            sieving_coefficient=0.0,
        ),
        tmp=tmp,
        crossflow_velocity=crossflow,
        feed_concentration=5.0,
        feed_volume=1.0,
        target_concentration=20.0,
    )


def test_ufdf_returns_expected_keys():
    config = _make_ufdf_config()
    result = ufdf.simulate(config, np.linspace(0, 3600, 50))
    for key in ("t", "flux", "retentate_concentration", "retentate_volume", "yield"):
        assert key in result


def test_ufdf_concentration_increases():
    """Retentate concentration must rise during UF (perfect retention)."""
    config = _make_ufdf_config()
    result = ufdf.simulate(config, np.linspace(0, 3600, 100))
    assert result["retentate_concentration"][-1] > config.feed_concentration


def test_ufdf_volume_decreases():
    """Retentate volume must decrease during UF."""
    config = _make_ufdf_config()
    result = ufdf.simulate(config, np.linspace(0, 3600, 100))
    assert result["retentate_volume"][-1] < config.feed_volume


def test_ufdf_mass_conservation():
    """Mass = C * V must be conserved (sieving = 0)."""
    config = _make_ufdf_config()
    result = ufdf.simulate(config, np.linspace(0, 1800, 100))
    C = result["retentate_concentration"]
    V = result["retentate_volume"]  # L
    initial_mass = config.feed_concentration * config.feed_volume
    final_mass = float(C[-1]) * float(V[-1])
    assert abs(final_mass - initial_mass) / initial_mass < 0.02  # <2% error


def test_ufdf_higher_crossflow_gives_higher_flux():
    """Higher cross-flow → larger mass-transfer coeff → higher permeate flux."""
    c_bulk = 5.0
    low = ufdf.permeate_flux(c_bulk, _make_ufdf_config(crossflow=0.5))
    high = ufdf.permeate_flux(c_bulk, _make_ufdf_config(crossflow=2.0))
    assert high > low
