"""Phase 1.3 perturbation tests."""

import numpy as np
import pytest

from vlab_doe.config import make_rng
from vlab_doe import perturbation as pert


def test_make_rng_is_reproducible():
    a = make_rng(123).standard_normal(10)
    b = make_rng(123).standard_normal(10)
    np.testing.assert_array_equal(a, b)


def test_noise_is_seed_reproducible():
    x = np.linspace(0, 1, 50)
    signal = np.sin(x)
    noise = pert.NoiseModel(additive_sd=0.05, proportional_cv=0.02)
    a = pert.add_measurement_noise(x, signal, noise, make_rng(7))
    b = pert.add_measurement_noise(x, signal, noise, make_rng(7))
    np.testing.assert_array_equal(a, b)


def test_different_seeds_give_different_noise():
    x = np.linspace(0, 1, 50)
    signal = np.ones(50)
    noise = pert.NoiseModel(additive_sd=0.1)
    a = pert.add_measurement_noise(x, signal, noise, make_rng(1))
    b = pert.add_measurement_noise(x, signal, noise, make_rng(2))
    assert not np.array_equal(a, b)


def test_additive_noise_has_correct_scale():
    """Mean absolute noise should be ≈ additive_sd * sqrt(2/π) for Gaussian."""
    x = np.zeros(10000)
    signal = np.zeros(10000)
    sd = 0.3
    noise = pert.NoiseModel(additive_sd=sd)
    observed = pert.add_measurement_noise(x, signal, noise, make_rng(0))
    assert abs(observed.std() - sd) < sd * 0.05  # within 5%


def test_drift_increases_linearly():
    """With only a drift term the observed signal should increase linearly."""
    x = np.linspace(0.0, 10.0, 100)
    signal = np.zeros(100)
    noise = pert.NoiseModel(drift_slope=0.5)
    observed = pert.add_measurement_noise(x, signal, noise, make_rng(0))
    np.testing.assert_allclose(observed, 0.5 * x, atol=1e-12)


def test_bias_shifts_signal():
    x = np.zeros(10)
    signal = np.ones(10)
    noise = pert.NoiseModel(bias=2.0)
    observed = pert.add_measurement_noise(x, signal, noise, make_rng(0))
    np.testing.assert_allclose(observed, 3.0 * np.ones(10), atol=1e-12)


def test_parameter_jitter_keeps_keys():
    params = {"k": 1.0, "nu": 4.0, "sigma": 10.0}
    out = pert.jitter_parameters(params, relative_sd=0.1, rng=make_rng(1))
    assert set(out) == set(params)


def test_parameter_jitter_changes_values():
    params = {"k": 1.0}
    out = pert.jitter_parameters(params, relative_sd=0.5, rng=make_rng(99))
    assert out["k"] != params["k"]


def test_parameter_jitter_preserves_sign():
    """Lognormal jitter preserves positive-definite parameter values."""
    params = {"a": 5.0, "b": 0.1, "c": 1000.0}
    for seed in range(20):
        out = pert.jitter_parameters(params, relative_sd=0.3, rng=make_rng(seed))
        assert all(v > 0 for v in out.values())
