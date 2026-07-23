"""Forward simulation of a single fermentation batch.

:func:`run_fermentation` integrates the :mod:`.kinetics` dynamics for one
:class:`FermentationSetup` and returns the **clean Mechanistic Truth**: the pH curve plus all
the latent states (biomass per strain, lactose, lactic acid, aroma).  Only the pH is ever
"measured" — see :mod:`.observe` — but the latent trajectories are kept so studies can check
what the indirect pH signal is actually standing in for.

Two integrators share one drift:

* ``process_noise_sd == 0`` → deterministic, solved with SciPy ``solve_ivp`` (LSODA, which
  copes with the stiff lag→growth transition).
* ``process_noise_sd > 0`` → an Itô SDE with multiplicative noise on biomass, integrated by
  fixed-step Euler–Maruyama; individual trajectories then wobble rather than being clean
  shifts of one another.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .kinetics import FermentationKinetics, make_kinetics
from .milk import Milk
from .strains import Consortium


@dataclass(frozen=True)
class FermentationSetup:
    """Everything needed to simulate one batch.

    Parameters
    ----------
    consortium:
        The starter culture (strains, inoculum fractions, interactions).
    milk:
        Milk base (lactose, buffering / titration curve).
    temperature:
        Incubation temperature (°C) — a primary process CPP.
    total_inoculum:
        Total starter biomass at t=0 (dimensionless scaled CFU), split across strains by the
        consortium fractions.
    process_noise_sd:
        Multiplicative biomass process-noise intensity ``sigma`` (1/√h).  ``0`` → deterministic.
    aroma_decay:
        First-order decay rate of the aroma proxy (1/h).
    """

    consortium: Consortium
    milk: Milk = Milk()
    temperature: float = 43.0
    total_inoculum: float = 0.02
    process_noise_sd: float = 0.0
    aroma_decay: float = 0.05


@dataclass
class FermentationResult:
    """Clean trajectories from one batch.

    Attributes
    ----------
    t:
        Time grid (h), shape ``(n_t,)``.
    ph:
        Milk pH, shape ``(n_t,)`` — the only quantity that is later "measured".
    biomass:
        Per-strain biomass, shape ``(n_strains, n_t)``.
    substrate, lactic_acid, aroma:
        Latent state trajectories, shape ``(n_t,)``.
    strain_names:
        Names matching the rows of ``biomass``.
    """

    t: np.ndarray
    ph: np.ndarray
    biomass: np.ndarray
    substrate: np.ndarray
    lactic_acid: np.ndarray
    aroma: np.ndarray
    strain_names: list[str]

    def as_dict(self) -> dict[str, np.ndarray]:
        """Return the trajectories as a plain dict (parallels the other models' output)."""
        return {
            "t": self.t,
            "ph": self.ph,
            "biomass": self.biomass,
            "substrate": self.substrate,
            "lactic_acid": self.lactic_acid,
            "aroma": self.aroma,
        }


def _lag_states(setup: FermentationSetup) -> np.ndarray:
    return np.array([s.lag_state for s in setup.consortium.strains], dtype=float)


def _states_to_result(
    kin: FermentationKinetics,
    t: np.ndarray,
    states: np.ndarray,
    names: list[str],
) -> FermentationResult:
    """Convert an ``(n_state, n_t)`` array into a :class:`FermentationResult`."""
    n = kin.n
    biomass = np.clip(states[:n], 0.0, None)
    substrate = np.clip(states[2 * n], 0.0, None)
    lactic_acid = np.clip(states[2 * n + 1], 0.0, None)
    aroma = np.clip(states[2 * n + 2], 0.0, None)
    from .milk import ph_from_acid

    ph = ph_from_acid(lactic_acid, kin.milk)
    return FermentationResult(
        t=t,
        ph=ph,
        biomass=biomass,
        substrate=substrate,
        lactic_acid=lactic_acid,
        aroma=aroma,
        strain_names=names,
    )


def run_fermentation(
    setup: FermentationSetup,
    t_eval: np.ndarray,
    rng: np.random.Generator | None = None,
) -> FermentationResult:
    """Simulate one batch over ``t_eval`` (hours).

    Parameters
    ----------
    setup:
        Batch configuration.
    t_eval:
        Output time grid (h); must be increasing and start at the inoculation time.
    rng:
        Generator for process noise.  Required when ``process_noise_sd > 0``; ignored
        otherwise.  Use :func:`vlab_doe.config.make_rng` for reproducibility.
    """
    t_eval = np.asarray(t_eval, dtype=float)
    kin = make_kinetics(
        setup.consortium, setup.milk, setup.temperature, aroma_decay=setup.aroma_decay
    )
    y0 = kin.initial_state(setup.total_inoculum, _lag_states(setup))
    names = setup.consortium.names

    if setup.process_noise_sd <= 0.0:
        sol = solve_ivp(
            kin.drift,
            t_span=(float(t_eval[0]), float(t_eval[-1])),
            y0=y0,
            method="LSODA",
            t_eval=t_eval,
            rtol=1e-6,
            atol=1e-9,
        )
        return _states_to_result(kin, sol.t, sol.y, names)

    # ── Stochastic: Euler–Maruyama on a fine grid, then sample at t_eval ──────────
    if rng is None:
        raise ValueError("rng is required when process_noise_sd > 0")
    states = _euler_maruyama(kin, y0, t_eval, setup.process_noise_sd, rng)
    return _states_to_result(kin, t_eval, states, names)


def _euler_maruyama(
    kin: FermentationKinetics,
    y0: np.ndarray,
    t_eval: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
    *,
    max_dt: float = 0.01,
) -> np.ndarray:
    """Integrate the SDE with a fixed internal step ≤ ``max_dt`` and sample at ``t_eval``.

    State variables are floored at 0 each step so multiplicative noise cannot push biomass
    negative.
    """
    out = np.empty((len(y0), len(t_eval)))
    y = y0.copy()
    out[:, 0] = y
    for k in range(1, len(t_eval)):
        t_start, t_end = t_eval[k - 1], t_eval[k]
        span = t_end - t_start
        n_steps = max(1, int(np.ceil(span / max_dt)))
        dt = span / n_steps
        sqrt_dt = np.sqrt(dt)
        t = t_start
        for _ in range(n_steps):
            drift = kin.drift(t, y)
            diff = kin.diffusion(y, sigma)
            dw = rng.normal(0.0, sqrt_dt, size=y.shape)
            y = y + drift * dt + diff * dw
            y = np.clip(y, 0.0, None)
            t += dt
        out[:, k] = y
    return out
