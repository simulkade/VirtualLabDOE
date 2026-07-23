# VlabDOE

**A virtual laboratory for downstream-bioprocess characterization and advanced Design of Experiments (DoE).**

<p align="center">
  <a href="https://github.com/simulkade/VirtualLabDOE/releases/latest/download/monograph.pdf">
    <img src="https://img.shields.io/badge/%F0%9F%93%95%20Read%20the%20Monograph-PDF-EC1C24?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" alt="Download the monograph (PDF)">
  </a>
  &nbsp;
  <a href="https://simulkade.github.io/VirtualLabDOE/">
    <img src="https://img.shields.io/badge/%F0%9F%8C%90%20Read%20the%20Monograph-HTML-E34F26?style=for-the-badge&logo=html5&logoColor=white" alt="Read the monograph (HTML, in browser)">
  </a>
</p>

<p align="center"><em>A book-length treatment of the science inside the package — every figure generated directly from the source.<br>Download the typeset <strong>PDF</strong>, or read the <strong>HTML</strong> edition straight in your browser (figures &amp; styling embedded).</em></p>

VlabDOE pairs first-principles process models with a modern DoE / optimization / uncertainty-quantification toolkit. Mechanistic models of **chromatography** and **ultrafiltration/diafiltration (UF/DF)** provide a fully-observable *"Mechanistic Truth"*; a perturbation layer turns that truth into noisy *"Virtual Experiments"*; and classical DoE, Bayesian optimization, and uncertainty quantification are exercised against that known ground truth. Because the truth is known, every method can be benchmarked for accuracy, efficiency, and robustness — something impossible with real wet-lab data alone.

The workflow mirrors a real **QbD / process-characterization** campaign: define critical process parameters (CPPs), screen them with a factorial design, map the design space with space-filling sampling, drive toward an optimum with Bayesian optimization, and finally quantify how much of the residual uncertainty is reducible (epistemic) versus irreducible process noise (aleatoric).

---

## What's in the box

| Domain | Module | Capability |
|---|---|---|
| **Mechanistic models** | `models/chromatography/` | Multi-mode column engine: **cation/anion exchange (CEX/AEX), HIC salting-out, RP-HPLC**, and nonlinear high-resolution IEX — one transport-dispersive (LDF) solver, mode set by the isotherm law. Multi-component, with **linear-gradient elution** and overload; computes chromatograms, **yield/purity/productivity**, resolution & plate count. The original single-component isocratic model is retained for backward compatibility. |
| | `models/ufdf.py` | Ultrafiltration/diafiltration with a combined pressure-limited + gel-polarisation flux model; cross-flow velocity is a physical CPP. Mass-balance ODEs for the UF (concentrating) and DF (buffer-exchange) phases. |
| | `models/fermentation/` | Stochastic milk-fermentation (yogurt) model: multi-strain Baranyi–Roberts growth with a Rosso temperature law, ST↔LB proto-cooperation, and Luedeking–Piret acid → milk-buffering pH. The only observable is the **pH time series**; three explicit randomness layers (batch **variability**, optional process-noise SDE, measurement **uncertainty**) plus a product **fingerprint** + distance for strain-replacement optimization. |
| | `models/benchtop/` | Four **simple benchtop systems** for practising DoE where the mechanism is one or two equations: **pipe pressure drop** (Hagen–Poiseuille / Darcy–Weisbach), **falling-ball viscometer** (Stokes + wall & inertia corrections), **yogurt back-extrusion** (Herschel–Bulkley probe rheometry), and **methyl-acetate hydrolysis** (reversible acid-catalysed kinetics). Each foregrounds a different response shape — power law, calibration, threshold, kinetics. |
| **Virtual experiments** | `perturbation.py` | Turns model output into realistic measurements: multiplicative + additive noise, drift, and calibration bias, plus lognormal batch-to-batch parameter jitter. |
| **Classical DoE** | `doe/factorial.py` | 2-level full-factorial designs with center points and randomised run order; a harness to run any design against the virtual lab. |
| | `doe/analysis.py` | OLS response-surface fitting with Type-II ANOVA (via `statsmodels`) and derivation of **Proven Acceptable Ranges (PARs)**. |
| | `doe/lhs.py` | Optimised Latin-Hypercube space-filling designs (low-discrepancy) with coverage diagnostics. |
| | `doe/multivariate.py` | PCA / PLS for exploring high-dimensional design-space data. |
| **Combinatorial DoE** | `doe/covering.py` | Block-size-constrained **covering arrays** for combinatorial screening: pick combinations of *k* items (e.g. strains) so every pair is co-tested, with greedy strength-2 coverage, balanced item usage, and honest coverage diagnostics. |
| | `doe/importance.py` | Tree-model screening analysis — **Random Forest** and **XGBoost** with cross-validated R² and model-agnostic **permutation importance** to rank which items drive the response. |
| **Optimization** | `optimization/surrogate.py` | Gaussian-Process surrogate (`botorch` `SingleTaskGP`, Matérn-5/2) with input normalisation and output standardisation. |
| | `optimization/bayesopt.py` | Bayesian-optimization loop using Log Expected Improvement, plus a head-to-head comparison against a DoE baseline. |
| **Uncertainty quantification** | `uq/inverse.py` | Inverse modeling / parameter estimation: trust-region least-squares **and** affine-invariant MCMC (`emcee`). |
| | `uq/uncertainty.py` | Posterior-predictive **aleatoric vs epistemic** variance decomposition via the law of total variance. |
| **Infrastructure** | `config.py`, `viz.py` | Seeded RNG and canonical artifact paths for fully-reproducible runs; shared plotting helpers. |

The mathematical derivation and engineering rationale behind every module are documented in **[`implementation.md`](implementation.md)**; the development plan and motivation are in **[`plan.md`](plan.md)**.

---

## Installation

VlabDOE targets **Python ≥ 3.13** and uses [`uv`](https://docs.astral.sh/uv/) for environment management.

```bash
# clone, then from the repo root:
uv sync                 # create the env from pyproject.toml / uv.lock
uv run pytest -q        # run the test suite (111 tests)
```

To use it as a library in your own environment:

```bash
pip install -e .        # or: uv pip install -e .
```

```python
import vlab_doe          # the importable package
print(vlab_doe.__version__)
```

---

## Quickstart

Run a single virtual chromatography experiment and read off its performance metrics:

```python
import numpy as np
from vlab_doe.models.chromatography import (
    ColumnGeometry, SMAParameters, ChromatographyConfig, simulate, pool_metrics,
)

config = ChromatographyConfig(
    geometry=ColumnGeometry(length=0.10, diameter=0.01, porosity=0.4),
    velocity=1e-3,            # interstitial velocity, m/s
    dispersion=1e-7,          # apparent axial dispersion, m²/s
    isotherm=SMAParameters(),
    salt=150.0,               # mobile-phase salt, mM     <- a CPP
    ph=7.0,                   # mobile-phase pH            <- a CPP
    load_density=20.0,        # mg protein / mL resin      <- a CPP
)

t = np.linspace(0, 5000, 2000)
result = simulate(config, t)
metrics = pool_metrics(t, result["c_outlet"], cut_start=1000, cut_end=3000)
print(metrics)   # -> {'yield': ..., 'purity': ..., 'productivity': ...}
```

From here you can wrap `simulate` + `pool_metrics` as an *oracle*, screen the CPPs
with `doe.factorial`, map the space with `doe.lhs`, optimise with
`optimization.bayesopt`, and quantify uncertainty with `uq` — the full campaign is
walked through, phase by phase, in the notebooks below.

---

## Guided notebooks

Each phase of the workflow has a self-contained notebook (`uv run jupyter lab`):

| Phase | Notebook | Modules exercised |
|---|---|---|
| 1 — Virtual laboratory | `notebooks/01_virtual_laboratory.ipynb` | `models/chromatography`, `models/ufdf`, `perturbation` |
| 2 — Full-factorial DoE | `notebooks/02_full_factorial_doe.ipynb` | `doe/factorial`, `doe/analysis` |
| 3 — LHS design space | `notebooks/03_lhs_design_space.ipynb` | `doe/lhs`, `doe/multivariate` |
| 4 — Bayesian optimization | `notebooks/04_bayesian_optimization.ipynb` | `optimization/surrogate`, `optimization/bayesopt` |
| 5 — UQ & parameter estimation | `notebooks/05_uq_parameter_estimation.ipynb` | `uq/inverse`, `uq/uncertainty` |
| 1.6 — Multi-mode chromatography | `notebooks/06_chromatography_modes.ipynb` | `models/chromatography` (modes, gradients, resolution) |
| 1.7 — Milk fermentation (yogurt) | `notebooks/07_fermentation_doe.ipynb` | `models/fermentation` (strains, variability, fingerprint matching) |
| 2.x — Covering-array strain screening | `notebooks/08_covering_array_strain_screening.ipynb` | `doe/covering`, `doe/importance` (RF + XGBoost), `models/fermentation` |
| 4 — Benchtop DoE sandbox | `notebooks/09_benchtop_doe.ipynb` | `models/benchtop`, `doe/factorial`, `doe/analysis`, `doe/lhs` |

---

## Project layout

```
src/vlab_doe/   installable package (config, viz, perturbation, models, doe, optimization, uq)
notebooks/            one notebook per phase
tests/                pytest suite
reports/              tech-transfer report (Phase 5 deliverable)
data/                 raw/ + processed/ synthetic datasets (gitignored)
results/              figures & fitted parameters (gitignored)
implementation.md     mathematical & engineering notes for every module
plan.md               development plan and rationale
```

---

## License

This project is released under the MIT License — see [LICENSE](LICENSE).
