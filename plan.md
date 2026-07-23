# API Process Characterization & Advanced DoE — Development Plan

A portfolio / skill-building project that builds **mechanistic process models as a "virtual laboratory,"** then exercises modern Design-of-Experiments (DoE), space-filling sampling, Bayesian optimization, and uncertainty quantification (UQ) against that ground truth.

The project deliberately bridges **mechanistic mathematical modeling + scientific computing** with the day-to-day toolkit of a pharmaceutical **development scientist** working on **API process characterization** and **downstream development** (e.g. Novo Nordisk). Every mechanistic model is treated as a *virtual experimental setup*: we know the "Mechanistic Truth," we observe a noisy "Virtual Experiment," and we use statistics/ML to recover and optimize the process — exactly the loop a CMC team runs in the lab, but fast, cheap, and fully observable.

---

## 1. Why this project (mapping to the job)

| Job-ad theme | Where it shows up here |
|---|---|
| API process characterization | Phase 1 mechanistic chromatography & UF/DF models |
| Process **robustness** & capacity | Phase 2 full-factorial DoE + ANOVA → proven acceptable ranges |
| **Latin Hypercube Sampling** | Phase 3 space-filling design of a 5+ factor space |
| Innovative experimental design & **optimization** | Phase 4 Bayesian optimization vs classical DoE |
| **Inverse modeling / Bayesian methods** | Phase 5 parameter estimation + aleatoric/epistemic UQ |
| CMC regulatory & **tech transfer** | Narrative outputs in Phases 2 & 5 (design space, transfer report) |

---

## 2. Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │              VIRTUAL LABORATORY               │
   process params → │  chromatography (ED + SMA/Langmuir)           │ → "Mechanistic Truth"
   (CPPs)           │  ultrafiltration / diafiltration (UF/DF)      │   (yield, purity, flux…)
                    └───────────────────────┬──────────────────────┘
                                            │
                                 perturbation.py  (measurement noise,
                                            │      drift/bias, batch effects)
                                            ▼
                                  "Virtual Experiment"  (noisy observations)
                                            │
        ┌───────────────┬───────────────────┼───────────────────┬───────────────┐
        ▼               ▼                   ▼                   ▼               ▼
  Phase 2          Phase 3             Phase 4             Phase 5         reports/
  full factorial   LHS design          GP surrogate +      inverse model   design space,
  + ANOVA/RSM      space mapping       Bayesian optim.     + UQ (MCMC)     tech transfer
```

**Recurring visual contract:** every notebook plots the **Mechanistic Truth vs the Virtual Experiment**, and where applicable the **recovered/optimized estimate**.

---

## 3. Phases

Each phase below lists: **objective → model/method → libraries → deliverable → definition of done (DoD)**.

### Phase 1 — Build the Virtual Laboratory
*Forward mechanistic models that serve as ground truth for all later synthetic data.*

**1.1 Chromatography** — `src/vlab_doe/models/chromatography.py`
- **Model:** Equilibrium-Dispersive (ED) model — 1D convection–dispersion PDE per component:
  `∂c/∂t + (1-ε)/ε · ∂q/∂t + u · ∂c/∂z = D_ap · ∂²c/∂z²`
  with binding isotherm `q = f(c)`.
- **Isotherms:** **Langmuir** (baseline) and **Steric Mass Action (SMA)** for ion-exchange so that **pH and salt concentration** enter as physical CPPs (these feed Phase 2).
- **Numerics:** method of lines — finite-volume spatial discretization with an upwind/flux-limited convective term (control numerical dispersion), integrated by `scipy.integrate.solve_ivp` (BDF/LSODA for stiffness).
- **Variables:** mobile-phase velocity, bed porosity, isotherm parameters, salt/pH, load density.
- **Outputs:** outlet chromatogram `c(t)`, multi-component separation, derived **yield / purity / productivity** with configurable pool cut-points.
- *Stretch:* General Rate Model (film + intraparticle pore diffusion).

**1.2 Ultrafiltration / Diafiltration** — `src/vlab_doe/models/ufdf.py`
- **Model:** permeate flux from combined osmotic-pressure / gel-polarization theory; concentration polarization via film theory `J = k · ln(C_wall / C_bulk)`, with mass-transfer coefficient `k` from a Sherwood correlation driven by cross-flow velocity. Mass-balance ODEs over **diavolumes** track retentate concentration and buffer exchange.
- **Variables:** transmembrane pressure (TMP), cross-flow velocity, feed concentration.
- **Outputs:** permeate flux vs time, retentate concentration trajectory, protein retention/yield, diavolumes to target.

**1.3 Perturbation module** — `src/vlab_doe/perturbation.py`
- Converts Truth → Experiment: proportional + additive **Gaussian measurement noise**, **systematic effects** (baseline drift, calibration bias), and **batch-to-batch random effects** (jitter on the "true" parameters between runs). Fully seedable.

**DoD:** models run for a nominal parameter set in `notebooks/01`, produce physically sensible curves, conserve mass within tolerance (tested), and the perturbation module yields reproducible noisy replicates for a fixed seed.

---

### Phase 2 — Traditional DoE: Full Factorial
*Robustness & capacity characterization — the classical CMC workhorse.*
- `src/vlab_doe/doe/factorial.py`, `src/vlab_doe/doe/analysis.py`
- **Design:** pick 2–3 CPPs (e.g. load pH, salt / gradient slope, load density); generate 2-level & 3-level full factorial **+ center points** (`pyDOE`); run the virtual lab (with perturbation) at each design point.
- **Analysis (`statsmodels`):** OLS with main effects + interactions (+ quadratics → response surface), **ANOVA table**, Pareto-of-effects, main-effect & interaction plots, response-surface contour plots.
- **Narrative:** translate significant effects into **proven acceptable ranges (PAR)** and a robustness statement suitable for a **CMC regulatory submission**.

**Deliverable:** `notebooks/02_full_factorial_doe.ipynb`.
**DoD:** ANOVA recovers the effects that are genuinely present in the mechanistic model; a documented design-space rationale.

---

### Phase 3 — Space-Filling Design: Latin Hypercube Sampling
*Efficient exploration of a higher-dimensional space (the job ad's named method).*
- `src/vlab_doe/doe/lhs.py`
- **Design:** scale to **5+ factors**; LHS via `scipy.stats.qmc.LatinHypercube` with maximin / low-correlation optimization (`pyDOE.lhs` cross-check). Quantify coverage vs a full grid (curse of dimensionality).
- **Run:** push LHS points through the UF/DF (and/or chromatography) model to **map the design space** for late-stage development.
- **Data management:** tidy `pandas` DataFrame → parquet/csv under `data/`; cleaning, pairplots, parallel-coordinates, correlation analysis.

**Deliverable:** `notebooks/03_lhs_design_space.ipynb`.
**DoD:** a clean, documented synthetic dataset + design-space visualizations; quantified sampling efficiency vs grid.

---

### Phase 4 — Advanced Optimization: Bayesian Optimization
*Innovative experimental design / optimization.*
- `src/vlab_doe/optimization/surrogate.py`, `src/vlab_doe/optimization/bayesopt.py`
- **Surrogate:** train a Gaussian-Process surrogate on Phase-3 LHS data using **bofire/botorch** (`SingleTaskGP`).
- **Objective:** **maximize API yield s.t. purity ≥ threshold** — constrained / multi-objective (`qNEHVI`).
- **Acquisition:** EI / qEI / qNEHVI; sequential BO loop using the virtual lab as the expensive oracle.
- **Comparison:** BO vs full-factorial/LHS — **experiments-to-optimum** and regret curves.

**Deliverable:** `notebooks/04_bayesian_optimization.ipynb`.
**DoD:** BO reaches a near-optimal operating point in markedly fewer virtual experiments than the grid, with a quantified comparison.

---

### Phase 5 — Uncertainty Quantification & Parameter Estimation
*Inverse modeling + Bayesian UQ — core mathematical-modeling expertise.*
- `src/vlab_doe/uq/inverse.py`, `src/vlab_doe/uq/uncertainty.py`
- **Inverse modeling:** from noisy synthetic data, recover mechanistic parameters (SMA characteristic charge & equilibrium constant, mass-transfer coefficient, membrane resistance). Deterministic via `scipy.optimize.least_squares`; **Bayesian via `emcee` MCMC** with **`arviz`** diagnostics/plots.
- **Aleatoric vs epistemic:** aleatoric = likelihood noise variance (from the perturbation model); epistemic = parameter-posterior spread + model-form uncertainty; propagate to predictions via the posterior predictive.
- **Reporting:** generate `reports/tech_transfer_report.md` — how the quantified uncertainty impacts **transfer to manufacturing** and the control strategy.

**Deliverable:** `notebooks/05_uq_parameter_estimation.ipynb` + the tech-transfer report.
**DoD:** posterior credible intervals bracket the known true parameters; aleatoric/epistemic split is reported and discussed.

---

## 4. Tooling, layout & reproducibility

- **Language / env:** Python ≥ 3.13, managed with **`uv`** (`uv sync`, `uv run`).
- **Core libs:** `numpy`/`scipy` (ODE/PDE, optimization, QMC), `pandas` (data), `scikit-learn` (regression utilities), `statsmodels` (ANOVA/RSM), `bofire`/`botorch`/`gpytorch` (GP + BO), `emcee` + `arviz` (Bayesian inference), `matplotlib`/`plotly` (viz).
- **Package:** `src/vlab_doe/` (src layout). `config.py` centralizes a `make_rng(seed)` helper and `DATA_DIR`/`RESULTS_DIR` paths so every phase is reproducible.
- **Repo layout:**
  ```
  src/vlab_doe/{config,viz,perturbation}.py
  src/vlab_doe/models/{chromatography,ufdf}.py
  src/vlab_doe/doe/{factorial,lhs,analysis}.py
  src/vlab_doe/optimization/{surrogate,bayesopt}.py
  src/vlab_doe/uq/{inverse,uncertainty}.py
  notebooks/0{1..5}_*.ipynb
  reports/tech_transfer_report.md
  data/{raw,processed}/   results/   tests/
  ```
- **Testing:** `pytest` — mass conservation & limiting cases for models, seed reproducibility & noise statistics for perturbation, design dimensions & LHS coverage for DoE.
- **Reproducibility:** fixed RNG seeds everywhere; `uv.lock` pins the environment; synthetic data and figures regenerable from notebooks.

## 5. Stretch goals
- General Rate Model upgrade for chromatography (film + pore diffusion).
- Multi-objective design-space mapping (yield/purity/cost Pareto front).
- A unified **Streamlit** capstone dashboard with live CPP sliders over the virtual lab.
