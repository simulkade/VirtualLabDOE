# VlabDOE — Monograph

A detailed, book-length treatment of the science inside the `VlabDOE`
package, in two parts.

**Part I — Foundations** brings a reader strong in one parent discipline up to
speed in the other:

1. **Statistical Foundations** — distributions, estimation and the standard
   error, least squares and the analysis of variance, cross-validation, the
   bootstrap, the Bayesian update, and the distinction between *variability* and
   *uncertainty*.
2. **Foundations of Downstream Separation** — the bioprocess train and its unit
   operations, the molecules and their impurities, the adsorption isotherm, the
   column (retention, plates, resolution), mass transfer, the modulator idea
   unifying the chromatography modes, and microbial-fermentation basics.

**Part II — The Virtual Laboratory** is the technical core:

3. **Mechanistic Models of Preparative Chromatography** — the governing
   conservation laws, the transport-dispersive engine and its numerics, the
   unified adsorption-isotherm core, a mode-by-mode treatment (CEX/AEX, HIC,
   RP-HPLC) with, for each: the full mathematical formulation, every parameter
   and its units, the real-world separation it represents, its limits of
   validity, and the experiments/data used to estimate its parameters; and a
   **general rate model** (finite-volume / PyFVTool) that resolves the film and
   intraparticle-diffusion mass transfer and is mass-conservative where the
   lumped engine is not.
4. **The Methods of Experimental Design** — full factorial designs with
   response-surface/ANOVA analysis, generalized linear models (logistic/Poisson)
   for the *probabilistic design space*, Latin hypercube sampling, PCA/PLS
   latent-variable analysis, and Gaussian-process Bayesian optimisation, each
   with its theory, mathematical formulation, and a frank account of strengths
   and weaknesses.
5. **The Living Reactor** — a stochastic milk-fermentation model and the
   covering-array designs and tree-ensemble analysis used to screen large strain
   libraries.

Every figure is generated **directly from the package source** —
`scripts/make_figures.py` for Part II and `scripts/make_foundations_figures.py`
for Part I (from the `vlab_doe.foundations` teaching module). The plots are
real numerical solutions of the models in the text, not schematics.

## Layout

```
doc/
├── main.tex                 # master document (preamble + front/back matter)
├── chapters/
│   ├── preface.tex
│   ├── foundations_stats.tex       # Part I, Ch.1 — statistical foundations
│   ├── foundations_separation.tex  # Part I, Ch.2 — downstream-separation foundations
│   ├── ch1_models.tex              # Part II, Ch.3 — chromatography models
│   ├── ch2_design.tex              # Part II, Ch.4 — experimental design methods
│   ├── ch3_fermentation.tex        # Part II, Ch.5 — fermentation & strain screening
│   └── bibliography.tex
├── figures/                 # generated PNGs (git-ignored)
├── scripts/
│   ├── make_figures.py             # builds the Part II figures from the package
│   ├── make_foundations_figures.py # builds the Part I figures from foundations/
│   ├── make_grm_figures.py  # GRM-vs-MoL comparison figures (called by make_figures)
│   ├── build_pdf.sh         # -> build/monograph.pdf   (pdflatex/latexmk)
│   ├── build_html.sh        # -> build/html/monograph.html (pandoc + MathJax)
│   └── monograph.css        # styling for the HTML build
├── Makefile
└── build/                   # all output (git-ignored)
```

## Building

From the `doc/` directory:

```bash
make            # figures + PDF + HTML
make pdf        # build/monograph.pdf
make html       # build/html/monograph.html
make figures    # just (re)generate the figures
```

Or call the scripts directly (they regenerate figures first; set `REGEN=0` to
skip that, and `PYTHON=...` to choose an interpreter):

```bash
bash scripts/build_pdf.sh
bash scripts/build_html.sh
```

## Requirements

- **PDF:** a TeX Live installation with `pdflatex` and `latexmk` (the document
  uses `amsmath`, `mathtools`, `booktabs`, `tcolorbox`, `hyperref`, `titlesec`,
  `fancyhdr`, `cleveref`, …).
- **HTML:** `pandoc` (≥ 3). Math is rendered client-side with MathJax.
- **Figures:** the project's Python environment (the package itself plus
  `matplotlib`, and `pyfvtool` for the general-rate-model figures); the scripts
  auto-detect `../.venv/bin/python`.

## Notes on the dual PDF/HTML source

The LaTeX source is written so a single set of files compiles richly with
`pdflatex` *and* converts cleanly with Pandoc:

- Custom commands are limited to **math macros** and **titled `tcolorbox`
  environments**; both survive the LaTeX→HTML conversion. Pandoc emits each
  call-out box as a `<div>` whose class matches the environment name and whose
  title text is preserved, so `monograph.css` can style them.
- Figures are plain `\includegraphics` (no conditional wrappers), so Pandoc
  embeds them.
- Cross-references use `\ref`/`\eqref`, which Pandoc resolves to numbered links.
