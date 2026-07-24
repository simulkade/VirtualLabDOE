"""vlab_doe — a virtual laboratory for bioprocess development, benchtop experiments & advanced DoE.

The package is organized to mirror the project's five phases:

* :mod:`vlab_doe.models`        — mechanistic forward models (the "Mechanistic Truth").
* :mod:`vlab_doe.perturbation`  — noise injection turning Truth into a "Virtual Experiment".
* :mod:`vlab_doe.doe`           — classical full-factorial and space-filling (LHS) designs.
* :mod:`vlab_doe.optimization`  — GP surrogates and Bayesian optimization.
* :mod:`vlab_doe.uq`            — inverse modeling and uncertainty quantification.

See ``plan.md`` at the repo root for the full development plan.
"""

__version__ = "0.4.0"

__all__ = ["__version__"]
