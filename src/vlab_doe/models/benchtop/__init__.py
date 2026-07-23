"""Benchtop physical & chemical systems — simple models for practising DoE.

Where the chromatography, UF/DF and fermentation models are distributed-parameter
or stochastic systems, these four are low-dimensional and (nearly) analytic, so
the mechanism never gets in the way of the experimental-design lesson:

* :mod:`~vlab_doe.models.benchtop.pipe_flow` — pressure drop for pipe flow
  (Hagen--Poiseuille / Darcy--Weisbach) — power-law effects and log transforms.
* :mod:`~vlab_doe.models.benchtop.falling_ball` — falling-ball viscometer
  (Stokes + wall & inertia corrections) — calibration and parameter estimation.
* :mod:`~vlab_doe.models.benchtop.back_extrusion` — yogurt back-extrusion
  probe rheometry (Herschel--Bulkley) — a yield-stress threshold for RSM/GLM.
* :mod:`~vlab_doe.models.benchtop.ester_hydrolysis` — acid-catalysed
  hydrolysis of methyl acetate (reversible kinetics) — time and temperature as
  factors, catalyst vs. equilibrium.
"""

from . import back_extrusion, ester_hydrolysis, falling_ball, pipe_flow

__all__ = ["pipe_flow", "falling_ball", "back_extrusion", "ester_hydrolysis"]
