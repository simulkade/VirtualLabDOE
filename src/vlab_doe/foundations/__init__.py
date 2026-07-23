"""Teaching companions for the monograph's two *Foundations* chapters.

These modules are deliberately small, dependency-light (NumPy/SciPy only), and
written to be *read*.  They re-derive, from scratch, the handful of statistical
and separation-science ideas that the rest of :mod:`vlab_doe` takes for
granted, so that a reader can move from the equations in Part I of the
monograph straight to a runnable implementation and back.

They are not used by the production models -- the engine has its own,
faster code paths.  Their job is pedagogical: every figure in the two
foundations chapters is generated from the functions here, keeping the book's
promise that nothing in it is a schematic.

* :mod:`vlab_doe.foundations.stats_demo` -- probability, least squares,
  the bootstrap, and the decomposition of variance into "between-batch" and
  "within-batch" parts.
* :mod:`vlab_doe.foundations.separation_demo` -- adsorption isotherms,
  the tanks-in-series picture of a column, peak resolution, and batch
  microbial growth.
"""

from __future__ import annotations

from . import separation_demo, stats_demo

__all__ = ["stats_demo", "separation_demo"]
