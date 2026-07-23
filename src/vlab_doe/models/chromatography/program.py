"""Elution program — the inlet modulator timeline and protein injection.

A run is described as an ordered list of :class:`Segment` phases (equilibrate, load,
wash, gradient, strip, …).  Each segment lasts a number of **column volumes (CV)** and
either *holds* the modulator (``m_end is None``) or *ramps* it linearly from ``m_start``
to ``m_end`` — this linear ramp is the "linearly changing eluate" / gradient elution.

The protein :class:`Injection` is a separate rectangular feed pulse.  Durations are given
in CV and converted to seconds against a specific column + flow when the program is
:meth:`ElutionProgram.compile`-d.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import ColumnGeometry


@dataclass(frozen=True)
class Segment:
    """One phase of the run.

    Parameters
    ----------
    name:
        Label (e.g. ``"equilibrate"``, ``"load"``, ``"gradient"``).
    duration_cv:
        Length of the phase in column volumes.
    m_start:
        Modulator value at the start of the phase (mM salt, or organic fraction φ).
    m_end:
        Modulator value at the end.  ``None`` → hold constant at ``m_start``.
    """

    name: str
    duration_cv: float
    m_start: float
    m_end: float | None = None

    @property
    def end_value(self) -> float:
        return self.m_start if self.m_end is None else self.m_end


@dataclass(frozen=True)
class Injection:
    """Rectangular protein feed pulse.

    Parameters
    ----------
    feed:
        Feed concentration of each component (g/L), shape ``(n_components,)``.
    start_cv:
        When loading starts, in CV from the beginning of the run.
    duration_cv:
        Loading duration in CV.
    """

    feed: np.ndarray
    start_cv: float
    duration_cv: float

    @classmethod
    def from_load_density(
        cls,
        load_density: float,
        feed: float | np.ndarray,
        porosity: float,
        start_cv: float = 0.0,
    ) -> "Injection":
        """Build an injection from a total *load_density* (g protein / L resin).

        The load volume that delivers ``load_density`` is
        ``V_inject = load_density·V_resin / c_feed_total``; expressed in CV this is
        ``load_density·(1−ε) / c_feed_total``.
        """
        feed_arr = np.atleast_1d(np.asarray(feed, dtype=float))
        feed_total = float(feed_arr.sum())
        duration_cv = load_density * (1.0 - porosity) / feed_total
        return cls(feed=feed_arr, start_cv=start_cv, duration_cv=duration_cv)


@dataclass
class CompiledProgram:
    """A program resolved against a column + flow, with second-based callables."""

    breakpoints_s: np.ndarray           # segment start times (s), shape (n_seg+1,)
    seg_m_start: np.ndarray             # modulator at each segment start
    seg_m_end: np.ndarray               # modulator at each segment end
    feed: np.ndarray                    # injection feed vector (n_components,)
    inject_start_s: float
    inject_end_s: float
    t_end_s: float
    n_components: int
    segment_names: list[str] = field(default_factory=list)

    def modulator(self, t: float) -> float:
        """Inlet modulator value at time *t* (s), piecewise-linear across segments."""
        bp = self.breakpoints_s
        if t <= bp[0]:
            return float(self.seg_m_start[0])
        if t >= bp[-1]:
            return float(self.seg_m_end[-1])
        i = int(np.searchsorted(bp, t, side="right") - 1)
        i = min(max(i, 0), len(self.seg_m_start) - 1)
        span = bp[i + 1] - bp[i]
        frac = 0.0 if span <= 0 else (t - bp[i]) / span
        return float(self.seg_m_start[i] + frac * (self.seg_m_end[i] - self.seg_m_start[i]))

    def feed_at(self, t: float) -> np.ndarray:
        """Inlet protein feed vector at time *t* (s) — feed during the load window, else 0."""
        if self.inject_start_s <= t < self.inject_end_s:
            return self.feed
        return np.zeros(self.n_components)


@dataclass(frozen=True)
class ElutionProgram:
    """An ordered list of modulator phases plus an optional protein injection."""

    segments: list[Segment]
    injection: Injection | None = None

    @property
    def total_cv(self) -> float:
        return float(sum(s.duration_cv for s in self.segments))

    def seconds_per_cv(self, geometry: ColumnGeometry, velocity: float) -> float:
        """Seconds to pass one column volume at interstitial *velocity* u.

        ``Q = u·A·ε`` is the volumetric flow through the pores and one CV is the
        geometric column volume ``V_c = A·L``, so ``t_CV = V_c/Q = L/(u·ε)``.
        """
        q_flow = velocity * geometry.area * geometry.porosity
        return geometry.volume / q_flow if q_flow > 0 else float("inf")

    def compile(self, geometry: ColumnGeometry, velocity: float) -> CompiledProgram:
        """Resolve CV-based phases into second-based modulator/feed callables."""
        spc = self.seconds_per_cv(geometry, velocity)

        breakpoints = [0.0]
        m_start, m_end, names = [], [], []
        t = 0.0
        for seg in self.segments:
            t_next = t + seg.duration_cv * spc
            m_start.append(seg.m_start)
            m_end.append(seg.end_value)
            names.append(seg.name)
            breakpoints.append(t_next)
            t = t_next

        if self.injection is not None:
            feed = np.atleast_1d(np.asarray(self.injection.feed, dtype=float))
            inj_start = self.injection.start_cv * spc
            inj_end = inj_start + self.injection.duration_cv * spc
        else:
            feed = np.zeros(1)
            inj_start = inj_end = 0.0

        return CompiledProgram(
            breakpoints_s=np.asarray(breakpoints, dtype=float),
            seg_m_start=np.asarray(m_start, dtype=float),
            seg_m_end=np.asarray(m_end, dtype=float),
            feed=feed,
            inject_start_s=inj_start,
            inject_end_s=inj_end,
            t_end_s=t,
            n_components=len(feed),
            segment_names=names,
        )

    # ── Convenience constructors ──────────────────────────────────────────────

    @classmethod
    def isocratic(
        cls,
        modulator: float,
        injection: Injection,
        *,
        run_cv: float = 20.0,
    ) -> "ElutionProgram":
        """A single constant-modulator phase (classic isocratic elution)."""
        return cls(
            segments=[Segment("run", run_cv, modulator)],
            injection=injection,
        )

    @classmethod
    def linear_gradient(
        cls,
        injection: Injection,
        *,
        m_start: float,
        m_end: float,
        gradient_cv: float,
        equilibrate_cv: float = 2.0,
        wash_cv: float = 2.0,
        strip_cv: float = 2.0,
    ) -> "ElutionProgram":
        """Equilibrate → load → wash → linear gradient → strip.

        The protein is loaded and washed at ``m_start`` (binding conditions), then the
        modulator is ramped linearly from ``m_start`` to ``m_end`` over ``gradient_cv``
        column volumes — increasing salt for IEX, decreasing salt for HIC, or rising
        organic fraction for RP-HPLC, depending on the isotherm mode.
        """
        load_cv = injection.duration_cv
        segments = [
            Segment("equilibrate", equilibrate_cv, m_start),
            Segment("load", load_cv, m_start),
            Segment("wash", wash_cv, m_start),
            Segment("gradient", gradient_cv, m_start, m_end),
            Segment("strip", strip_cv, m_end),
        ]
        # Re-anchor the injection to start at the end of equilibration.
        inj = Injection(feed=injection.feed, start_cv=equilibrate_cv, duration_cv=load_cv)
        return cls(segments=segments, injection=inj)
