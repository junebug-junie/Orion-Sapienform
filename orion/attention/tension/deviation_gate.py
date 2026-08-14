"""Deviation gate: turn a stream of per-dimension observations into admissions
that fire on *change*, not on presence.

Restored 2026-08-14 from `d6a4e892b^:orion/autonomy/deviation_gate.py`, deleted
2026-07-30 as collateral in the drives sweep (`d6a4e892b`). It was never
drive-coupled -- its whole interface is `(kind, dimension, value) -> deviation`
and it has never referenced a drive, bucket, or category. It was deleted because
it lived in `orion/autonomy/` next to `signal_drive_map.yaml`, not because it was
measured bad. See `docs/superpowers/specs/2026-08-14-field-deviation-tension-
sensing-design.md`.

Per ``(kind, dimension)`` we hold an EWMA baseline ``(mu, var)`` and a warm-up
counter. An observation is admitted only when it deviates from its own learned
baseline past ``z_threshold`` in the *worse* direction. Steady input settles to
its own mean and admits nothing -- this is what starves a high-rate flood.

**Two changes from the restored original**:

1. ``impulse_k`` is gone. The original returned ``impulse_k * excess * confidence``.
   Downstream combination is rank-based (`orion.attention.rank_aggregation`), and a
   monotonic scaling constant cannot change a rank -- it was a tunable that provably
   did nothing, so carrying it forward would be re-importing a hand-set number for no
   behavioural effect. What is returned is ``excess`` in z-units (optionally scaled by
   a per-observation ``confidence``, which does *not* cancel because it varies between
   targets within one channel's ballot).
2. The absolute ``sigma_floor`` is replaced by ``relative_sigma_floor``. The absolute
   version silently broke scale-freedom and made small-variance channels structurally
   unable to admit -- see the class docstring for the live evidence.

Pure and synchronous: no bus, no clock, no I/O. Never raises on a bad *observation*
(degrades to a 0.0 deviation); does raise on invalid *construction* parameters, which
are operator error and better surfaced immediately than thousands of ticks in.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple

Worse = Literal["up", "down"]


@dataclass
class _Baseline:
    mu: float = 0.0
    var: float = 0.0
    count: int = 0


# Numerical backstop only -- prevents division by zero for a dimension whose
# baseline mean is itself ~0. Deliberately far below any real channel's scale so
# it never acts as a de facto admission threshold the way an absolute
# `sigma_floor` did (see `relative_sigma_floor`).
_MIN_SIGMA = 1e-12


@dataclass
class DeviationGate:
    """Adaptive per-dimension deviation detector.

    Args:
        alpha: EWMA weight for new observations (0<alpha<=1). Larger = faster
            adaptation, shorter memory.
        z_threshold: minimum |z| (in the worse direction) before any admission.
        relative_sigma_floor: minimum std used in the z denominator, expressed as
            a fraction of the dimension's own |mean|. See below for why this is
            relative and not absolute.
        warmup: observations to learn a baseline before admitting (cold start
            admits nothing).

    **`relative_sigma_floor`, and the absolute floor it replaced (fixed in review,
    2026-08-14).** The restored original used `sigma_floor: float = 0.02` --
    an absolute constant applied to every dimension regardless of that
    dimension's natural scale. That silently broke the scale-freedom this whole
    design rests on, in the direction that matters most: a channel whose real
    variation is *smaller* than the floor can never admit anything, no matter how
    large its relative move. Live evidence from the review: `node:atlas` /
    `memory_pressure` over 10,528 ticks ranges 0.1109-0.1171 (pstdev 8.8e-4) --
    real, continuous variation -- and admitted **0 of 10,528** under
    `sigma_floor=0.02`, because clearing it would have required a ~0.03 absolute
    jump (a 26% relative move). Worse, it then appeared in the measurement's
    `never_admitted` list indistinguishable from a genuinely calm channel: an
    instrument structurally incapable of reading the state it reports.

    A floor proportional to the dimension's own mean scales with the dimension,
    so multiplying a channel by any k>0 scales `mu`, `sqrt(var)` and the floor
    identically and leaves `z` -- and therefore the rank -- unchanged. That is the
    invariance the design claims, now true in both directions rather than only
    upward.
    """

    alpha: float = 0.1
    z_threshold: float = 1.5
    relative_sigma_floor: float = 0.01
    warmup: int = 5
    _baselines: Dict[Tuple[str, str], _Baseline] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Loud on construction rather than a ZeroDivisionError thousands of ticks
        # into a run, or a silently non-adapting baseline. These are public
        # dataclass fields and CLI flags; a bad value is an operator error worth
        # surfacing immediately.
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha!r}")
        if self.z_threshold < 0.0:
            raise ValueError(f"z_threshold must be >= 0, got {self.z_threshold!r}")
        if self.relative_sigma_floor < 0.0:
            raise ValueError(
                f"relative_sigma_floor must be >= 0, got {self.relative_sigma_floor!r}"
            )
        if self.warmup < 1:
            raise ValueError(f"warmup must be >= 1, got {self.warmup!r}")

    def observe(
        self,
        kind: str,
        dimension: str,
        x: float,
        *,
        confidence: float = 1.0,
        worse: Worse = "up",
    ) -> float:
        """Return the admitted deviation in z-units (>=0, 0.0 == not admitted)
        for this observation, then fold it into the baseline. Warm-up
        observations return 0 but still train."""
        try:
            x = float(x)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(x):
            return 0.0

        key = (kind, dimension)
        b = self._baselines.get(key)
        if b is None:
            # Seed the mean to the first observation (var stays 0). Drifting up
            # from mu=0 would pump artificial variance and desensitise the gate.
            self._baselines[key] = _Baseline(mu=x, var=0.0, count=1)
            return 0.0

        # Compute deviation against the CURRENT baseline (before folding x in),
        # so a step change registers before it moves the mean.
        deviation = 0.0
        if b.count >= self.warmup:
            sigma = max(
                math.sqrt(max(b.var, 0.0)),
                self.relative_sigma_floor * abs(b.mu),
                _MIN_SIGMA,
            )
            z = (x - b.mu) / sigma
            direction = 1.0 if worse == "up" else -1.0
            excess = direction * z - self.z_threshold
            if excess > 0.0:
                conf = min(1.0, max(0.0, float(confidence)))
                deviation = excess * conf

        # EWMA update (mean + variance) -- West's incremental form.
        delta = x - b.mu
        b.mu += self.alpha * delta
        b.var = (1.0 - self.alpha) * (b.var + self.alpha * delta * delta)
        b.count += 1
        return max(0.0, deviation)

    def baseline_count(self) -> int:
        return len(self._baselines)

    def is_warm(self, kind: str, dimension: str) -> bool:
        """True once this dimension has enough observations to admit anything."""
        b = self._baselines.get((kind, dimension))
        return b is not None and b.count >= self.warmup
