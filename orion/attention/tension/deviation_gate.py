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

**Changed from the restored original**: the original returned
``impulse_k * excess * confidence``. ``impulse_k`` is gone. Downstream combination
is rank-based (`orion.attention.rank_aggregation`), and a monotonic scaling
constant cannot change a rank -- it was a tunable that provably did nothing, so
carrying it forward would be re-importing a hand-set number for no behavioural
effect. What is returned is ``excess`` in z-units (optionally scaled by a
per-observation ``confidence``, which does *not* cancel because it varies between
targets within one channel's ballot).

Pure and synchronous: no bus, no clock, no I/O. Never raises on bad input;
degrades to a 0.0 deviation.
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


@dataclass
class DeviationGate:
    """Adaptive per-dimension deviation detector.

    Args:
        alpha: EWMA weight for new observations (0<alpha<=1). Larger = faster
            adaptation, shorter memory.
        z_threshold: minimum |z| (in the worse direction) before any admission.
        sigma_floor: minimum std used in the z denominator; prevents blow-up
            when a dimension is briefly constant.
        warmup: observations to learn a baseline before admitting (cold start
            admits nothing).
    """

    alpha: float = 0.1
    z_threshold: float = 1.5
    sigma_floor: float = 0.02
    warmup: int = 5
    _baselines: Dict[Tuple[str, str], _Baseline] = field(default_factory=dict)

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
            sigma = max(math.sqrt(max(b.var, 0.0)), self.sigma_floor)
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
