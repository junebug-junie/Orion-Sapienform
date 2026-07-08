"""Deviation gate: turn a stream of per-dimension observations into impulses
that fire on *change*, not on presence (spec 2026-07-07 §The unified model).

Per ``(signal_kind, dimension)`` we hold an EWMA baseline ``(mu, var)`` and a
warm-up counter. An observation impulses only when it deviates from its own
learned baseline past ``z_threshold`` in the *worse* direction. Steady input —
including the 55/s ``scene_state`` flood — settles to its mean and mints zero.

Pure and synchronous: no bus, no clock, no I/O. Never raises on bad input;
degrades to a 0.0 impulse.
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
        z_threshold: minimum |z| (in the worse direction) before any impulse.
        sigma_floor: minimum std used in the z denominator; prevents blow-up
            when a dimension is briefly constant.
        impulse_k: scales deviation into impulse magnitude.
        warmup: observations to learn a baseline before impulsing (cold start
            mints nothing).
    """

    alpha: float = 0.1
    z_threshold: float = 1.5
    sigma_floor: float = 0.02
    impulse_k: float = 0.25
    warmup: int = 5
    _baselines: Dict[Tuple[str, str], _Baseline] = field(default_factory=dict)

    def observe(
        self,
        signal_kind: str,
        dimension: str,
        x: float,
        *,
        confidence: float = 1.0,
        worse: Worse = "up",
    ) -> float:
        """Return the deviation impulse (>=0) for this observation, then fold it
        into the baseline. Warm-up observations return 0 but still train."""
        try:
            x = float(x)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(x):
            return 0.0

        key = (signal_kind, dimension)
        b = self._baselines.get(key)
        if b is None:
            # Seed the mean to the first observation (var stays 0). Drifting up
            # from mu=0 would pump artificial variance and desensitise the gate.
            self._baselines[key] = _Baseline(mu=x, var=0.0, count=1)
            return 0.0

        # Compute impulse against the CURRENT baseline (before folding x in), so
        # a step change registers before it moves the mean.
        impulse = 0.0
        if b.count >= self.warmup:
            sigma = max(math.sqrt(max(b.var, 0.0)), self.sigma_floor)
            z = (x - b.mu) / sigma
            direction = 1.0 if worse == "up" else -1.0
            excess = direction * z - self.z_threshold
            if excess > 0.0:
                conf = min(1.0, max(0.0, float(confidence)))
                impulse = self.impulse_k * excess * conf

        # EWMA update (mean + variance) — West's incremental form.
        delta = x - b.mu
        b.mu += self.alpha * delta
        b.var = (1.0 - self.alpha) * (b.var + self.alpha * delta * delta)
        b.count += 1
        return max(0.0, impulse)

    def baseline_count(self) -> int:
        return len(self._baselines)
