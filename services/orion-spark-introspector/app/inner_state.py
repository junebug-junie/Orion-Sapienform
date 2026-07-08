"""Honest inner-state feature assembly for spark-introspector.

Replaces the geometric-mean φ (which a single saturated infra signal could
floor) with a decontaminated, robust-scaled feature vector plus an arithmetic
cold-start headline and a degeneracy freeze (the GIGO guard).
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from statistics import median as _median
from typing import Deque, Dict, List, Optional, Tuple

# NOTE: do NOT import orion.schemas.telemetry.inner_state here. That schema is
# created in Task 4; the scaler in this task does not need it. The extractor in
# Task 5 adds the import at that point (Task 4 has run by then). Importing it now
# would make the Task 3 test's importlib module-load raise ModuleNotFoundError.

SCALE_CLIP = 4.0


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    idx = pct * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def robust_scale(value: float, *, median: float, iqr: float) -> float:
    if iqr <= 1e-9:
        return 0.0
    scaled = (float(value) - median) / iqr
    return max(-SCALE_CLIP, min(SCALE_CLIP, scaled))


class RollingRobustScaler:
    """Bounded rolling median/IQR standardizer, one window per feature name."""

    def __init__(self, maxlen: int = 256) -> None:
        self._maxlen = max(2, int(maxlen))
        self._windows: Dict[str, Deque[float]] = {}

    def observe(self, name: str, value: float) -> None:
        w = self._windows.get(name)
        if w is None:
            w = deque(maxlen=self._maxlen)
            self._windows[name] = w
        w.append(float(value))

    def scale(self, name: str, value: float) -> float:
        w = self._windows.get(name)
        if not w or len(w) < 2:
            return 0.0
        vals = list(w)
        med = _median(vals)
        iqr = _percentile(vals, 0.75) - _percentile(vals, 0.25)
        return robust_scale(value, median=med, iqr=iqr)

    def observe_and_scale(self, name: str, value: float) -> float:
        self.observe(name, value)
        return self.scale(name, value)
