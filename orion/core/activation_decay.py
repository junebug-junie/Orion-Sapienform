"""Half-life activation decay — stdlib only (safe for lightweight service imports)."""

from __future__ import annotations

import math


def decay_activation(
    *,
    current: float,
    elapsed_seconds: float,
    half_life_seconds: int | None,
    floor: float,
) -> float:
    if elapsed_seconds <= 0:
        return max(0.0, min(1.0, current))
    if not half_life_seconds:
        return max(0.0, min(1.0, max(floor, current)))
    decay_multiplier = math.pow(0.5, elapsed_seconds / float(half_life_seconds))
    return max(0.0, min(1.0, max(floor, current * decay_multiplier)))
