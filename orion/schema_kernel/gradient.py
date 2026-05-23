"""Gradient utilities — small helpers for the canonical gradient vector."""

from __future__ import annotations

from typing import Final


DEFAULT_GRADIENT_KEYS: Final[tuple[str, ...]] = (
    "salience",
    "contradiction",
    "novelty",
    "coherence",
)


def empty_gradient_vector() -> dict[str, float]:
    """Return a fresh gradient vector seeded at 0.0 for the canonical keys."""

    return {key: 0.0 for key in DEFAULT_GRADIENT_KEYS}


def clamp_gradient(value: float, *, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a single gradient component to [lo, hi]."""

    if value < lo:
        return lo
    if value > hi:
        return hi
    return value
