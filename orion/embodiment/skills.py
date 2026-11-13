"""Game-free action vocabulary for every Orion body (docs/superpowers/specs/2026-10-23-embodied-experience-learning-design.md).

A body (a Retro game, the AI Town sprite, a real motor) maps its own goals onto these skills and
these context features before asking orion-prediction for an action-bet quote. Bets pool at the
skill level across every body and domain: that pooling is what lets a new game or body start from
what earlier ones taught. Context values are categorical, never ids, so they carry over too; a
domain-specific place key (e.g. 'zelda:0:119:S') is the only thing that does not.
"""
from __future__ import annotations

from typing import Iterable, Optional

SKILLS = ('traverse', 'engage', 'withdraw', 'acquire', 'enter', 'hold')
SIDE_STATES = ('open_unvisited', 'open_visited', 'closed', 'unknown')
HEALTH_BANDS = ('full', 'hurt', 'low', 'unknown')


def health_band(current: Optional[float], containers: Optional[float]) -> str:
    """'full' at current >= containers, 'low' at <= 1 unit, else 'hurt'; 'unknown' when not read."""
    if not isinstance(current, (int, float)) or not isinstance(containers, (int, float)) or containers <= 0:
        return 'unknown'
    if current >= containers:
        return 'full'
    return 'low' if current <= 1 else 'hurt'


def threat_band(distances: Iterable[float], near: float) -> str:
    """'near' when any threat is within ``near`` (the body's own reach unit), else 'none'."""
    return 'near' if any(isinstance(d, (int, float)) and d <= near for d in distances) else 'none'


def count_band(n: int) -> str:
    return 'many' if n > 1 else 'one' if n == 1 else 'none'


def context(**features: Optional[str]) -> dict[str, str]:
    """A context dict with keys sorted and None features dropped (orion-prediction matches it exactly)."""
    return {k: str(v) for k, v in sorted(features.items()) if v is not None}
