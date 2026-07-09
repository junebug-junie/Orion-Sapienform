"""Dynamic memory weight for crystallizations: encode weakly, strengthen on
reinforcement/recall, decay on disuse.

Deterministic (§4): pure functions over a crystallization + a clock. No LLM, no I/O.
Reuses the substrate half-life decay math (`orion.core.activation_decay.decay_activation`)
so crystallization memory ages on the same curve the substrate graph already uses.

Live wiring (M1): `seed_dynamics` / `seed_weak_dynamics` on formation, `reinforce` on dedup,
`governor.approve` seeds dynamics. M3 follow-on: recall_boost on fetch, decay reaper, retire.
"""

from __future__ import annotations

from datetime import datetime, timezone

from orion.core.activation_decay import decay_activation
from orion.memory.crystallization.schemas import MemoryCrystallizationV1

_SECONDS_PER_DAY = 86400.0


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def seed_dynamics(
    crystallization: MemoryCrystallizationV1,
    *,
    now: datetime,
) -> MemoryCrystallizationV1:
    """Initial encoding: dynamic weight starts at the intrinsic salience prior."""
    updated = crystallization.model_copy(deep=True)
    updated.dynamics.activation = _clamp(crystallization.salience)
    updated.dynamics.formed_at = _aware(now)
    updated.updated_at = _aware(now)
    return updated


def seed_weak_dynamics(
    crystallization: MemoryCrystallizationV1,
    *,
    now: datetime,
    ratio: float = 0.4,
    min_activation: float = 0.05,
    max_activation: float = 0.35,
) -> MemoryCrystallizationV1:
    """Auto-encode at a fraction of salience — weak initial footprint."""
    updated = crystallization.model_copy(deep=True)
    raw = _clamp(crystallization.salience * _clamp(ratio))
    updated.dynamics.activation = max(min_activation, min(max_activation, raw))
    updated.dynamics.formed_at = _aware(now)
    updated.updated_at = _aware(now)
    return updated


def reinforce(
    crystallization: MemoryCrystallizationV1,
    *,
    now: datetime,
    boost: float = 0.2,
) -> MemoryCrystallizationV1:
    """Repetition strengthens: move activation toward 1.0 and refresh recency.

    Multiplicative-toward-ceiling so repeated reinforcement has diminishing returns
    (never overshoots 1.0), matching how rehearsal consolidates but saturates.
    """
    updated = crystallization.model_copy(deep=True)
    current = _clamp(updated.dynamics.activation)
    updated.dynamics.activation = _clamp(current + (1.0 - current) * _clamp(boost))
    updated.dynamics.reinforcement_count += 1
    updated.dynamics.last_reinforced_at = _aware(now)
    updated.updated_at = _aware(now)
    return updated


def recall_boost(
    crystallization: MemoryCrystallizationV1,
    *,
    now: datetime,
    boost: float = 0.08,
) -> MemoryCrystallizationV1:
    """Being recalled is a weaker reinforcement signal than recurrence."""
    updated = crystallization.model_copy(deep=True)
    current = _clamp(updated.dynamics.activation)
    updated.dynamics.activation = _clamp(current + (1.0 - current) * _clamp(boost))
    updated.dynamics.last_recalled_at = _aware(now)
    updated.updated_at = _aware(now)
    return updated


def _reference_time(crystallization: MemoryCrystallizationV1) -> datetime | None:
    dyn = crystallization.dynamics
    ref = dyn.last_reinforced_at or dyn.last_recalled_at or dyn.formed_at
    return _aware(ref) if ref is not None else None


def decayed_activation(
    crystallization: MemoryCrystallizationV1,
    *,
    now: datetime,
    floor: float = 0.0,
) -> float:
    """Activation decayed to `now` on its half-life. Pure read — does not mutate."""
    ref = _reference_time(crystallization)
    if ref is None:
        return _clamp(crystallization.dynamics.activation)
    elapsed = max(0.0, (_aware(now) - ref).total_seconds())
    half_life_seconds = int(crystallization.dynamics.decay_half_life_days * _SECONDS_PER_DAY)
    return decay_activation(
        current=_clamp(crystallization.dynamics.activation),
        elapsed_seconds=elapsed,
        half_life_seconds=half_life_seconds,
        floor=floor,
    )


def decay(
    crystallization: MemoryCrystallizationV1,
    *,
    now: datetime,
    floor: float = 0.0,
) -> MemoryCrystallizationV1:
    """Apply half-life decay in place (returns updated copy)."""
    updated = crystallization.model_copy(deep=True)
    updated.dynamics.activation = decayed_activation(crystallization, now=now, floor=floor)
    updated.updated_at = _aware(now)
    return updated


def should_retire(
    crystallization: MemoryCrystallizationV1,
    *,
    now: datetime,
    floor: float = 0.05,
) -> bool:
    """A memory retires when its decayed weight falls below the recall floor.

    Forgetting is the default: without reinforcement, activation eventually drops under
    `floor` and the crystallization should be de-projected from recall (Phase 4 reaper).
    """
    return decayed_activation(crystallization, now=now) < floor
