from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math

from orion.core.schemas.cognitive_substrate import BaseSubstrateNodeV1


@dataclass(frozen=True)
class ActivationConfig:
    recency_horizon_seconds: int = 3600
    recency_weight: float = 0.45
    salience_weight: float = 0.35
    pressure_weight: float = 0.2
    attenuation: float = 0.6
    min_delta: float = 0.02
    max_hops: int = 2
    allowed_predicates: frozenset[str] = frozenset(
        {
            "supports",
            "refines",
            "associated_with",
            "observed_in",
            "activates",
            "causes",
            "seeks",
            "blocks",
            "satisfies",
            "co_occurs_with",
        }
    )


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def recency_score(node: BaseSubstrateNodeV1, *, now: datetime, horizon_seconds: int) -> float:
    observed = node.temporal.observed_at
    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=timezone.utc)
    age_seconds = max(0.0, (now - observed).total_seconds())
    if horizon_seconds <= 0:
        return 0.0
    return _clamp(1.0 - (age_seconds / float(horizon_seconds)))


def decay_activation(*, current: float, elapsed_seconds: float, half_life_seconds: int | None, floor: float) -> float:
    if elapsed_seconds <= 0:
        return _clamp(current)
    if not half_life_seconds:
        return _clamp(max(floor, current))
    decay_multiplier = math.pow(0.5, elapsed_seconds / float(half_life_seconds))
    return _clamp(max(floor, current * decay_multiplier))


def seed_activation(
    node: BaseSubstrateNodeV1,
    *,
    now: datetime,
    config: ActivationConfig,
    pressure: float,
    contradiction_boost: float,
) -> float:
    recent = recency_score(node, now=now, horizon_seconds=config.recency_horizon_seconds)
    activation = (
        (recent * config.recency_weight)
        + (node.signals.salience * config.salience_weight)
        + (pressure * config.pressure_weight)
        + contradiction_boost
    )
    if node.node_kind == "tension":
        activation += float(getattr(node, "intensity", 0.0)) * 0.25
    if node.node_kind == "state_snapshot":
        activation += 0.15
    return _clamp(activation)
