from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from orion.core.activation_decay import decay_activation
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


# Re-export shared decay helper (implementation in orion.core.activation_decay).


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


# --- Decay bookkeeping (L6, 2026-10-06) -------------------------------------
#
# A node's stored ``signals.activation.activation`` is a value that has
# ALREADY been decayed up to some moment. Decaying it again by the full time
# since ``temporal.observed_at`` on every tick (the pre-2026-10-06 behavior,
# still available as ``legacy``) re-applies decay that was already applied, so
# the loss compounds: ~2.2% per 30 s tick on a 30-day half-life at 23 h of
# age, instead of ~0.0008%. ``activation_decayed_at`` records the moment the
# stored value is valid as of, so the next decay covers only the time since
# then. Only the dynamics tick recomputes activation and sets it (the Hub's
# second decay writer was removed 2026-10-06: one owner, so writes cannot land
# out of order); every other writer leaves it untouched.
ACTIVATION_DECAYED_AT_KEY = "activation_decayed_at"

DECAY_MODE_SINCE_LAST = "since_last"
DECAY_MODE_LEGACY = "legacy"
DECAY_MODES: frozenset[str] = frozenset({DECAY_MODE_SINCE_LAST, DECAY_MODE_LEGACY})


def normalize_decay_mode(value: object) -> str:
    """Return a valid decay mode; raise on anything unknown (a typo in the
    rollback flag must not silently pick a mode)."""
    mode = str(value or DECAY_MODE_SINCE_LAST).strip().lower()
    if mode not in DECAY_MODES:
        raise ValueError(f"unknown substrate decay mode {value!r}; expected one of {sorted(DECAY_MODES)}")
    return mode


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def parse_activation_decayed_at(raw: object) -> datetime | None:
    """Parse a stored ``activation_decayed_at`` stamp. Unparseable -> None
    (caller falls back to ``observed_at``), never raises."""
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return _as_utc(raw)
    try:
        return _as_utc(datetime.fromisoformat(str(raw).strip()))
    except (TypeError, ValueError):
        return None


def activation_decay_anchor(node: BaseSubstrateNodeV1) -> datetime:
    """The moment the node's stored activation is valid as of.

    ``max(stamp, observed_at)``: a missing/garbled stamp falls back to
    ``observed_at``; a node re-observed after the last decay (a producer wrote
    a fresh activation with a newer ``observed_at``) is decayed only from that
    newer observation, never from the older stamp.
    """
    observed = _as_utc(node.temporal.observed_at)
    stamp = parse_activation_decayed_at((node.metadata or {}).get(ACTIVATION_DECAYED_AT_KEY))
    if stamp is None:
        return observed
    return max(stamp, observed)
