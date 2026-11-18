"""What a reader concludes from Orion's arousal reading, pure (Temporal Self rev 4, R3 readers).

durable-runs writes one ``RegulationStateV1`` to Redis ``orion:regulation:latest`` every
regulate step (120 s, and on every Juniper turn; 360 s TTL). A reader turns that raw value
into one of four levels at its own ``now``:

* ``engaged`` / ``idle`` / ``strained`` -- a fresh, parseable reading at that level;
* ``unknown`` -- absent, unparseable, stale, stamped in the future, the producer's own
  ``unknown`` (stale inputs, or ``ORION_REGULATION_AROUSAL_ENABLED=false`` -> ``disabled``),
  or the reader's flag off.

**Unknown changes nothing.** Every reader treats it exactly like "arousal does not exist", so
an outage in durable-runs or Redis can never stop a loop or a sleep.

**One direction only** (spec R3, risk 10): a reader may use a level only to be *more*
conservative -- hold optional work while ``strained``, avoid self-occupation while
``engaged``. No level ever loosens a gate, and no safety, courtesy, or spend gate reads it
(enforced at branch level by ``tests/test_regulation_must_not_readers.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Optional, Tuple

from orion.schemas.regulation import parse_regulation_state

Level = Literal["engaged", "idle", "strained", "unknown"]

# A reading stamped this far ahead of the reader is a clock fault, not evidence.
MAX_FUTURE_SKEW_SEC = 300.0
# Same as the producer's Redis TTL (ORION_REGULATION_REDIS_TTL_SEC): three regulate steps.
DEFAULT_MAX_AGE_SEC = 360.0

# Refusal reasons readers log and return. One spelling across services so a grep over logs
# finds every arousal refusal at once.
AROUSAL_STRAINED_REASON = "arousal_strained"
AROUSAL_ENGAGED_REASON = "arousal_engaged"
AROUSAL_ENGAGED_COOLDOWN_REASON = "arousal_engaged_cooldown"


def _utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class ArousalView:
    """What a reader concluded at ``now``. ``reason`` is for logs and status pages."""

    level: Level
    reason: str
    since: Optional[datetime] = None
    age_sec: Optional[float] = None
    reasons: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def known(self) -> bool:
        return self.level != "unknown"

    def as_dict(self) -> dict:
        return {
            "level": self.level,
            "reason": self.reason,
            "since": None if self.since is None else self.since.isoformat(),
            "age_sec": self.age_sec,
            "reasons": list(self.reasons),
        }


UNKNOWN_DISABLED = ArousalView(level="unknown", reason="disabled")


def arousal_view(raw: object, *, now: datetime, max_age_sec: float = DEFAULT_MAX_AGE_SEC) -> ArousalView:
    """Redis value -> the level a reader may act on. Anything doubtful is ``unknown``."""
    state = parse_regulation_state(raw)
    if state is None:
        return ArousalView(level="unknown", reason="absent" if raw is None else "unparseable")
    reading = state.arousal
    age = (_utc(now) - _utc(state.generated_at)).total_seconds()
    base = dict(since=_utc(reading.since), age_sec=round(age, 1), reasons=tuple(reading.reasons))
    if age < -MAX_FUTURE_SKEW_SEC:
        return ArousalView(level="unknown", reason="future_stamped", **base)
    if age > float(max_age_sec):
        return ArousalView(level="unknown", reason="stale", **base)
    if reading.arousal_level == "unknown":
        first = reading.reasons[0] if reading.reasons else "unspecified"
        return ArousalView(level="unknown", reason=f"producer_unknown:{first}", **base)
    return ArousalView(level=reading.arousal_level, reason=reading.arousal_level, **base)


def engaged_cooldown_sec(base_sec: float, view: ArousalView, multiplier: float) -> Optional[float]:
    """The longer cooldown a reader keeps while Juniper is talking with Orion, else None.

    None (not ``base_sec``) so callers can tell "stretched" from "unchanged". A multiplier at
    or below 1 never changes anything (it can never shorten a cooldown)."""
    if view.level != "engaged":
        return None
    m = float(multiplier)
    if not m > 1.0:
        return None
    return float(base_sec) * m


__all__ = [
    "AROUSAL_ENGAGED_COOLDOWN_REASON",
    "AROUSAL_ENGAGED_REASON",
    "AROUSAL_STRAINED_REASON",
    "ArousalView",
    "DEFAULT_MAX_AGE_SEC",
    "Level",
    "MAX_FUTURE_SKEW_SEC",
    "UNKNOWN_DISABLED",
    "arousal_view",
    "engaged_cooldown_sec",
]
