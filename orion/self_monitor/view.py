"""What a reader concludes from the self-monitor's published state, pure.

The host (patch 2: a ``self_monitor`` node in orion-durable-runs' ``temporal_self.update`` graph)
writes one ``SelfMonitorStateV1`` JSON to Redis ``SELF_MONITOR_REDIS_KEY`` every evaluation
(hourly) with a TTL of ``DEFAULT_MAX_AGE_SEC``. A reader turns that raw value into a
``SelfMonitorView`` at its own ``now``.

**Unknown changes nothing.** An absent, unparseable, stale or future-stamped state is ``known=False``
and every reader behaves exactly as if the organ did not exist, so an outage in durable-runs or Redis
can never stop a loop.

**One direction only** (spec, "Effects"): a reader may use a faculty's status only to be *more*
cautious -- refuse or re-test a failing act, ask why, tell Juniper. No status ever loosens a gate,
raises a budget or lifts a refusal the organ did not impose.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Mapping, Optional

from orion.schemas.self_monitor import FacultyRecordV1, SelfMonitorStateV1

SELF_MONITOR_REDIS_KEY = "orion:self_monitor:latest"
# Three hourly evaluations: one missed evaluation never blanks the state, three do.
DEFAULT_MAX_AGE_SEC = 3 * 3600.0
MAX_FUTURE_SKEW_SEC = 300.0
# One spelling across services for the refusal an act gate logs (E1).
TRACK_RECORD_FAILING_REASON = "track_record_failing"


def _utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def parse_state(raw: object) -> Optional[SelfMonitorStateV1]:
    if raw is None:
        return None
    try:
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        data = json.loads(raw) if isinstance(raw, str) else raw
        return SelfMonitorStateV1.model_validate(data)
    except Exception:  # noqa: BLE001 -- any doubt reads as absent
        return None


@dataclass(frozen=True)
class SelfMonitorView:
    known: bool
    reason: str
    age_sec: Optional[float] = None
    faculties: Mapping[str, FacultyRecordV1] = field(default_factory=dict)

    def faculty(self, name: str) -> Optional[FacultyRecordV1]:
        """The faculty's record, or None when the view is unknown or the faculty is not tracked."""
        return self.faculties.get(name) if self.known else None

    def status(self, name: str) -> str:
        record = self.faculty(name)
        return "unknown" if record is None else record.status


UNKNOWN_DISABLED = SelfMonitorView(known=False, reason="disabled")


def self_monitor_view(raw: object, *, now: datetime, max_age_sec: float = DEFAULT_MAX_AGE_SEC) -> SelfMonitorView:
    """Redis value -> what a reader may act on. Anything doubtful is ``known=False``."""
    state = parse_state(raw)
    if state is None:
        return SelfMonitorView(known=False, reason="absent" if raw is None else "unparseable")
    age = (_utc(now) - _utc(state.computed_at)).total_seconds()
    if age < -MAX_FUTURE_SKEW_SEC:
        return SelfMonitorView(known=False, reason="future_stamped", age_sec=round(age, 1))
    if age > float(max_age_sec):
        return SelfMonitorView(known=False, reason="stale", age_sec=round(age, 1))
    return SelfMonitorView(known=True, reason="fresh", age_sec=round(age, 1),
                           faculties={r.faculty: r for r in state.faculties})


__all__ = [
    "DEFAULT_MAX_AGE_SEC",
    "SELF_MONITOR_REDIS_KEY",
    "SelfMonitorView",
    "TRACK_RECORD_FAILING_REASON",
    "UNKNOWN_DISABLED",
    "parse_state",
    "self_monitor_view",
]
