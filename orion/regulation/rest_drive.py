"""The rest drive, pure (Temporal Self rev 4, R2). No IO.

Two halves, one module, so producer and readers cannot drift:

* `read_rest_drive` -- the dream service turns one sleep-pressure check into a
  `DriveReadingV1`, from exactly the values its own gate in
  services/orion-dream/app/cycle.py uses (`should_sleep`'s threshold, the
  `overdue` backstop, the `too_soon` refractory clock).
* `rest_drive_view` / `eased_cooldown_sec` -- a reader decides whether Orion is
  tired *now*. Only a fresh `due` reading is tired. Absent, stale, malformed,
  `no_reading`, or from the future is UNKNOWN, and UNKNOWN changes nothing.

What "ease off" means is deliberately small and one-directional: a reader may
only stretch its own cooldown by a multiplier >= 1. It never refuses outright,
never touches a daily cap, quiet hours, a forced/operator run, or a safety gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

from orion.schemas.dream_cycle import SleepPressureV1
from orion.schemas.drive_reading import DriveReadingV1, parse_drive_reading

Verdict = Literal["tired", "not_tired", "unknown"]

# A reading stamped this far ahead of the reader is a clock fault, not evidence.
MAX_FUTURE_SKEW_SEC = 300.0


def _utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def read_rest_drive(
    pressure: SleepPressureV1,
    *,
    now: datetime,
    source_ref: str,
    last_attempt_end: Optional[datetime],
    min_interval_hours: float,
    overdue: bool,
    has_candidates: bool,
    source_errors: tuple[str, ...] | list[str] = (),
) -> DriveReadingV1:
    """One dream check -> one reading. State precedence:

    no_reading (a source read failed: the level would be an undercount)
    > refractory (the dream cannot sleep yet, whatever the level)
    > due (level >= threshold, or the overdue backstop with material to replay)
    > resting (level at the declared rest, 0.0)
    > building.
    """
    now = _utc(now)
    last_end = _utc(last_attempt_end)
    refractory_until = (
        last_end + timedelta(hours=min_interval_hours) if last_end is not None else None
    )
    common = dict(
        observed_at=now,
        threshold=pressure.threshold,
        accumulating_since=_utc(pressure.since),
        last_discharge_at=last_end,
        refractory_until=refractory_until,
        source_ref=source_ref,
    )
    if source_errors:
        reason = "source_errors:" + ",".join(sorted(set(source_errors)))
        return DriveReadingV1(state="no_reading", no_reading_reason=reason[:200], **common)
    level = float(pressure.pressure)
    if refractory_until is not None and now < refractory_until:
        return DriveReadingV1(state="refractory", level=level, **common)
    if level >= pressure.threshold:
        return DriveReadingV1(state="due", due_reason="threshold", level=level, **common)
    if overdue and has_candidates:
        return DriveReadingV1(state="due", due_reason="overdue", level=level, **common)
    if level <= 0.0:
        return DriveReadingV1(state="resting", level=level, **common)
    return DriveReadingV1(state="building", level=level, **common)


def no_rest_reading(*, now: datetime, source_ref: str, threshold: float, reason: str) -> DriveReadingV1:
    """The dream could not read its pressure at all this check."""
    return DriveReadingV1(
        observed_at=_utc(now), threshold=threshold, state="no_reading",
        no_reading_reason=reason[:200], source_ref=source_ref,
    )


@dataclass(frozen=True)
class RestDriveView:
    """What a reader concluded at `now`. `reason` is for logs and status pages."""

    verdict: Verdict
    reason: str
    state: Optional[str] = None
    level: Optional[float] = None
    age_sec: Optional[float] = None
    source_ref: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict, "reason": self.reason, "state": self.state,
            "level": self.level, "age_sec": self.age_sec, "source_ref": self.source_ref,
        }


UNKNOWN_DISABLED = RestDriveView(verdict="unknown", reason="disabled")


def rest_drive_view(raw: object, *, now: datetime, max_age_sec: float) -> RestDriveView:
    """Is Orion tired right now? Only a fresh, parseable `due` reading says yes."""
    reading = parse_drive_reading(raw)
    if reading is None:
        return RestDriveView(verdict="unknown", reason="absent" if raw is None else "unparseable")
    age = (_utc(now) - reading.observed_at).total_seconds()
    base = dict(state=reading.state, level=reading.level, age_sec=round(age, 1), source_ref=reading.source_ref)
    if age < -MAX_FUTURE_SKEW_SEC:
        return RestDriveView(verdict="unknown", reason="future_stamped", **base)
    if age > max_age_sec:
        return RestDriveView(verdict="unknown", reason="stale", **base)
    if reading.state == "no_reading":
        return RestDriveView(verdict="unknown", reason="no_reading", **base)
    if reading.state == "due":
        return RestDriveView(verdict="tired", reason=f"due:{reading.due_reason}", **base)
    return RestDriveView(verdict="not_tired", reason=reading.state, **base)


def eased_cooldown_sec(base_sec: float, view: RestDriveView, multiplier: float) -> Optional[float]:
    """The longer cooldown a tired reader keeps, or None when nothing changes.

    None (not `base_sec`) so callers can tell "eased" from "unchanged" and log a
    distinct reason. A multiplier below 1 never shortens anything."""
    if view.verdict != "tired":
        return None
    m = float(multiplier)
    if not m > 1.0:
        return None
    return float(base_sec) * m


__all__ = [
    "MAX_FUTURE_SKEW_SEC",
    "RestDriveView",
    "UNKNOWN_DISABLED",
    "eased_cooldown_sec",
    "no_rest_reading",
    "read_rest_drive",
    "rest_drive_view",
]
