"""One live evaluation of the organ, pure: candidate state + carried hysteresis -> held state + events.

The host (orion-durable-runs' ``self_monitor`` node in the ``temporal_self.update`` graph) reads the bet
sources (``orion.self_monitor.collect``), then calls ``evaluate`` here with the compact per-faculty carry
from its checkpoint. Nothing here does I/O or reads a clock.

The carry, per faculty (all JSON):

* ``status``, ``status_since``, ``candidate_status``, ``candidate_held``: ``record.carry_of``, the
  hysteresis memory ``record.advance_carry`` needs;
* ``sources``: the faculty's bet sources, so a faculty missing because its source failed to read keeps
  its held status instead of reading as gone;
* ``recorded``: the held status last written to ``temporal_self_event`` (``None``: nothing recorded);
* ``row_at``: when that row was written.

Events (one ``temporal_self_event`` row each, kind ``faculty_status_changed``):

* **change**: the held status differs from ``recorded``. A faculty seen for the first time writes a
  row only when its status is notable (broken, unreliable, miscalibrated); otherwise its status is
  simply noted as recorded. A row that failed to write leaves ``recorded`` alone, so the next
  evaluation writes it again with the same ``from``.
* **keepalive**: while a notable status holds, a row at least every ``FACULTY_KEEPALIVE_SEC``. The
  chronology closes a faculty arc with no row for ``faculty_stale_sec`` (2.5x), so a stopped organ
  (flag off, crash) never leaves "broken since" standing past its evidence -- the same rule as strain.
* **dropped**: a faculty with no bets in the window (and no source error that could explain it) leaves
  the state; if its recorded status was notable, one row moves it to ``unknown`` and the carry forgets it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, Optional

from orion.schemas.self_monitor import FacultyRecordV1, SelfMonitorStateV1
from orion.self_monitor.collect import erred_sources
from orion.self_monitor.record import HOLD, advance_carry, carry_of

NOTABLE_STATUSES = frozenset({"broken", "unreliable", "miscalibrated"})
FACULTY_EVENT_KIND = "faculty_status_changed"
FACULTY_EVENT_PREFIX = f"{FACULTY_EVENT_KIND}:"
FACULTY_EVENT_TABLE = "self_monitor"
FACULTY_KEEPALIVE_SEC = 12 * 3600.0
DROPPED_REASON = "no bets in the window"


@dataclass(frozen=True)
class FacultyChange:
    faculty: str
    from_status: Optional[str]
    to_status: str
    since: datetime          # when the held status began (the row's occurred_at for a change)
    keepalive: bool = False
    dropped: bool = False
    record: Optional[FacultyRecordV1] = None   # None for a dropped faculty


@dataclass(frozen=True)
class Evaluation:
    state: SelfMonitorStateV1          # held statuses, what is published
    carry: dict[str, dict]             # as if no event row was written (see ``mark_recorded``)
    changes: list[FacultyChange]


def _utc(dt: datetime) -> datetime:
    return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _carry_since(carry: Mapping) -> Optional[datetime]:
    raw = carry.get("status_since")
    if not raw:
        return None
    try:
        return _utc(datetime.fromisoformat(str(raw).replace("Z", "+00:00")))
    except ValueError:
        return None


def _row_age(carry: Mapping, now: datetime) -> Optional[float]:
    raw = carry.get("row_at")
    if not raw:
        return None
    try:
        at = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    return (now - _utc(at)).total_seconds()


def evaluate(candidate: SelfMonitorStateV1, carry: Optional[Mapping[str, Mapping]], *, now: datetime,
             hold: int = HOLD, keepalive_sec: float = FACULTY_KEEPALIVE_SEC) -> Evaluation:
    prev_all: dict[str, Mapping] = {k: v for k, v in (carry or {}).items() if isinstance(v, Mapping)}
    held_records: list[FacultyRecordV1] = []
    out: dict[str, dict] = {}
    changes: list[FacultyChange] = []
    erred = erred_sources(candidate)
    for record in candidate.faculties:
        prev = prev_all.get(record.faculty)
        if prev is not None and erred.intersection(record.sources) and _carry_since(prev) is not None:
            # Part of this faculty's evidence is unread: publish its held status unchanged and do not
            # count this evaluation toward a new candidate (review finding).
            held = record.model_copy(update={"status": prev.get("status") or record.status,
                                             "status_since": _carry_since(prev),
                                             "candidate_held": int(prev.get("candidate_held") or 0)})
            held_records.append(held)
            out[held.faculty] = dict(prev)
            continue
        held = advance_carry(prev, record, now=now, hold=hold)
        held_records.append(held)
        entry = {**carry_of(held), "sources": list(held.sources),
                 "recorded": None if prev is None else prev.get("recorded"),
                 "row_at": None if prev is None else prev.get("row_at")}
        recorded = entry["recorded"]
        since = held.status_since or now
        if prev is None or "recorded" not in prev:
            # First sight of this faculty (or a carry from before this field): say only what is notable.
            if held.status in NOTABLE_STATUSES:
                changes.append(FacultyChange(held.faculty, None, held.status, since, record=held))
            else:
                entry["recorded"] = held.status
        elif held.status != recorded:
            changes.append(FacultyChange(held.faculty, recorded, held.status, since, record=held))
        elif held.status in NOTABLE_STATUSES:
            age = _row_age(prev, now)
            if age is None or age >= keepalive_sec:
                changes.append(FacultyChange(held.faculty, recorded, held.status, since, keepalive=True,
                                             record=held))
        out[held.faculty] = entry

    for faculty, prev in prev_all.items():
        if faculty in out:
            continue
        if erred.intersection(prev.get("sources") or []) or (erred and not prev.get("sources")):
            out[faculty] = dict(prev)  # unread, not gone: keep its memory untouched
            continue
        if prev.get("recorded") in NOTABLE_STATUSES:
            # Keyed on the held status's start, so a retried drop row keeps its id.
            since = _carry_since(prev) or now
            changes.append(FacultyChange(faculty, prev.get("recorded"), "unknown", since, dropped=True))
            out[faculty] = dict(prev)  # forgotten once the closing row is written (mark_recorded)
    state = candidate.model_copy(update={"faculties": held_records})
    return Evaluation(state=state, carry=out, changes=changes)


def mark_recorded(carry: Mapping[str, Mapping], changes: list[FacultyChange], *, now: datetime) -> dict[str, dict]:
    """The carry after ``changes`` were written to ``temporal_self_event``."""
    out = {k: dict(v) for k, v in carry.items()}
    for change in changes:
        if change.dropped:
            out.pop(change.faculty, None)
            continue
        entry = out.get(change.faculty)
        if entry is not None:
            entry["recorded"] = change.to_status
            entry["row_at"] = _utc(now).isoformat()
    return out


def event_row(change: FacultyChange, *, day_id: str, now: datetime) -> dict:
    """One ``temporal_self_event`` row. Deterministic id: a change (and a drop) is keyed by its
    held-status start, a keepalive also by its write time, so a retried write never adds a copy."""
    since = _utc(change.since)
    at = _utc(now)
    event_id = f"{FACULTY_EVENT_PREFIX}{change.faculty}:{since.isoformat()}:{change.to_status}"
    if change.keepalive:
        event_id += f":at:{at.isoformat()}"
    elif change.dropped:
        event_id += ":dropped"
    r = change.record
    payload = {
        "faculty": change.faculty, "from": change.from_status, "to": change.to_status,
        "since": since.isoformat(), "keepalive": change.keepalive, "dropped": change.dropped,
        "reasons": [DROPPED_REASON] if r is None else list(r.reasons),
        "gradeability": None if r is None else r.gradeability,
        "sources": [] if r is None else list(r.sources),
        "n_bets": 0 if r is None else r.n_bets,
        "n_graded": 0 if r is None else r.n_graded,
        "n_failed_to_execute": 0 if r is None else r.n_failed_to_execute,
        "reliability": None if r is None else r.reliability,
        "execution_failure": None if r is None else r.execution_failure,
        "stated_mean": None if r is None else r.stated_mean,
    }
    return {
        "event_id": event_id,
        "day_id": day_id,
        # A change happened when the status was adopted; a keepalive or a drop when it was observed.
        # Every row is stamped when it was WRITTEN, never at ``since``: the chronicle reads rows past its
        # watermark (now - 300 s), so a change row retried an hour later and stamped at ``since`` would
        # sit behind it and never fold (review finding). ``since`` rides in the payload, and the fold
        # keeps it as the arc's episode_began_at.
        "occurred_at": at,
        "source_ref": f"self_monitor.state.v1:{at.isoformat()}",
        "subject_ref": f"faculty:{change.faculty}",
        "label": change.to_status,
        "payload": json.dumps(payload),
    }


def summary(state: SelfMonitorStateV1) -> dict:
    """One line for the checkpoint and /health: counts by held status, and the source errors."""
    counts: dict[str, int] = {}
    for r in state.faculties:
        counts[r.status] = counts.get(r.status, 0) + 1
    return {"computed_at": _utc(state.computed_at).isoformat(), "faculties": len(state.faculties),
            "by_status": dict(sorted(counts.items())), "source_errors": list(state.source_errors)}


__all__ = [
    "DROPPED_REASON",
    "Evaluation",
    "FACULTY_EVENT_KIND",
    "FACULTY_EVENT_PREFIX",
    "FACULTY_EVENT_TABLE",
    "FACULTY_KEEPALIVE_SEC",
    "FacultyChange",
    "NOTABLE_STATUSES",
    "evaluate",
    "event_row",
    "mark_recorded",
    "summary",
]
