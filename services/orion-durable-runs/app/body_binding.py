"""Felt state: one ``body_binding`` row per change in whether a body-economy constraint binds. Pure.

Body economy design Organ 5 "Felt state", "Where Orion sees it"
(docs/superpowers/specs/2026-10-23-organism-body-economy-design.md): Temporal Self arcs
``binding:<constraint>``. The physiology node of ``temporal_self.update`` prices every constraint
each tick (``orion/physiology/budget.py``). This module decides, per constraint and per tick,
whether that priced reading starts, changes, refreshes or ends a binding episode, and shapes the
row the node writes into ``temporal_self_event`` (``source_kind='body_binding'``,
``source_table='physiology'``). The chronicle folds those rows into one ``binding`` arc per
constraint episode (``orion/temporal_self/arcs.py`` ``_fold_binding``).

Bound means the price's ``bind_state`` is one of ``BOUND_STATES``: the set the body report already
calls BINDING (``orion/physiology/report_text.py`` ``BIND_WORDS``):

* ``binding``: forecast use is over the budget, so the price climbs.
* ``exogenous_bind``: over the line even with Orion's share at 0; the price is clamped.
* ``stuck_at_ceiling``: at the ceiling for more than one horizon, unflagged.

``slack`` is not bound. ``no_reading`` is unknown, never a release: nothing is written, the pending
timers restart, and an open arc closes by the reducer's stale timeout at its last row.

Rules, per constraint (``plan_constraint``):

* onset: bound continuously for ``BINDING_ONSET_SEC`` (240 s, two 120 s ticks): anti-flap.
* change: the bound state changed (e.g. binding -> stuck_at_ceiling), at most every
  ``BINDING_CHANGE_MIN_SEC`` so a state flapping between two bound states cannot flood the table.
* keepalive: every ``BINDING_KEEPALIVE_SEC`` while bound.
* release: slack continuously for ``BINDING_RELEASE_SEC``. The arc ends at the first slack reading.
* A gap in readings longer than ``max_gap_sec`` (three ticks) breaks a pending run (the evidence is
  not continuous). An open episode that went unread (no_reading or a gap) writes a keepalive with
  ``after_gap`` as soon as it reads bound again, and its release carries the flag too: the reducer
  starts a new segment, so the unread stretch is never counted as binding time. An open episode whose last row is ``BINDING_STALE_SEC`` old was already closed by the
  reducer: the next bound reading starts a new one.

No I/O, no clock. The caller supplies ``now``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

BOUND_STATES = frozenset({"binding", "exogenous_bind", "stuck_at_ceiling"})
SLACK = "slack"
BINDING_ONSET_SEC = 240.0
BINDING_RELEASE_SEC = 240.0
BINDING_KEEPALIVE_SEC = 1800.0
BINDING_CHANGE_MIN_SEC = 600.0
# The reducer's ReducerConfig.binding_stale_sec: 2.5 keepalives.
BINDING_STALE_SEC = 4500.0
# Three ticks (the regulation thread's own max_prev_gap rule).
MAX_READING_GAP_SEC = 360.0
EVENT_PREFIX = "body_binding:"
SUBJECT_PREFIX = "orion:body:"
WHY_MAX = 300


def _dt(v: Any) -> Optional[datetime]:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    try:
        d = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _iso(d: datetime) -> str:
    return d.astimezone(timezone.utc).isoformat()


@dataclass(frozen=True)
class Plan:
    """What one tick does for one constraint. ``row``: the row kind to write, or None.
    ``written``: the carry entry if the row was written (or no row was due); ``failed``: the
    entry if the write failed, which keeps the episode as it was so the next tick retries."""

    row: Optional[str]
    written: dict
    failed: dict
    began_at: Optional[datetime] = None
    slack_since: Optional[datetime] = None
    # The episode went unread since its last row: the reducer starts a new segment instead of
    # counting the gap as bound time.
    after_gap: bool = False


def plan_constraint(entry: Mapping[str, Any], bind_state: Optional[str], now: datetime, *,
                    max_gap_sec: float = MAX_READING_GAP_SEC) -> Plan:
    """One tick of one constraint's episode. ``entry`` is its carry: ``open`` (an onset row was
    written), ``began_at`` (the first bound reading of the episode), ``state`` (the bound state the
    last row carried), ``row_at`` (when it was written), ``bound_since`` / ``slack_since`` (the start
    of a pending onset / release run), ``seen_at`` (the last known reading), ``after_gap`` (an open
    episode went unread: the next row says so, and the reducer does not count the gap as bound).
    ``max_gap_sec``: the longest reading gap that keeps a run continuous (three ticks)."""
    e = dict(entry)
    seen = _dt(e.get("seen_at"))
    row_at = _dt(e.get("row_at"))
    stale = row_at is None or (now - row_at).total_seconds() >= BINDING_STALE_SEC
    if e.get("open") and (stale or _dt(e.get("began_at")) is None):
        # No row for the reducer's stale timeout (it closed this arc at its last row already), or a
        # malformed carry: start over rather than assert on it every tick.
        e = {"seen_at": e.get("seen_at")}
        row_at = None
    gap = seen is None or not (0.0 <= (now - seen).total_seconds() <= max_gap_sec)
    if gap:
        e["bound_since"] = e["slack_since"] = None  # the readings had a gap: a pending run restarts
        if e.get("open"):
            e["after_gap"] = True
    if bind_state not in BOUND_STATES and bind_state != SLACK:
        # no_reading, no price, or the constraint is absent this tick: unknown is not a release.
        e.update(bound_since=None, slack_since=None, seen_at=None)
        if e.get("open"):
            e["after_gap"] = True
        return Plan(None, e, e)
    e["seen_at"] = _iso(now)
    began = _dt(e.get("began_at"))
    if bind_state in BOUND_STATES:
        e["slack_since"] = None
        if not e.get("open"):
            since = _dt(e.get("bound_since")) or now
            e["bound_since"] = _iso(since)
            if (now - since).total_seconds() < BINDING_ONSET_SEC:
                return Plan(None, e, e)
            written = {"open": True, "began_at": _iso(since), "state": bind_state, "row_at": _iso(now),
                       "seen_at": _iso(now)}
            return Plan("onset", written, e, began_at=since)
        assert row_at is not None and began is not None
        age = (now - row_at).total_seconds()
        if bind_state != e.get("state") and age >= BINDING_CHANGE_MIN_SEC:
            kind = "change"
        elif age >= BINDING_KEEPALIVE_SEC or (e.get("after_gap") and age >= BINDING_CHANGE_MIN_SEC):
            # After an unread stretch: as soon as the change rate limit allows (a reading flapping to
            # no_reading cannot flood the table). Until then the flag waits for the next row.
            kind = "keepalive"
        else:
            return Plan(None, e, e)
        written = {**e, "state": bind_state, "row_at": _iso(now)}
        written.pop("after_gap", None)
        return Plan(kind, written, e, began_at=began, after_gap=bool(e.get("after_gap")))
    # slack
    e["bound_since"] = None
    if not e.get("open"):
        return Plan(None, e, e)
    since = _dt(e.get("slack_since")) or now
    e["slack_since"] = _iso(since)
    if (now - since).total_seconds() < BINDING_RELEASE_SEC:
        return Plan(None, e, e)
    return Plan("release", {"seen_at": _iso(now)}, e, began_at=began, slack_since=since,
                after_gap=bool(e.get("after_gap")))


def keep_entry(entry: Mapping[str, Any]) -> bool:
    """Carry only what can still matter: an open episode or a pending run."""
    return bool(entry.get("open") or entry.get("bound_since") or entry.get("slack_since"))


def binding_row(plan: Plan, reading: Any, now: datetime, day_id: str, *, self_spent: bool) -> dict:
    """The ``temporal_self_event`` row for ``plan`` (``reading``: the priced
    ``PhysiologyConstraintReadingV1`` of this tick). Deterministic event id."""
    assert plan.row is not None and plan.began_at is not None
    price = reading.price
    c = reading.constraint
    began = _iso(plan.began_at)
    at = _iso(now)
    payload = {
        "row": plan.row, "constraint": c, "bind_state": price.bind_state if price is not None else None,
        "began_at": began, "slack_since": _iso(plan.slack_since) if plan.slack_since else None,
        "unit": reading.unit, "level": reading.level, "line": reading.budget,
        "price": price.value if price is not None else None, "ceiling": price.ceiling if price is not None else None,
        "why": (price.why if price is not None else "")[:WHY_MAX], "self_spent": bool(self_spent),
        "after_gap": bool(plan.after_gap),
    }
    return {
        "event_id": f"{EVENT_PREFIX}{c}:{began}:{plan.row}:{at}",
        "day_id": day_id,
        "occurred_at": now.astimezone(timezone.utc),
        "source_ref": f"physiology.state.v1:{at}",
        "subject_ref": f"{SUBJECT_PREFIX}{c}",
        "label": c,
        "payload": json.dumps(payload),
    }


__all__ = [
    "BINDING_CHANGE_MIN_SEC",
    "BINDING_KEEPALIVE_SEC",
    "BINDING_ONSET_SEC",
    "BINDING_RELEASE_SEC",
    "BINDING_STALE_SEC",
    "BOUND_STATES",
    "MAX_READING_GAP_SEC",
    "Plan",
    "binding_row",
    "keep_entry",
    "plan_constraint",
]
