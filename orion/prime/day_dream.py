"""The day dream's plan reaches Prime (day-dream Patch 2b). CPU only, never a model call.

Design: docs/superpowers/specs/2026-11-17-nightly-day-dream-design.md, "Patch 2". The nightly ``dream.day`` run
(orion-durable-runs) stores a ref-checked plan in its ``dream_day`` row: keep or drop each concern Prime held when
the night began, and at most one new concern. Once per Denver day Prime reads that row
(``GET /dream-day/result``) and applies it here.

Rules, so the plan never undoes Prime's evidence clock (docs/superpowers/specs/2026-11-17-prime-concerns-decided-
by-attention-design.md, #3574/#3610 patch 1):

* ``drop`` closes the concern now, end ``dropped``, note ``dropped: dream <day>: <reason>``. Its own note, not
  ``faded``, so the fade evidence #3648 is waiting on stays readable.
* ``keep`` changes nothing. A keep is a model saying so; only a sent question or a take-on renews a concern
  (``state.touch_concern``). The keep and its reason stay in the ``dream_day`` row.
* ``new`` opens one concern the way a letter thread does: deterministic id, inside ``MAX_CONCERNS``, no
  displacement (that is #3610 patch 2). No room: counted, not opened.

Applied at most once per day (``PrimeDreamDayStateV1.applied``), so a restart or a replay opens nothing twice.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from typing import Any, Optional

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from orion.prime.state import MAX_CONCERNS, PrimeConcernV1, PrimeStateV1, close_concern, note_outcome
from orion.schemas.dream_day import DreamDayPlanItemV1

DREAM_DAY_KEY = "orion:prime:dream_day"
DREAM_EVENT_PREFIX = "dream_day:"
CHECK_INTERVAL_SEC = 900.0
NEW_MAX = 1
MAX_APPLIED = 14
REASON_CAP = 200


class PrimeDreamDayAppliedV1(BaseModel):
    """What Prime did with one night's plan."""

    model_config = ConfigDict(extra="forbid")

    day: date
    run_id: Optional[str] = None
    applied_at: AwareDatetime
    dropped: list[str] = Field(default_factory=list)
    opened: list[str] = Field(default_factory=list)
    kept: int = 0
    # Plan items naming a concern Prime no longer holds, a second new, or a new with no room.
    skipped: list[str] = Field(default_factory=list)


class PrimeDreamDayStateV1(BaseModel):
    """Prime's record of applied nights, saved under ``DREAM_DAY_KEY`` (not in ``orion:prime:state``, which Hub
    validates: an added field there needs a rollout order)."""

    model_config = ConfigDict(extra="forbid")

    applied: list[PrimeDreamDayAppliedV1] = Field(default_factory=list)


def is_applied(st: PrimeDreamDayStateV1, day: date) -> bool:
    return any(a.day == day for a in st.applied)


def record(st: PrimeDreamDayStateV1, done: PrimeDreamDayAppliedV1) -> PrimeDreamDayStateV1:
    return st.model_copy(update={"applied": [*[a for a in st.applied if a.day != done.day], done][-MAX_APPLIED:]})


def dream_concern_id(day: date, index: int) -> str:
    return hashlib.sha256(f"{DREAM_EVENT_PREFIX}{day.isoformat()}#new{index}".encode()).hexdigest()[:16]


def parse_plan(raw: Any) -> list[DreamDayPlanItemV1]:
    """The stored plan's items; an unreadable item is left out, never guessed."""
    out = []
    for item in raw if isinstance(raw, list) else []:
        try:
            out.append(DreamDayPlanItemV1.model_validate(item))
        except Exception:  # noqa: BLE001
            continue
    return out


def apply_plan(state: PrimeStateV1, day: date, plan: list[DreamDayPlanItemV1], now: datetime, *,
               run_id: Optional[str] = None) -> tuple[PrimeStateV1, PrimeDreamDayAppliedV1]:
    """Apply one night's plan to Prime's concerns (rules in the module docstring)."""
    concerns = list(state.concerns)
    dropped: list[str] = []
    opened: list[str] = []
    skipped: list[str] = []
    kept = 0
    news = 0
    for i, item in enumerate(plan):
        reason = " ".join((item.reason or "").split())[:REASON_CAP] or "no reason given"
        if item.action == "keep":
            kept += 1
            continue
        if item.action == "drop":
            gone = next((c for c in concerns if c.concern_id == item.concern_id), None)
            if gone is None:
                skipped.append(f"drop {item.concern_id}: not held")
                continue
            state = close_concern(state, gone, end="dropped", now=now,
                                  note=f"dropped: dream {day.isoformat()}: {reason}")
            concerns = [c for c in concerns if c.concern_id != gone.concern_id]
            dropped.append(gone.concern_id)
            continue
        # new
        gist = " ".join((item.gist or "").split())
        if not gist:
            skipped.append("new: no gist")
            continue
        if news >= NEW_MAX:
            skipped.append("new: only one per night")
            continue
        news += 1
        cid = dream_concern_id(day, i)
        if any(c.concern_id == cid for c in concerns):
            continue
        if len(concerns) >= MAX_CONCERNS:
            skipped.append("new: no room")
            continue
        concerns.append(PrimeConcernV1(
            concern_id=cid, text=f"from my dream on {day.isoformat()}: {gist}"[:300],
            event_ref=f"{DREAM_EVENT_PREFIX}{day.isoformat()}#new{i}", opened_at=now, last_evidence_at=now))
        opened.append(cid)
    state = state.model_copy(update={"concerns": concerns})
    done = PrimeDreamDayAppliedV1(day=day, run_id=run_id, applied_at=now, dropped=dropped, opened=opened,
                                  kept=kept, skipped=skipped)
    if dropped or opened or kept:
        state = note_outcome(state, (
            f"my dream on {day.isoformat()} looked at my concerns: dropped {len(dropped)}, opened {len(opened)}, "
            f"kept {kept} (a dream's keep renews nothing)" + (f"; {len(skipped)} skipped" if skipped else "")))
    return state, done
