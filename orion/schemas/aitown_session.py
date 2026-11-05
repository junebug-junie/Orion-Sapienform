"""AI Town sessions: the town runs only while a session is open (2026-10-30).

Juniper asked (2026-10-30): AI Town is a want Orion chooses, capped at 2 hours a day. Before this,
the town ran all day: its NPC dialogue made ~1,100 LLM calls and held ~3.2 GPU lease-hours a day on
circe, whether or not anyone (Orion included) was there.

Two kinds of session:

* ``orion`` -- Orion chose to go. orion-embodiment's own timer (``app/town_self_start.py``, ladder
  retirement PR 3, 2026-11-05; before that the motor allocator's ``aitown_session`` dispatch and a
  bus RPC) places the shadow ``aitown.session`` bid, then asks the session controller's gate (kill
  switch, the daily cap, the gap, a session already open). A pass resumes the town and opens the
  ``play`` act (actor ``aitown.session``). The controller pauses the town at the session cap, the
  daily cap, or when the switch goes off. Counts toward the daily cap.
* ``manual`` -- the town is running and Orion did not open it: Juniper pressed Unfreeze in Hub (or
  her open page woke a town that was idle, not paused; a town the controller paused needs Unfreeze). Logged, never paused by the controller, and NOT counted toward
  Orion's cap.

The session ledger is one Redis hash (``AITOWN_SESSION_LEDGER_REDIS_KEY``, field = session id,
value = ``AITownSessionRowV1`` JSON). The controller writes it; orion-durable-runs reads it into
``GET /physiology/report`` (``wants.town``).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

#: The want family and act-ledger actor of an Orion-chosen session (config/physiology/families.yaml).
AITOWN_SESSION_FAMILY = "aitown.session"
#: The act ledger key (grammar_acts.test_before) of the ``play`` act a started session opens:
#: ``aitown_session_dispatch:<act id>``. The prefix is kept from the dispatch era (the act id is now
#: ``aitown_self_start:<uuid>``), so every ``aitown.session`` act has one shape.
AITOWN_SESSION_OBJECT_PREFIX = "aitown_session_dispatch:"
#: The session ledger: Redis hash, field = session_id, value = AITownSessionRowV1 JSON.
AITOWN_SESSION_LEDGER_REDIS_KEY = "orion:aitown:sessions"
#: The controller's live limits (AITownSessionConfigV1 JSON), written at its boot, so the report
#: shows the cap the controller enforces instead of a second copy of the setting.
AITOWN_SESSION_CONFIG_REDIS_KEY = "orion:aitown:sessions:config"

SessionKind = Literal["orion", "manual"]
#: Why a session started: Orion's own want (``want_funded``: the allocator's admission until
#: 2026-11-05, embodiment's timer since; the value is kept so existing ledger readers parse every
#: row), or the town was opened by hand.
StartReason = Literal["want_funded", "juniper_override"]
#: Why a session ended.
#:   session_cap      Orion's session reached its per-session limit.
#:   daily_cap        Orion's sessions today reached the daily limit (America/Denver day).
#:   disabled         the controller's kill switch went off mid-session.
#:   town_paused      the town stopped by itself or by hand (Freeze, or the idle stop).
#:   embodiment_restart  the controller restarted with this session open; closed at its last tick.
StopReason = Literal["session_cap", "daily_cap", "disabled", "town_paused", "embodiment_restart"]


def aitown_session_object_ref(dispatch_id: str) -> str:
    return f"{AITOWN_SESSION_OBJECT_PREFIX}{dispatch_id}"


class AITownSessionRowV1(BaseModel):
    """One session in the ledger. Times are epoch seconds (UTC). ``ended_at`` is None while open;
    ``last_seen_at`` moves every controller tick, so a crash is closed at its last tick, not later."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(..., min_length=1, max_length=256)
    kind: SessionKind
    started_at: float
    last_seen_at: float
    ended_at: Optional[float] = None
    start_reason: StartReason
    stop_reason: Optional[StopReason] = None
    dispatch_id: Optional[str] = Field(None, max_length=256)
    #: Lines Orion spoke in town during the session (the play act's artifact).
    utterances: int = Field(0, ge=0)


class AITownSessionConfigV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    session_max_sec: float
    daily_cap_sec: float
    min_gap_sec: float
    tz: str
    written_at: float


__all__ = [
    "AITOWN_SESSION_CONFIG_REDIS_KEY",
    "AITownSessionConfigV1",
    "AITOWN_SESSION_FAMILY",
    "AITOWN_SESSION_LEDGER_REDIS_KEY",
    "AITOWN_SESSION_OBJECT_PREFIX",
    "AITownSessionRowV1",
    "SessionKind",
    "StartReason",
    "StopReason",
    "aitown_session_object_ref",
]
