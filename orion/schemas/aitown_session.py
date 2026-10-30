"""AI Town sessions: the town runs only while a session is open (2026-10-30).

Juniper asked (2026-10-30): AI Town is a want Orion chooses, capped at 2 hours a day. Before this,
the town ran all day: its NPC dialogue made ~1,100 LLM calls and held ~3.2 GPU lease-hours a day on
circe, whether or not anyone (Orion included) was there.

Two kinds of session:

* ``orion`` -- Orion chose to go. The motor allocator admits the ``aitown_session`` dispatch
  (orion-execution-dispatch-runtime, executor key ``orion.aitown_session.start.v1``), which sends
  one ``orion:aitown_session:start:request`` RPC. orion-embodiment's session controller runs its own
  gate (kill switch, the daily cap, a session already open), resumes the town, and opens the
  ``play`` act (actor ``aitown.session``). It pauses the town at the session cap, the daily cap, or
  when the switch goes off. Counts toward the daily cap.
* ``manual`` -- the town is running and Orion did not open it: Juniper pressed Unfreeze in Hub (or
  her open page woke a town that was idle, not paused; a town the controller paused needs Unfreeze). Logged, never paused by the controller, and NOT counted toward
  Orion's cap.

The session ledger is one Redis hash (``AITOWN_SESSION_LEDGER_REDIS_KEY``, field = session id,
value = ``AITownSessionRowV1`` JSON). The controller writes it; orion-durable-runs reads it into
``GET /physiology/report`` (``wants.town``).
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

AITOWN_SESSION_START_REQUEST_CHANNEL = "orion:aitown_session:start:request"
AITOWN_SESSION_START_REPLY_PREFIX = "orion:aitown_session:start:reply:"
AITOWN_SESSION_START_REQUEST_KIND = "aitown_session.start.request.v1"
AITOWN_SESSION_START_RESULT_KIND = "aitown_session.start.result.v1"

#: The want family and act-ledger actor of an Orion-chosen session (config/physiology/families.yaml).
AITOWN_SESSION_FAMILY = "aitown.session"
#: The act ledger key (grammar_acts.test_before) of the ``play`` act a started session opens.
AITOWN_SESSION_OBJECT_PREFIX = "aitown_session_dispatch:"
#: The session ledger: Redis hash, field = session_id, value = AITownSessionRowV1 JSON.
AITOWN_SESSION_LEDGER_REDIS_KEY = "orion:aitown:sessions"
#: The controller's live limits (AITownSessionConfigV1 JSON), written at its boot, so the report
#: shows the cap the controller enforces instead of a second copy of the setting.
AITOWN_SESSION_CONFIG_REDIS_KEY = "orion:aitown:sessions:config"

SessionKind = Literal["orion", "manual"]
#: Why a session started: the allocator funded Orion's want, or the town was opened by hand.
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


StartState = Literal["started", "refused"]


class AITownSessionStartRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dispatch_id: str = Field(..., min_length=1, max_length=256)
    correlation: dict[str, Any] = Field(default_factory=dict)


class AITownSessionStartResultV1(BaseModel):
    """``refusal`` is set only for ``state=refused``: the controller's named reason
    (``disabled`` | ``daily_cap`` | ``cooldown`` | ``session_open`` | ``manual_session_open`` |
    ``ledger_unknown`` | ``town_unreachable`` | ``resume_failed`` | ``already_ran``). ``retry_after_sec`` is how long the
    same refusal will hold, from the controller's own clock (the next local midnight for
    ``daily_cap``)."""

    model_config = ConfigDict(extra="forbid")

    dispatch_id: str
    state: StartState
    session_id: Optional[str] = Field(None, max_length=256)
    object_ref: Optional[str] = None
    refusal: Optional[str] = Field(None, max_length=200)
    retry_after_sec: Optional[float] = Field(None, ge=0.0)
    gates: dict[str, Any] = Field(default_factory=dict)


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
    "AITOWN_SESSION_START_REPLY_PREFIX",
    "AITOWN_SESSION_START_REQUEST_CHANNEL",
    "AITOWN_SESSION_START_REQUEST_KIND",
    "AITOWN_SESSION_START_RESULT_KIND",
    "AITownSessionRowV1",
    "AITownSessionStartRequestV1",
    "AITownSessionStartResultV1",
    "SessionKind",
    "StartReason",
    "StartState",
    "StopReason",
    "aitown_session_object_ref",
]
