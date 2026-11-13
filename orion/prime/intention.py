"""Intentions Orion takes on or lets go (Step -1, docs/superpowers/specs/2026-11-13-choice-as-expected-free-energy-design.md).

Pure. CPU only. One primitive with one adapter per source; promises are the first source.

The three choices, on what already exists:

* **take on** -- the one new Prime move, ``{"kind": "take_on", "concern_id", "why", "not_before"?}`` (orion/prime/
  thought.py ``TakeOnMoveV1``). Offered only on a concern whose source has an adapter (``ADAPTERS``). Prime opens a
  ``make`` act (actor ``prime``, ``object_ref intention:<concern_id>``, ``make_act``): taking on is the graded act,
  since a turn concern places no bet of its own (orion/prime/handback.py). Prime holds the order until
  ``not_before`` itself (nothing reads ``WantWindowV1.earliest``), then orders one carry run from Hub.
* **let go** -- the existing concern ``drop`` with its why. With an open take-on act, the act closes 0 with
  ``let go: <why>`` and the reply is never delivered.
* **not yet** -- the existing ``keep``, or no edit. A source's concern ages out at its horizon (``age_out``).

The act closes 1 when the carry run completed and Hub stored its reply on the adapter's route; 0 when the run
failed, Hub refused, the reply could not be stored, Prime let it go, or the horizon passed. While an intention is
open its concern is kept alive (``keep_taken_on``): Orion settled the whether; the run settles the concern.

Adding a source: one adapter module in orion/prime/intention_sources/ (``PREFIX``, ``SOURCE``, ``HORIZON_SEC``,
``horizon_from``, ``fetch``, ``adapt``) and one entry in ``ADAPTERS``. Nothing else changes (acceptance S-1.9).

State lives in its own Redis key (``INTENTIONS_KEY``), apart from ``orion:prime:state``, so an older image still
loads Prime's state after a rollback. It never holds a promise's quote: the quote reaches only the carry turn's
prompt, which Hub builds (acceptance S-1.8: ordinary turns' motor prefix carries no promise text from this path).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from types import ModuleType
from typing import Any, Literal, Optional

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from orion.grammar.acts import build_act
from orion.prime.acts import ACTOR
from orion.prime.intention_sources import turn_promise
from orion.prime.state import PrimeConcernV1, PrimeStateV1, close_concern, note_change
from orion.schemas.grammar import ActV1
from orion.schemas.intention import IntentionRoute, IntentionSource

# prefix of a concern's event_ref -> its adapter. One line per source.
ADAPTERS: dict[str, ModuleType] = {
    turn_promise.PREFIX: turn_promise,
}

INTENTIONS_KEY = "orion:prime:intentions"
INTENTION_COMPONENT = "prime.intention"
OBJECT_PREFIX = "intention:"
# The act's horizon past the concern's own: the run's terminal and the delivery land inside it.
ACT_GRACE_SEC = 3600.0
# A not_before this close to the horizon leaves the run no room (the hold releases 1800 s before the deadline).
MIN_RUN_ROOM_SEC = 7200.0
# How often the tick looks at an open intention (HTTP reads).
CHECK_INTERVAL_SEC = 60.0
MAX_DONE = 30
NOTE_CAP = 300
TERMINAL = frozenset({"completed", "failed", "cancelled", "abandoned"})

MAKE_BASIS = (
    "artifact_produced = the one intention.carry durable run this take-on ordered ended completed AND Hub stored "
    "its reply on the source's route (chat_thread: a chat_history_log row in the originating session; read from "
    "Hub POST /api/intention/deliver). 0 when the run failed or was cancelled, Hub refused the order, the reply "
    "could not be stored, Prime let the concern go, or the concern's horizon passed (turn promises: promise time + "
    "72 h). horizon = the concern's horizon + 3600 s"
)

IntentionOutcome = Literal["carried", "not_carried", "let_go", "expired"]


class PrimeIntentionOpenV1(BaseModel):
    """One concern Prime took on, from the take-on until its act closes."""

    model_config = ConfigDict(extra="forbid")

    concern_id: str
    event_ref: str
    source: IntentionSource
    why: str = Field(max_length=600)
    thought_id: str
    act_id: str
    act_horizon_sec: float
    taken_at: AwareDatetime
    # Prime orders no run before this (None: at once).
    not_before: Optional[AwareDatetime] = None
    # Prime's estimate at take-on; Hub's answer (the adapter's) replaces it.
    horizon_at: AwareDatetime
    run_id: str
    correlation_id: str
    # Hub's last answer (None = not ordered yet, or not answered: send it again).
    order_status: Optional[str] = None
    order_detail: Optional[str] = Field(default=None, max_length=500)
    ordered_at: Optional[AwareDatetime] = None
    route: Optional[IntentionRoute] = None
    route_target: Optional[str] = None
    # The run's last read status, and the delivery's last answer.
    run_status: Optional[str] = None
    deliver_status: Optional[str] = None
    deliver_detail: Optional[str] = Field(default=None, max_length=500)


class PrimeIntentionDoneV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    concern_id: str
    source: IntentionSource
    outcome: IntentionOutcome
    note: str = Field(default="", max_length=NOTE_CAP)
    taken_at: AwareDatetime
    closed_at: AwareDatetime
    act_id: str
    run_id: Optional[str] = None
    pinged: Optional[bool] = None


class PrimeIntentionsStateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["prime.intentions.v1"] = "prime.intentions.v1"
    open: list[PrimeIntentionOpenV1] = Field(default_factory=list)
    done: list[PrimeIntentionDoneV1] = Field(default_factory=list)


# --- adapters ---------------------------------------------------------------------------------------------------


def adapter_for(event_ref: Optional[str]) -> Optional[ModuleType]:
    ref = event_ref or ""
    return next((a for prefix, a in ADAPTERS.items() if ref.startswith(prefix)), None)


def has_adapter(concern: PrimeConcernV1) -> bool:
    return adapter_for(concern.event_ref) is not None


def horizon_for(concern: PrimeConcernV1) -> Optional[datetime]:
    """Prime's estimate of the concern's deadline (Hub's adapter answers the exact one)."""
    a = adapter_for(concern.event_ref)
    return a.horizon_from(concern.opened_at) if a is not None else None


def carry_prompt(concern_id: str, body: str) -> str:
    """The carry turn's prompt: the concern ref first, so the hand-back settles it when the turn completes
    (orion/prime/handback.py reads ``prime_concern:<id>`` from the PROMPT only)."""
    return f"prime_concern:{concern_id}\n\n{body}"


# --- the take-on ------------------------------------------------------------------------------------------------


def find_open(st: PrimeIntentionsStateV1, concern_id: str) -> Optional[PrimeIntentionOpenV1]:
    return next((o for o in st.open if o.concern_id == concern_id), None)


def take_on_refusal(concern_id: str, state: PrimeStateV1, st: PrimeIntentionsStateV1, now: datetime, *,
                    enabled: bool) -> Optional[str]:
    """Why a take-on cannot go, or None. Never silent: the reason is shown in Prime's next frame (S-1.1)."""
    if not enabled:
        return "taking a concern on is switched off"
    concern = next((c for c in state.concerns if c.concern_id == concern_id), None)
    if concern is None:
        return f"I hold no concern {concern_id[:40]!r}"
    if concern.graded == "settled":
        return "that concern is already settled"
    if not has_adapter(concern):
        return ("nothing can carry that concern out: only promises I made in a chat with Juniper can be taken on "
                "today")
    if find_open(st, concern_id) is not None:
        return "I already took that concern on; its run is not finished"
    if any(d.concern_id == concern_id for d in st.done):
        return "I already took that concern on once (one run per concern)"
    horizon = horizon_for(concern)
    if horizon is not None and now >= horizon - timedelta(seconds=MIN_RUN_ROOM_SEC):
        return f"its horizon ({horizon.strftime('%Y-%m-%d %H:%MZ')}) leaves no room for a run"
    return None


def parse_not_before(raw: Any, now: datetime, horizon: Optional[datetime]) -> tuple[Optional[datetime], Optional[str]]:
    """(not_before or None = at once, refusal). A time already past means at once. A naive time is UTC."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None, None
    try:
        when = raw if isinstance(raw, datetime) else datetime.fromisoformat(str(raw).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None, f"not_before {str(raw)[:60]!r} is not an ISO time"
    if when.tzinfo is None:
        from datetime import timezone

        when = when.replace(tzinfo=timezone.utc)
    if when <= now:
        return None, None
    if horizon is not None and when >= horizon - timedelta(seconds=MIN_RUN_ROOM_SEC):
        return None, (f"not_before {when.strftime('%Y-%m-%d %H:%MZ')} is too close to the concern's horizon "
                      f"({horizon.strftime('%Y-%m-%d %H:%MZ')}) to leave a run room")
    return when, None


def make_act(concern_id: str, *, source: str, horizon_sec: float) -> ActV1:
    return build_act(actor=ACTOR, verb="make", key=f"{OBJECT_PREFIX}{concern_id}", signal_id="artifact_produced",
                     horizon_sec=float(horizon_sec), basis=MAKE_BASIS, object_ref=f"{OBJECT_PREFIX}{concern_id}",
                     test_before=f"intention_source:{source}")


def act_horizon_sec(horizon_at: datetime, now: datetime) -> float:
    return max(ACT_GRACE_SEC, (horizon_at - now).total_seconds() + ACT_GRACE_SEC)


def take_on(st: PrimeIntentionsStateV1, concern: PrimeConcernV1, *, why: str, thought_id: str, act: ActV1,
            horizon_sec: float, not_before: Optional[datetime], now: datetime) -> PrimeIntentionsStateV1:
    from orion.schemas.intention import intention_correlation_id, intention_run_id

    adapter = adapter_for(concern.event_ref)
    op = PrimeIntentionOpenV1(
        concern_id=concern.concern_id, event_ref=concern.event_ref or "", source=adapter.SOURCE, why=why[:600],
        thought_id=thought_id, act_id=act.act_id, act_horizon_sec=horizon_sec, taken_at=now, not_before=not_before,
        horizon_at=horizon_for(concern), run_id=intention_run_id(concern.concern_id),
        correlation_id=intention_correlation_id(concern.concern_id))
    return st.model_copy(update={"open": [*st.open, op]})


def update_open(st: PrimeIntentionsStateV1, op: PrimeIntentionOpenV1) -> PrimeIntentionsStateV1:
    return st.model_copy(update={"open": [op if o.concern_id == op.concern_id else o for o in st.open]})


def finish(st: PrimeIntentionsStateV1, op: PrimeIntentionOpenV1, *, outcome: IntentionOutcome, note: str,
           now: datetime, pinged: Optional[bool] = None) -> PrimeIntentionsStateV1:
    done = PrimeIntentionDoneV1(concern_id=op.concern_id, source=op.source, outcome=outcome, note=note[:NOTE_CAP],
                                taken_at=op.taken_at, closed_at=now, act_id=op.act_id,
                                run_id=op.run_id if op.order_status in ("submitted", "exists") else None, pinged=pinged)
    return st.model_copy(update={"open": [o for o in st.open if o.concern_id != op.concern_id],
                                 "done": [*st.done, done][-MAX_DONE:]})


def order_due(op: PrimeIntentionOpenV1, now: datetime) -> bool:
    """The order should be sent (again) now: not answered yet (or unavailable) and not held by not_before."""
    if op.order_status in ("submitted", "exists", "refused"):
        return False
    return op.not_before is None or now >= op.not_before


def past_horizon(op: PrimeIntentionOpenV1, now: datetime) -> bool:
    return now >= op.horizon_at + timedelta(seconds=ACT_GRACE_SEC)


def run_terminal(run: Optional[dict[str, Any]]) -> tuple[Optional[str], str]:
    """(terminal status or None while running, a short note) from ``GET /runs/<run_id>``."""
    if not isinstance(run, dict):
        return None, "unread"
    status = str(run.get("status") or "")
    if status not in TERMINAL:
        return None, status or "unknown"
    if status == "completed":
        return status, "the carry run completed"
    err = str(run.get("error") or "no error recorded")
    return status, f"the carry run ended {status}: {err}"[:NOTE_CAP]


# --- concern upkeep ---------------------------------------------------------------------------------------------


def keep_taken_on(state: PrimeStateV1, st: PrimeIntentionsStateV1, now: datetime) -> PrimeStateV1:
    """A concern Prime took on does not fade while its run is open: the whether is settled, only the doing
    remains. Bounded: the intention itself closes at its horizon."""
    taken = {o.concern_id for o in st.open}
    if not taken:
        return state
    return state.model_copy(update={"concerns": [
        c.model_copy(update={"last_evidence_at": now}) if c.concern_id in taken else c for c in state.concerns]})


def age_out(state: PrimeStateV1, st: PrimeIntentionsStateV1, now: datetime) -> PrimeStateV1:
    """"Not yet" has an end: a concern with an adapter whose horizon passed, not taken on, leaves Prime's list."""
    taken = {o.concern_id for o in st.open}
    kept: list[PrimeConcernV1] = []
    for c in state.concerns:
        horizon = horizon_for(c)
        if horizon is None or c.concern_id in taken or now < horizon:
            kept.append(c)
            continue
        hours = (horizon - c.opened_at).total_seconds() / 3600.0
        state = close_concern(state, c, end="lapsed", now=now,
                              note=f"aged out: its {hours:.0f} h horizon passed and I never took it on")
        state = note_change(state, f"a concern aged out at its horizon, never taken on: {c.text[:120]!r}")
    return state.model_copy(update={"concerns": kept})


# --- the frame --------------------------------------------------------------------------------------------------


def _age(now: datetime, then: datetime) -> str:
    secs = max(0, int((now - then).total_seconds()))
    return f"{secs // 60} min ago" if secs < 7200 else f"{secs // 3600} h ago"


def frame_lines(st: PrimeIntentionsStateV1, now: datetime) -> list[str]:
    out = []
    for o in st.open:
        if o.order_status is None and o.not_before is not None and now < o.not_before:
            doing = f"waiting until {o.not_before.strftime('%Y-%m-%d %H:%MZ')} as I chose"
        elif o.order_status in ("submitted", "exists"):
            doing = f"run {o.run_id} {o.run_status or 'ordered'}"
            if o.deliver_status:
                doing += f", reply {o.deliver_status}"
        else:
            doing = f"order not taken yet ({o.order_detail or 'unanswered'})"
        out.append(f"- concern {o.concern_id} ({o.source}), taken on {_age(now, o.taken_at)}: {doing}; horizon "
                   f"{o.horizon_at.strftime('%Y-%m-%d %H:%MZ')}")
    for d in st.done[-3:]:
        out.append(f"- concern {d.concern_id} ({d.source}) ended {d.outcome} {_age(now, d.closed_at)}: {d.note}")
    return out or ["(nothing taken on yet)"]


__all__ = [
    "ACT_GRACE_SEC", "ADAPTERS", "CHECK_INTERVAL_SEC", "INTENTIONS_KEY", "INTENTION_COMPONENT", "MAKE_BASIS",
    "MIN_RUN_ROOM_SEC", "OBJECT_PREFIX", "PrimeIntentionDoneV1", "PrimeIntentionOpenV1", "PrimeIntentionsStateV1",
    "act_horizon_sec", "adapter_for", "age_out", "carry_prompt", "find_open", "finish", "frame_lines", "has_adapter",
    "horizon_for", "keep_taken_on", "make_act", "order_due", "parse_not_before", "past_horizon", "run_terminal",
    "take_on", "take_on_refusal", "update_open",
]
