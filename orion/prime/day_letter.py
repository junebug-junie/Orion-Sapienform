"""The day letter is Prime's end-of-day turn (living Prime design #3224, Part 3). Pure parts; CPU only.

Design: docs/superpowers/specs/2026-11-12-living-prime-one-orion-design.md, "Part 3". Juniper's answers to the
spec's missing questions: a refused letter is acceptable (no letter that day; the next letter says it could not
write it); reminiscence is not part of this.

What Prime does, on its tick (services/orion-mind/app/prime_loop.py ``_day_letter_upkeep``; this module decides,
the runtime does the I/O):

1. **Day close.** Yesterday (America/Denver) is closed when the temporal chronicle drained it
   (``GET {durable}/temporal-self/day/<date>`` answers 200) and Hub has no ``orion_day_letter`` row for it
   (``GET {hub}/api/orion-day/letters/<date>``). Prime holds that as its own end-of-day concern
   (``PrimeDayLetterOpenV1``). It is tracked here, not in the 3-slot list of concerns a thought keeps or drops:
   the tick decides it, not a thought, and a full concern list must never crowd out the day's letter.
2. **The bid.** One ``WantV1`` of Prime's family (``prime``), verb ``make``, on the letter turn's lane, priced at
   the turn's enforced maximum (the same cost-ceiling rule as Prime's thought). Deferred: no act, re-bid after the
   market's ``retry_after_sec`` (at least ``REBID_MIN_SEC``) until the bid window ends. Never funded inside the
   window -> no letter that day (``not_written`` with the last reason). Prime's self-monitor gate on
   ``act.prime.make`` is asked first, the same way it is for Prime's other acts.
3. **The act and the turn.** Funded: Prime opens the ``make`` act (actor ``prime``) and orders the run from Hub
   (``OrionDayLetterOrderV1``); Hub builds the day's input and submits ONE durable unified turn
   (``orion_day.letter``). The run's correlation id is ``letter_correlation_id(letter_date)``.
4. **Settle.** The run's terminal state (``GET {durable}/runs/<run_id>``) closes the act: 1 = this run wrote the
   day's letter row, else 0. The letter's threads (the run's finish detail) become Prime concerns, at most
   ``LETTER_CONCERNS_MAX`` and never past ``MAX_CONCERNS``; each places the usual concern bet, so a later Prime
   ``ask`` about one (curiosity's only way to hear the letter) can settle it. Threads with no room are counted.
5. **A missed day** (never funded, the run failed, or Hub refused) is recorded; the next order lists it in
   ``missed_letters`` so the next letter says it could not write that day. Hub adds the day before on its own when
   it has no row and Prime did not list it (Prime was down).

State lives in its own Redis key (``DAY_LETTER_KEY``), apart from ``orion:prime:state``, so an older image still
loads Prime's state after a rollback.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta
from typing import Literal, Optional
from uuid import NAMESPACE_URL, uuid5
from zoneinfo import ZoneInfo

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from orion.grammar.acts import build_act
from orion.prime.acts import ACTOR
from orion.prime.state import MAX_CONCERNS, PrimeConcernV1, PrimeStateV1, note_change
from orion.schemas.grammar import ActCostV2, ActV1
from orion.schemas.orion_day import ORION_DAY_LLM_ROUTE, ORION_DAY_TIMEZONE, MissedLetterV1, orion_day_run_id

DAY_LETTER_KEY = "orion:prime:day_letter"
DAY_LETTER_COMPONENT = "prime.day_letter"
LETTER_EVENT_PREFIX = "orion_day_letter:"
# How often the tick looks at the day (HTTP reads), and the floor between two bids.
CHECK_INTERVAL_SEC = 300.0
REBID_MIN_SEC = 900.0
# The act's horizon past the run's own deadline: the run's terminal lands inside it.
ACT_HORIZON_GRACE_SEC = 3600.0
LETTER_CONCERNS_MAX = 2
MAX_DONE = 14
MAX_MISSED_TOLD = 7
REASON_CAP = 300
TERMINAL = frozenset({"completed", "failed", "cancelled"})

MAKE_BASIS = (
    "artifact_produced = the orion_day.letter durable run this act ordered ended completed and wrote the day's "
    "orion_day_letter row (persist_outcome written; read from GET /runs/<run_id> on orion-durable-runs); 0 when "
    "the run failed or was cancelled, a different run's row already existed, or Hub refused the order. horizon = "
    "the run's admission deadline (PRIME_DAY_LETTER_RUN_DEADLINE_HOURS after the funded bid) + 3600s"
)

DoneOutcome = Literal["written", "not_written", "already_written", "empty"]


class PrimeDayLetterOpenV1(BaseModel):
    """The day Prime is trying to write about, from the moment it saw the day closed with no letter."""

    model_config = ConfigDict(extra="forbid")

    letter_date: date
    seen_at: AwareDatetime
    bid_until: AwareDatetime
    bids: int = 0
    next_bid_at: Optional[AwareDatetime] = None
    last_refusal: Optional[str] = Field(default=None, max_length=REASON_CAP)
    # Set once a bid is funded and the act opened.
    want_id: Optional[str] = None
    act_id: Optional[str] = None
    act_opened_at: Optional[AwareDatetime] = None
    run_id: Optional[str] = None
    correlation_id: Optional[str] = None
    deadline_at: Optional[AwareDatetime] = None
    # Hub's last answer to the order (None = not answered yet: send it again).
    order_status: Optional[str] = None
    order_detail: Optional[str] = Field(default=None, max_length=500)
    missed_sent: list[date] = Field(default_factory=list)


class PrimeDayLetterDoneV1(BaseModel):
    """How one day's letter ended, as Prime holds it."""

    model_config = ConfigDict(extra="forbid")

    letter_date: date
    outcome: DoneOutcome
    reason: str = Field(default="", max_length=REASON_CAP)
    closed_at: AwareDatetime
    act_id: Optional[str] = None
    run_id: Optional[str] = None
    turn_correlation_id: Optional[str] = None
    threads: int = 0
    concern_ids: list[str] = Field(default_factory=list)
    threads_without_room: int = 0
    # A not_written day: a later order already told the next letter about it.
    told: bool = False


class PrimeDayLetterStateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["prime.day_letter.v1"] = "prime.day_letter.v1"
    open: Optional[PrimeDayLetterOpenV1] = None
    done: list[PrimeDayLetterDoneV1] = Field(default_factory=list)


# --- dates ------------------------------------------------------------------------------------------------------


def yesterday(now: datetime, tz_name: str = ORION_DAY_TIMEZONE) -> date:
    """The local day before ``now``'s local day: the day whose letter is due."""
    return now.astimezone(ZoneInfo(tz_name)).date() - timedelta(days=1)


def handled(st: PrimeDayLetterStateV1, day: date) -> bool:
    return (st.open is not None and st.open.letter_date == day) or any(d.letter_date == day for d in st.done)


# --- the concern, the bid and the act ---------------------------------------------------------------------------


def start(st: PrimeDayLetterStateV1, day: date, now: datetime, *, window_hours: float) -> PrimeDayLetterStateV1:
    """Yesterday is closed and has no letter: Prime's own end-of-day concern opens."""
    return st.model_copy(update={"open": PrimeDayLetterOpenV1(
        letter_date=day, seen_at=now, bid_until=now + timedelta(hours=float(window_hours)), next_bid_at=now)})


def want_id(day: date, bid_no: int) -> str:
    return f"prime-day-letter-{day.isoformat()}-{int(bid_no)}"


def letter_correlation_id(day: date) -> str:
    """The letter run's correlation id: derived from the day, so a re-sent order is the same request."""
    return str(uuid5(NAMESPACE_URL, f"orion:prime:day_letter:{day.isoformat()}"))


def letter_run_id(day: date) -> str:
    """One run per day (attempt 1). Prime orders at most one run for a day; its retries live inside the run."""
    return orion_day_run_id(day, 1)


def letter_cost(*, lane: str, turn_timeout_sec: float) -> ActCostV2:
    """The letter turn at its enforced maximum on its lane (the mindloop cost-ceiling rule)."""
    return ActCostV2(gpu_slot_s_by_lane={lane: float(turn_timeout_sec)},
                     basis={"gpu_slot_s_by_lane": f"ceiling: PRIME_DAY_LETTER_TURN_TIMEOUT_SEC "
                                                  f"{float(turn_timeout_sec):g} s per letter turn"})


def make_act(day: date, *, horizon_sec: float) -> ActV1:
    return build_act(actor=ACTOR, verb="make", key=f"orion_day:{day.isoformat()}", signal_id="artifact_produced",
                     horizon_sec=float(horizon_sec), basis=MAKE_BASIS, object_ref=f"{LETTER_EVENT_PREFIX}{day.isoformat()}")


def deferred(op: PrimeDayLetterOpenV1, now: datetime, reason: str,
             retry_after_sec: Optional[float]) -> PrimeDayLetterOpenV1:
    wait = max(REBID_MIN_SEC, float(retry_after_sec or 0.0))
    return op.model_copy(update={"last_refusal": reason[:REASON_CAP], "next_bid_at": now + timedelta(seconds=wait)})


def window_over(op: PrimeDayLetterOpenV1, now: datetime) -> bool:
    return op.act_id is None and now >= op.bid_until


# --- missed days ------------------------------------------------------------------------------------------------


def missed_for(st: PrimeDayLetterStateV1, day: date) -> list[MissedLetterV1]:
    """Days before ``day`` with no letter that no letter was told about yet (newest MAX_MISSED_TOLD). An empty day
    is listed too, with its own reason, so Hub never names it "missed for no known reason"."""
    out = [MissedLetterV1(letter_date=d.letter_date, reason=d.reason or "no letter was written")
           for d in sorted(st.done, key=lambda d: d.letter_date)
           if d.outcome in ("not_written", "empty") and not d.told and d.letter_date < day]
    return out[-MAX_MISSED_TOLD:]


def mark_told(st: PrimeDayLetterStateV1, days: list[date]) -> PrimeDayLetterStateV1:
    told = set(days)
    return st.model_copy(update={"done": [d.model_copy(update={"told": True}) if d.letter_date in told else d
                                          for d in st.done]})


def finish(st: PrimeDayLetterStateV1, done: PrimeDayLetterDoneV1) -> PrimeDayLetterStateV1:
    keep = [d for d in st.done if d.letter_date != done.letter_date]
    days = sorted([*keep, done], key=lambda d: d.letter_date)
    return st.model_copy(update={"open": None, "done": days[-MAX_DONE:]})


# --- settling the run -------------------------------------------------------------------------------------------


def run_outcome(run: dict) -> tuple[Optional[bool], str, dict]:
    """(produced, note, the run's orion_day detail) from ``GET /runs/<run_id>``; produced None = not terminal."""
    status = str(run.get("status") or "")
    detail = run.get("orion_day") if isinstance(run.get("orion_day"), dict) else {}
    if status not in TERMINAL:
        return None, status or "unknown", detail
    outcome = detail.get("persist_outcome")
    if status == "completed" and outcome == "written":
        return True, "the run wrote the day's letter", detail
    if status == "completed":
        return False, f"the run completed but did not write the row (persist_outcome {outcome})", detail
    err = str(run.get("error") or detail.get("error") or "no error recorded")
    return False, f"the run ended {status}: {err}"[:REASON_CAP], detail


def done_outcome(produced: bool, detail: dict) -> "DoneOutcome":
    """How the day ended when the run is terminal: a row another run wrote is a letter, not a missed day."""
    if produced:
        return "written"
    return "already_written" if detail.get("persist_outcome") == "already_written" else "not_written"


def thread_concern_id(day: date, index: int) -> str:
    return hashlib.sha256(f"{LETTER_EVENT_PREFIX}{day.isoformat()}#thread{index}".encode()).hexdigest()[:16]


def thread_concerns(state: PrimeStateV1, day: date, threads: list[str], now: datetime
                    ) -> tuple[PrimeStateV1, list[str], int]:
    """Open concerns from the letter's threads, in the letter's order, within the caps. Returns
    (state, opened concern ids, threads left without room). Deterministic ids: a replay opens nothing twice."""
    held = {c.concern_id for c in state.concerns}
    opened: list[str] = []
    no_room = 0
    for i, thread in enumerate(t.strip() for t in threads):
        if not thread:
            continue
        cid = thread_concern_id(day, i)
        if cid in held:
            continue
        if len(opened) >= LETTER_CONCERNS_MAX or len(state.concerns) >= MAX_CONCERNS:
            no_room += 1
            continue
        text = f"a thread from my letter for {day.isoformat()}: {thread}"[:300]
        concern = PrimeConcernV1(concern_id=cid, text=text, event_ref=f"{LETTER_EVENT_PREFIX}{day.isoformat()}#thread{i}",
                                 opened_at=now, last_evidence_at=now)
        state = state.model_copy(update={"concerns": [*state.concerns, concern]})
        opened.append(cid)
    if opened:
        state = note_change(state, f"my letter for {day.isoformat()} named {len([t for t in threads if t.strip()])} "
                                   f"thread(s); I now hold {len(opened)} of them as concerns")
    return state, opened, no_room


def status_lines(st: PrimeDayLetterStateV1, now: datetime) -> list[str]:
    """Short lines for logs and the frame: what Prime is doing about the day's letter."""
    out = []
    op = st.open
    if op is not None:
        if op.act_id is None:
            out.append(f"letter for {op.letter_date}: bidding ({op.bids} bid(s); last: {op.last_refusal or '-'})")
        else:
            out.append(f"letter for {op.letter_date}: writing (act {op.act_id}, run {op.run_id}, "
                       f"order {op.order_status or 'unanswered'})")
    for d in st.done[-3:]:
        out.append(f"letter for {d.letter_date}: {d.outcome}" + (f" ({d.reason})" if d.reason else ""))
    return out


__all__ = [
    "ACT_HORIZON_GRACE_SEC", "CHECK_INTERVAL_SEC", "DAY_LETTER_COMPONENT", "DAY_LETTER_KEY", "LETTER_CONCERNS_MAX",
    "MAKE_BASIS", "PrimeDayLetterDoneV1", "PrimeDayLetterOpenV1", "PrimeDayLetterStateV1", "REBID_MIN_SEC",
    "deferred", "finish", "handled", "letter_correlation_id", "letter_cost", "letter_run_id", "make_act",
    "mark_told", "missed_for", "run_outcome", "start", "status_lines", "thread_concern_id", "thread_concerns",
    "want_id", "window_over", "yesterday",
]

# The letter turn's lane (the durable run's admission route): the bid prices this lane.
DEFAULT_LANE = ORION_DAY_LLM_ROUTE
