"""Prime's persisted state and the tick (design section 8.1): CPU only, never calls a model.

The tick reads a ``PrimeSnapshotV1`` (what the body, attention and curiosity look like now),
compares it with what Prime last saw, and records each difference as a pending change. A thought
is worth bidding for only while there is at least one pending change: "nothing changed" is the
normal tick and costs nothing.

What counts as a change (each one is a fact Prime can be shown, never a score):

* attention moved: Temporal Self's ``active_arc`` is a different arc than last tick;
* the set of BINDING body constraints changed (``prices[*].bind_state == "binding"`` from GET
  /physiology/state: the body's own label; slack and "out of my hands" (exogenous) are not news);
* one of Prime's own acts settled (written by the runtime when a nomination closes);
* a concern faded: Prime has not kept it for ``CONCERN_TTL`` (a thought keeps a concern by naming it
  under ``concerns.keep``; nothing outside a thought renews it -- 2026-10-30, Juniper: an arc the model
  tied a concern to is not evidence for it; the first live concern, about energy, was tied to a
  conversation arc and was being renewed by that conversation);
* first wake: Prime has never thought.

Concerns are graded acts (Prime step 1, 2026-10-30; orion/prime/acts.py ``concern_act``). Every concern
Prime holds has one act; when the act closes, one ``PrimeConcernRecordV1`` is appended here: settled,
dropped, faded or lapsed. The record is also the act's outbox: the runtime publishes the close of every
unpublished record, then marks it published, so a restart between the two re-publishes it and the
ledger keeps the first close (exactly one close per concern act).

The model's hunger (``WantV1.intensity``) is NOT computed here: the body-economy market fills it
from the ``prime`` family's own act history (orion/physiology/appetite.py). This module only
answers "is there anything to think about at all".
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal, Optional
from uuid import uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

MAX_CONCERNS = 3
MAX_PENDING_CHANGES = 20
MAX_RECENT_OUTCOMES = 10
# How many closed concerns Prime's frame summarises as its track record ("last 20").
MAX_CONCERN_RECORD = 20
# "Dropped within an hour" in the track record: the live problem (2026-10-30) was concerns replaced
# each time attention moved, most of them inside an hour.
QUICK_DROP = timedelta(hours=1)
MAX_UNPUBLISHED = 200
CONCERN_TEXT_CAP = 300
# A concern Prime has not kept for a day fades. One day, because Temporal Self's arcs are per day
# (``/temporal-self/frame`` day_id): an arc not seen for a whole day is no longer part of today.
CONCERN_TTL = timedelta(hours=24)

Arm = Literal["treated", "control"]


class PrimeConcernV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    concern_id: str
    text: str = Field(min_length=1, max_length=CONCERN_TEXT_CAP)
    # The Temporal Self arc the model said this concern is about. A reference only, never evidence.
    arc_id: Optional[str] = None
    opened_at: AwareDatetime
    # When a thought last kept it (or opened it). Name kept for the persisted state's schema.
    last_evidence_at: AwareDatetime
    # Its act (orion/prime/acts.py concern_act): when the runtime published the open (None = not yet;
    # a concern from before 2026-10-30 is adopted on the first tick), and how it closed while Prime
    # still holds it ("settled" or "lapsed"; None = still open). Defaults keep old persisted state loading.
    act_opened_at: Optional[AwareDatetime] = None
    graded: Optional[Literal["settled", "lapsed"]] = None


ConcernEnd = Literal["settled", "dropped", "faded", "lapsed"]


class PrimeConcernRecordV1(BaseModel):
    """One concern act's close: Prime's track record, and the outbox the runtime publishes from."""

    model_config = ConfigDict(extra="forbid")

    concern_id: str
    text: str = Field(max_length=CONCERN_TEXT_CAP)
    opened_at: AwareDatetime
    act_opened_at: Optional[AwareDatetime] = None
    closed_at: AwareDatetime
    end: ConcernEnd
    produced: bool
    note: str = ""
    published: bool = False


class PrimeNominationV1(BaseModel):
    """One question Prime asked curiosity to look at (or held back), until its act closes."""

    model_config = ConfigDict(extra="forbid")

    act_id: str
    thought_id: str
    focus_id: str
    question: str
    why: str
    arm: Arm
    opened_at: AwareDatetime
    concern_id: Optional[str] = None
    # Treated only: what Hub answered. None until it answered; False with queue_error on refusal.
    queued: Optional[bool] = None
    queue_error: Optional[str] = None
    # Consecutive settle reads that found this treated focus neither pending nor taken by a run.
    gone_reads: int = 0


class PrimeStateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["prime.state.v1"] = "prime.state.v1"
    concerns: list[PrimeConcernV1] = Field(default_factory=list)
    open_nominations: list[PrimeNominationV1] = Field(default_factory=list)
    pending_changes: list[str] = Field(default_factory=list)
    recent_outcomes: list[str] = Field(default_factory=list)
    concern_record: list[PrimeConcernRecordV1] = Field(default_factory=list)
    last_thought_at: Optional[AwareDatetime] = None
    last_reading: str = ""
    thoughts_total: int = 0
    # Cursors: what the last tick saw, so the next one can tell what moved.
    seen_active_arc_id: Optional[str] = None
    seen_binding: Optional[list[str]] = None


class PrimeSnapshotV1(BaseModel):
    """What the tick reads. ``None`` fields are unknown (the read failed), never "calm"."""

    model_config = ConfigDict(extra="forbid")

    active_arc_id: Optional[str] = None
    active_arc_label: Optional[str] = None
    # Body constraints whose bind_state is "binding", sorted. None when the body was not read.
    binding: Optional[list[str]] = None
    temporal_self_ok: bool = False
    body_ok: bool = False


def _push(items: list[str], text: str, cap: int) -> list[str]:
    out = [*items, text]
    return out[-cap:]


def note_change(state: PrimeStateV1, text: str) -> PrimeStateV1:
    return state.model_copy(update={"pending_changes": _push(state.pending_changes, text, MAX_PENDING_CHANGES)})


def note_outcome(state: PrimeStateV1, text: str) -> PrimeStateV1:
    """An own act settled: something to think about, and a line for the next frames."""
    state = note_change(state, text)
    return state.model_copy(update={"recent_outcomes": _push(state.recent_outcomes, text, MAX_RECENT_OUTCOMES)})


def close_concern(state: PrimeStateV1, concern: PrimeConcernV1, *, end: ConcernEnd, now: datetime,
                  note: str) -> PrimeStateV1:
    """Record the close of ``concern``'s act (once: a concern already graded has no open act).
    ``settled`` is the only produced=1 end. The record is unpublished until the runtime sends it."""
    if concern.graded is not None:
        return state
    rec = PrimeConcernRecordV1(
        concern_id=concern.concern_id, text=concern.text, opened_at=concern.opened_at,
        act_opened_at=concern.act_opened_at, closed_at=now, end=end, produced=end == "settled", note=note[:300])
    return state.model_copy(update={"concern_record": trim_record([*state.concern_record, rec])})


def trim_record(records: list[PrimeConcernRecordV1]) -> list[PrimeConcernRecordV1]:
    """The newest MAX_UNPUBLISHED unpublished records (the outbox) and the newest MAX_CONCERN_RECORD
    published ones, in order. The outbox only outgrows its cap if the bus is down for hundreds of concern
    ends; an act whose close is dropped then is closed 0 by the ledger at its horizon (``expired``)."""
    pub = [r for r in records if r.published][-MAX_CONCERN_RECORD:]
    unpub = [r for r in records if not r.published][-MAX_UNPUBLISHED:]
    keep = {id(r) for r in (*pub, *unpub)}
    return [r for r in records if id(r) in keep]


def track_record(state: PrimeStateV1) -> str:
    """One line for the frame: how Prime's last MAX_CONCERN_RECORD concerns ended."""
    recs = state.concern_record[-MAX_CONCERN_RECORD:]
    if not recs:
        return "(no concern has ended yet)"
    n = {k: sum(1 for r in recs if r.end == k) for k in ("settled", "dropped", "faded", "lapsed")}
    quick = sum(1 for r in recs if r.end == "dropped" and r.closed_at - r.opened_at <= QUICK_DROP)
    return (f"last {len(recs)} concerns: {n['settled']} settled (a question about it got a run that wrote "
            f"something), {n['dropped']} dropped ({quick} of them within an hour of opening), "
            f"{n['faded']} faded (not kept for a day), {n['lapsed']} lapsed (held past the horizon with no "
            f"question about it settling)")


def observe(state: PrimeStateV1, snap: PrimeSnapshotV1, now: datetime) -> PrimeStateV1:
    """One tick: record what moved since the last one. Unknown reads change nothing."""
    if state.last_thought_at is None and not state.pending_changes:
        state = note_change(state, "first wake: I have not thought yet")

    if snap.temporal_self_ok and snap.active_arc_id and snap.active_arc_id != state.seen_active_arc_id:
        label = snap.active_arc_label or snap.active_arc_id
        state = note_change(state, f"attention moved to {label!r} (arc {snap.active_arc_id})")
        state = state.model_copy(update={"seen_active_arc_id": snap.active_arc_id})

    if snap.body_ok and snap.binding is not None and snap.binding != state.seen_binding:
        now_binding = ", ".join(snap.binding) or "nothing"
        state = note_change(state, f"body: binding constraints now {now_binding}")
        state = state.model_copy(update={"seen_binding": snap.binding})

    kept: list[PrimeConcernV1] = []
    for concern in state.concerns:
        if now - concern.last_evidence_at > CONCERN_TTL:
            state = note_change(state, f"concern faded, not kept for a day: {concern.text!r}")
            state = close_concern(state, concern, end="faded", now=now, note="faded: not kept for a day")
        else:
            kept.append(concern)
    return state.model_copy(update={"concerns": kept})


def wants_to_think(state: PrimeStateV1, now: datetime, *, refractory: timedelta) -> bool:
    """True when there is something to think about and the refractory interval has passed.
    Price, heat and lane contention are not decided here: the bid decides those."""
    if not state.pending_changes:
        return False
    return state.last_thought_at is None or now - state.last_thought_at >= refractory


def new_id() -> str:
    return uuid4().hex[:16]


__all__ = [
    "Arm",
    "CONCERN_TTL",
    "MAX_CONCERNS",
    "ConcernEnd",
    "MAX_CONCERN_RECORD",
    "PrimeConcernRecordV1",
    "PrimeConcernV1",
    "PrimeNominationV1",
    "PrimeSnapshotV1",
    "PrimeStateV1",
    "close_concern",
    "new_id",
    "note_change",
    "note_outcome",
    "observe",
    "track_record",
    "trim_record",
    "wants_to_think",
]
