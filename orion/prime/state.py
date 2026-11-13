"""Prime's persisted state and the tick (design section 8.1): CPU only, never calls a model.

The tick reads a ``PrimeSnapshotV1`` (what the body, attention and curiosity look like now),
compares it with what Prime last saw, and records each difference as a pending change. A thought
is worth bidding for only while there is at least one pending change: "nothing changed" is the
normal tick and costs nothing.

What counts as a change (each one is a fact Prime can be shown, never a score):

* one of Prime's own acts settled (written by the runtime when a nomination closes);
* a concern faded: Prime has not kept it for ``concern_ttl(concern)`` -- ``CONCERN_TTL`` (24 h), or
  ``PROMISE_CONCERN_TTL`` (72 h) for a promise concern (a thought keeps a concern by naming it
  under ``concerns.keep``; nothing outside a thought renews it -- 2026-10-30, Juniper: an arc the model
  tied a concern to is not evidence for it; the first live concern, about energy, was tied to a
  conversation arc and was being renewed by that conversation);
* first wake: Prime has never thought;
* quiet: nothing else woke Prime for ``QUIET_FLOOR`` (a quarter of ``CONCERN_TTL``), so a held concern
  gets several chances to be kept or dropped before it fades for lack of any thought at all.

Which body constraints bind is NOT a change either (2026-11-04). Live 11-03 09:43..21:29, after
attention was removed, Prime thought 26 times, almost every one at the 5-minute floor: 21 did nothing,
4 dropped a concern "because lane X is no longer the binding constraint", 1 asked a question. Over 3
days the binding set changed 25 times a day (median spell 12 min, 29 of 75 under 5 min): the market
moving its prices, not news for a mind. It stays visible in every frame's BODY section, read live from the body.

Attention moving is NOT a change (2026-11-03). Live 10-30..11-03 Prime's concerns churned with it:
22 of the 38 concern drops gave "attention has moved ..." as the reason, and 1 of 37 graded concern
acts settled. Attention is still in every frame, as background
(orion/prime/thought.py ``_attention_lines``); it just does not wake Prime on its own.

Tries that never reached the model (2026-11-03; the code is orion/mindloop/backoff.py since 2026-11-04): a try that got no pool seat, no gateway reply, or an
upstream 5xx (after one immediate retry) is not a thought. ``note_no_model`` records it without
starting the refractory interval, and the next tick re-bids (that is the retry). ``NO_MODEL_FREE_TRIES``
in a row are free; after that each one backs off (refractory x 2^n, capped at ``NO_MODEL_BACKOFF_CAP``),
and a gateway shed answer's own ``retry_after_sec`` is always honoured. A no-seat try also sends the
next one on the ``starved`` pool priority (``wants_starved_route``).

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
from typing import Any, Literal, Optional
from uuid import uuid4

from pydantic import (AwareDatetime, BaseModel, ConfigDict, Field, GetJsonSchemaHandler, SerializerFunctionWrapHandler,
                      model_serializer)

# Tries that never reached the model: the loop library's, re-exported here so Prime's callers and tests keep
# their imports (moved to orion/mindloop/backoff.py on 2026-11-04, unchanged).
from orion.mindloop.backoff import (  # noqa: F401 -- re-exports
    GATEWAY_NO_SEAT_ERRORS,
    GATEWAY_UPSTREAM_5XX,
    NO_MODEL_BACKOFF_CAP,
    NO_MODEL_FREE_TRIES,
    TryKind,
    classify_try,
    no_model_backoff,
    note_model_reached,
    note_no_model,
    wants_starved_route,
)

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
# The quiet floor. Live 10-30..11-03, with attention moves removed, the longest stretch with no other
# trigger was 12.3 h (p90 1.6 h): without a floor a concern could fade with no thought in between.
QUIET_FLOOR = CONCERN_TTL / 4
# A promise concern: one the turn hand-back opened because Orion promised something in a turn (orion/prime/handback.py,
# event_ref ``turn:<correlation_id>``). It fades after 72 h, the horizon the intention ledger gives a promise
# (docs/superpowers/specs/2026-11-11-intention-ledger-design.md; orion/prime/intention_sources/turn_promise.py
# HORIZON_SEC, a test pins the two together). Juniper approved 2026-11-13. Live 11-12 a promise to "take a lazy
# browse through the last 24 hours" faded after one day with no move. The kind is read from event_ref, so a
# concern saved before this change gets the right TTL with no new field. QUIET_FLOOR stays CONCERN_TTL / 4.
PROMISE_EVENT_PREFIX = "turn:"
PROMISE_CONCERN_TTL = timedelta(hours=72)
PROMISE_TEXT_CAP = 600

Arm = Literal["treated", "control"]


class PrimeConcernV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    concern_id: str
    text: str = Field(min_length=1, max_length=CONCERN_TEXT_CAP)
    # The Temporal Self arc the model said this concern is about. A reference only, never evidence.
    arc_id: Optional[str] = None
    # Event mode (PRIME_CONCERNS_FROM_EVENT_WANTS, attention as the want competition patch 2): the funded
    # event want (its event_ref) this concern came from. Left out of the saved state when None.
    event_ref: Optional[str] = Field(default=None, max_length=500)
    # Personality rev 4 (orion/prime/choice.py): the event want's family and whether it was on the list only
    # through care, as the list stood when this concern opened. The choice point keys the option by this family
    # and marks a pick care_promoted from this flag, so a concern that outlives its want's place on the list
    # keeps both. Left out of the saved state when None, so a state with no such concern saves as before.
    # A concern that HAS them cannot be read by an orion-mind or Hub built before them (extra="forbid"):
    # rebuild Hub before orion-mind, and drop these keys from orion:prime:state before rolling mind back.
    event_family: Optional[str] = Field(default=None, max_length=200)
    event_care_promoted: Optional[bool] = None
    # A promise concern (``is_promise_concern``): the promise sentence(s) the 5b judge quoted from Orion's reply,
    # labelled with who it was made to and when (orion/prime/handback.py ``promise_text``). Shown only in Prime's
    # own thinking frame (orion/prime/thought.py); never in ``text``, so it never reaches the Prime block in every
    # turn's motor prefix or the published concerns view (choice-as-EFE acceptance S-1.8). Left out of the saved
    # state when None; same rollback rule as event_family (rebuild Hub before orion-mind).
    promise_text: Optional[str] = Field(default=None, max_length=PROMISE_TEXT_CAP)
    opened_at: AwareDatetime
    # When a thought last kept it (or opened it). Name kept for the persisted state's schema.
    last_evidence_at: AwareDatetime
    # Its act (orion/prime/acts.py concern_act): when the runtime published the open (None = not yet;
    # a concern from before 2026-10-30 is adopted on the first tick), and how it closed while Prime
    # still holds it ("settled" or "lapsed"; None = still open). Defaults keep old persisted state loading.
    act_opened_at: Optional[AwareDatetime] = None
    graded: Optional[Literal["settled", "lapsed"]] = None

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: Any, handler: GetJsonSchemaHandler) -> Any:
        if handler.mode == "serialization":
            return handler({k: v for k, v in core_schema.items() if k != "serialization"})
        return handler(core_schema)

    @model_serializer(mode="wrap")
    def _omit_unset_event(self, handler: SerializerFunctionWrapHandler) -> Any:
        # A concern with no event saves byte-for-byte as before (a rollback still loads the state).
        out = handler(self)
        for key in ("event_ref", "event_family", "event_care_promoted", "promise_text"):
            if out.get(key, 0) is None:
                del out[key]
        return out


def is_promise_concern(concern: PrimeConcernV1) -> bool:
    return (concern.event_ref or "").startswith(PROMISE_EVENT_PREFIX)


def concern_ttl(concern: PrimeConcernV1) -> timedelta:
    """How long ``concern`` lives unkept: 72 h for a promise concern, 24 h for every other."""
    return PROMISE_CONCERN_TTL if is_promise_concern(concern) else CONCERN_TTL


def _ttl_words(ttl: timedelta) -> str:
    days = ttl / timedelta(days=1)
    return "a day" if days == 1 else f"{days:g} days"


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


class PrimeRegulationOpenV1(BaseModel):
    """One regulate signal Prime sent (or held back), until its act closes (Orion Prime step 3).

    The signal itself lives in Redis (orion/physiology/prime_charge.py) where the market reads it; this is
    Prime's own record of it: the act, the arm, and whether the write reached Redis."""

    model_config = ConfigDict(extra="forbid")

    act_id: str
    signal_id: str
    thought_id: str
    family: str
    direction: Literal["slow", "quicken"]
    why: str
    arm: Arm
    magnitude: float
    opened_at: AwareDatetime
    expires_at: AwareDatetime
    # None until the write was tried; False with write_error when it failed (the act then closes 0 at once).
    written: Optional[bool] = None
    write_error: Optional[str] = None


class PrimeStateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["prime.state.v1"] = "prime.state.v1"
    concerns: list[PrimeConcernV1] = Field(default_factory=list)
    open_nominations: list[PrimeNominationV1] = Field(default_factory=list)
    pending_changes: list[str] = Field(default_factory=list)
    recent_outcomes: list[str] = Field(default_factory=list)
    concern_record: list[PrimeConcernRecordV1] = Field(default_factory=list)
    # Regulate signals not yet graded (step 3). Default keeps older persisted state loading.
    open_regulations: list[PrimeRegulationOpenV1] = Field(default_factory=list)
    # family -> no new signal to it before this time: one rest as long as the last signal, counted from its
    # end. Caps Prime's charge on one organ at half of any stretch of time (the loop guard, design section 11).
    regulation_rest_until: dict[str, AwareDatetime] = Field(default_factory=dict)
    last_thought_at: Optional[AwareDatetime] = None
    last_reading: str = ""
    # The thought that wrote ``last_reading`` (its ``prime_thought:<id>`` ref, act ``thought_act(id)``), set with
    # it. For the Prime block (living Prime Part 1) to cite a real id. None until the first answered thought.
    last_thought_id: Optional[str] = None
    thoughts_total: int = 0
    # Retired 2026-11-03 (attention moves no longer wake Prime); kept so older persisted state loads.
    seen_active_arc_id: Optional[str] = None
    # Cursor: what the last tick saw, so the next one can tell what moved.
    seen_binding: Optional[list[str]] = None
    # Tries that never reached the model (2026-11-03, ``note_no_model``). Defaults keep old state loading.
    no_model_streak: int = 0
    no_model_tries_total: int = 0
    last_try_kind: Optional[TryKind] = None
    retry_not_before: Optional[AwareDatetime] = None

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: Any, handler: GetJsonSchemaHandler) -> Any:
        if handler.mode == "serialization":
            return handler({k: v for k, v in core_schema.items() if k != "serialization"})
        return handler(core_schema)

    @model_serializer(mode="wrap")
    def _omit_unset_thought_id(self, handler: SerializerFunctionWrapHandler) -> Any:
        # Until a thought sets it, the state saves byte-for-byte as before (tests/test_prime_replay.py).
        out = handler(self)
        if out.get("last_thought_id", 0) is None:
            del out["last_thought_id"]
        return out


class PrimeSnapshotV1(BaseModel):
    """What the tick reads. ``None`` fields are unknown (the read failed), never "calm"."""

    model_config = ConfigDict(extra="forbid")

    # Body constraints whose bind_state is "binding", sorted. None when the body was not read.
    binding: Optional[list[str]] = None
    body_ok: bool = False


# Prime's first published call (curiosity replica design 2026-11-04, section 2): a read-only view of its
# open concerns, so another loop instance (the curiosity replica) can read what Prime is concerned about
# without touching Prime's private state. Written to this Redis key as JSON; the writer lands with the
# replica that reads it (patch 3): PrimeRuntime.save (services/orion-mind/app/prime_loop.py) writes it after every
# state save, and the replica (services/orion-mind/app/replica_loop.py) reads it.
PRIME_CONCERNS_KEY = "orion:prime:concerns"


class PrimeConcernViewV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    concern_id: str
    text: str = Field(min_length=1, max_length=CONCERN_TEXT_CAP)
    # Hours since Prime opened it.
    age_h: float = Field(ge=0.0)
    arc_id: Optional[str] = None


class PrimeConcernsViewV1(BaseModel):
    """What Prime publishes about its open concerns: ids, words, age and arc. No bets, no outcomes, no
    track record -- those stay Prime's."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["prime.concerns.view.v1"] = "prime.concerns.view.v1"
    as_of: AwareDatetime
    concerns: list[PrimeConcernViewV1] = Field(default_factory=list, max_length=MAX_CONCERNS)


def concerns_view(state: PrimeStateV1, *, now: datetime) -> PrimeConcernsViewV1:
    """The published view of ``state``'s open concerns (a concern already graded is closed, so left out)."""
    return PrimeConcernsViewV1(
        as_of=now,
        concerns=[
            PrimeConcernViewV1(
                concern_id=c.concern_id,
                text=c.text,
                age_h=max(0.0, (now - c.opened_at).total_seconds() / 3600.0),
                arc_id=c.arc_id,
            )
            for c in state.concerns
            if c.graded is None
        ][:MAX_CONCERNS],
    )


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
            f"{n['faded']} faded (not kept for a day, or 3 days for a promise), {n['lapsed']} lapsed "
            f"(held past the horizon with no "
            f"question about it settling)")


def observe(state: PrimeStateV1, snap: PrimeSnapshotV1, now: datetime) -> PrimeStateV1:
    """One tick: record what moved since the last one. Unknown reads change nothing."""
    if state.last_thought_at is None and not state.pending_changes:
        state = note_change(state, "first wake: I have not thought yet")

    if (state.last_thought_at is not None and not state.pending_changes
            and now - state.last_thought_at >= QUIET_FLOOR):
        hours = (now - state.last_thought_at).total_seconds() / 3600.0
        state = note_change(state, f"quiet: nothing has woken me for {hours:.0f} hours")

    if snap.body_ok and snap.binding is not None and snap.binding != state.seen_binding:
        # Kept only as a record of the last binding set Prime saw (persisted state field). The frame's BODY
        # section reads the live body, not this; it is never a wake (module docstring, 2026-11-04).
        state = state.model_copy(update={"seen_binding": snap.binding})

    kept: list[PrimeConcernV1] = []
    for concern in state.concerns:
        ttl = concern_ttl(concern)
        if now - concern.last_evidence_at > ttl:
            words = _ttl_words(ttl)
            state = note_change(state, f"concern faded, not kept for {words}: {concern.text!r}")
            state = close_concern(state, concern, end="faded", now=now, note=f"faded: not kept for {words}")
        else:
            kept.append(concern)
    return state.model_copy(update={"concerns": kept})


def wants_to_think(state: PrimeStateV1, now: datetime, *, refractory: timedelta) -> bool:
    """True when there is something to think about, the refractory interval has passed and no
    no-model backoff is running. Price, heat and lane contention are not decided here: the bid does."""
    if not state.pending_changes:
        return False
    if state.retry_not_before is not None and now < state.retry_not_before:
        return False
    return state.last_thought_at is None or now - state.last_thought_at >= refractory


def new_id() -> str:
    return uuid4().hex[:16]


__all__ = [
    "Arm",
    "CONCERN_TTL",
    "PROMISE_CONCERN_TTL",
    "PROMISE_EVENT_PREFIX",
    "PROMISE_TEXT_CAP",
    "concern_ttl",
    "is_promise_concern",
    "NO_MODEL_FREE_TRIES",
    "QUIET_FLOOR",
    "TryKind",
    "classify_try",
    "no_model_backoff",
    "note_model_reached",
    "note_no_model",
    "wants_starved_route",
    "MAX_CONCERNS",
    "ConcernEnd",
    "MAX_CONCERN_RECORD",
    "PRIME_CONCERNS_KEY",
    "PrimeConcernRecordV1",
    "PrimeConcernV1",
    "PrimeConcernViewV1",
    "PrimeConcernsViewV1",
    "PrimeNominationV1",
    "PrimeRegulationOpenV1",
    "PrimeSnapshotV1",
    "PrimeStateV1",
    "close_concern",
    "concerns_view",
    "new_id",
    "note_change",
    "note_outcome",
    "observe",
    "track_record",
    "trim_record",
    "wants_to_think",
]
