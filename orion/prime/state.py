"""Prime's persisted state and the tick (design section 8.1): CPU only, never calls a model.

The tick reads a ``PrimeSnapshotV1`` (what the body, attention and curiosity look like now),
compares it with what Prime last saw, and records each difference as a pending change. A thought
is worth bidding for only while there is at least one pending change: "nothing changed" is the
normal tick and costs nothing.

What counts as a change (each one is a fact Prime can be shown, never a score):

* one of Prime's own acts settled (written by the runtime when a nomination closes);
* a concern faded: nothing renewed it for ``concern_ttl(concern)`` -- ``CONCERN_TTL`` (24 h), or
  ``PROMISE_CONCERN_TTL`` (72 h) for a promise concern. What renews it (``last_evidence_at``), with the evidence
  clock on (PRIME_CONCERN_EVIDENCE_CLOCK_ENABLED, 2026-11-17, design
  docs/superpowers/specs/2026-11-17-prime-concerns-decided-by-attention-design.md patch 1, #3574): only real
  evidence -- a question Prime sent about it (the move's concern_id) or Prime taking it on (``touch_concern``).
  A turn whose prompt cites ``prime_concern:<id>`` settles it (orion/prime/handback.py) but does not renew it:
  a day letter's prompt quotes Orion's own earlier text, refs included (#3644). A bare "keep" no longer renews it: from 10-30 to
  11-17 the model kept every held concern on every thought, so 0 of 54 concern acts ever faded and new letter
  threads found no room. With the clock off, a keep renews it as before. An arc the model tied a concern to is
  never evidence (2026-10-30, Juniper: the first live concern, about energy, was tied to a conversation arc and
  was being renewed by that conversation);
* first wake: Prime has never thought.

Besides a change, a **need** can wake Prime (integration design Part B, patch 5; orion/prime/hunger.py): a
concern nearing its horizon, an unexplained surprise, Juniper's silence past her usual gap. The 6-hour quiet floor
that used to add a "quiet" change is DELETED (2026-11-16): 36 of 48 thoughts 11-05..11-15 woke only from it and most
did nothing. The horizon need does its job (a held concern gets a thought before it fades) with content.

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
from orion.prime.sitting import PrimeSittingSeenV1
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
from orion.schemas.promise_horizon import PROMISE_HORIZON_SEC

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
# A concern nothing renewed for a day fades (orion/prime/state.py observe; what renews it: touch_concern). One day, because Temporal Self's arcs are per day
# (``/temporal-self/frame`` day_id): an arc not seen for a whole day is no longer part of today.
CONCERN_TTL = timedelta(hours=24)
# A promise concern: one the turn hand-back opened because Orion promised something in a turn (orion/prime/handback.py,
# event_ref ``turn:<correlation_id>``). It fades after 72 h, the horizon the intention ledger gives a promise
# (docs/superpowers/specs/2026-11-11-intention-ledger-design.md; orion/prime/intention_sources/turn_promise.py
# HORIZON_SEC, a test pins the two together). Juniper approved 2026-11-13. Live 11-12 a promise to "take a lazy
# browse through the last 24 hours" faded after one day with no move. The kind is read from event_ref, so a
# concern saved before this change gets the right TTL with no new field.
PROMISE_EVENT_PREFIX = "turn:"
PROMISE_CONCERN_TTL = timedelta(seconds=PROMISE_HORIZON_SEC)  # 72 h, the promise act's clock (#3595)
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
    # When evidence last touched it (or it opened): ``touch_concern``. With PRIME_CONCERN_EVIDENCE_CLOCK_ENABLED off,
    # also when a thought last kept it (the old meaning, before 2026-11-17).
    last_evidence_at: AwareDatetime
    # Its act (orion/prime/acts.py concern_act): when the runtime published the open (None = not yet;
    # a concern from before 2026-10-30 is adopted on the first tick), and how it closed while Prime
    # still holds it ("settled" or "lapsed"; None = still open). Defaults keep old persisted state loading.
    act_opened_at: Optional[AwareDatetime] = None
    graded: Optional[Literal["settled", "lapsed"]] = None
    # The concern contest (orion/attention/concern_contest.py, PRIME_CONCERN_CONTEST_ENABLED, 2026-11-24): the
    # freshest evidence's place in its own source's history (0..1). Set at open from a funded event want's own
    # ``salience`` (orion/physiology/event_wants.py source_salience). None = unscored, never 0. Left out of the saved
    # state when None; same rollback rule as event_family (rebuild Hub and durable-runs before orion-mind).
    evidence_salience: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    # A promise concern: the kind of turn that made the promise (orion/prime/handback.py ``TurnWho``, e.g.
    # "chat_juniper", "curiosity"), set by ``fold_turn_signals``. The take-on check reads it (#3785) instead of
    # parsing ``text``; a concern saved before this field has None and falls back to the text. A plain str, not the
    # Literal: a turn kind added later must not stop an older image loading the whole state. Left out of the saved
    # state when None; same rollback rule as event_family (rebuild Hub and durable-runs before orion-mind).
    turn_who: Optional[str] = Field(default=None, max_length=50)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: Any, handler: GetJsonSchemaHandler) -> Any:
        if handler.mode == "serialization":
            return handler({k: v for k, v in core_schema.items() if k != "serialization"})
        return handler(core_schema)

    @model_serializer(mode="wrap")
    def _omit_unset_event(self, handler: SerializerFunctionWrapHandler) -> Any:
        # A concern with no event saves byte-for-byte as before (a rollback still loads the state).
        out = handler(self)
        for key in ("event_ref", "event_family", "event_care_promoted", "promise_text", "evidence_salience",
                    "turn_who"):
            if out.get(key, 0) is None:
                del out[key]
        return out


def is_promise_concern(concern: PrimeConcernV1) -> bool:
    return (concern.event_ref or "").startswith(PROMISE_EVENT_PREFIX)


def concern_ttl(concern: PrimeConcernV1) -> timedelta:
    """How long ``concern`` lives unkept: 72 h for a promise concern, 24 h for every other."""
    return PROMISE_CONCERN_TTL if is_promise_concern(concern) else CONCERN_TTL


def touch_concern(state: "PrimeStateV1", concern_id: Optional[str], now: datetime) -> "PrimeStateV1":
    """Real evidence touched a held concern: its clock (``last_evidence_at``) restarts at ``now``. Callers: a
    question Prime sent about it (services/orion-mind/app/prime_loop.py ``_nominate``; Prime taking it on renews it
    through orion/prime/intention.py ``keep_taken_on`` on every tick while the run is open).
    Keyed on ``concern_id`` only, never on the concern's text: a letter thread's id is a hash of its stable
    ``orion_day_letter:<day>#thread<i>`` event_ref (orion/prime/day_letter.py ``thread_concern_id``), while its
    text is cut. An unknown id, or a clock already at or after ``now``, changes nothing."""
    if not concern_id:
        return state
    if not any(c.concern_id == concern_id and c.last_evidence_at < now for c in state.concerns):
        return state
    return state.model_copy(update={"concerns": [
        c.model_copy(update={"last_evidence_at": now}) if c.concern_id == concern_id else c for c in state.concerns]})


def _ttl_words(ttl: timedelta) -> str:
    days = ttl / timedelta(days=1)
    return "a day" if days == 1 else f"{days:g} days"


# ``displaced``: the concern contest (orion/prime/slots.py) pushed it out of its slot for a waiting candidate or a new
# promise; the note gives both scores and the reason. A record with it cannot be read by an image built before it
# (extra="forbid"): rebuild Hub and orion-durable-runs before orion-mind.
ConcernEnd = Literal["settled", "dropped", "faded", "lapsed", "displaced"]


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


# --- Prime hunger's guards (orion/prime/hunger.py), persisted in PrimeStateV1.needs ---------------------------


class NeedDampV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    factor: float = Field(ge=0.0, le=1.0)  # the multiplier right after the last halving
    at: AwareDatetime
    recover_sec: float = Field(gt=0.0)


class NeedFiredV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # The raw drive when this content line last woke Prime, lowered to the lowest seen since (hunger.track_troughs).
    drive: float = Field(ge=0.0, le=1.0)
    at: AwareDatetime


class PrimeNeedsStateV1(BaseModel):
    """Guards 1, 4 and 5, persisted inside ``PrimeStateV1.needs``."""

    model_config = ConfigDict(extra="forbid")

    damp: dict[str, NeedDampV1] = Field(default_factory=dict)  # damp key (need name, or horizon:event:<family>) -> last halving
    fired: dict[str, NeedFiredV1] = Field(default_factory=dict)  # need key -> when it last woke Prime
    thoughts_at: list[AwareDatetime] = Field(default_factory=list)  # thought starts in the budget window

    def is_empty(self) -> bool:
        return not (self.damp or self.fired or self.thoughts_at)


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
    # Hunger's guards (integration design Part B, orion/prime/hunger.py): halvings, the content lines that woke
    # Prime, thought starts in the budget window. Left out of the saved state while empty, so a state with no
    # need history saves byte-for-byte as before. A state that HAS it cannot be read by an orion-mind or Hub built
    # before it (extra="forbid"): rebuild Hub before orion-mind; drop the key before rolling mind back.
    needs: PrimeNeedsStateV1 = Field(default_factory=PrimeNeedsStateV1)
    # Concerns with no slot yet (orion/prime/slots.py, the concern contest): letter threads, dream concerns and concerns
    # Prime named while all 3 slots were full, and promises that found every slot held by a promise or by something
    # Prime took on. No act is open for a waiting concern; it opens when the concern gets a slot. At most
    # ``WAITING_MAX``. Left out of the saved state while empty (same rollback rule as ``needs``).
    waiting: list[PrimeConcernV1] = Field(default_factory=list)
    # Game sittings Prime has seen (game company patch 4, orion/prime/sitting.py): which of the two wakes fired and
    # how many thoughts named each. Left out of the saved state while empty (an older reader reads a state without it);
    # a state that HAS it is unreadable to a Hub, orion-durable-runs (dream_day_store) or orion-mind built before it
    # (extra="forbid"): rebuild Hub and durable-runs before orion-mind, the same rule as ``needs``.
    sittings: dict[str, PrimeSittingSeenV1] = Field(default_factory=dict)

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
        if self.needs.is_empty():
            out.pop("needs", None)
        if not self.waiting:
            out.pop("waiting", None)
        if not self.sittings:
            out.pop("sittings", None)
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
    n = {k: sum(1 for r in recs if r.end == k) for k in ("settled", "dropped", "faded", "lapsed", "displaced")}
    quick = sum(1 for r in recs if r.end == "dropped" and r.closed_at - r.opened_at <= QUICK_DROP)
    return (f"last {len(recs)} concerns: {n['settled']} settled (a question about it got a run that wrote "
            f"something), {n['dropped']} dropped ({quick} of them within an hour of opening), "
            f"{n['faded']} faded (nothing renewed it for a day, or 3 days for a promise), {n['lapsed']} lapsed "
            f"(held past the horizon with no "
            f"question about it settling)"
            + (f", {n['displaced']} displaced (a waiting concern or a new promise took its slot)"
               if n["displaced"] else ""))


def observe(state: PrimeStateV1, snap: PrimeSnapshotV1, now: datetime) -> PrimeStateV1:
    """One tick: record what moved since the last one. Unknown reads change nothing."""
    if state.last_thought_at is None and not state.pending_changes:
        state = note_change(state, "first wake: I have not thought yet")

    if snap.body_ok and snap.binding is not None and snap.binding != state.seen_binding:
        # Kept only as a record of the last binding set Prime saw (persisted state field). The frame's BODY
        # section reads the live body, not this; it is never a wake (module docstring, 2026-11-04).
        state = state.model_copy(update={"seen_binding": snap.binding})

    kept: list[PrimeConcernV1] = []
    for concern in state.concerns:
        ttl = concern_ttl(concern)
        if now - concern.last_evidence_at > ttl:
            words = _ttl_words(ttl)
            state = note_change(state, f"concern faded, nothing renewed it for {words}: {concern.text!r}")
            state = close_concern(state, concern, end="faded", now=now, note=f"faded: nothing renewed it for {words}")
        else:
            kept.append(concern)
    return state.model_copy(update={"concerns": kept})


def wants_to_think(state: PrimeStateV1, now: datetime, *, refractory: timedelta, need_wake: bool = False) -> bool:
    """True when there is something to think about (a pending change, or ``need_wake``: a need crossed its
    threshold, orion/prime/hunger.py ``pick_wake``, which already holds the daily budget), the refractory interval
    has passed and no no-model backoff is running. Price, heat and lane contention are not decided here: the bid
    does."""
    if not state.pending_changes and not need_wake:
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
    "NeedDampV1",
    "NeedFiredV1",
    "PrimeNeedsStateV1",
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
    "touch_concern",
    "track_record",
    "trim_record",
    "wants_to_think",
]
