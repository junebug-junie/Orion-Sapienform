"""Prime's hunger: needs that carry content (integration design Part B, patch 5). Pure: no I/O, no model.

Spec: docs/superpowers/specs/2026-11-15-orion-integration-and-a-life-design.md, "Part B: Prime hunger, built as
designed". Juniper approved the bracketed defaults (threshold 0.5, budget 24 thoughts a day).

Hunger is a short list of **needs**. Each need has a **drive** (0..1, computed here by code) and a **content line**
(the specific thing to think about, with its ref). Prime's frame shows the content lines, never the numbers. A need
with drive 0 is not shown. Total hunger is the strongest need (max), not a sum: a sum would let six weak needs look
like one strong one.

The needs built here (each has a live producer; see the PR report's metric gate for the live checks):

* ``horizon`` -- a held concern (24 h) or a promise (72 h, or its intention adapter's own deadline) nears its end.
  drive = (elapsed / horizon - 0.5) / 0.5, clamped to 0..1: 0 until half the horizon has passed, 0.5 at three
  quarters, 1 at the end. Source: Prime's own state. This is the job the 6-hour quiet floor did, with content.
  Also a funded event want Prime does not hold yet (event mode, PRIME_CONCERNS_FROM_EVENT_WANTS: Prime's only concern
  source) nearing its ``expires_at``, counted from its ``decided_at``. Its drive stops at ``EVENT_DRIVE_CAP`` (0.5):
  the event is offered to one thought before it expires, never re-offered as it rises. Without this, an event with a
  short window (a reading's look 3 h, an office visit 2 h, a curiosity finding 12 h) could expire between thoughts
  once the floor is gone: the replay's longest stretch with no thought is 20-27 h. Only an event in the top tenth of
  its own source (``salience`` >= ``EVENT_WAKE_SALIENCE``) may wake Prime this way. An ordinary funded event is
  shown at Prime's next thought only if it is still open then and in the top ``PRIME_EVENT_WANTS_LIMIT`` by net:
  a 3 h read or a 15 min cooling act below the band is often never seen by Prime (the cooling act itself runs on
  its own clock either way). Deliberate: salience (how unusual for its source), not net, decides the wake. Live 11-16 18:06Z..11-17 03:00Z, with no gate, 11 of 12 thoughts were woken by
  funded reading wants (3-day mean salience 0.09: reads that made no link) and all 11 ended "no move".
* ``surprise`` -- an experience of another organ whose ``surprise_nats`` is strictly above its own source kind's
  30-day 90th percentile, that no organ has cited yet, and whose surprisal is at least ln 2 (the outcome had
  probability 1/2 or less: more unlikely than not). drive = 1 - exp(-nats) = 1 - P(observed): exactly 0.5 at ln 2.
  Source: orion-durable-runs GET /temporal-self/hunger (temporal_self_event, experience_use).
* ``connection`` -- Juniper has not written to Orion for longer than her own usual gap between conversations.
  Gaps shorter than ``CONVERSATION_PAUSE`` are pauses inside one conversation and are not gaps between them.
  drive = the share of her past gaps between conversations (30 days, at least ``MIN_HISTORY``) strictly shorter
  than the current one: 0.5 at her median gap. Source: the same endpoint (chat_history_log, source hub_orion).

* ``learning`` -- a forecast series' skill against its baseline moved this week (Oudeyer-Kaplan learning progress;
  choice-as-EFE design Step 1). Source: orion-prediction GET /learning-progress, read by the Prime loop. Only a
  ``better`` or ``worse`` reading is a need; ``flat`` (the rest point: under Student's two-sided 5% cut at the
  Welch df, or under 0.05 skill) and ``insufficient`` are not. drive = 1 - the producer's two-sided p-value. A move
  already passed p < 0.05, so in practice the drive is 0.95-1: a gate that wakes Prime once per move, not a graded
  intensity (it is not "P(the change is real)"). The ref names the series and the direction
  (``learning:<series_id>:<better|worse>``), so a flip to the opposite direction is a new line. One move is one line
  until the series reads flat again (guard 5's trough). Each series is damped on its own key (``learning:<series>``):
  refs are stable, so one shown series must not hold another's wake. An unread source is unknown, never flat.
  Halving recovers over the 7-day window.

Not built here, each because its source does not exist yet (no placeholder need ships; the PR report links the
tickets): ``coherence`` (the contradiction finder) and ``project`` (Part C projects).

When Prime thinks (``pick_wake`` + orion/prime/state.py ``wants_to_think``): a pending change, OR a need whose
damped drive is at least the threshold; AND the refractory interval passed; AND no back-off; AND, for a need wake
only, the daily budget is not spent. Anti-swamp guards, all here:

1. A thought that shows needs and changes nothing (no move sent, no concern added or dropped) halves each shown
   need's drive, except a need whose own ref the thought answered (``touched``: the concern it kept or dropped, the
   event it took up as a concern, the experience it cited). With the evidence clock off, keeping a concern is the
   answer to its horizon need, so it is never "nothing". With it on (PRIME_CONCERN_EVIDENCE_CLOCK_ENABLED, design
   2026-11-17 patch 1, #3574) a bare keep answers nothing: the runtime leaves kept ids out of ``answered_refs``, and
   a thought that only keeps halves the horizon need like any other untouched need. A sent move, a take-on or a
   drop still answers it. Live 11-16..11-17, 17 of 18 thoughts ended "kept 3, no move". The halving is kept per **damp key** (``NeedV1.damp_key``): the need name, except an
   unheld funded event want's horizon need, which is kept under ``horizon:event:<its family>``
   (``event_damp_key``). A key with any answered shown need is not halved. Why not per name: keeping a held concern
   answers ``horizon``, and an event want is also ``horizon``, so before #3552 a thought that kept its 3 concerns
   and ignored the reading want that woke it halved nothing; live, every new reading (a new ref, so guard 5 cannot
   hold it) woke Prime again, about 1 an hour. Why not per ref: every reading is a new ref, so per-ref halving would
   not hold the next one either. Per family: an ignored reading damps the next readings, not an office visit, and
   not a held concern's own horizon. At the default threshold (= ``EVENT_DRIVE_CAP``; ``PRIME_HUNGER_THRESHOLD`` can
   lower it) an event need never exceeds the threshold, so any halving holds that family until it fully recovers: at
   most one event wake per family per recovery window. Any thought that shows a family and changes nothing halves it
   again and restarts its window (the same rule as every other need). The
   halving recovers linearly over the fired content's own horizon. Kept in ``PrimeNeedsStateV1`` inside Prime's
   persisted state, so it survives a restart (older saved state, keyed by name only, still loads and means the same).
2. Concern TTLs stay (24 h, 72 h for a promise): orion/prime/state.py.
3. The refractory interval stays: orion/prime/state.py ``wants_to_think``.
4. Daily budget: at most ``DAILY_BUDGET`` thoughts in any 24 hours. Past that only a pending change wakes Prime.
5. The same content line (need name + ref) cannot wake Prime twice unless its raw drive rose since it last did, by at
   least ``MIN_RISE``, measured from the lowest the drive went since then (``track_troughs``: a concern kept falls
   to 0 and may rise again; a need pinned at 1.0 never re-arms). Without the margin a stepped drive refires on every step: the connection drive moves in
   1/40 steps (40 conversation gaps in 30 days, live), and the replay eval had it wake Prime 20 times in 10 days.

The trace (stable, read by the daily roundup and the 14-day check): a thought a need woke has an act close note
(``grammar_acts.outcome_note``) that starts ``woke: need:<name> <ref>`` (``woke_prefix``, ``WOKE_NEED_RE``; when a
change was pending too: ``woke: need:<name> <ref> (also: <changes>)``; a ref never holds a space, ``;``, ``|`` or a
parenthesis). A thought
a pending change woke starts ``woke: <the changes>`` as before. The same string is the ``prime_thought`` experience's
label in temporal_self_event.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta
from typing import Any, Iterable, Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from orion.prime.state import (NeedDampV1, NeedFiredV1, PrimeNeedsStateV1, PrimeStateV1, concern_ttl,
                               is_promise_concern)

NeedName = Literal["horizon", "surprise", "connection", "learning"]
NEED_NAMES: tuple[str, ...] = ("horizon", "surprise", "connection", "learning")

THRESHOLD = 0.5  # [default, accepted 2026-11-15]: a need wakes Prime at this damped drive
DAILY_BUDGET = 24  # [default, accepted 2026-11-15]: thoughts in any 24 hours before only changes wake Prime
BUDGET_WINDOW = timedelta(hours=24)
HORIZON_START = 0.5  # the horizon drive counts from half the horizon
PRIME_NEED_EVENT_PREFIX = "prime_need:"
EVENT_DAMP_PREFIX = "horizon:event:"  # guard 1's key for an unheld event want's horizon need, per family
EVENT_DRIVE_CAP = 0.5  # an unheld funded event want is offered once, at three quarters of its window
EVENT_WAKE_SALIENCE = 0.9  # only the top tenth of an event's own source may wake Prime (curiosity wants' band)
SURPRISE_FLOOR_NATS = math.log(2.0)  # the outcome had P <= 1/2
SURPRISE_RECOVERY = timedelta(hours=24)  # the now's use window: an experience counts as used within 24 h
CONVERSATION_PAUSE = timedelta(hours=2)
LEARNING_RECOVERY = timedelta(days=7)  # learning progress compares 7-day windows
LEARNING_LABELS = {"athena.cabinet_temp_c": "cabinet temperature", "office.temp_c": "office temperature"}
MIN_HISTORY = 20  # the salience rule's own minimum (orion.physiology.event_wants.MIN_SOURCE_HISTORY)
DAMPING = 0.5
MIN_RISE = 0.1  # guard 5: how much a content line's raw drive must rise before it may wake Prime again
MAX_SHOWN = 5
MAX_FIRED = 50
CONTENT_CAP = 300
REF_CAP = 120  # also bounds the trace prefix, so the close note keeps room for what the thought did
# The trace's stable shape (module docstring): "woke: need:<name> <ref>".
WOKE_NEED_RE = re.compile(r"^woke: need:(horizon|surprise|connection|learning) ([^\s;|()]+)")


class NeedV1(BaseModel):
    """One need now. ``drive``: the raw drive. ``damped``: after guard 1 (what is compared to the threshold)."""

    model_config = ConfigDict(extra="forbid")

    name: NeedName
    drive: float = Field(ge=0.0, le=1.0)
    damped: float = Field(ge=0.0, le=1.0)
    content: str = Field(min_length=1, max_length=CONTENT_CAP)
    ref: str = Field(min_length=1, max_length=REF_CAP)
    # How long a halving caused by this need takes to recover (its own horizon).
    recover_sec: float = Field(gt=0.0)
    # The experience this need's content cites, when it is not the ref itself (connection: the ref is the chat
    # message, stable for the whole silence; the experience appears once the chronicle ingests the chat turn).
    experience: Optional[str] = Field(default=None, max_length=REF_CAP)
    # Guard 1's key, when it is not the need name (an unheld event want: ``event_damp_key(family)``).
    damp_as: Optional[str] = Field(default=None, max_length=REF_CAP)

    @property
    def key(self) -> str:
        return f"{self.name}:{self.ref}"

    @property
    def damp_key(self) -> str:
        """Guard 1's key: the halving this need reads and writes (module docstring)."""
        return self.damp_as or self.name

    @property
    def experience_id(self) -> Optional[str]:
        if self.experience:
            return self.experience
        return self.ref[len("experience:"):] if self.ref.startswith("experience:") else None


def _clamp(x: float) -> float:
    return 0.0 if not math.isfinite(x) else min(1.0, max(0.0, x))


def _hours(td: timedelta) -> float:
    return td.total_seconds() / 3600.0


# --- the drives ---------------------------------------------------------------------------------------------


def horizon_drive(elapsed: timedelta, horizon: timedelta) -> float:
    """0 until half the horizon, then linear to 1 at the horizon."""
    if horizon.total_seconds() <= 0:
        return 0.0
    frac = elapsed / horizon
    return _clamp((frac - HORIZON_START) / (1.0 - HORIZON_START))


def _when(v: Any) -> Optional[datetime]:
    if isinstance(v, datetime):
        return v
    try:
        t = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return t if t.tzinfo is not None else None


def event_damp_key(family: Any) -> str:
    """Guard 1's key for an unheld event want of ``family`` (module docstring)."""
    fam = "".join(str(family or "").split())[:REF_CAP - len(EVENT_DAMP_PREFIX)] or "unknown"
    return f"{EVENT_DAMP_PREFIX}{fam}"


def event_horizon_needs(state: PrimeStateV1, now: datetime,
                        events: Optional[dict[str, Any]]) -> list[tuple[Any, ...]]:
    """Funded event wants (GET /physiology/event-wants, FundedEventWantsV1) Prime does not hold as a concern, nearing
    their expiry. Drive capped at EVENT_DRIVE_CAP (module docstring)."""
    held = {c.event_ref for c in state.concerns if c.event_ref}
    out = []
    for w in (events or {}).get("wants") or []:
        ref = str(w.get("event_ref") or "")
        start, end = _when(w.get("decided_at")), _when(w.get("expires_at"))
        if not ref or ref in held or start is None or end is None or end <= start:
            continue
        if w.get("family") == "prime" or ref.startswith(PRIME_NEED_EVENT_PREFIX):
            continue  # Prime's own thought bid is never an event for Prime (a self-feeding loop)
        salience = w.get("salience")
        if not isinstance(salience, (int, float)) or salience < EVENT_WAKE_SALIENCE:
            continue  # an ordinary event waits in the frame for the next thought; it does not wake Prime
        drive = min(EVENT_DRIVE_CAP, horizon_drive(now - start, end - start))
        if drive <= 0.0:
            continue
        label = " ".join(str(w.get("label") or w.get("family") or "").split())[:160]
        content = (f"{label} ({w.get('family') or '?'}) stops being worth effort in "
                   f"{max(0.0, _hours(end - now)):.0f} h unless you take it up [event_id {ref}]")
        # (name, drive, content, ref, recover_sec, experience, damp_as)
        out.append(("horizon", drive, content, ref[:REF_CAP], (end - start).total_seconds(), None,
                    event_damp_key(w.get("family"))))
    return out


def horizon_needs(state: PrimeStateV1, now: datetime, *, taken_on: Iterable[str] = (),
                  horizon_of: Optional[Any] = None,
                  evidence_clock: bool = False) -> list[tuple[NeedName, float, str, str, float]]:
    """(name, drive, content, ref, recover_sec) per open concern. A concern taken on (its run is open) or already
    settled has nothing left to decide. ``horizon_of(concern)``: the intention adapter's deadline, or None.
    ``evidence_clock``: the content says a keep will not save it (PRIME_CONCERN_EVIDENCE_CLOCK_ENABLED)."""
    skip = set(taken_on)
    out = []
    for c in state.concerns:
        if c.concern_id in skip or c.graded == "settled":
            continue
        ttl = concern_ttl(c)
        fade_at = c.last_evidence_at + ttl
        drive = horizon_drive(now - c.last_evidence_at, ttl)
        end, verb = fade_at, ("fades unless I act on it (keeping it does not renew it)" if evidence_clock
                              else "fades unless I keep it")
        deadline = horizon_of(c) if horizon_of is not None else None
        if deadline is not None and deadline > c.opened_at:
            d2 = horizon_drive(now - c.opened_at, deadline - c.opened_at)
            if d2 > drive:
                drive, end, verb = d2, deadline, "ends"
        if drive <= 0.0:
            continue
        left = max(0.0, _hours(end - now))
        what = "promise" if is_promise_concern(c) else "concern"
        content = f"{what} {c.text[:160]!r} {verb} in {left:.0f} h [prime_concern:{c.concern_id}]"
        out.append(("horizon", drive, content, f"prime_concern:{c.concern_id}", ttl.total_seconds()))
    return out


def surprise_drive(nats: float, p90: float, n: int) -> float:
    """1 - P(observed), only for a surprisal above its own source's 90th percentile (with enough history) and at
    least ln 2; else 0."""
    if n < MIN_HISTORY or not math.isfinite(nats) or nats <= p90 or nats < SURPRISE_FLOOR_NATS:
        return 0.0
    return _clamp(1.0 - math.exp(-nats))


def surprise_needs(inputs: Optional[dict[str, Any]]) -> list[tuple[NeedName, float, str, str, float]]:
    out = []
    for s in (inputs or {}).get("surprises") or []:
        drive = surprise_drive(float(s.get("surprise_nats") or 0.0), float(s.get("source_p90") or 0.0),
                               int(s.get("source_n") or 0))
        if drive <= 0.0:
            continue
        label = " ".join(str(s.get("label") or s.get("source_kind") or "").split())[:160]
        content = (f"{s.get('organ') or '?'}: {label} -- {float(s['surprise_nats']):.2f} nats, more than its source's "
                   f"usual 90th percentile of {float(s['source_p90']):.2f}; nobody has used it yet "
                   f"[experience:{s['event_id']}]")
        out.append(("surprise", drive, content, f"experience:{s['event_id']}", SURPRISE_RECOVERY.total_seconds()))
    return out


def conversation_gaps(times: Sequence[datetime]) -> list[timedelta]:
    """Gaps between Juniper's conversations: consecutive message gaps of at least ``CONVERSATION_PAUSE``."""
    ts = sorted(times)
    return [b - a for a, b in zip(ts, ts[1:]) if b - a >= CONVERSATION_PAUSE]


def connection_drive(times: Sequence[datetime], now: datetime) -> Optional[float]:
    """Share of her past gaps between conversations strictly shorter than the current gap. None: too little
    history to say what her usual gap is (unknown, never "calm"). 0 while the current gap is a pause."""
    gaps = conversation_gaps(times)
    if len(gaps) < MIN_HISTORY or not times:
        return None
    current = now - max(times)
    if current < CONVERSATION_PAUSE:
        return 0.0
    return sum(1 for g in gaps if g < current) / len(gaps)


def connection_needs(inputs: Optional[dict[str, Any]], now: datetime) -> list[tuple[NeedName, float, str, str, float]]:
    j = (inputs or {}).get("juniper") or {}
    times = [t if isinstance(t, datetime) else datetime.fromisoformat(str(t).replace("Z", "+00:00"))
             for t in j.get("message_times") or []]
    drive = connection_drive(times, now)
    if not drive:
        return []
    gaps = sorted(conversation_gaps(times))
    median = gaps[len(gaps) // 2]
    since = _hours(now - max(times))
    quote = " ".join(str(j.get("last_prompt") or "").split())[:200]
    # The ref (guard 5's key, the trace) is the message itself, so one silence is one line even after the chronicle
    # ingests the chat turn and it becomes citable.
    ref = f"chat_history_log:{j.get('last_chat_id') or 'unknown'}"
    exp = str(j["last_event_id"]) if j.get("last_event_id") else None
    tail = f"[experience:{exp}]" if exp else f"(chat {j.get('last_chat_id') or '?'}, not yet an experience you can cite)"
    content = (f"Juniper last wrote {since:.0f} h ago; her usual gap between conversations is {_hours(median):.0f} h. "
               f"Her last message: {quote!r} {tail}")
    return [("connection", drive, content[:CONTENT_CAP], ref[:REF_CAP], max(median.total_seconds(), 3600.0), exp)]


def learning_drive(direction: Any, p_value: Any) -> float:
    """1 - the two-sided p-value of the skill change. 0 unless better or worse; 0 when the p-value is unreadable."""
    if direction not in ("better", "worse"):
        return 0.0
    try:
        p = float(p_value)
    except (TypeError, ValueError):
        return 0.0
    return _clamp(1.0 - p) if math.isfinite(p) else 0.0


def learning_needs(learning: Optional[dict[str, Any]]) -> list[tuple[Any, ...]]:
    """``learning``: orion-prediction GET /learning-progress as read (None = unread)."""
    out = []
    for s in (learning or {}).get("series") or []:
        sid = str(s.get("series_id") or "")
        drive = learning_drive(s.get("direction"), s.get("p_value"))
        if drive <= 0.0 or not sid:
            continue
        recent, prior = s.get("recent") or {}, s.get("prior") or {}
        try:
            rs, ps = float(recent["mean_skill"]), float(prior["mean_skill"])
        except (KeyError, TypeError, ValueError):
            continue
        name = LEARNING_LABELS.get(sid, sid)
        word = "better" if s.get("direction") == "better" else "worse"
        ref = f"learning:{sid}:{word}"
        content = (f"Your {name} forecasts got {word} this week: skill against the naive baseline went from "
                   f"{ps:.2f} ({prior.get('start')} to {prior.get('end')}) to {rs:.2f} ({recent.get('start')} to "
                   f"{recent.get('end')}), with the same models on both sides [{ref}]")
        out.append(("learning", drive, content, ref, LEARNING_RECOVERY.total_seconds(), None, f"learning:{sid}"))
    return out


# --- guards ---------------------------------------------------------------------------------------------------


def damp_factor(needs: PrimeNeedsStateV1, key: str, now: datetime) -> float:
    """Guard 1: the multiplier now on the drive of needs with damp key ``key`` (``NeedV1.damp_key``: the need name,
    or ``event_damp_key(family)``; 1 = undamped), recovering linearly over its horizon."""
    d = needs.damp.get(key)
    if d is None:
        return 1.0
    left = 1.0 - (now - d.at).total_seconds() / d.recover_sec
    return 1.0 if left <= 0 else _clamp(1.0 - (1.0 - d.factor) * left)


def compute_needs(state: PrimeStateV1, now: datetime, inputs: Optional[dict[str, Any]], *,
                  taken_on: Iterable[str] = (), horizon_of: Optional[Any] = None,
                  events: Optional[dict[str, Any]] = None, evidence_clock: bool = False,
                  learning: Optional[dict[str, Any]] = None) -> list[NeedV1]:
    """Every need with drive above 0, strongest (damped) first. ``inputs``: the durable-runs hunger read
    (TemporalHungerInputsV1 as a dict; None = unread). ``events``: the funded event wants Prime read this tick.
    ``learning``: the orion-prediction GET /learning-progress read (None = unread)."""
    raw = [*horizon_needs(state, now, taken_on=taken_on, horizon_of=horizon_of, evidence_clock=evidence_clock),
           *event_horizon_needs(state, now, events),
           *surprise_needs(inputs), *connection_needs(inputs, now), *learning_needs(learning)]
    out = []
    for name, drive, content, ref, recover, *extra in raw:
        drive = _clamp(drive)
        if drive <= 0.0:
            continue
        experience = extra[0] if extra else None
        damp_as = extra[1] if len(extra) > 1 else None
        damped = _clamp(drive * damp_factor(state.needs, damp_as or name, now))
        out.append(NeedV1(name=name, drive=drive, damped=damped, content=content[:CONTENT_CAP], ref=ref[:REF_CAP],
                          recover_sec=recover, experience=experience, damp_as=damp_as))
    out.sort(key=lambda n: (-n.damped, -n.drive, n.key))
    return out


def thoughts_in_window(needs: PrimeNeedsStateV1, now: datetime) -> int:
    return sum(1 for t in needs.thoughts_at if now - t < BUDGET_WINDOW)


def budget_spent(needs: PrimeNeedsStateV1, now: datetime, budget: int = DAILY_BUDGET) -> bool:
    return thoughts_in_window(needs, now) >= budget


def may_fire(need: NeedV1, needs: PrimeNeedsStateV1) -> bool:
    """Guard 5: this content line has not woken Prime yet, or its raw drive rose by at least ``MIN_RISE`` since."""
    f = needs.fired.get(need.key)
    return f is None or need.drive >= f.drive + MIN_RISE - 1e-9


def track_troughs(state: PrimeStateV1, needs_now: Sequence[NeedV1], *, events_read: bool = True,
                  hunger_read: bool = True, learning_read: bool = True) -> PrimeStateV1:
    """Guard 5's memory: for every content line that fired, keep the lowest raw drive seen since. A line absent now
    counts as drive 0 only where its absence means it fell:

    * a held concern (``prime_concern:``): kept, dropped or faded -- always known;
    * an event want's line: only when the event wants were read this tick (``events_read``);
    * connection: only when the hunger read is current (``hunger_read``);
    * learning: only when the learning-progress read is current (``learning_read``): the series read flat again;
    * surprise: never. An experience's surprisal is fixed; it leaves the read when cited, aged out or cut by the
      read's cap, none of which is the same surprise becoming calm.
    An unread source is unknown, not calm, and must not re-arm its lines."""
    if not state.needs.fired:
        return state
    now_drive = {n.key: n.drive for n in needs_now}
    fired = {}
    changed = False
    for key, f in state.needs.fired.items():
        name, _, ref = key.partition(":")
        if key in now_drive:
            d = now_drive[key]
        elif name == "horizon" and ref.startswith("prime_concern:"):
            d = 0.0
        elif name == "horizon":
            d = 0.0 if events_read else f.drive
        elif name == "connection":
            d = 0.0 if hunger_read else f.drive
        elif name == "learning":
            d = 0.0 if learning_read else f.drive
        else:
            d = f.drive
        if d < f.drive:
            fired[key] = f.model_copy(update={"drive": d})
            changed = True
        else:
            fired[key] = f
    if not changed:
        return state
    return state.model_copy(update={"needs": state.needs.model_copy(update={"fired": fired})})


def pick_wake(needs_now: Sequence[NeedV1], state: PrimeStateV1, now: datetime, *, threshold: float = THRESHOLD,
              budget: int = DAILY_BUDGET) -> Optional[NeedV1]:
    """The need that wakes Prime now, or None: the strongest damped need at or over the threshold that guard 5
    allows; never while the daily budget is spent (guard 4)."""
    if budget_spent(state.needs, now, budget):
        return None
    for n in needs_now:
        if n.damped >= threshold and may_fire(n, state.needs):
            return n
    return None


def note_thought(state: PrimeStateV1, now: datetime, fired: Optional[NeedV1]) -> PrimeStateV1:
    """A thought started (the model was reached): count it for the budget; record the content line that woke it."""
    n = state.needs
    thoughts = [t for t in n.thoughts_at if now - t < BUDGET_WINDOW] + [now]
    fired_map = dict(n.fired)
    if fired is not None:
        fired_map[fired.key] = NeedFiredV1(drive=fired.drive, at=now)
        if len(fired_map) > MAX_FIRED:
            fired_map = dict(sorted(fired_map.items(), key=lambda kv: kv[1].at)[-MAX_FIRED:])
    return state.model_copy(update={"needs": n.model_copy(update={"thoughts_at": thoughts, "fired": fired_map})})


def after_answer(state: PrimeStateV1, shown: Sequence[NeedV1], now: datetime, *, acted: bool,
                 touched: Iterable[str] = ()) -> PrimeStateV1:
    """Guard 1: a thought that was shown needs and changed nothing halves each shown need (by damp key,
    ``NeedV1.damp_key``), except needs whose ref is in ``touched`` (the thought answered that one). A damp key with
    any answered need is not halved. Kept concerns answer ``horizon`` (evidence clock off; with it on the runtime
    passes no kept ids); they never shield an event want's key (#3552)."""
    if acted or not shown:
        return state
    hit = set(touched)
    answered = {n.damp_key for n in shown if n.ref in hit}
    damp = dict(state.needs.damp)
    seen: set[str] = set(answered)
    for n in shown:
        k = n.damp_key
        if k in seen:
            continue
        seen.add(k)
        damp[k] = NeedDampV1(factor=_clamp(damp_factor(state.needs, k, now) * DAMPING), at=now,
                             recover_sec=n.recover_sec)
    return state.model_copy(update={"needs": state.needs.model_copy(update={"damp": damp})})


def answered_refs(*, kept: Iterable[str] = (), dropped: Iterable[str] = (), added_event_refs: Iterable[str] = (),
                  cited: Iterable[str] = ()) -> set[str]:
    """The need refs a thought answered: ``prime_concern:<id>`` for each concern kept or dropped (``dropped`` items
    may carry a reason after the id), each event it took up as a concern, ``experience:<id>`` for each cite."""
    out = {f"prime_concern:{c}" for c in kept}
    out |= {f"prime_concern:{str(d).split(' ', 1)[0]}" for d in dropped}
    out |= {str(e) for e in added_event_refs if e}
    out |= {f"experience:{c}" for c in cited}
    return out


def prune(state: PrimeStateV1, now: datetime) -> PrimeStateV1:
    """Drop recovered halvings and budget entries outside the window (keeps the saved state small)."""
    n = state.needs
    damp = {k: d for k, d in n.damp.items() if (now - d.at).total_seconds() < d.recover_sec}
    thoughts = [t for t in n.thoughts_at if now - t < BUDGET_WINDOW]
    if damp == n.damp and thoughts == n.thoughts_at:
        return state
    return state.model_copy(update={"needs": n.model_copy(update={"damp": damp, "thoughts_at": thoughts})})


# --- what Prime sees, and the trace ------------------------------------------------------------------------------


def shown_needs(needs_now: Sequence[NeedV1], woke: Optional[NeedV1] = None) -> list[NeedV1]:
    """At most MAX_SHOWN, strongest first, and always the need that woke the thought (first)."""
    if woke is None:
        return list(needs_now[:MAX_SHOWN])
    rest = [n for n in needs_now if n.key != woke.key]
    return [woke, *rest[:MAX_SHOWN - 1]]


def frame_lines(needs_now: Sequence[NeedV1], woke: Optional[NeedV1], *, unread: Optional[str] = None,
                learning_unread: Optional[str] = None) -> list[str]:
    """Content lines only, strongest first; the one that woke this thought is marked. No numbers (spec)."""
    lines = [f"- [{n.name}] {n.content}" + (" (this is what woke you)" if woke is not None and n.key == woke.key else "")
             for n in shown_needs(needs_now, woke)]
    if unread:
        lines.append(f"(surprise and connection could not be read: {unread})")
    if learning_unread:
        lines.append(f"(whether your forecasts are improving could not be read: {learning_unread})")
    return lines or ["(nothing is pulling at you right now)"]


def woke_prefix(need: NeedV1) -> str:
    """The trace (module docstring): ``woke: need:<name> <ref>``."""
    return f"woke: need:{need.name} {need.ref}"


def parse_woke_need(note: Optional[str]) -> Optional[tuple[str, str]]:
    """(need name, ref) from a thought act's close note, or None when a pending change woke it."""
    m = WOKE_NEED_RE.match(note or "")
    return (m.group(1), m.group(2)) if m else None


__all__ = [
    "BUDGET_WINDOW", "CONVERSATION_PAUSE", "EVENT_DAMP_PREFIX", "EVENT_DRIVE_CAP", "EVENT_WAKE_SALIENCE",
    "PRIME_NEED_EVENT_PREFIX", "event_damp_key", "event_horizon_needs", "DAILY_BUDGET", "DAMPING", "MIN_HISTORY",
    "LEARNING_RECOVERY", "learning_drive", "learning_needs",
    "MIN_RISE", "NEED_NAMES", "NeedDampV1",
    "NeedFiredV1", "NeedName", "NeedV1", "PrimeNeedsStateV1", "SURPRISE_FLOOR_NATS", "THRESHOLD", "WOKE_NEED_RE",
    "after_answer", "answered_refs", "budget_spent", "compute_needs", "connection_drive", "conversation_gaps", "damp_factor",
    "frame_lines", "horizon_drive", "horizon_needs", "may_fire", "note_thought", "parse_woke_need", "pick_wake",
    "prune", "shown_needs", "surprise_drive", "thoughts_in_window", "track_troughs", "woke_prefix",
]
