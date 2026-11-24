"""Prime's 3 concern slots, decided by the concern contest (design 2026-11-17 patch 2, #3574; #3656). Pure, CPU only.

Design: docs/superpowers/specs/2026-11-17-prime-concerns-decided-by-attention-design.md. The rules live in
orion/attention/concern_contest.py; this module applies them to ``PrimeStateV1``.

* **Nothing is lost for lack of room.** A letter thread, a dream concern or a concern Prime named while all 3 slots
  are full goes to ``state.waiting`` (at most ``WAITING_MAX``), not to oblivion. Live 11-17..11-23, 4 of 6 letters
  got 0 threads held ("5 threads, 0 now held as concerns, 5 without room") and the threads were forgotten.
* **A waiting concern gets a free slot** on the next contest, best first, a letter at most ``LETTER_GROUP_CAP``.
* **The contest** (``PRIME_CONCERN_CONTEST_ENABLED``): a waiting concern takes the slot of the weakest held concern
  that has been quiet for half its life, has no act in flight, and that Prime has thought about at least once since
  it went quiet (Prime's last thought is after that moment), if it beats it (``concern_contest.displacements``).
* **A new promise always gets a slot** (#3656, Juniper 2026-11-24). Live 11-17 21:55Z and six times on 11-23 a promise
  was refused with "already holding 3 concerns" and lost. It now pushes out the weakest held concern that is not a
  promise and was not taken on (``concern_contest.intake_victim``); no quiet rule. A promise never pushes out a
  promise. If every slot holds a promise or something Prime took on, the promise waits, and the turn says so.
* **Every displacement is a close**: end ``displaced``, the note gives both scores and the reason, and Prime's next
  frame is told (``note_change``). A waiting concern that outlives its own life (24 h; 72 h for a promise) leaves
  the list with a line in the frame; a promise that expires that way is a change that wakes Prime.

A concern that is waiting has no act. Its act opens when it gets a slot (services/orion-mind/app/prime_loop.py
``_flush_concern_acts``), so a waiting concern never places a bet that it cannot win.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

from orion.attention.concern_contest import (
    DEFAULT_MARGIN,
    MAX_DISPLACEMENTS_PER_TICK,
    ContestEntry,
    displacements,
    intake_victim,
    score,
    score_words,
    standings,
    strength,
)
from orion.prime.state import (
    MAX_CONCERNS,
    MAX_RECENT_OUTCOMES,
    PrimeConcernV1,
    PrimeStateV1,
    close_concern,
    concern_ttl,
    is_promise_concern,
    note_change,
)

WAITING_MAX = 10
# orion/prime/day_letter.py LETTER_EVENT_PREFIX and LETTER_CONCERNS_MAX (a test pins them together): a letter holds
# at most 2 slots, so one letter never fills all 3.
LETTER_EVENT_PREFIX = "orion_day_letter:"
LETTER_GROUP_CAP = 2

PromiseBound = Callable[[PrimeStateV1, PrimeConcernV1], Optional[str]]


def _outcome_line(state: PrimeStateV1, text: str) -> PrimeStateV1:
    """A line for the next frame that does not wake Prime."""
    return state.model_copy(update={"recent_outcomes": [*state.recent_outcomes, text][-MAX_RECENT_OUTCOMES:]})


def letter_group(c: PrimeConcernV1) -> Optional[str]:
    ref = c.event_ref or ""
    return ref.split("#", 1)[0] if ref.startswith(LETTER_EVENT_PREFIX) else None


def _group_full(concerns: list[PrimeConcernV1], c: PrimeConcernV1) -> bool:
    group = letter_group(c)
    return group is not None and sum(1 for h in concerns if letter_group(h) == group) >= LETTER_GROUP_CAP


def known_ids(state: PrimeStateV1) -> set[str]:
    """Held, waiting, or already ended: such an id is never queued or opened again."""
    return ({c.concern_id for c in state.concerns} | {c.concern_id for c in state.waiting}
            | {r.concern_id for r in state.concern_record})


def entry(c: PrimeConcernV1, state: PrimeStateV1, *, taken_on: frozenset[str] = frozenset()) -> ContestEntry:
    never = ("I took it on" if c.concern_id in taken_on
             else "a promise" if is_promise_concern(c) and c.graded is None else None)
    asked = any(n.concern_id == c.concern_id for n in state.open_nominations)
    ttl = concern_ttl(c)
    went_quiet = c.last_evidence_at + ttl / 2
    unseen = state.last_thought_at is None or state.last_thought_at < went_quiet
    return ContestEntry(concern_id=c.concern_id, last_evidence_at=c.last_evidence_at, ttl=ttl,
                        salience=c.evidence_salience, never_displaced=never,
                        in_flight="a question I asked about it is open" if asked else None,
                        done=c.graded is not None, unseen_quiet=unseen)


def _seat(state: PrimeStateV1, c: PrimeConcernV1, now: datetime) -> PrimeStateV1:
    seated = c.model_copy(update={"opened_at": now})
    return state.model_copy(update={"concerns": [*state.concerns, seated],
                                    "waiting": [w for w in state.waiting if w.concern_id != c.concern_id]})


def queue(state: PrimeStateV1, c: PrimeConcernV1) -> PrimeStateV1:
    """Put ``c`` on the waiting list (newest kept). Over ``WAITING_MAX`` the oldest non-promise leaves first; a
    promise that has to leave is a change Prime is told about, never silent."""
    if c.concern_id in {w.concern_id for w in state.waiting}:
        return state
    waiting = [*state.waiting, c]
    while len(waiting) > WAITING_MAX:
        gone = next((w for w in waiting if not is_promise_concern(w)), waiting[0])
        waiting = [w for w in waiting if w is not gone]
        line = f"no room on my waiting list, a waiting concern left it: {gone.text[:120]!r}"
        state = note_change(state, line) if is_promise_concern(gone) else _outcome_line(state, line)
    return state.model_copy(update={"waiting": waiting})


def seat_or_wait(state: PrimeStateV1, c: PrimeConcernV1, now: datetime) -> tuple[PrimeStateV1, bool]:
    """A free slot (and room in its letter's share) seats ``c``; otherwise it waits. (state, seated). An id that is
    already held, waiting or ended changes nothing."""
    if c.concern_id in known_ids(state):
        return state, False
    if len(state.concerns) < MAX_CONCERNS and not _group_full(state.concerns, c):
        return _seat(state, c, now), True
    return queue(state, c), False


def displace(state: PrimeStateV1, held: PrimeConcernV1, by: PrimeConcernV1, *, now: datetime,
             held_score: Optional[float], by_score: Optional[float], reason: str) -> PrimeStateV1:
    """Close ``held`` (end ``displaced``, both scores and the reason in the note) and free its slot for ``by``."""
    note = (f"displaced: by {by.concern_id}; score {score_words(held_score)} held vs {score_words(by_score)} new; "
            f"{reason}")
    state = close_concern(state, held, end="displaced", now=now, note=note)
    state = state.model_copy(update={"concerns": [c for c in state.concerns if c.concern_id != held.concern_id]})
    return note_change(state, f"concern displaced ({note[len('displaced: '):][:160]}): {held.text[:100]!r} gave its "
                              f"slot to {by.text[:100]!r}")


def seat_promise(state: PrimeStateV1, promise: PrimeConcernV1, now: datetime, *, taken_on: frozenset[str] = frozenset(),
                 contest: bool = True) -> tuple[PrimeStateV1, bool, Optional[str]]:
    """A new promise gets a slot (#3656). (state, seated, why it waits). A free slot seats it. Full: with the contest
    on it pushes out the weakest held concern that is not a promise and was not taken on; with none such (or the
    contest off) it waits."""
    if promise.concern_id in known_ids(state) and promise.concern_id not in {w.concern_id for w in state.waiting}:
        return state, False, "already held or ended"
    if len(state.concerns) < MAX_CONCERNS:
        return _seat(state, promise, now), True, None
    if contest:
        victim = intake_victim([entry(c, state, taken_on=taken_on) for c in state.concerns], now)
        if victim is not None:
            held = next(c for c in state.concerns if c.concern_id == victim.concern_id)
            state = displace(state, held, promise, now=now, held_score=score(victim, now), by_score=None,
                             reason="a new promise always gets a slot; this was my weakest concern that is not a "
                                    "promise and not taken on")
            return _seat(state, promise, now), True, None
        why = f"waiting: all {MAX_CONCERNS} concerns I hold are promises or taken on"
    else:
        why = f"waiting: already holding {MAX_CONCERNS} concerns (the concern contest is off)"
    return queue(state, promise), False, why


@dataclass
class ContestReport:
    seated: list[str] = field(default_factory=list)
    # (held id, by id)
    displaced: list[tuple[str, str]] = field(default_factory=list)
    expired: list[str] = field(default_factory=list)


def run_contest(state: PrimeStateV1, now: datetime, *, taken_on: frozenset[str] = frozenset(), contest: bool = True,
                margin: float = DEFAULT_MARGIN, max_displacements: int = MAX_DISPLACEMENTS_PER_TICK,
                promise_bound: Optional[PromiseBound] = None) -> tuple[PrimeStateV1, ContestReport]:
    """One pass over the waiting list, on Prime's tick. Order: expire, waiting promises (oldest first: a free slot,
    else the intake rule), free slots for the rest (best first), then the contest. ``promise_bound(state, promise)``:
    why a waiting promise may not open yet (the hand-back's loop guard, orion/prime/handback.py), or None.
    ``contest`` off: no displacement; waiting concerns still take free slots."""
    rep = ContestReport()
    ended = {r.concern_id for r in state.concern_record} | {c.concern_id for c in state.concerns}
    keep: list[PrimeConcernV1] = []
    for w in state.waiting:
        if w.concern_id in ended:
            continue
        if now - w.last_evidence_at > concern_ttl(w):
            rep.expired.append(w.concern_id)
            line = f"a concern waited for a slot until its own life ran out and left my list: {w.text[:120]!r}"
            state = note_change(state, line) if is_promise_concern(w) else _outcome_line(state, line)
            continue
        keep.append(w)
    state = state.model_copy(update={"waiting": keep})

    for p in [w for w in state.waiting if is_promise_concern(w)]:
        if promise_bound is not None and promise_bound(state, p) is not None:
            continue
        before = {c.concern_id for c in state.concerns}
        state, seated, _why = seat_promise(state, p, now, taken_on=taken_on, contest=contest)
        if seated:
            rep.seated.append(p.concern_id)
            rep.displaced.extend((cid, p.concern_id) for cid in before - {c.concern_id for c in state.concerns})

    def ranked_rest() -> list[PrimeConcernV1]:
        rest = [w for w in state.waiting if not is_promise_concern(w)]
        order = {id(w): i for i, w in enumerate(rest)}
        return sorted(rest, key=lambda w: (strength(entry(w, state), now), -order[id(w)]), reverse=True)

    for w in ranked_rest():
        if len(state.concerns) >= MAX_CONCERNS:
            break
        if _group_full(state.concerns, w):
            continue
        state = _seat(state, w, now)
        rep.seated.append(w.concern_id)

    if not contest or len(state.concerns) < MAX_CONCERNS:
        return state, rep
    n = 0
    while n < max(0, int(max_displacements)):
        candidates = [w for w in ranked_rest() if not _group_full(state.concerns, w)]
        held = [entry(c, state, taken_on=taken_on) for c in state.concerns]
        found = displacements(held, [entry(w, state) for w in candidates], now, margin=margin, max_n=1)
        if not found:
            break
        d = found[0]
        h = next(c for c in state.concerns if c.concern_id == d.held_id)
        w = next(c for c in candidates if c.concern_id == d.by_id)
        state = displace(state, h, w, now=now, held_score=d.held_score, by_score=d.by_score, reason=d.reason)
        state = _seat(state, w, now)
        rep.seated.append(w.concern_id)
        rep.displaced.append((d.held_id, d.by_id))
        n += 1
    return state, rep


def standing_lines(state: PrimeStateV1, now: datetime, *, taken_on: frozenset[str] = frozenset()) -> dict[str, str]:
    """concern_id -> its standing in the contest, for Prime's frame."""
    out = {}
    for s in standings([entry(c, state, taken_on=taken_on) for c in state.concerns], now):
        out[s.concern_id] = f"contest: score {score_words(s.score)}, quiet {s.quiet_h:.1f} h, {s.why}"
    return out


def waiting_lines(state: PrimeStateV1, now: datetime) -> list[str]:
    """The waiting list for Prime's frame, best first."""
    if not state.waiting:
        return ["(nothing is waiting for a slot)"]
    lines = []
    for w in sorted(state.waiting, key=lambda w: (is_promise_concern(w), strength(entry(w, state), now)),
                    reverse=True):
        age = max(0.0, (now - w.last_evidence_at).total_seconds() / 3600.0)
        kind = "a promise, it opens as soon as a slot can be freed" if is_promise_concern(w) else "waiting"
        lines.append(f"- {w.concern_id}: {w.text[:200]} ({kind}; score {score_words(score(entry(w, state), now))}, "
                     f"evidence {age:.1f} h ago)")
    return lines


__all__ = [
    "ContestReport",
    "LETTER_EVENT_PREFIX",
    "LETTER_GROUP_CAP",
    "WAITING_MAX",
    "displace",
    "entry",
    "known_ids",
    "letter_group",
    "queue",
    "run_contest",
    "seat_or_wait",
    "seat_promise",
    "standing_lines",
    "waiting_lines",
]
