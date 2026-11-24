"""Which concern keeps a slot: attention's rules applied to Prime's 3 concern slots. Pure, no I/O.

Design: docs/superpowers/specs/2026-11-17-prime-concerns-decided-by-attention-design.md, patch 2 (#3574, #3610).
Juniper 2026-11-17: Q1 yes (attention's rule, Prime's tick), Q3 yes (a promise is never displaced), Q4 no (the
contest never seats a funded event want on its own; Prime must name it). 2026-11-24 (#3656): a new promise always
gets a slot; it pushes out the weakest held concern that is not a promise.

The rules are world-first's (orion/attention/world_first.py), moved to an hours scale:

* **Score.** ``salience x event_fade(age, ttl, grace = ttl / 2)``. ``salience`` is the freshest evidence's place in
  its own source's history (0..1; for a funded event want, orion/physiology/event_wants.py ``source_salience``).
  None means **unscored**, never 0: silence is not calm. Unscored ranks below every scored entry; among unscored,
  newer evidence ranks higher.
* **Quiet first.** A held concern can be displaced only when nothing touched it for half its life (12 h; 36 h for a
  promise, which is never displaced anyway), no act about it is in flight, and its holder has thought at least once
  since it went quiet (``unseen_quiet``). Half its life is the moment Prime's horizon need starts to pull
  (orion/prime/hunger.py); the last rule makes "Prime always gets a chance to act first" true, not just likely: the
  replay of 11-17..11-24 without it displaced concerns at exactly 12.0 h, on the same tick they went quiet. This is
  the churn guard: 10-30..11-03 concerns followed attention around and most drops came within an hour.
* **Beat it.** A waiting candidate takes the weakest displaceable slot only if it beats it: both scored and ahead by
  ``margin``; or the held one unscored and the candidate scored; or both unscored and the candidate's evidence newer.
  Ties go to the incumbent.

A concern whose act already closed (``done``: settled, or lapsed) has nothing left to protect: it is the weakest
entry and can always yield, unless it is ``never_displaced``.

``intake_victim`` is the promise rule (#3656): a new promise does not wait for quiet. It pushes out the weakest held
entry that is not ``never_displaced`` (a promise, or a concern Prime took on), preferring one with no act in flight.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Sequence

from orion.attention.world_first import event_fade

# The score margin a scored candidate needs over a scored incumbent. A bracketed default (design: "a knob to grade on
# live data, not a finding"); PRIME_CONCERN_CONTEST_MARGIN overrides it.
DEFAULT_MARGIN = 0.1
# At most this many contest displacements per tick (the design's LETTER_CONCERNS_MAX).
MAX_DISPLACEMENTS_PER_TICK = 2


@dataclass(frozen=True)
class ContestEntry:
    """One concern, held or waiting, as the contest sees it."""

    concern_id: str
    last_evidence_at: datetime
    ttl: timedelta
    # The freshest evidence's own-source percentile (0..1). None = unscored.
    salience: Optional[float] = None
    # Why it can never be displaced (a promise, taken on), else None.
    never_displaced: Optional[str] = None
    # Why an act about it is in flight (a question about it is open), else None. Protects it in the contest; at
    # promise intake it only ranks it last.
    in_flight: Optional[str] = None
    # Its act already closed (settled or lapsed) while it is still held.
    done: bool = False
    # It went quiet past half its life, and its holder has not looked at it since (Prime has not thought since).
    # Protects it in the contest: the holder always gets one chance to act on a quiet concern before it can lose
    # its slot. Promise intake ignores it.
    unseen_quiet: bool = False


@dataclass(frozen=True)
class Displacement:
    held_id: str
    by_id: str
    held_score: Optional[float]
    by_score: Optional[float]
    reason: str


@dataclass(frozen=True)
class Standing:
    concern_id: str
    score: Optional[float]
    quiet_h: float
    displaceable: bool
    why: str


def score(e: ContestEntry, now: datetime) -> Optional[float]:
    """``salience x event_fade``; None when unscored."""
    if e.salience is None:
        return None
    ttl = e.ttl.total_seconds()
    age = max(0.0, (now - e.last_evidence_at).total_seconds())
    return max(0.0, min(1.0, float(e.salience))) * event_fade(age, ttl, ttl / 2.0)


def score_words(value: Optional[float]) -> str:
    return "unscored" if value is None else f"{value:.2f}"


def quiet_for(e: ContestEntry, now: datetime) -> timedelta:
    return max(timedelta(0), now - e.last_evidence_at)


def displaceable(e: ContestEntry, now: datetime) -> bool:
    """Can the contest push ``e`` out now: not protected, and quiet for half its life (or its act closed)."""
    if e.never_displaced:
        return False
    if e.done:
        return True
    if e.in_flight or e.unseen_quiet:
        return False
    return quiet_for(e, now) >= e.ttl / 2


def strength(e: ContestEntry, now: datetime) -> tuple:
    """Sort key, weakest first: done, then unscored (older evidence weaker), then scored (lower weaker)."""
    s = score(e, now)
    return (0 if e.done else 1, 0 if s is None else 1, s or 0.0, e.last_evidence_at.timestamp())


def beats(w: ContestEntry, h: ContestEntry, now: datetime, *, margin: float = DEFAULT_MARGIN) -> Optional[str]:
    """Why waiting ``w`` beats held ``h``, or None (a tie keeps the incumbent)."""
    if h.done:
        return "its act already closed"
    sw, sh = score(w, now), score(h, now)
    if sw is not None and sh is not None:
        return f"score {sw:.2f} >= {sh:.2f} + margin {margin:g}" if sw >= sh + margin else None
    if sh is None and sw is not None:
        return f"scored {sw:.2f} beats unscored"
    if sh is None and sw is None:
        return "both unscored, newer evidence" if w.last_evidence_at > h.last_evidence_at else None
    return None


def _hours(td: timedelta) -> str:
    return f"{td.total_seconds() / 3600.0:.1f} h"


def displacements(held: Sequence[ContestEntry], waiting: Sequence[ContestEntry], now: datetime, *,
                  margin: float = DEFAULT_MARGIN, max_n: int = MAX_DISPLACEMENTS_PER_TICK) -> list[Displacement]:
    """The best waiting candidates against the weakest displaceable held concerns, at most ``max_n``. ``waiting``
    is in its own priority order for equal strength (stable sort)."""
    pool = sorted((h for h in held if displaceable(h, now)), key=lambda h: strength(h, now))
    ranked = sorted(waiting, key=lambda w: strength(w, now), reverse=True)
    out: list[Displacement] = []
    for w in ranked:
        if len(out) >= max(0, int(max_n)) or not pool:
            break
        h = pool[0]
        why = beats(w, h, now, margin=margin)
        if why is None:
            continue
        quiet = "its act already closed" if h.done else (
            f"quiet {_hours(quiet_for(h, now))}, past half its life ({_hours(h.ttl / 2)})")
        out.append(Displacement(held_id=h.concern_id, by_id=w.concern_id, held_score=score(h, now),
                                by_score=score(w, now), reason=f"{quiet}; {why}"))
        pool.pop(0)
    return out


def intake_victim(held: Sequence[ContestEntry], now: datetime) -> Optional[ContestEntry]:
    """The held concern a new promise pushes out: the weakest one that is not ``never_displaced``, preferring one
    with no act in flight. None when every held concern is protected (the promise then waits)."""
    pool = [h for h in held if not h.never_displaced]
    if not pool:
        return None
    return min(pool, key=lambda h: (h.in_flight is not None and not h.done, strength(h, now)))


def standings(held: Sequence[ContestEntry], now: datetime) -> list[Standing]:
    """One line of fact per held concern: score, how long it has been quiet, and whether it can be displaced."""
    out = []
    for h in held:
        quiet = quiet_for(h, now)
        if h.never_displaced:
            why = f"never displaced: {h.never_displaced}"
        elif h.done:
            why = "can be displaced: its act already closed"
        elif h.in_flight:
            why = f"protected: {h.in_flight}"
        elif h.unseen_quiet and quiet >= h.ttl / 2:
            why = "quiet past half its life; it can be displaced after this thought unless I act on it"
        elif quiet >= h.ttl / 2:
            why = f"can be displaced: quiet past half its life ({_hours(h.ttl / 2)})"
        else:
            why = f"protected until quiet {_hours(h.ttl / 2)}"
        out.append(Standing(concern_id=h.concern_id, score=score(h, now), quiet_h=quiet.total_seconds() / 3600.0,
                            displaceable=displaceable(h, now), why=why))
    return out


__all__ = [
    "DEFAULT_MARGIN",
    "MAX_DISPLACEMENTS_PER_TICK",
    "ContestEntry",
    "Displacement",
    "Standing",
    "beats",
    "displaceable",
    "displacements",
    "intake_victim",
    "quiet_for",
    "score",
    "score_words",
    "standings",
    "strength",
]
