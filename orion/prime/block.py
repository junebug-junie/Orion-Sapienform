"""The Prime block: Prime's present, rendered by code into every unified turn (living Prime design, Part 1).

Design: docs/superpowers/specs/2026-11-12-living-prime-one-orion-design.md, "Part 1". Prime
(services/orion-mind/app/prime_loop.py) holds concerns between turns in Redis ``orion:prime:state``. A
unified turn (chat, curiosity, consult, self-sense) never read it. This module turns that state into a
short, cited block for the turn's harness prefix. No model writes it.

What it shows, each line with its own ref:

* concerns held now: age, act state, and the Temporal Self arc ``[arc:<id>]`` when there is one
  (``[prime_concern:<id>]``);
* the last thought: its reading, how long ago, and the move it made or "no move"
  (``[thought:<id>]``; see ``_thought_ref`` for how the id is known);
* open asks (questions to curiosity) and regulate signals, with their settle state;
* Prime's last body read: the binding constraints it last saw (``[prime_tick:<time>]``);
* at most 2 personality tendencies, ``tending: <label> [disposition:<id>]`` (orion/prime/tending.py,
  PRIME_PERSONALITY_LINE_ENABLED): what Orion's past choices show Orion tends to pick over what.

Rules (all deterministic):

* **Capped** at ``BLOCK_CAP`` characters. Free text (concern words, reading, questions) is cut to fit;
  refs are never cut. If even the refs do not fit, lower lines are dropped, in this order: tending,
  body, "asks:
  none open", concerns after the first, open asks, the last thought. The first concern is dropped last.
* **Stale means absent.** Prime's last tick time (``orion:prime:concerns`` ``as_of``: Prime writes it on
  every tick, after its state) older than ``stale_after`` renders only "Prime has not ticked since <time>".
* **Missing or unreadable** state renders one labelled absent line. Never raises into a turn.

``check_prime_citations`` is the citation gate for a reply written with this block in view: refs of the
block's kinds that were not shown are counted as rejected, the same way ``check_arc_citations``
(orion/temporal_self/citations.py) counts unknown arc ids.

Pure except ``fetch_prime_block``, the one Redis read (fail-open), which takes any object with an async
``get``. Callers: Hub's unified turn (orion/hub/turn_orchestrator.py). Part 3 (Prime's end-of-day turn)
calls ``render_prime_block`` on state it already holds.
"""

from __future__ import annotations

import asyncio
import inspect
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Literal, Optional

from orion.prime.state import PRIME_CONCERNS_KEY, PrimeConcernsViewV1, PrimeStateV1
from orion.prime.tending import TendingV1

PRIME_STATE_KEY = "orion:prime:state"  # same key as services/orion-mind/app/prime_loop.py STATE_KEY
BLOCK_CAP = 600
# Prime ticks every PRIME_TICK_SEC (15 s) and saves on every tick, also while it yields to a Juniper turn.
# Ten minutes without a tick is forty missed ticks: Prime is down, not busy.
DEFAULT_STALE_AFTER = timedelta(minutes=10)
FETCH_TIMEOUT_SEC = 1.0
HEADER = "PRIME (my slow loop between turns)"

BlockStatus = Literal["ok", "stale", "absent"]
# Ref kinds the block writes. The citation gate only judges these kinds.
PRIME_REF_KINDS = ("prime_concern", "thought", "curiosity_focus", "prime_regulate", "prime_tick", "disposition")
# The id ends on a letter, digit, '#' or 'Z' (prime_tick:04:22Z), so sentence punctuation after an
# unbracketed ref ("thought:#5.") is not read as part of it.
_PRIME_REF_RE = re.compile(r"\[?\b(" + "|".join(PRIME_REF_KINDS) + r"):([A-Za-z0-9#_.:\-]*[A-Za-z0-9#_])\]?")
_ELLIPSIS = "…"


@dataclass(frozen=True)
class PrimeBlockV1:
    """The rendered block. ``text`` goes into the prompt verbatim; ``refs`` are every ref it shows."""

    text: str
    status: BlockStatus
    refs: tuple[str, ...] = ()
    ticked_at: Optional[datetime] = None
    dropped_lines: int = 0
    # Personality tending lines that made it into ``text`` (after the cap).
    tending: int = 0

    def log_fields(self) -> str:
        return (f"status={self.status} chars={len(self.text)} refs={len(self.refs)} "
                f"dropped={self.dropped_lines} tending={self.tending} "
                f"ticked_at={self.ticked_at.isoformat() if self.ticked_at else '-'}")


@dataclass
class _Line:
    """One block line: ``lead`` + cut-to-fit ``body`` + ``refs``. Lower ``rank`` is dropped last."""

    lead: str
    body: str
    refs: list[str]
    rank: int
    min_body: int = 0

    def render(self, body_len: Optional[int] = None) -> str:
        body = self.body if body_len is None else _cut(self.body, body_len)
        head = self.lead + body if body else self.lead.rstrip()
        cites = " ".join(f"[{r}]" for r in self.refs)
        return " ".join(p for p in (head, cites) if p)


def _cut(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= n:
        return text
    if n <= 1:
        return ""
    return text[: n - 1].rstrip() + _ELLIPSIS


def _utc(at: datetime) -> datetime:
    return at.replace(tzinfo=timezone.utc) if at.tzinfo is None else at.astimezone(timezone.utc)


def _ago(now: datetime, at: datetime) -> str:
    sec = max(0.0, (_utc(now) - _utc(at)).total_seconds())
    if sec < 60:
        return "under a minute ago"
    minutes = int(sec // 60)
    if minutes < 60:
        return f"{minutes} min ago"
    hours = minutes // 60
    if hours < 48:
        return f"{hours} h {minutes % 60} min ago"
    return f"{hours // 24} days ago"


def _age(now: datetime, at: datetime) -> str:
    return _ago(now, at).replace(" ago", "")


def _clock(at: datetime) -> str:
    return f"{_utc(at):%b %d %H:%M} UTC"


def _tick_ref(at: datetime) -> str:
    return f"prime_tick:{_utc(at):%H:%MZ}"


def absent_block(reason: str) -> PrimeBlockV1:
    return PrimeBlockV1(text=f"{HEADER}: absent ({_cut(reason, 160)}). Do not guess what Prime holds.",
                        status="absent")


def stale_block(ticked_at: datetime, now: datetime) -> PrimeBlockV1:
    return PrimeBlockV1(
        text=(f"{HEADER}: Prime has not ticked since {_clock(ticked_at)} ({_ago(now, ticked_at)}). "
              "Its last state is not shown: it may no longer be true."),
        status="stale", ticked_at=_utc(ticked_at))


def _thought_ref(state: PrimeStateV1) -> tuple[Optional[str], str]:
    """(ref id, move words) for the last thought.

    The ref is ``last_thought_id`` (the id Prime's runtime saves with ``last_reading``, living Prime Part 2) when
    the state has it. A state saved before that field existed falls back to Prime's own thought count (``#235`` =
    its 235th thought). The move is the open question or regulate signal the last thought made (opened at the
    thought's own time, carrying its id), else "no move"."""
    at = state.last_thought_at
    if at is None:
        return None, ""
    # Exact match by construction: Prime's _conclude uses one `now` for apply_answer's last_thought_at and
    # for the move's opened_at (services/orion-mind/app/prime_loop.py _conclude/_nominate/_regulate).
    move_tid, move = None, "no move"
    for n in state.open_nominations:
        if n.opened_at >= at:
            move_tid, move = n.thought_id, "asked curiosity a question"
            break
    else:
        for r in state.open_regulations:
            if r.opened_at >= at:
                move_tid, move = r.thought_id, f"regulate {r.direction} {r.family}"
                break
    if state.last_thought_id:
        return state.last_thought_id, move
    return (move_tid or f"#{state.thoughts_total}"), move


# A tending label is cut to this many characters in the block (the frame's labels are <= 160).
TENDING_LABEL_SHOWN = 90
MAX_TENDING_LINES = 2


def _lines(state: PrimeStateV1, now: datetime, ticked_at: datetime) -> list[_Line]:
    out: list[_Line] = []
    # Concerns: rank 1 for the first (never dropped before the reading), 4 for the rest.
    if not state.concerns:
        out.append(_Line("- Holding: no concerns now.", "", [_tick_ref(ticked_at)], rank=1))
    for i, c in enumerate(state.concerns):
        meta = f"{_age(now, c.opened_at)}"
        if c.graded == "lapsed":
            meta += ", its act lapsed"
        elif c.graded == "settled":
            meta += ", settled"
        refs = [f"prime_concern:{c.concern_id}"] + ([f"arc:{c.arc_id}"] if c.arc_id else [])
        out.append(_Line(f"- Holding ({meta}): ", c.text, refs, rank=1 if i == 0 else 4, min_body=24))

    tid, move = _thought_ref(state)
    if state.last_thought_at is None or tid is None:
        out.append(_Line("- Last thought: none yet.", "", [_tick_ref(ticked_at)], rank=2))
    else:
        lead = f"- Last thought ({_ago(now, state.last_thought_at)}; {move}): "
        out.append(_Line(lead, state.last_reading or "(no reading)", [f"thought:{tid}"], rank=2, min_body=40))

    for n in state.open_nominations:
        if n.arm == "control":
            settle = "held back (control arm, not sent)"
        elif n.queued is None:
            settle = "sent, no answer yet"
        elif n.queued:
            settle = "queued, waiting for a curiosity run"
        else:
            settle = f"refused ({_cut(n.queue_error or 'no reason', 40)})"
        refs = [f"curiosity_focus:{n.focus_id}", f"thought:{n.thought_id}"]
        out.append(_Line(f"- Asked curiosity, {settle}: ", n.question, refs, rank=3, min_body=24))
    for r in state.open_regulations:
        if r.arm == "control":
            settle = "held back (control arm)"
        elif r.written is None:
            settle = "not written yet"
        elif r.written:
            settle = f"in force until {_utc(r.expires_at):%H:%M} UTC"
        else:
            settle = f"not delivered ({_cut(r.write_error or 'write failed', 40)})"
        out.append(_Line(f"- Regulate {r.direction} {r.family}, {settle}", "",
                         [f"prime_regulate:{r.signal_id}", f"thought:{r.thought_id}"], rank=3))
    if not state.open_nominations and not state.open_regulations:
        out.append(_Line("- Asks: none open.", "", [_tick_ref(ticked_at)], rank=5))

    if state.seen_binding is None:
        body = "- Body: not read yet."
    elif state.seen_binding:
        body = "- Body: binding now: " + ", ".join(state.seen_binding[:4])
    else:
        body = "- Body: nothing binding."
    out.append(_Line(body, "", [_tick_ref(ticked_at)], rank=6))
    return out


def _tending_line(t: TendingV1) -> str:
    return _Line("- tending: ", _cut(t.label, TENDING_LABEL_SHOWN), [t.ref], rank=7).render()


def _fit(header: str, lines: list[_Line], cap: int) -> tuple[list[str], list[_Line], int]:
    """Cut free text so the block fits ``cap``; drop the highest-rank lines when refs alone do not fit."""
    lines = list(lines)
    dropped = 0
    while True:
        fixed = len(header) + sum(len(ln.render(0)) + 1 for ln in lines)  # +1: newline
        need = sum(ln.min_body for ln in lines if ln.body)
        if fixed + need <= cap or len(lines) <= 1:
            break
        worst = max(range(len(lines)), key=lambda i: (lines[i].rank, i))
        lines.pop(worst)
        dropped += 1
    room = max(0, cap - (len(header) + sum(len(ln.render(0)) + 1 for ln in lines)))
    # Water-fill the room: short bodies take only what they need (shortest first), and every char they
    # leave goes to the longer ones. +1 per body for the space render() puts before its refs.
    wants = {i: len(" ".join(ln.body.split())) + 1 for i, ln in enumerate(lines) if ln.body}
    order = sorted(wants, key=lambda i: (wants[i], i))
    budgets: dict[int, int] = {}
    left = room
    for k, i in enumerate(order):
        budgets[i] = min(left // (len(order) - k), wants[i])
        left -= budgets[i]
    rendered = [ln.render(max(0, budgets.get(i, 1) - 1) if ln.body else None) for i, ln in enumerate(lines)]
    return rendered, lines, dropped


def render_prime_block(
    state: Optional[PrimeStateV1],
    *,
    ticked_at: Optional[datetime],
    now: datetime,
    stale_after: timedelta = DEFAULT_STALE_AFTER,
    cap: int = BLOCK_CAP,
    absent_reason: Optional[str] = None,
    tending: Iterable[TendingV1] = (),
) -> PrimeBlockV1:
    """Render Prime's present for a turn. Pure.

    ``state``: Prime's state (``orion:prime:state``, ``PrimeStateV1``). ``ticked_at``: when Prime last
    ticked (``orion:prime:concerns`` ``as_of``). ``absent_reason``: why the state could not be read; it
    wins over everything else. The block never exceeds ``cap`` characters."""
    if absent_reason:
        return absent_block(absent_reason)
    if state is None:
        return absent_block("Prime has no saved state")
    if ticked_at is None:
        return absent_block("Prime's last tick time is unknown, so its state may be old")
    if _utc(now) - _utc(ticked_at) > stale_after:
        return stale_block(ticked_at, now)
    header = f"{HEADER}, ticked {_ago(now, ticked_at)}:"
    lines = _lines(state, now, ticked_at)
    text, refs, dropped = _render(header, lines, cap)
    # Tending lines are dropped first: k lines are shown only when the rest of the block, fitted into the
    # room they leave, still keeps every line it keeps without them (its free text may be cut shorter).
    shown = list(tending)[:MAX_TENDING_LINES]
    for k in range(len(shown), 0, -1):
        extra = [_tending_line(t) for t in shown[:k]]
        t_text, t_refs, _t_dropped = _render(header, lines, cap - sum(len(e) + 1 for e in extra))
        if set(t_refs) >= set(refs) and len(t_text) + sum(len(e) + 1 for e in extra) <= cap:
            return PrimeBlockV1(text="\n".join([t_text, *extra]), status="ok",
                                refs=tuple(dict.fromkeys(t_refs + tuple(t.ref for t in shown[:k]))),
                                ticked_at=_utc(ticked_at),
                                dropped_lines=dropped + len(shown) - k, tending=k)
    return PrimeBlockV1(text=text, status="ok", refs=refs, ticked_at=_utc(ticked_at),
                        dropped_lines=dropped + len(shown), tending=0)


def _render(header: str, lines: list[_Line], cap: int) -> tuple[str, tuple[str, ...], int]:
    """(text, refs shown, lines dropped) for the block's own lines fitted into ``cap``."""
    rendered, kept, dropped = _fit(header, lines, cap)
    text = "\n".join([header, *rendered])
    if len(text) > cap:  # only if one line alone outgrows the cap: never send more than promised
        # Cut back to a line boundary so no half ref is left for the model to copy.
        cut = text.rfind("\n", 0, cap)
        text = text[:cut] if cut > 0 else text[: cap - 1] + _ELLIPSIS
    # Refs from the lines themselves (never parsed back out of free text), and only those still in the text.
    refs = tuple(dict.fromkeys(r for ln in kept for r in ln.refs if f"[{r}]" in text))
    return text, refs, dropped


def parse_prime_redis(raw_state: Any, raw_view: Any) -> tuple[Optional[PrimeStateV1], Optional[datetime],
                                                               Optional[str]]:
    """(state, ticked_at, absent_reason) from the two raw Redis values. Pure, never raises."""
    if not raw_state:
        return None, None, "Prime has no saved state in orion:prime:state"
    try:
        state = PrimeStateV1.model_validate_json(raw_state)
    except Exception as exc:  # noqa: BLE001
        return None, None, f"Prime's state is unreadable ({type(exc).__name__})"
    ticked_at: Optional[datetime] = None
    if raw_view:
        try:
            ticked_at = PrimeConcernsViewV1.model_validate_json(raw_view).as_of
        except Exception:  # noqa: BLE001 -- unknown tick time renders absent, never as current
            ticked_at = None
    return state, ticked_at, None


async def fetch_prime_block(
    redis: Any,
    *,
    now: Optional[datetime] = None,
    stale_after: timedelta = DEFAULT_STALE_AFTER,
    cap: int = BLOCK_CAP,
    timeout_sec: float = FETCH_TIMEOUT_SEC,
    tending: Any = (),
) -> PrimeBlockV1:
    """Read Prime's state and tick time from Redis and render the block. Fail-open: any read error or
    timeout renders the labelled absent block, never raises.

    ``tending``: the tending lines, or an awaitable (e.g. an already-running task) that yields them, so the
    caller's HTTP read runs alongside the Redis reads. A failed awaitable means no tending lines."""
    if inspect.isawaitable(tending):
        tending_src, tending = tending, ()
    else:
        tending_src = None
    try:
        raw_state, raw_view = await asyncio.wait_for(
            asyncio.gather(redis.get(PRIME_STATE_KEY), redis.get(PRIME_CONCERNS_KEY)), timeout=timeout_sec)
    except asyncio.TimeoutError:
        return absent_block(f"Prime's state did not load within {timeout_sec:g} s")
    except Exception as exc:  # noqa: BLE001
        return absent_block(f"Prime's state could not be read ({type(exc).__name__})")
    if tending_src is not None:
        try:
            tending = tuple(await tending_src)
        except Exception:  # noqa: BLE001 -- the tending lines never break the block
            tending = ()
    state, ticked_at, reason = parse_prime_redis(raw_state, raw_view)
    return render_prime_block(state, ticked_at=ticked_at, now=now or datetime.now(timezone.utc),
                              stale_after=stale_after, cap=cap, absent_reason=reason, tending=tending)


@dataclass(frozen=True)
class PrimeCitationCheck:
    """What a reply did with the block's refs. Counted, never blocked (same as the arc gate's counts)."""

    shown: int
    cited_valid: tuple[str, ...] = ()
    rejected_unknown: tuple[str, ...] = ()
    label: Literal["verified", "UNVERIFIED", "no_prime_claims"] = "no_prime_claims"
    examples: tuple[str, ...] = field(default_factory=tuple)

    def log_fields(self) -> str:
        return (f"label={self.label} shown={self.shown} cited={len(self.cited_valid)} "
                f"rejected={len(self.rejected_unknown)}")


def check_prime_citations(text: str, shown_refs: Iterable[str]) -> PrimeCitationCheck:
    """Gate a reply's Prime refs against the refs the block showed.

    A ref of a Prime kind (``prime_concern:``, ``thought:``, ...) that was not shown is rejected: Orion
    cited a concern or thought it was never shown. Arc refs are not judged here (the arc gate owns them)."""
    shown = {r for r in shown_refs if r.split(":", 1)[0] in PRIME_REF_KINDS}
    valid: list[str] = []
    rejected: list[str] = []
    for m in _PRIME_REF_RE.finditer(text or ""):
        ref = f"{m.group(1)}:{m.group(2)}"
        bucket = valid if ref in shown else rejected
        if ref not in bucket:
            bucket.append(ref)
    label: Literal["verified", "UNVERIFIED", "no_prime_claims"]
    label = "UNVERIFIED" if rejected else ("verified" if valid else "no_prime_claims")
    return PrimeCitationCheck(shown=len(shown), cited_valid=tuple(valid), rejected_unknown=tuple(rejected),
                              label=label, examples=tuple(rejected[:3]))


__all__ = [
    "BLOCK_CAP",
    "DEFAULT_STALE_AFTER",
    "PRIME_REF_KINDS",
    "PRIME_STATE_KEY",
    "PrimeBlockV1",
    "PrimeCitationCheck",
    "absent_block",
    "check_prime_citations",
    "fetch_prime_block",
    "parse_prime_redis",
    "render_prime_block",
    "stale_block",
]
