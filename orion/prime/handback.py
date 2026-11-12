"""Every turn ends back in Prime (living Prime design, Part 2): the turn hand-back. CPU only, never a model call.

Design: docs/superpowers/specs/2026-11-12-living-prime-one-orion-design.md, "Part 2".

Source (verified live 2026-11-12, see the PR report): the cockpit hops Hub already publishes on the bus channel
``orion:cockpit:hop`` (``CockpitHopV1``) for every unified turn -- chat, curiosity, consult and every system
session. Two hops of a turn carry all Prime needs, joined by ``correlation_id``:

* ``ingress`` (first hop of the turn): ``session_id``, the prompt (``user_message``) and, for a turn Juniper
  typed in Hub, ``mode``;
* ``finalize`` (last hop of a turn that produced a reply): the whole ``HarnessRunV1`` -- ``final_text``,
  ``compliance_verdict`` and the 5b judge's ``reflection`` with the intention ledger's ``promises_made`` /
  ``promises_kept`` (PR #3217, D0).

No new bus event: the hand-back is a second consumer of an existing one.

The fold is deterministic: the same hops in the same order give the same state. Per finished turn Prime keeps:

* who: ``chat_juniper`` (a turn Juniper typed), ``curiosity``, ``consult`` or ``letter``. Every other session
  (world-pulse reads, self-sense evals, PR field notes, reminiscence, outreach) is not handed back: those are
  ~90% of turns (live 11-09..11-12) and are not threads of Orion's own life with anyone;
* a one-line gist, cut from the prompt and the reply (code, not a model);
* links, never copies, to the promises the turn made: ``harness_turn_trace:<correlation_id>#promises_made[<i>]``,
  the same reference the intention ledger's promise acts use (design 2026-11-11, D2 ``test_before``), and the
  promise act ids it kept;
* the correlation id.

Two things can change Prime's state, and only these two wake a thought (the existing funded-thought rule then
decides whether Prime actually thinks):

* **A turn opened a thread** -- it made at least one counted promise (an offer is not counted until accepted,
  the intention ledger's rule). Prime opens a concern whose ``event_ref`` is ``turn:<correlation_id>``. The
  concern is an ordinary concern: its act is a graded bet, it fades after ``CONCERN_TTL`` unless a thought keeps
  it, and it counts toward ``MAX_CONCERNS``.
* **A turn answered a concern** -- the turn's prompt cites ``prime_concern:<id>`` of a concern Prime holds open
  and the turn finished with a reply (verdict completed or partial). That is the existing meaning of "settled"
  (a question about it got a run that wrote something). Only the PROMPT counts, never the reply: once the Prime
  block (Part 1) puts concern ids into every turn's prefix, a reply that lists what Prime holds would otherwise
  settle every concern it names.

Loop guard (turns open concerns, concerns steer turns). On top of ``MAX_CONCERNS`` and ``CONCERN_TTL``:
at most ``TURN_CONCERNS_MAX_OPEN`` turn-opened concerns held at once (so at least two slots stay for Prime's own
thoughts), and at most ``TURN_CONCERNS_PER_DAY`` opened in any 24 hours. A thread refused by a bound is recorded
on the turn, never silently dropped.

Turns Prime started (its curiosity question) still close the act that started them through the existing
nomination grading (orion/prime/acts.py ``settle_nomination``); the runtime links the run's turn correlation id
to the nomination's act on the turn record (``link_started_turn``).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Any, Literal, Optional, Sequence

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from orion.prime.acts import settle_concern
from orion.prime.state import MAX_CONCERNS, PrimeConcernV1, PrimeStateV1, new_id, note_change

TURNS_KEY = "orion:prime:turns"
HANDBACK_CHANNEL = "orion:cockpit:hop"

TurnWho = Literal["chat_juniper", "curiosity", "consult", "letter"]
WHO_WORDS: dict[str, str] = {
    "chat_juniper": "chat with Juniper",
    "curiosity": "curiosity run",
    "consult": "consult with Claude",
    "letter": "day letter",
}

CURIOSITY_SESSION_ID = "orion_curiosity"
# orion/consult/core.py ORION_SESSION_PREFIX (a test pins the two together).
CONSULT_SESSION_PREFIX = "consult-"
# The day letter as Prime's end-of-day turn (design Part 3) runs under this session id. Nothing sends it yet.
LETTER_SESSION_ID = "orion_day_letter"
# Orion's own system sessions are named orion_*; a Hub chat turn Juniper typed carries ``mode`` on ingress.
SYSTEM_SESSION_PREFIX = "orion_"

TURN_EVENT_PREFIX = "turn:"
CONCERN_REF = re.compile(r"prime_concern:([0-9a-f]{16})\b")
COUNTED_VERDICTS = frozenset({"completed", "partial"})

MAX_PENDING = 30
PENDING_TTL = timedelta(hours=6)
MAX_RECENT_TURNS = 20
MAX_SEEN = 300
TURN_CONCERNS_MAX_OPEN = 1
TURN_CONCERNS_PER_DAY = 4
TURN_CONCERN_WINDOW = timedelta(hours=24)
MAX_PROMPT_CONCERN_IDS = 5
PROMPT_EXCERPT_CAP = 300
FINAL_EXCERPT_CAP = 600
GIST_CAP = 200


class TurnSignalV1(BaseModel):
    """One cockpit hop, cut to what the fold reads (``signal_from_hop``)."""

    model_config = ConfigDict(extra="forbid")

    correlation_id: str = Field(min_length=1, max_length=200)
    stage: Literal["ingress", "finalize"]
    at: AwareDatetime
    # ingress
    who: Optional[TurnWho] = None
    prompt_excerpt: str = Field(default="", max_length=PROMPT_EXCERPT_CAP)
    prompt_concern_ids: list[str] = Field(default_factory=list, max_length=MAX_PROMPT_CONCERN_IDS)
    # finalize
    final_excerpt: str = Field(default="", max_length=FINAL_EXCERPT_CAP)
    verdict: Optional[str] = Field(default=None, max_length=40)
    promise_count: int = Field(default=0, ge=0)
    promises_kept: list[str] = Field(default_factory=list, max_length=3)


class PrimeTurnPendingV1(BaseModel):
    """A turn that started (ingress seen) and has not handed back yet."""

    model_config = ConfigDict(extra="forbid")

    correlation_id: str
    who: TurnWho
    started_at: AwareDatetime
    prompt_excerpt: str = Field(default="", max_length=PROMPT_EXCERPT_CAP)
    prompt_concern_ids: list[str] = Field(default_factory=list, max_length=MAX_PROMPT_CONCERN_IDS)


class PrimeTurnV1(BaseModel):
    """One finished turn as Prime holds it."""

    model_config = ConfigDict(extra="forbid")

    correlation_id: str
    who: TurnWho
    gist: str = Field(max_length=GIST_CAP)
    started_at: AwareDatetime
    finished_at: AwareDatetime
    verdict: Optional[str] = None
    # Links into the intention ledger, never the promise text.
    promise_refs: list[str] = Field(default_factory=list)
    promises_kept: list[str] = Field(default_factory=list)
    opened_concern_id: Optional[str] = None
    # Why a thread this turn opened did not become a concern (a loop-guard bound), else None.
    thread_refused: Optional[str] = None
    settled_concern_ids: list[str] = Field(default_factory=list)
    # A turn Prime started: the act that started it (its nomination), linked when that act settles.
    started_by_act: Optional[str] = None


class PrimeTurnsStateV1(BaseModel):
    """The hand-back's own Redis record (``TURNS_KEY``), apart from ``orion:prime:state`` so an older image
    still loads Prime's state after a rollback."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["prime.turns.v1"] = "prime.turns.v1"
    pending: list[PrimeTurnPendingV1] = Field(default_factory=list)
    recent: list[PrimeTurnV1] = Field(default_factory=list)
    # Correlation ids already folded (replay / duplicate-hop safety), newest last.
    seen: list[str] = Field(default_factory=list)
    # When each turn-opened concern opened, for the per-day bound.
    turn_concerns_opened_at: list[AwareDatetime] = Field(default_factory=list)
    turns_total: int = 0


# --- hop -> signal ---------------------------------------------------------------------------------------


def classify_who(session_id: Optional[str], raw: dict[str, Any]) -> Optional[TurnWho]:
    sid = str(session_id or "")
    if sid == CURIOSITY_SESSION_ID:
        return "curiosity"
    if sid.startswith(CONSULT_SESSION_PREFIX):
        return "consult"
    if sid == LETTER_SESSION_ID:
        return "letter"
    if sid and "mode" in raw and not sid.startswith(SYSTEM_SESSION_PREFIX):
        return "chat_juniper"
    return None


def _concern_ids(text: str) -> list[str]:
    out: list[str] = []
    for cid in CONCERN_REF.findall(text or ""):
        if cid not in out:
            out.append(cid)
    return out[:MAX_PROMPT_CONCERN_IDS]


def _counted_promises(reflection: Any) -> tuple[int, list[str]]:
    if not isinstance(reflection, dict):
        return 0, []
    made = reflection.get("promises_made") or []
    count = sum(1 for p in made if isinstance(p, dict) and p.get("quote") and p.get("offer") is not True)
    kept = [str(k)[:200] for k in (reflection.get("promises_kept") or []) if isinstance(k, str) and k][:3]
    return count, kept


def signal_from_hop(hop: dict[str, Any]) -> Optional[TurnSignalV1]:
    """A ``CockpitHopV1`` payload (dict) -> the fold's signal, or None when the hop is not one the fold reads.
    Ingress hops of sessions that are not handed back are dropped here, so the inbox stays small."""
    stage = hop.get("stage")
    corr = str(hop.get("correlation_id") or "")
    at = hop.get("ts")
    if stage not in ("ingress", "finalize") or not corr or not at:
        return None
    raw = hop.get("raw") if isinstance(hop.get("raw"), dict) else {}
    if stage == "ingress":
        who = classify_who(raw.get("session_id"), raw)
        if who is None:
            return None
        prompt = str(raw.get("user_message") or "")
        return TurnSignalV1(correlation_id=corr[:200], stage="ingress", at=at, who=who,
                            prompt_excerpt=prompt[:PROMPT_EXCERPT_CAP], prompt_concern_ids=_concern_ids(prompt))
    count, kept = _counted_promises(raw.get("reflection"))
    verdict = raw.get("compliance_verdict")
    return TurnSignalV1(correlation_id=corr[:200], stage="finalize", at=at,
                        final_excerpt=str(raw.get("final_text") or "")[:FINAL_EXCERPT_CAP],
                        verdict=str(verdict)[:40] if verdict else None, promise_count=count, promises_kept=kept)


# --- the gist ----------------------------------------------------------------------------------------------

_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```\s*$")
_LEAD = re.compile(r"^[#>*\-\s]+")


def _first_line(text: str) -> str:
    for line in (text or "").splitlines():
        line = _LEAD.sub("", line.replace("**", "").replace("__", "")).strip()
        if line and not line.startswith("```"):
            return " ".join(line.split())
    return ""


def _reply_line(final: str) -> str:
    """First meaningful line of a reply. A JSON reply (curiosity writes one) gives its first string field."""
    stripped = _FENCE.sub("", (final or "").strip())
    if stripped.startswith("{"):
        try:
            data = json.loads(stripped)
        except ValueError:
            data = None
        if isinstance(data, dict):
            for value in data.values():
                if isinstance(value, str) and value.strip():
                    return _first_line(value)
    return _first_line(stripped)


def _cut(text: str, cap: int) -> str:
    return text if len(text) <= cap else text[:cap - 3].rstrip() + "..."


def turn_gist(who: TurnWho, prompt: str, final: str) -> str:
    reply = _reply_line(final) or "(no reply text)"
    if who == "chat_juniper":
        return _cut(f'Juniper: "{_cut(_first_line(prompt), 80)}" / me: "{_cut(reply, 100)}"', GIST_CAP)
    return _cut(reply, GIST_CAP)


# --- the fold ------------------------------------------------------------------------------------------------


def promise_refs(correlation_id: str, count: int) -> list[str]:
    return [f"harness_turn_trace:{correlation_id}#promises_made[{i}]" for i in range(count)]


def _open_turn_concerns(state: PrimeStateV1) -> int:
    return sum(1 for c in state.concerns if c.graded is None and (c.event_ref or "").startswith(TURN_EVENT_PREFIX))


def thread_refusal(state: PrimeStateV1, turns: PrimeTurnsStateV1, now: datetime) -> Optional[str]:
    """Why a thread a turn opened may not become a concern now (the loop guard), or None."""
    if len(state.concerns) >= MAX_CONCERNS:
        return f"already holding {MAX_CONCERNS} concerns"
    if _open_turn_concerns(state) >= TURN_CONCERNS_MAX_OPEN:
        return f"already holding {TURN_CONCERNS_MAX_OPEN} concern opened by a turn"
    recent = [t for t in turns.turn_concerns_opened_at if now - t < TURN_CONCERN_WINDOW]
    if len(recent) >= TURN_CONCERNS_PER_DAY:
        return f"{TURN_CONCERNS_PER_DAY} turn concerns opened in the last 24 hours"
    return None


def _push_seen(seen: list[str], corr: str) -> list[str]:
    return [*seen, corr][-MAX_SEEN:]


def fold_turn_signals(
    state: PrimeStateV1, turns: PrimeTurnsStateV1, signals: Sequence[TurnSignalV1], now: datetime,
) -> tuple[PrimeStateV1, PrimeTurnsStateV1, list[PrimeTurnV1]]:
    """Fold hops, in arrival order, into Prime's state and the hand-back record. Returns the turns folded now.
    Pure and deterministic; a hop seen twice, or a finalize with no ingress, changes nothing."""
    pending = list(turns.pending)
    recent = list(turns.recent)
    seen = list(turns.seen)
    opened_at = [t for t in turns.turn_concerns_opened_at if now - t < TURN_CONCERN_WINDOW]
    folded: list[PrimeTurnV1] = []
    for sig in signals:
        corr = sig.correlation_id
        if corr in seen:
            continue
        if sig.stage == "ingress":
            if sig.who is None or any(p.correlation_id == corr for p in pending):
                continue
            pending.append(PrimeTurnPendingV1(correlation_id=corr, who=sig.who, started_at=sig.at,
                                              prompt_excerpt=sig.prompt_excerpt,
                                              prompt_concern_ids=sig.prompt_concern_ids))
            continue
        start = next((p for p in pending if p.correlation_id == corr), None)
        if start is None:
            continue
        pending = [p for p in pending if p.correlation_id != corr]
        seen = _push_seen(seen, corr)
        words = WHO_WORDS[start.who]
        gist = turn_gist(start.who, start.prompt_excerpt, sig.final_excerpt)
        replied = bool(sig.final_excerpt.strip()) and (sig.verdict or "") in COUNTED_VERDICTS
        settled: list[str] = []
        if replied:
            for cid in start.prompt_concern_ids:
                state, line = settle_concern(state, cid, arm="treated", produced=True, now=now,
                                             note=f"answered in {words} turn {corr}")
                if line:
                    settled.append(cid)
                    state = note_change(state, f"a {words} turn ({corr}) answered my concern: {line}")
        refs = promise_refs(corr, sig.promise_count)
        opened: Optional[str] = None
        refused: Optional[str] = None
        if refs:
            refused = thread_refusal(state, turns.model_copy(update={"turn_concerns_opened_at": opened_at}), now)
            if refused is None:
                n = len(refs)
                text = _cut(f"a thread I opened in a {words} ({n} promise{'s' if n != 1 else ''}, "
                            f"{refs[0]}{' ...' if n > 1 else ''}): {gist}", 300)
                concern = PrimeConcernV1(concern_id=new_id(), text=text, event_ref=f"{TURN_EVENT_PREFIX}{corr}",
                                         opened_at=now, last_evidence_at=now)
                state = state.model_copy(update={"concerns": [*state.concerns, concern]})
                state = note_change(state, f"a {words} turn ({corr}) opened a thread: {gist[:120]}")
                opened = concern.concern_id
                opened_at.append(now)
        turn = PrimeTurnV1(correlation_id=corr, who=start.who, gist=gist, started_at=start.started_at,
                           finished_at=sig.at, verdict=sig.verdict, promise_refs=refs,
                           promises_kept=list(sig.promises_kept), opened_concern_id=opened, thread_refused=refused,
                           settled_concern_ids=settled)
        recent = [*recent, turn][-MAX_RECENT_TURNS:]
        folded.append(turn)
    pending = [p for p in pending if now - p.started_at <= PENDING_TTL][-MAX_PENDING:]
    turns = turns.model_copy(update={"pending": pending, "recent": recent, "seen": seen,
                                     "turn_concerns_opened_at": opened_at,
                                     "turns_total": turns.turns_total + len(folded)})
    return state, turns, folded


def link_started_turn(turns: PrimeTurnsStateV1, correlation_id: Optional[str], act_id: str) -> PrimeTurnsStateV1:
    """A nomination Prime opened settled on a run whose turn is ``correlation_id``: link the turn to that act.
    The act itself is closed by the existing nomination grading, not here."""
    if not correlation_id:
        return turns
    recent = [t.model_copy(update={"started_by_act": act_id}) if t.correlation_id == correlation_id else t
              for t in turns.recent]
    return turns.model_copy(update={"recent": recent})


def turn_lines(turns: PrimeTurnsStateV1, now: datetime, *, limit: int = 5) -> list[str]:
    """Frame lines: the latest finished turns, newest first (shown when Prime thinks; never a wake)."""
    out = []
    for t in reversed(turns.recent[-limit:]):
        mins = max(0, int((now - t.finished_at).total_seconds() // 60))
        age = f"{mins} min ago" if mins < 120 else f"{mins // 60} h ago"
        bits = []
        if t.opened_concern_id:
            bits.append(f"opened concern {t.opened_concern_id}")
        if t.thread_refused:
            bits.append(f"made {len(t.promise_refs)} promise(s), no concern: {t.thread_refused}")
        if t.settled_concern_ids:
            bits.append(f"answered concern {', '.join(t.settled_concern_ids)}")
        if t.started_by_act:
            bits.append(f"started by my act {t.started_by_act}")
        tail = f" [{'; '.join(bits)}]" if bits else ""
        out.append(f"- {age}, {WHO_WORDS[t.who]} (turn {t.correlation_id}): {t.gist}{tail}")
    return out or ["(no turn has handed back yet)"]


__all__ = [
    "HANDBACK_CHANNEL",
    "LETTER_SESSION_ID",
    "PrimeTurnPendingV1",
    "PrimeTurnV1",
    "PrimeTurnsStateV1",
    "TURNS_KEY",
    "TURN_CONCERNS_MAX_OPEN",
    "TURN_CONCERNS_PER_DAY",
    "TurnSignalV1",
    "classify_who",
    "fold_turn_signals",
    "link_started_turn",
    "promise_refs",
    "signal_from_hop",
    "thread_refusal",
    "turn_gist",
    "turn_lines",
]
