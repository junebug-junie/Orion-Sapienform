"""Reveries that ask, Patch 3: turned-away questions wait, digs are read back, threads continue.

Spec: docs/superpowers/specs/2026-11-18-reverie-questions-design.md, "Threads (idea 9) and read-back".
Migration: services/orion-sql-db/manual_migration_reverie_question_threads.sql.

Everything here is pure (rows in, decisions out); app/store.py reads and writes, app/reverie.py runs it
once per poll. No model call, no scheduler: the reverie worker's own poll does the work.

1. **Waiting.** The curiosity focus slot holds one pending question. Hub turns a reverie question away
   while any focus is pending (orion/curiosity/focus.py PENDING_KEPT, or JUNIPER_PROTECTED). Such a
   question is not wrong, only early: it is stored ``offer_status = waiting`` and offered again, oldest
   first, on a later poll when the slot is free. A waiting question older than the lapse window lapses.
2. **Read-back.** When the curiosity run that took a reverie focus ends, ``reverie_question.outcome`` is
   filled from the focus row (status, disposition, run id), the finding register (register, reason,
   novelty), the run's finish state (``continue_line``, the finding text) and the run's attention-schema
   row (``predicted_next``: the run's own ``:TurnOutcome.continue_note``, mirrored to Postgres by
   orion-durable-runs; the worldview graph itself is not reachable from this thin service).
3. **Threads.** A dig that ended with ``continue_line`` true makes its thread the next reverie's seed,
   before the picker rotates kinds. A thread closes ``concluded`` (the dig ended with no continue),
   ``exhausted`` (two reveries in a row on it produced no usable question) or ``lapsed`` (no activity
   for the lapse window).

The inquiry ledger (#3442 path (b), ``curiosity_inquiry``) is not built. When it lands, a thread becomes an
inquiry with ``question_ref = reverie:<thread_id>`` (fold-in point: ``thread_book`` below).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence

# reverie_question.offer_status
OFFER_WAITING = "waiting"
OFFER_SENT = "sent"
OFFER_LAPSED = "lapsed"
OFFER_DROPPED = "dropped"

# reverie_thread.status
THREAD_OPEN = "open"
THREAD_CONCLUDED = "concluded"
THREAD_EXHAUSTED = "exhausted"
THREAD_LAPSED = "lapsed"
CLOSED_THREAD_STATUSES = (THREAD_CONCLUDED, THREAD_EXHAUSTED, THREAD_LAPSED)

# A run marks its focus `ran` about 30 s before its finish state and attention-schema row land
# (live 2026-11-25: 14:14:48 ran, 14:14:53 register, 14:15:22 finish). The outcome waits for the finish
# row, but never longer than this: a run whose finish row never lands still gets an outcome.
OUTCOME_GRACE = timedelta(minutes=30)
FINDING_EXCERPT_CHARS = 600
CONTINUE_NOTE_CHARS = 600
# Focus statuses that mean the focus will not change again (orion/curiosity/focus.py).
FOCUS_TERMINAL = ("ran", "failed", "expired", "cleared", "replaced")
FOCUS_REFUSED_PREFIX = "focus refused: "


def _slot_full_messages() -> tuple[str, ...]:
    from orion.curiosity.focus import JUNIPER_PROTECTED, PENDING_KEPT

    return (PENDING_KEPT, JUNIPER_PROTECTED)


def is_slot_full_refusal(route_note: str | None) -> bool:
    """True when Hub refused the focus only because the one slot is held by another pending focus.

    Every other refusal (the cap, routing off, a bad trace id, an unknown answer) is not "early", so the
    question does not wait."""
    note = str(route_note or "")
    if not note.startswith(FOCUS_REFUSED_PREFIX):
        return False
    reason = note[len(FOCUS_REFUSED_PREFIX):]
    return any(reason.startswith(msg) for msg in _slot_full_messages())


def waiting_note(since: datetime | None = None) -> str:
    tail = f" since {since.astimezone(timezone.utc):%Y-%m-%d %H:%M}Z" if since else ""
    return f"waiting for the focus slot{tail}: another focus was pending; offered again when it is free"


def _aware(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _clip(text: Any, limit: int) -> str | None:
    s = " ".join(str(text or "").split())
    if not s:
        return None
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"true", "t", "1", "yes"}:
            return True
        if low in {"false", "f", "0", "no"}:
            return False
    return None


# --- 1. waiting --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ReofferDecision:
    """What the re-offer step does this poll: nothing (``reason``), or offer ``question_id``."""

    question_id: str | None
    reason: str


def pick_reoffer(
    waiting: Sequence[Mapping[str, Any]], *, slot_pending: bool, routed_24h: int, cap: int,
) -> ReofferDecision:
    """The oldest waiting question, when the slot is free and the 24 h focus cap has room."""
    if not waiting:
        return ReofferDecision(None, "none_waiting")
    if cap <= 0:
        return ReofferDecision(None, "focus_routing_off")
    if slot_pending:
        return ReofferDecision(None, "slot_full")
    if routed_24h >= cap:
        return ReofferDecision(None, "focus_cap")
    oldest = min(waiting, key=lambda r: (_aware(r.get("created_at")) or datetime.max.replace(tzinfo=timezone.utc),
                                         str(r.get("question_id"))))
    return ReofferDecision(str(oldest["question_id"]), "offer")


# --- 2. read-back ------------------------------------------------------------------------------------


def build_outcome(row: Mapping[str, Any], *, now: datetime, grace: timedelta = OUTCOME_GRACE) -> dict[str, Any] | None:
    """The ``reverie_question.outcome`` for one sent question, or None while its focus is still open.

    ``row`` is one line of store.RECONCILE_SQL: the focus row (``focus_status``, ``status_at``,
    ``offered_run_id``, ``disposition``, ``refused_reason``), the finding register (``register``,
    ``register_reason``, ``novelty``), the finish state (``finish_found``, ``continue_line``,
    ``finding_text``) and the attention-schema note (``continue_note``).

    Keys ``disposition``, ``finding`` and ``continue_note`` are the ones orion/reverie/questions.py
    outcome_words already renders for the day letter and the situation brief."""
    status = row.get("focus_status")
    if status is None:
        return {"dug": False, "focus_status": "missing"}
    status = str(status)
    if status not in FOCUS_TERMINAL:
        return None  # pending or taken: the run has not ended
    if status != "ran":
        out: dict[str, Any] = {"dug": False, "focus_status": status}
        if row.get("offered_run_id"):
            out["run_id"] = str(row["offered_run_id"])
        if row.get("focus_refused_reason"):
            out["refused_reason"] = _clip(row["focus_refused_reason"], 300)
        return out
    finish_found = bool(row.get("finish_found"))
    status_at = _aware(row.get("status_at"))
    if not finish_found and status_at is not None and now - status_at < grace:
        return None  # ran, but the finish state is still being written
    continue_line = _bool(row.get("continue_line")) if finish_found else None
    out = {
        "dug": True,
        "focus_status": status,
        "run_id": str(row.get("offered_run_id") or "") or None,
        "disposition": row.get("disposition") or None,
        "finding": row.get("register") or None,
        "register_reason": _clip(row.get("register_reason"), 300),
        "novelty": float(row["novelty"]) if row.get("novelty") is not None else None,
        "continue_line": continue_line,
        "continue_note": _clip(row.get("continue_note"), CONTINUE_NOTE_CHARS),
        "finding_excerpt": _clip(row.get("finding_text"), FINDING_EXCERPT_CHARS),
        "finish_found": finish_found,
    }
    return out


# --- 3. threads --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ThreadContinuation:
    """A thread whose last dig said continue: the next reverie's seed and what the prompt shows."""

    thread_id: str
    thread_key: str
    seed_kind: str
    seed_refs: tuple[str, ...]
    seed_material: str
    earlier_question: str
    dig_found: str
    still_open: str
    dug_question_id: str
    dug_at: datetime | None


@dataclass(frozen=True)
class ThreadStatus:
    thread_id: str
    thread_key: str
    seed_kind: str
    status: str
    reason: str
    last_question_id: str
    last_activity_at: datetime


@dataclass
class ThreadBook:
    statuses: list[ThreadStatus] = field(default_factory=list)
    continuation: ThreadContinuation | None = None


def _is_accepted(row: Mapping[str, Any]) -> bool:
    return bool(str(row.get("question") or "").strip()) and not row.get("refused_reason") and not row.get("none_reason")


def _no_question(row: Mapping[str, Any]) -> bool:
    """The reverie produced nothing usable: a none_reason, or a question code refused."""
    return not _is_accepted(row)


def _outcome(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    out = row.get("outcome")
    return out if isinstance(out, Mapping) else None


def _activity(row: Mapping[str, Any]) -> datetime:
    times = [_aware(row.get(k)) for k in ("created_at", "routed_at", "outcome_at")]
    return max(t for t in times if t is not None) if any(times) else datetime.min.replace(tzinfo=timezone.utc)


def thread_status(rows: Sequence[Mapping[str, Any]], *, now: datetime, lapse: timedelta) -> tuple[str, str, bool]:
    """(status, reason, wants_continue) for one thread's rows (any order).

    Checked in order: concluded (the latest dig ended with no continue and nothing was asked after it),
    exhausted (the last two reveries on the thread produced no usable question), lapsed (no activity for
    the lapse window), else open. ``wants_continue``: open, the latest dig said continue, and no accepted
    question was written on the thread after it (a waiting or sent follow-up already carries the thread)."""
    ordered = sorted(rows, key=lambda r: (_aware(r.get("created_at")) or datetime.min.replace(tzinfo=timezone.utc),
                                          str(r.get("question_id"))))
    dug = [r for r in ordered if (_outcome(r) or {}).get("dug")]
    latest_dug = dug[-1] if dug else None
    after = ordered[ordered.index(latest_dug) + 1:] if latest_dug is not None else []
    asked_after = any(_is_accepted(r) for r in after)
    if latest_dug is not None and not asked_after and (_outcome(latest_dug) or {}).get("continue_line") is False:
        return THREAD_CONCLUDED, f"the dig on reverie_question:{latest_dug.get('question_id')} ended with no continue", False
    if len(ordered) >= 2 and _no_question(ordered[-1]) and _no_question(ordered[-2]):
        return THREAD_EXHAUSTED, "two reveries in a row on this thread left no usable question", False
    last = max(_activity(r) for r in ordered)
    if now - last > lapse:
        return THREAD_LAPSED, f"no activity since {last:%Y-%m-%d}", False
    wants = (latest_dug is not None and not asked_after
             and (_outcome(latest_dug) or {}).get("continue_line") is True)
    return THREAD_OPEN, "continue" if wants else "", wants


def thread_book(
    rows: Iterable[Mapping[str, Any]], stored: Mapping[str, Mapping[str, Any]], *, now: datetime, lapse: timedelta,
) -> ThreadBook:
    """Every open (or not yet recorded) thread's status, and the one continuation to offer first.

    ``rows``: reverie_question rows of threads that are open or have no reverie_thread row yet.
    ``stored``: reverie_thread rows by thread_id. Returned statuses are only the ones that changed, so the
    store writes nothing on a quiet poll. The continuation is the thread whose continue-dig is oldest.
    Fold-in point for the inquiry ledger (#3442): an open thread here is the inquiry
    ``question_ref = reverie:<thread_id>``."""
    by_thread: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        by_thread.setdefault(str(r["thread_id"]), []).append(r)
    book = ThreadBook()
    candidates: list[ThreadContinuation] = []
    for thread_id, trows in by_thread.items():
        prior = stored.get(thread_id) or {}
        if prior.get("status") in CLOSED_THREAD_STATUSES:
            continue  # closed is final; a new seed on the same thread_key starts a new thread
        status, reason, wants = thread_status(trows, now=now, lapse=lapse)
        newest = max(trows, key=lambda r: (_aware(r.get("created_at")) or now, str(r.get("question_id"))))
        last_activity = max(_activity(r) for r in trows)
        if (status != prior.get("status") or reason != (prior.get("status_reason") or "")
                or str(newest.get("question_id")) != str(prior.get("last_question_id") or "")):
            book.statuses.append(ThreadStatus(
                thread_id=thread_id, thread_key=str(newest.get("thread_key") or ""),
                seed_kind=str(newest.get("seed_kind") or ""), status=status, reason=reason,
                last_question_id=str(newest.get("question_id")), last_activity_at=last_activity,
            ))
        if wants:
            dug = [r for r in trows if (_outcome(r) or {}).get("dug")]
            d = max(dug, key=lambda r: (_aware(r.get("created_at")) or now, str(r.get("question_id"))))
            o = _outcome(d) or {}
            refs = tuple(str(x) for x in (d.get("seed_refs") or []))
            if not refs or not str(d.get("seed_material") or "").strip():
                continue  # nothing to re-seed from; the thread lapses on its own
            found = " ".join(x for x in (
                f"{o.get('disposition')}." if o.get("disposition") else "",
                f"Finding register: {o.get('finding')}." if o.get("finding") else "",
                str(o.get("finding_excerpt") or ""),
            ) if x).strip()
            candidates.append(ThreadContinuation(
                thread_id=thread_id, thread_key=str(d.get("thread_key") or ""), seed_kind=str(d.get("seed_kind")),
                seed_refs=refs, seed_material=str(d.get("seed_material")),
                earlier_question=str(d.get("question") or ""),
                dig_found=found or "the dig left no readable finding",
                still_open=str(o.get("continue_note") or "") or
                "the dig said to continue this line; it left no note on what is open",
                dug_question_id=str(d.get("question_id")), dug_at=_aware(d.get("outcome_at")) or _aware(d.get("created_at")),
            ))
    if candidates:
        book.continuation = min(candidates, key=lambda c: (c.dug_at or now, c.thread_id))
    return book


__all__ = [
    "CLOSED_THREAD_STATUSES",
    "OFFER_DROPPED",
    "OFFER_LAPSED",
    "OFFER_SENT",
    "OFFER_WAITING",
    "OUTCOME_GRACE",
    "THREAD_CONCLUDED",
    "THREAD_EXHAUSTED",
    "THREAD_LAPSED",
    "THREAD_OPEN",
    "ReofferDecision",
    "ThreadBook",
    "ThreadContinuation",
    "ThreadStatus",
    "build_outcome",
    "is_slot_full_refusal",
    "pick_reoffer",
    "thread_book",
    "thread_status",
    "waiting_note",
]
