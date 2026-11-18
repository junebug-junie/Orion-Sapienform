"""Session threads: coding sessions write to Orion; Orion asks questions once an hour.

Design: docs/superpowers/specs/2026-11-15-session-threads-design.md.
Pure rules (caps, the hour bucket, receipt, prompt, parser, PR comment text): ``orion/session_threads/core.py``.

1. **Intake** (``SessionRoundupLoop.intake``, called by ``POST /api/session-threads/message``): a
   coding session sends a message, usually a pull request's design spec. Hub stores it as a
   ``session`` turn, stores its own receipt as a ``hub`` turn, and shows both in Hub chat. The receipt
   is Hub's, never Orion's, and says Orion has NOT read the message. No GPU.
2. **The hourly round-up** (``tick``): every bucket (an hour by default) Hub takes every held thread
   whose message arrived before the bucket started and submits ONE durable run
   (``session_roundup.orion_turn``, run id ``session-roundup-<utc hour>``) whose prompt holds them
   all. No held thread: nothing is submitted. The threads are marked ``reading`` BEFORE the submit,
   so a Hub restart finds the round and never submits a second one for the same hour.
3. **Orion's reply** is cut into one section per thread by code and stored as an ``orion`` turn on
   each thread. A reply with no usable section for a thread is marked ``unparsed`` and ``requeued``:
   it is sent nowhere, it does NOT use the thread's one question round, and the thread is read again
   next hour (a parse accident is not Orion's silence). The whole reply is stored on the thread only
   in a round of one thread; in a larger round it holds other threads' sections, so it stays only on
   the round's durable run. It counts as a
   failure toward the cap below. Empty text, a cut-short turn or a failed run fails the round: the
   threads go back to ``held`` for the next hour, and after 3 failures in a row a thread stays
   ``failed`` until a new session message re-holds it. A run the body never gave a turn in time
   (``core.is_body_refusal``: past its admission deadline before any turn, or still waiting for the
   pool 30 min after it) is not a failure: the threads are held again with their count unchanged,
   until the NEWEST unread message is ``core.BODY_REFUSAL_MAX_AGE_SEC`` old (a new message restarts
   that clock and zeroes the count). A prompt the GPU pool refuses as too big for the lane
   (``core.is_sizing_refusal``) is Hub's sizing: no strike, and Hub halves its prompt budget.
4. **The PR comment**: for a thread with a pull request and real questions Hub posts one comment on
   the PR with Orion's GitHub token, only while the PR is open. Code does it, not the model. The
   prompt tells Orion when a thread's PR is closed or merged.
6. **Delivery**: Orion's questions are shown once to each session that asks for them (sender or
   branch), and stay pending for new sessions until a session replies on the thread.
5. **One round of questions per thread** (``core.MAX_QUESTION_ROUNDS_PER_THREAD``): once Orion has asked
   on a thread, a session reply is still read at the next round-up, but the prompt marks the thread
   ANSWER PHASE and what Orion writes for it is stored as a closing note (status ``closed``): never a
   PR comment, never handed to a session as questions. Question-like lines in it are counted in the log.

Not the consult queue (scripts/consult_loop.py): no Claude, no weekly cap, its own workflow and act
family. Turns are NOT written to ``chat_history_log``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

from orion.session_threads import core

logger = logging.getLogger("orion-hub.session_threads")

MIGRATION = "services/orion-sql-db/manual_migration_orion_session_thread_v1.sql"
FRAME_KIND = "session_thread_turn"

_THREAD_COLUMNS = (
    "thread_id, source_ref, kind, title, session_id, session_name, worktree, branch, pr_number, pr_url, "
    "spec_paths, status, status_reason, last_roundup_id, roundup_failures, created_at, updated_at"
)
_TURN_COLUMNS = (
    "turn_id, seq, thread_id, speaker, body, session_id, roundup_id, delivered_at, pr_comment_url, meta, created_at"
)
_CUT_SHORT_RE = re.compile(r"^\s*\[cut short\b", re.IGNORECASE)
# Makes each round's data-fence lines unguessable for a session (core.fence_lines). Per process: after a
# restart a resubmitted prompt differs by its fence only, and durable-runs keeps the run it already has.
_FENCE_SECRET = secrets.token_hex(16)
# A run with no end in sight this long after its own deadline: Hub stops waiting for it.
OVERDUE_GRACE_SEC = 1800.0
MIN_PROMPT_CHARS = 8000
BUDGET_RESTORE_SEC = 86400.0
UNPARSED_REASON = "orion_reply_unparsed"


class IntakeRefused(ValueError):
    """A message Hub will not store. ``too_many``: a cap was hit (HTTP 429), not a bad message."""

    def __init__(self, detail: str, *, too_many: bool = False) -> None:
        super().__init__(detail)
        self.too_many = too_many


def fence_nonce(roundup_id: str) -> str:
    return hashlib.sha256(f"{_FENCE_SECRET}:{roundup_id}".encode()).hexdigest()[:12]


def message_digest(body: str, specs: Sequence[Dict[str, Any]]) -> str:
    h = hashlib.sha256(body.encode("utf-8"))
    for spec in specs:
        h.update(b"\0" + str(spec.get("path")).encode("utf-8") + b"\0" + str(spec.get("text")).encode("utf-8"))
    return h.hexdigest()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _json_field(row: Dict[str, Any], key: str) -> None:
    if isinstance(row.get(key), str):
        try:
            row[key] = json.loads(row[key])
        except ValueError:
            pass


def _thread(rec: Any) -> Dict[str, Any]:
    out = dict(rec)
    _json_field(out, "spec_paths")
    return out


def _turn(rec: Any) -> Dict[str, Any]:
    out = dict(rec)
    _json_field(out, "meta")
    return out


# --- store (asyncpg) ---------------------------------------------------------------------------


async def intake_precheck(pool, *, thread_id: Optional[str], source_ref: str) -> Dict[str, Any]:
    """What intake needs before it stores: how many threads Hub holds, and for the target thread (when
    it exists) its row, its unread session messages and the digest of its newest session message."""
    held = await pool.fetchrow(
        "SELECT count(*) AS n FROM orion_session_thread WHERE status IN ('held', 'reading')")
    rec = await pool.fetchrow(
        f"SELECT {_THREAD_COLUMNS} FROM orion_session_thread WHERE thread_id = $1::text OR "
        "($1::text IS NULL AND source_ref = $2)", thread_id, source_ref)
    out: Dict[str, Any] = {"held_threads": int(held["n"]) if held else 0, "thread": _thread(rec) if rec else None,
                           "unread": 0, "last_digest": None, "answer_phase": False}
    if rec is not None:
        orion = await pool.fetch(
            "SELECT speaker, roundup_id, meta FROM orion_session_turn WHERE thread_id = $1 AND speaker = 'orion'",
            rec["thread_id"])
        out["answer_phase"] = core.in_answer_phase([_turn(r) for r in orion])
        row = await pool.fetchrow(
            "SELECT (SELECT count(*) FROM orion_session_turn t WHERE t.thread_id = $1 AND t.speaker = 'session' "
            "AND t.roundup_id IS NULL) AS unread, (SELECT t.meta->>'digest' FROM orion_session_turn t "
            "WHERE t.thread_id = $1 AND t.speaker = 'session' ORDER BY t.seq DESC LIMIT 1) AS last_digest",
            rec["thread_id"])
        out.update(unread=int(row["unread"]), last_digest=row["last_digest"])
    return out


async def add_message(pool, *, thread_id: Optional[str], source_ref: str, kind: str, title: str, session_id: str,
                      session_name: Optional[str], worktree: Optional[str], branch: Optional[str],
                      pr_number: Optional[int], pr_url: Optional[str], spec_paths: List[str], body: str,
                      meta: Dict[str, Any], receipt: Callable[[str], str]) -> Optional[Dict[str, Any]]:
    """One session message and Hub's receipt for it, in one transaction.

    ``thread_id`` given: a turn on that thread (None when it does not exist). Otherwise the thread
    for ``source_ref``, created when new. The thread goes back to ``held`` (a thread in a submitted
    round stays ``reading``; the new turn waits for the next round) and its failure count resets."""
    async with pool.acquire() as conn:
        async with conn.transaction():
            if thread_id:
                rec = await conn.fetchrow(
                    "UPDATE orion_session_thread SET status = CASE WHEN status = 'reading' THEN 'reading' ELSE 'held' END, "
                    "status_reason = NULL, roundup_failures = 0, updated_at = now() WHERE thread_id = $1 "
                    f"RETURNING {_THREAD_COLUMNS}", thread_id)
                if rec is None:
                    return None
            else:
                rec = await conn.fetchrow(
                    "INSERT INTO orion_session_thread (thread_id, source_ref, kind, title, session_id, session_name, "
                    "worktree, branch, pr_number, pr_url, spec_paths) "
                    "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb) "
                    "ON CONFLICT (source_ref) DO UPDATE SET title = EXCLUDED.title, session_id = EXCLUDED.session_id, "
                    "session_name = EXCLUDED.session_name, worktree = EXCLUDED.worktree, branch = EXCLUDED.branch, "
                    "pr_url = EXCLUDED.pr_url, spec_paths = EXCLUDED.spec_paths, "
                    "status = CASE WHEN orion_session_thread.status = 'reading' THEN 'reading' ELSE 'held' END, "
                    "status_reason = NULL, roundup_failures = 0, updated_at = now() "
                    f"RETURNING {_THREAD_COLUMNS}",
                    _new_id(), source_ref, kind, title, session_id, session_name, worktree, branch, pr_number,
                    pr_url, json.dumps(spec_paths))
            thread = _thread(rec)
            turn = _turn(await conn.fetchrow(
                "INSERT INTO orion_session_turn (turn_id, thread_id, speaker, body, session_id, meta, created_at) "
                f"VALUES ($1, $2, 'session', $3, $4, $5::jsonb, clock_timestamp()) RETURNING {_TURN_COLUMNS}",
                _new_id(), thread["thread_id"], body, session_id, json.dumps(meta)))
            hub = _turn(await conn.fetchrow(
                "INSERT INTO orion_session_turn (turn_id, thread_id, speaker, body, meta, created_at) "
                f"VALUES ($1, $2, 'hub', $3, $4::jsonb, clock_timestamp()) RETURNING {_TURN_COLUMNS}",
                _new_id(), thread["thread_id"], receipt(thread["thread_id"]), json.dumps({"receipt": True})))
    return {"thread": thread, "turn": turn, "receipt": hub}


async def recent_threads(pool, *, hours: float, limit: int = 100) -> List[Dict[str, Any]]:
    rows = await pool.fetch(
        f"SELECT {_THREAD_COLUMNS}, (SELECT count(*) FROM orion_session_turn t WHERE t.thread_id = h.thread_id) AS turns "
        "FROM orion_session_thread h WHERE updated_at >= $1 ORDER BY updated_at DESC LIMIT $2",
        _now() - timedelta(hours=float(hours)), limit)
    return [_thread(r) for r in rows]


async def get_thread(pool, thread_id: str) -> Optional[Dict[str, Any]]:
    rec = await pool.fetchrow(f"SELECT {_THREAD_COLUMNS} FROM orion_session_thread WHERE thread_id = $1", thread_id)
    return _thread(rec) if rec else None


async def thread_turns(pool, thread_ids: Sequence[str]) -> List[Dict[str, Any]]:
    rows = await pool.fetch(
        f"SELECT {_TURN_COLUMNS} FROM orion_session_turn WHERE thread_id = ANY($1::text[]) ORDER BY seq",
        list(thread_ids))
    return [_turn(r) for r in rows]


async def threads_with_status(pool, status: str) -> List[Dict[str, Any]]:
    rows = await pool.fetch(
        f"SELECT {_THREAD_COLUMNS} FROM orion_session_thread WHERE status = $1 ORDER BY updated_at, thread_id", status)
    return [_thread(r) for r in rows]


async def roundup_exists(pool, roundup_id: str) -> bool:
    rec = await pool.fetchrow("SELECT 1 AS x FROM orion_session_thread WHERE last_roundup_id = $1 LIMIT 1", roundup_id)
    return rec is not None


async def mark_round(pool, roundup_id: str, thread_ids: Sequence[str], turn_ids: Sequence[str]) -> None:
    """The threads and session turns of one round. Written BEFORE the run is submitted."""
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "UPDATE orion_session_thread SET status = 'reading', last_roundup_id = $1, updated_at = now() "
                "WHERE thread_id = ANY($2::text[]) AND status = 'held'", roundup_id, list(thread_ids))
            await conn.execute(
                "UPDATE orion_session_turn SET roundup_id = $1 WHERE turn_id = ANY($2::text[]) AND roundup_id IS NULL",
                roundup_id, list(turn_ids))


async def release_threads(pool, roundup_id: str, thread_ids: Sequence[str]) -> None:
    """Threads marked for a round whose prompt cannot hold them after all: held again, untouched."""
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "UPDATE orion_session_turn SET roundup_id = NULL WHERE roundup_id = $1 AND speaker = 'session' "
                "AND thread_id = ANY($2::text[])", roundup_id, list(thread_ids))
            await conn.execute(
                "UPDATE orion_session_thread SET status = 'held', last_roundup_id = NULL, updated_at = now() "
                "WHERE thread_id = ANY($2::text[]) AND last_roundup_id = $1 AND status = 'reading'",
                roundup_id, list(thread_ids))


async def store_answer(pool, *, thread_id: str, roundup_id: str, body: str, meta: Dict[str, Any],
                       status: str) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
    """Orion's turn for one thread of a round, once. Returns (thread, turn, created). The thread
    ends ``status``, or ``held`` when a session message arrived while Orion was reading."""
    async with pool.acquire() as conn:
        async with conn.transaction():
            rec = await conn.fetchrow(
                f"SELECT {_TURN_COLUMNS} FROM orion_session_turn WHERE thread_id = $1 AND roundup_id = $2 "
                "AND speaker = 'orion' LIMIT 1", thread_id, roundup_id)
            created = rec is None
            if created:
                rec = await conn.fetchrow(
                    "INSERT INTO orion_session_turn (turn_id, thread_id, speaker, body, roundup_id, meta, created_at) "
                    f"VALUES ($1, $2, 'orion', $3, $4, $5::jsonb, clock_timestamp()) RETURNING {_TURN_COLUMNS}",
                    _new_id(), thread_id, body, roundup_id, json.dumps(meta))
            thread = await conn.fetchrow(
                "UPDATE orion_session_thread SET status = CASE WHEN EXISTS (SELECT 1 FROM orion_session_turn t "
                "WHERE t.thread_id = $1 AND t.speaker = 'session' AND t.roundup_id IS NULL) THEN 'held' ELSE $2 END, "
                "status_reason = NULL, roundup_failures = 0, updated_at = now() "
                f"WHERE thread_id = $1 AND status = 'reading' AND last_roundup_id = $3 RETURNING {_THREAD_COLUMNS}",
                thread_id, status, roundup_id)
            if thread is None:
                thread = await conn.fetchrow(
                    f"SELECT {_THREAD_COLUMNS} FROM orion_session_thread WHERE thread_id = $1", thread_id)
    return _thread(thread), _turn(rec), created


async def fail_round(pool, roundup_id: str, *, reason: str, text_for: Callable[[int], str],
                     strike: bool = True, refusal_text_for: Optional[Callable[[int], str]] = None
                     ) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """A failed round: its session turns are unread again, and each thread gets a Hub line that says so.

    ``strike`` (the round's own failure): the thread counts one failure; under the cap it is ``held``
    again, at the cap it stays ``failed``. Not a strike (the body never ran it, ``core.is_body_refusal``):
    ``held`` again, count unchanged. A thread whose previous round was ALSO a body refusal (its
    ``status_reason`` is one; a message or an answer clears it) gets no second Hub line: a hot week
    retries quietly instead of filling every chat window. ``text_for`` writes a strike's line,
    ``refusal_text_for`` (default ``text_for``) a body refusal's. Returns (thread, Hub turn) per line."""
    out: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    async with pool.acquire() as conn:
        async with conn.transaction():
            # A body refusal on a thread whose NEWEST message of this round is older than the backstop
            # counts after all (``core.BODY_REFUSAL_MAX_AGE_SEC``): ``o.n`` is the strike per thread. The
            # newest, not the oldest: a fresh session message restarts the age clock (and ``add_message``
            # zeroes the strike count), so a thread a session still writes on is never ended by the body's
            # heat. Read before the round's turns are released below (the age is taken from them).
            rows = await conn.fetch(
                "UPDATE orion_session_thread h SET roundup_failures = h.roundup_failures + o.n, "
                "status = CASE WHEN h.roundup_failures + o.n >= $2 THEN 'failed' ELSE 'held' END, status_reason = $3, "
                "updated_at = now() FROM (SELECT x.thread_id, x.status_reason, CASE WHEN $4::int = 1 OR "
                "(SELECT max(s.created_at) FROM orion_session_turn s WHERE s.thread_id = x.thread_id "
                "AND s.roundup_id = $1 AND s.speaker = 'session') < now() - make_interval(secs => $5::float8) "
                "THEN 1 ELSE 0 END AS n FROM orion_session_thread x "
                "WHERE x.last_roundup_id = $1 AND x.status = 'reading' FOR UPDATE OF x) o "
                f"WHERE h.thread_id = o.thread_id RETURNING {', '.join('h.' + c.strip() for c in _THREAD_COLUMNS.split(','))}, "
                "o.status_reason AS prior_reason, o.n AS strike",
                roundup_id, core.MAX_CONSECUTIVE_FAILURES, reason[:300], 1 if strike else 0,
                float(core.BODY_REFUSAL_MAX_AGE_SEC))
            await conn.execute(
                "UPDATE orion_session_turn SET roundup_id = NULL WHERE roundup_id = $1 AND speaker = 'session'",
                roundup_id)
            for rec in rows:
                thread = _thread(rec)
                prior = thread.pop("prior_reason", None)
                struck = bool(thread.pop("strike", 1))
                if not struck and core.is_body_refusal(prior or ""):
                    continue
                turn = await conn.fetchrow(
                    "INSERT INTO orion_session_turn (turn_id, thread_id, speaker, body, roundup_id, meta, created_at) "
                    f"VALUES ($1, $2, 'hub', $3, $4, $5::jsonb, clock_timestamp()) RETURNING {_TURN_COLUMNS}",
                    _new_id(), thread["thread_id"],
                    (text_for if struck else (refusal_text_for or text_for))(int(thread["roundup_failures"])), roundup_id,
                    json.dumps({"roundup_failed": True, "reason": reason[:300],
                                **({} if struck else {"body_refusal": True})}))
                out.append((thread, _turn(turn)))
    return out


async def requeue_unparsed(pool, *, thread_id: str, roundup_id: str, body: Optional[str], meta: Dict[str, Any],
                           reason: str, text_for: Callable[[int], str]
                           ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]], bool]:
    """A thread of the round whose reply Hub could not cut: Orion's whole reply is kept as an
    ``orion`` turn (``meta.requeued``: never handed to a session, never a question round), the
    thread's session turns of this round are unread again, and the thread counts one failure (held
    under the cap, ``failed`` at it) with a Hub line. Once per (thread, round). ``body`` None: no
    Orion turn is stored (the reply holds other threads' sections; it stays on the durable run).
    Returns (thread, Orion turn or None, Hub turn or None, Orion turn created)."""
    meta = {**meta, "unparsed": True, "requeued": True}
    async with pool.acquire() as conn:
        async with conn.transaction():
            rec = await conn.fetchrow(
                f"SELECT {_TURN_COLUMNS} FROM orion_session_turn WHERE thread_id = $1 AND roundup_id = $2 "
                "AND speaker = 'orion' LIMIT 1", thread_id, roundup_id)
            created = rec is None and body is not None
            if created:
                rec = await conn.fetchrow(
                    "INSERT INTO orion_session_turn (turn_id, thread_id, speaker, body, roundup_id, meta, created_at) "
                    f"VALUES ($1, $2, 'orion', $3, $4, $5::jsonb, clock_timestamp()) RETURNING {_TURN_COLUMNS}",
                    _new_id(), thread_id, body, roundup_id, json.dumps(meta))
            thread = await conn.fetchrow(
                "UPDATE orion_session_thread SET roundup_failures = roundup_failures + 1, "
                "status = CASE WHEN roundup_failures + 1 >= $3 THEN 'failed' ELSE 'held' END, status_reason = $4, "
                f"updated_at = now() WHERE thread_id = $1 AND status = 'reading' AND last_roundup_id = $2 "
                f"RETURNING {_THREAD_COLUMNS}",
                thread_id, roundup_id, core.MAX_CONSECUTIVE_FAILURES, reason[:300])
            hub = None
            if thread is None:   # already applied (or the thread left the round): nothing more to do
                thread = await conn.fetchrow(
                    f"SELECT {_THREAD_COLUMNS} FROM orion_session_thread WHERE thread_id = $1", thread_id)
            else:
                await conn.execute(
                    "UPDATE orion_session_turn SET roundup_id = NULL WHERE roundup_id = $1 AND thread_id = $2 "
                    "AND speaker = 'session'", roundup_id, thread_id)
                hub = _turn(await conn.fetchrow(
                    "INSERT INTO orion_session_turn (turn_id, thread_id, speaker, body, roundup_id, meta, created_at) "
                    f"VALUES ($1, $2, 'hub', $3, $4, $5::jsonb, clock_timestamp()) RETURNING {_TURN_COLUMNS}",
                    _new_id(), thread_id, text_for(int(thread["roundup_failures"])), roundup_id,
                    json.dumps({"reply_unparsed": True, "reason": reason[:300],
                                **({} if body is not None else {"reply_not_stored": True})})))
    return _thread(thread), (_turn(rec) if rec is not None else None), hub, created


async def set_pr_comment_url(pool, turn_id: str, url: str) -> None:
    await pool.execute("UPDATE orion_session_turn SET pr_comment_url = $2 WHERE turn_id = $1", turn_id, url)


# Orion's questions stay pending until the thread moves on: a session that pulled them and exited
# without answering must not lose them for every later session on the branch. Each session is shown
# them once (``meta.delivered_to`` holds the session ids that claimed them), so a session's own later
# prompts do not repeat them. The thread moves on when a session writes on it after the questions, or
# Orion writes a later turn there (a closing note: a reply that arrived while Orion was reading sits
# BEFORE the question turn). "No questions" and closing notes ask nothing: shown once, to one session.
# Only turns claimed under this rule (``delivered_to`` present) and younger than
# ``REDELIVERY_MAX_AGE_SEC`` are handed out again: questions delivered before it shipped stay delivered.
MAX_DELIVERIES_PER_TURN = 20
REDELIVERY_MAX_AGE_SEC = 14 * 86400
_REDELIVERABLE_SQL = (
    "t.meta ? 'delivered_to' "
    "AND NOT (t.meta @> '{\"no_questions\": true}'::jsonb) AND NOT (t.meta @> '{\"closing_note\": true}'::jsonb) "
    "AND jsonb_array_length(t.meta->'delivered_to') < " + str(MAX_DELIVERIES_PER_TURN) + " "
    "AND t.created_at > now() - make_interval(secs => " + str(REDELIVERY_MAX_AGE_SEC) + ") "
    "AND NOT EXISTS (SELECT 1 FROM orion_session_turn r WHERE r.thread_id = t.thread_id AND r.seq > t.seq "
    "AND (r.speaker = 'session' OR (r.speaker = 'orion' "
    "AND NOT (COALESCE(r.meta, '{}'::jsonb) @> '{\"requeued\": true}'::jsonb))))"
)


def _claimable_sql(session_param: str) -> str:
    """A turn a caller may be shown: never handed back, or (for a named session) a question turn no
    session has replied to and this session has not been shown."""
    return (f"(t.delivered_at IS NULL OR ({session_param} <> '' AND {_REDELIVERABLE_SQL} "
            f"AND NOT (COALESCE(t.meta->'delivered_to', '[]'::jsonb) ? {session_param})))")


async def pending_for(pool, *, session_id: str = "", branch: str = "", limit: int = 20) -> List[Dict[str, Any]]:
    """Orion's turns not yet handed back to THIS session, for the session that sent the thread (or any
    session that wrote on it) OR for the thread's branch. A question turn stays pending for a new
    session until a session replies on the thread (``_REDELIVERABLE_SQL``)."""
    rows = await pool.fetch(
        "SELECT t.turn_id, t.thread_id, t.body, t.meta, t.roundup_id, t.pr_comment_url, t.created_at, t.delivered_at, "
        "h.title, h.branch, h.pr_number, h.pr_url, h.status "
        "FROM orion_session_turn t JOIN orion_session_thread h ON h.thread_id = t.thread_id "
        f"WHERE t.speaker = 'orion' AND {_claimable_sql('$1')} "
        "AND NOT (COALESCE(t.meta, '{}'::jsonb) @> '{\"requeued\": true}'::jsonb) AND ("
        " ($1 <> '' AND (h.session_id = $1 OR EXISTS (SELECT 1 FROM orion_session_turn s WHERE s.thread_id = t.thread_id"
        " AND s.speaker = 'session' AND s.session_id = $1)))"
        " OR ($2 <> '' AND h.branch = $2)) ORDER BY (t.delivered_at IS NOT NULL), t.seq LIMIT $3",
        session_id or "", branch or "", limit)
    return [_turn(r) for r in rows]


async def mark_delivered(pool, turn_ids: Sequence[str], *, session_id: str = "") -> int:
    """Claim turns for one caller. Returns how many this caller claimed (a turn is claimed once per
    session; without a session id, once in total). Atomic per row, so two hooks of one session never
    both show a turn."""
    rows = await pool.fetch(
        "UPDATE orion_session_turn t SET delivered_at = COALESCE(t.delivered_at, now()), meta = CASE WHEN $2 = '' "
        "THEN t.meta ELSE jsonb_set(COALESCE(t.meta, '{}'::jsonb), '{delivered_to}', "
        "COALESCE(t.meta->'delivered_to', '[]'::jsonb) || to_jsonb($2::text)) END "
        f"WHERE t.turn_id = ANY($1::text[]) AND t.speaker = 'orion' AND {_claimable_sql('$2')} "
        "AND NOT (COALESCE(t.meta, '{}'::jsonb) @> '{\"requeued\": true}'::jsonb) "
        "RETURNING t.turn_id", list(turn_ids), session_id or "")
    return len(rows)


# --- GitHub ------------------------------------------------------------------------------------


class GitHubCommenter:
    """One issue comment on a pull request, as Orion (GH_TOKEN / ORION_GITHUB_TOKEN in Hub's
    compose). Only ever on the configured repo: a session's ``pr_url`` never picks the target."""

    def __init__(self, repo: str, *, token_provider: Callable[[], str] = lambda: (
            os.environ.get("ORION_GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""),
            timeout_sec: float = 15.0, api_base: str = "https://api.github.com",
            guard_loader: Optional[Callable[[], Any]] = None) -> None:
        self.repo = repo.strip().strip("/")
        self._token_provider = token_provider
        self.timeout_sec = timeout_sec
        self.api_base = api_base.rstrip("/")
        # Mesh-edge check (orion/memory/mesh_edge.py, issue #3409): None = load from Postgres.
        self._guard_loader = guard_loader

    async def mesh_edge_refusal(self, body: str) -> Optional[str]:
        """Why this body must not be posted publicly, or None when it may (fail closed)."""
        from orion.memory.mesh_edge import MeshEdgeUnavailable, load_guard

        try:
            guard = await asyncio.to_thread(self._guard_loader or load_guard)
        except MeshEdgeUnavailable:
            return "mesh_edge_unavailable"
        hits = guard.find(body)
        return f"mesh_edge_leak:{len(hits)}" if hits else None

    async def post(self, pr_number: int, body: str, *, branch: Optional[str] = None) -> Optional[str]:
        """The comment's URL, or None when it was not posted (no token, no repo, API error, or the
        number is not an OPEN pull request whose head branch is ``branch``: a session names the PR
        number, so Hub checks it before Orion's account writes anywhere)."""
        import httpx

        refusal = await self.mesh_edge_refusal(body)
        if refusal:
            logger.warning("session_thread_pr_comment_skipped pr=%s why=%s", pr_number, refusal)
            return None
        token = (self._token_provider() or "").strip()
        if not token or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", self.repo):
            logger.warning("session_thread_pr_comment_skipped pr=%s why=%s", pr_number,
                           "no_token" if not token else "bad_repo")
            return None
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                   "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "orion-hub"}
        try:
            async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
                pull = await client.get(f"{self.api_base}/repos/{self.repo}/pulls/{int(pr_number)}", headers=headers)
                why = self.target_refusal(pull.status_code, pull.json() if pull.status_code == 200 else {}, branch)
                if why:
                    logger.warning("session_thread_pr_comment_skipped pr=%s why=%s", pr_number, why)
                    return None
                resp = await client.post(
                    f"{self.api_base}/repos/{self.repo}/issues/{int(pr_number)}/comments",
                    headers=headers, json={"body": body})
        except Exception as exc:  # noqa: BLE001 -- the turn is already stored
            logger.warning("session_thread_pr_comment_failed pr=%s err=%s", pr_number, type(exc).__name__)
            return None
        if resp.status_code >= 300:
            logger.warning("session_thread_pr_comment_refused pr=%s http=%s", pr_number, resp.status_code)
            return None
        try:
            return str(resp.json().get("html_url") or "") or None
        except ValueError:
            return None


    async def pr_state(self, pr_number: int, *, timeout_sec: float = 5.0) -> Optional[str]:
        """``open``, ``closed`` or ``merged`` for a pull request of the configured repo, or None when
        Hub cannot tell (no token, an API error). Read-only; never raises."""
        import httpx

        token = (self._token_provider() or "").strip()
        if not token or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", self.repo):
            return None
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                   "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "orion-hub"}
        try:
            async with httpx.AsyncClient(timeout=min(self.timeout_sec, timeout_sec)) as client:
                pull = await client.get(f"{self.api_base}/repos/{self.repo}/pulls/{int(pr_number)}", headers=headers)
            return self.state_of(pull.status_code, pull.json() if pull.status_code == 200 else {})
        except Exception as exc:  # noqa: BLE001 -- the prompt is still built, without the line
            logger.warning("session_thread_pr_state_failed pr=%s err=%s", pr_number, type(exc).__name__)
            return None

    @staticmethod
    def state_of(status_code: int, pull: Dict[str, Any]) -> Optional[str]:
        """A pull request's state from GitHub's ``GET /pulls/<n>``; None when it is not readable."""
        if status_code != 200:
            return None
        state = str(pull.get("state") or "")
        if state == "closed" and (pull.get("merged") or pull.get("merged_at")):
            return "merged"
        return state if state in ("open", "closed") else None

    @staticmethod
    def target_refusal(status_code: int, pull: Dict[str, Any], branch: Optional[str]) -> Optional[str]:
        """Why this number must not get Orion's comment, or None when it may."""
        if status_code != 200:
            return f"not_a_pull_request:http_{status_code}"
        if str(pull.get("state") or "") != "open":
            return f"pull_request_{pull.get('state') or 'unknown'}"
        head = str(((pull.get("head") or {}).get("ref")) or "")
        if not branch or head != branch:
            return "branch_mismatch"
        return None


# --- the loop ----------------------------------------------------------------------------------


class SessionRoundupLoop:
    def __init__(
        self,
        *,
        enabled: bool,
        pool_provider: Callable[[], Any],
        durable_client: Any,
        connections_provider: Callable[[], Dict[str, Any]] = lambda: {},
        commenter: Any = None,
        pr_comments: bool = True,
        interval_sec: float = 3600.0,
        tick_sec: float = 120.0,
        orion_timeout_sec: float = 1800.0,
        run_deadline_sec: float = 10800.0,
        prompt_max_chars: int = 100000,
        tz_name: str = "America/Denver",
        clock: Callable[[], datetime] = _now,
    ) -> None:
        self.enabled = enabled
        self._pool_provider = pool_provider
        self._durable = durable_client
        self._connections_provider = connections_provider
        self._commenter = commenter
        self.pr_comments = pr_comments
        self.interval_sec = max(60.0, float(interval_sec))
        self.tick_sec = tick_sec
        self.orion_timeout_sec = orion_timeout_sec
        self.run_deadline_sec = run_deadline_sec
        self.prompt_max_chars = max(MIN_PROMPT_CHARS, int(prompt_max_chars))
        # The budget a round is built to. Cut (down to MIN_PROMPT_CHARS) each time the GPU pool refuses a
        # round's prompt as too big for the lane (``core.is_sizing_refusal``); back to the setting
        # BUDGET_RESTORE_SEC after the last such refusal (the refusal can be brief: a role away) or on restart.
        self.budget_chars = self.prompt_max_chars
        self._budget_cut_at: Optional[datetime] = None
        self.tz_name = tz_name
        self._clock = clock
        self._lock = asyncio.Lock()
        self._tasks: List[asyncio.Task] = []
        self.last_tick: Dict[str, Any] = {}

    # -- lifecycle ------------------------------------------------------------------------------

    async def start(self) -> None:
        if not self.enabled:
            logger.info("session_roundup_loop_disabled")
            return
        self._tasks.append(asyncio.create_task(self._tick_loop(), name="session_roundup_tick"))
        logger.info("session_roundup_loop_started interval=%ss tick=%ss", self.interval_sec, self.tick_sec)

    async def stop(self) -> None:
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    async def _tick_loop(self) -> None:
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 -- one bad tick must not stop the loop
                logger.exception("session_roundup_tick_failed")
            await asyncio.sleep(self.tick_sec)

    # -- intake ---------------------------------------------------------------------------------

    async def intake(self, *, session_id: str, title: str, body: str, spec: Sequence[Dict[str, str]] = (),
                     session_name: Optional[str] = None, worktree: Optional[str] = None, branch: Optional[str] = None,
                     pr_number: Optional[int] = None, pr_url: Optional[str] = None,
                     thread_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Store one session message with Hub's receipt and show both. None: ``thread_id`` unknown.
        Raises ``ValueError`` for a message with nothing in it. No GPU, no model."""
        pool = self._pool_provider()
        if pool is None:
            raise RuntimeError("postgres_pool_unavailable")
        notes: List[str] = []
        body_text, body_cut = core.cap_text((body or "").strip(), core.BODY_MAX_CHARS, what="message")
        if body_cut:
            notes.append(f"Hub cut the message to {core.BODY_MAX_CHARS} characters.")
        specs: List[Dict[str, Any]] = []
        for item in list(spec or [])[: core.MAX_SPECS]:
            text = str((item or {}).get("text") or "").strip()
            path = core.one_line((item or {}).get("path"), 500)
            if not text:
                continue
            text, cut = core.cap_spec(text, core.SPEC_MAX_CHARS, path=path, branch=core.one_line(branch, 300))
            if cut:
                notes.append(f"Hub cut the spec {path} to at most {core.SPEC_MAX_CHARS} characters "
                             "(its last line says where, and where the full text is).")
            specs.append({"path": path, "text": text, "cut": cut})
        if len(list(spec or [])) > core.MAX_SPECS:
            notes.append(f"Hub kept the first {core.MAX_SPECS} spec files and dropped the rest.")
        if not body_text and not specs:
            raise IntakeRefused("empty_message")
        title = core.one_line(title, core.TITLE_MAX_CHARS)
        if not title:
            raise IntakeRefused("empty_title")
        source_ref = f"pr:{int(pr_number)}" if pr_number else f"note:{_new_id()}"
        digest = message_digest(body_text, specs)
        before = await intake_precheck(pool, thread_id=thread_id, source_ref=source_ref)
        existing = before["thread"]
        if thread_id and existing is None:
            return None
        now = self._clock()
        next_at = core.next_roundup_at(now, self.interval_sec)
        if existing is not None and before["last_digest"] == digest:
            # The same message again (a hook that fired twice, a retried send): nothing new to read.
            text = (f"Hub already has this exact message as the newest one on thread {existing['thread_id']} "
                    f"(status: {existing['status']}). Nothing new was stored.")
            return {"thread": existing, "turn": None, "receipt": {"turn_id": None, "body": text}, "duplicate": True,
                    "next_roundup_at": next_at, "roundup_enabled": self.enabled}
        if existing is None and before["held_threads"] >= core.MAX_HELD_THREADS:
            raise IntakeRefused(f"too_many_held_threads:{before['held_threads']}", too_many=True)
        if before["unread"] >= core.MAX_UNREAD_TURNS_PER_THREAD:
            raise IntakeRefused(f"too_many_unread_messages_on_thread:{before['unread']}", too_many=True)
        kind = core.KIND_PR_SPEC if pr_number else core.KIND_NOTE
        result = await add_message(
            pool, thread_id=thread_id, source_ref=source_ref,
            kind=kind, title=title, session_id=core.one_line(session_id, 128) or "unknown",
            session_name=core.one_line(session_name, 200) or None, worktree=core.one_line(worktree, 500) or None,
            branch=core.one_line(branch, 300) or None, pr_number=int(pr_number) if pr_number else None,
            pr_url=core.one_line(pr_url, 300) or None, spec_paths=[s["path"] for s in specs], body=body_text,
            meta={"spec": specs, "digest": digest} if specs else {"digest": digest},
            receipt=lambda tid: core.receipt_text(thread_id=tid, next_at=next_at, roundup_enabled=self.enabled,
                                                  cut_notes=notes, tz_name=self.tz_name,
                                                  answer_phase=bool(before["answer_phase"]),
                                                  reading_now=bool(existing and existing["status"] == core.STATUS_READING)))
        if result is None:
            return None
        thread = result["thread"]
        self._push_turn(thread, result["turn"])
        self._push_turn(thread, result["receipt"])
        if existing is not None and existing["session_id"] != thread["session_id"]:
            logger.info("session_thread_sender_changed thread=%s from=%s to=%s", thread["thread_id"],
                        existing["session_id"], thread["session_id"])
        logger.info("session_thread_message thread=%s kind=%s session=%s pr=%s specs=%s", thread["thread_id"],
                    thread["kind"], thread["session_id"], thread.get("pr_number"), len(specs))
        return {**result, "duplicate": False, "next_roundup_at": next_at, "roundup_enabled": self.enabled,
                "answer_phase": bool(before["answer_phase"])}

    # -- the tick -------------------------------------------------------------------------------

    async def tick(self) -> Dict[str, Any]:
        """Advance the round in flight, or start this hour's round. One thing per tick."""
        pool = self._pool_provider()
        if pool is None:
            self.last_tick = {"at": self._clock().isoformat(), "skipped": "no_pool"}
            return self.last_tick
        if self._lock.locked():
            self.last_tick = {"at": self._clock().isoformat(), "skipped": "busy"}
            return self.last_tick
        async with self._lock:
            result = await self._tick(pool)
        self.last_tick = {"at": self._clock().isoformat(), **result}
        return self.last_tick

    async def _tick(self, pool) -> Dict[str, Any]:
        if self._budget_cut_at is not None and (self._clock() - self._budget_cut_at).total_seconds() >= BUDGET_RESTORE_SEC:
            logger.info("session_roundup_budget_restored budget=%s", self.prompt_max_chars)
            self.budget_chars, self._budget_cut_at = self.prompt_max_chars, None
        reading = await threads_with_status(pool, core.STATUS_READING)
        if reading:
            # One round at a time: the next hour's round waits for this one to end.
            roundup_id = sorted({t["last_roundup_id"] or "" for t in reading})[0]
            return await self._poll(pool, roundup_id)
        held = await threads_with_status(pool, core.STATUS_HELD)
        if not held:
            return {"skipped": "nothing_held"}
        turns = await thread_turns(pool, [t["thread_id"] for t in held])
        # The clock is taken AFTER the read: a message stored while the read ran is newer than a
        # clock taken before it, and "arrived before this hour started" would be judged on a stale now.
        now = self._clock()
        start = core.bucket_start(now, self.interval_sec)
        due = [c for c in self._candidates(held, turns) if c[1] and c[1][0]["created_at"] < start]
        if not due:
            return {"skipped": "nothing_due", "held": len(held), "next_roundup_at": core.next_roundup_at(now, self.interval_sec).isoformat()}
        from orion.schemas.session_roundup_run import session_roundup_run_id

        roundup_id = session_roundup_run_id(start)
        if await roundup_exists(pool, roundup_id):
            return {"skipped": "roundup_already_ran_this_hour", "roundup_id": roundup_id, "due": len(due)}
        due.sort(key=lambda c: (c[1][0]["created_at"], c[0]["thread_id"]))
        selection = core.select_for_prompt(due, budget_chars=self.budget_chars, nonce=fence_nonce(roundup_id))
        if not selection.sections:
            logger.warning("session_roundup_nothing_fits roundup=%s due=%s budget=%s", roundup_id, len(due),
                           self.budget_chars)
            return {"skipped": "nothing_fits", "due": len(due)}
        if selection.left_held:
            logger.info("session_roundup_left_held roundup=%s threads=%s", roundup_id, ",".join(selection.left_held))
        chosen = set(selection.thread_ids)
        await mark_round(pool, roundup_id, selection.thread_ids,
                         [t["turn_id"] for c in due if c[0]["thread_id"] in chosen for t in c[1]])
        logger.info("session_roundup_started roundup=%s threads=%s left_held=%s", roundup_id, len(chosen),
                    len(selection.left_held))
        return {"started": roundup_id, "left_held": selection.left_held, **await self._submit(pool, roundup_id)}

    @staticmethod
    def _candidates(threads: Sequence[Dict[str, Any]], turns: Sequence[Dict[str, Any]], *,
                    roundup_id: Optional[str] = None):
        """(thread, new session turns, earlier turns) per thread. New = not yet shown to Orion, or
        (``roundup_id`` given) the turns of that round. Hub's own lines are never shown to Orion.

        The thread comes back as a copy with ``answer_phase`` set (``core.in_answer_phase`` over ALL its
        Orion turns except this round's own): the prompt and the parser both read that one flag."""
        out = []
        for thread in threads:
            # A re-queued uncut reply is never shown again: it holds other threads' sections.
            mine = [t for t in turns if t["thread_id"] == thread["thread_id"] and t["speaker"] != core.SPEAKER_HUB
                    and not (t.get("meta") or {}).get("requeued")]
            new = [t for t in mine if t["speaker"] == core.SPEAKER_SESSION and t.get("roundup_id") == roundup_id]
            first = min((t["seq"] for t in new), default=None)
            phase = core.in_answer_phase(mine, exclude_roundup=roundup_id)
            if phase and first is not None:
                # Orion's question turn may sit AFTER a new message (one sent while Orion was reading):
                # every turn that is not new is shown, so Orion always sees its own questions.
                new_ids = {t["turn_id"] for t in new}
                earlier = [t for t in mine if t["turn_id"] not in new_ids
                           and not (t["speaker"] == core.SPEAKER_ORION and t.get("roundup_id") == roundup_id
                                    and roundup_id is not None)]
            else:
                earlier = [t for t in mine if first is not None and t["seq"] < first]
            out.append(({**thread, "answer_phase": phase}, new, earlier))
        return out

    async def _with_pr_states(self, cands):
        """Each candidate's thread with ``pr_state`` set from GitHub when Hub can read it: the prompt tells
        Orion when a thread's pull request is closed or merged (Hub comments only on an open one). Only
        when PR comments are on (otherwise Orion's questions never reach a PR anyway). Best effort."""
        state_of = getattr(self._commenter, "pr_state", None)
        numbers = sorted({int(c[0]["pr_number"]) for c in cands if c[0].get("pr_number")})
        if not self.pr_comments or state_of is None or not numbers:
            return cands
        results = await asyncio.gather(*(state_of(n) for n in numbers), return_exceptions=True)
        states = {n: r for n, r in zip(numbers, results) if isinstance(r, str)}
        return [({**c[0], "pr_state": states.get(int(c[0]["pr_number"]))} if c[0].get("pr_number") else c[0], c[1], c[2])
                for c in cands]

    async def _round(self, pool, roundup_id: str):
        threads = [t for t in await threads_with_status(pool, core.STATUS_READING) if t["last_roundup_id"] == roundup_id]
        turns = await thread_turns(pool, [t["thread_id"] for t in threads]) if threads else []
        cands = [c for c in self._candidates(threads, turns, roundup_id=roundup_id) if c[1]]
        cands.sort(key=lambda c: (c[1][0]["created_at"], c[0]["thread_id"]))
        return threads, cands

    async def _submit(self, pool, roundup_id: str) -> Dict[str, Any]:
        """Submit the round's one durable run. The prompt is built from what ``mark_round`` stored, plus
        the threads' PR states and this process's budget, so a resubmit after a restart may differ; a 409
        then adopts the run durable-runs already has. Never a retry queue here: durable-runs down is
        simply tried again next tick."""
        from orion.schemas.durable_run import DurableRunRequestV1
        from orion.schemas.resource_admission import ResourceRequirementV1
        from orion.schemas.session_roundup_run import (
            SESSION_ROUNDUP_WORKFLOW, SessionRoundupBriefV1, session_roundup_session_id,
        )

        threads, cands = await self._round(pool, roundup_id)
        if not cands:
            if threads:   # marked, but no turn of the round is left: nothing to ask Orion
                await release_threads(pool, roundup_id, [t["thread_id"] for t in threads])
            return {"skipped": "round_has_no_turns", "roundup_id": roundup_id}
        nonce = fence_nonce(roundup_id)
        cands = await self._with_pr_states(cands)
        selection = core.select_for_prompt(cands, budget_chars=self.budget_chars, nonce=nonce)
        if selection.left_held:
            await release_threads(pool, roundup_id, selection.left_held)
        if not selection.sections:
            return {"skipped": "nothing_fits", "roundup_id": roundup_id}
        titles = "; ".join(core.one_line(c[0].get("title"), 120) for c in cands if c[0]["thread_id"] in selection.thread_ids)
        try:
            bucket = datetime.strptime(roundup_id.rsplit("-", 1)[1], "%Y%m%dT%H%M").replace(tzinfo=timezone.utc)
            request = DurableRunRequestV1(
                run_id=roundup_id, workflow=SESSION_ROUNDUP_WORKFLOW,
                correlation_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"orion:session_roundup:{roundup_id}")),
                brief=SessionRoundupBriefV1(
                    roundup="session_roundup.orion_turn", roundup_id=roundup_id, thread_ids=selection.thread_ids,
                    prompt=core.roundup_prompt(selection.sections, nonce=nonce),
                    session_id=session_roundup_session_id(roundup_id), retrieval_query=titles[:1000] or None,
                    timeout_sec=self.orion_timeout_sec,
                ),
                # The deadline bounds one round: a run still waiting or retrying then is failed by
                # durable-runs, and the threads go back to held. Fixed per bucket, so a resubmit is identical.
                admission=ResourceRequirementV1(preferred_lane="agent", resource="llm.route.agent", priority="background",
                                                deadline_at=bucket + timedelta(seconds=self.run_deadline_sec)),
            )
        except Exception as exc:  # noqa: BLE001 -- a brief that cannot be built would jam every tick
            logger.warning("session_roundup_brief_invalid roundup=%s err=%s", roundup_id, str(exc)[:300])
            return await self._fail(pool, roundup_id, f"brief_invalid:{type(exc).__name__}")
        try:
            code, body = await self._durable.submit(request.model_dump(mode="json"))
        except Exception as exc:  # noqa: BLE001 -- durable-runs down: the next tick submits again
            logger.warning("session_roundup_submit_failed roundup=%s err=%s", roundup_id, exc)
            return {"waiting": roundup_id, "for": "durable_runs"}
        if code == 409:
            # The run already exists (a restart between submit and the read): adopt it.
            try:
                existing = await self._durable.get_run(roundup_id)
            except Exception:  # noqa: BLE001
                existing = None
            if existing is None:
                # Refused as a different request, and the run could not be read: ask again next tick
                # rather than fail a round whose run may be live.
                return {"waiting": roundup_id, "for": "durable_runs:conflict_unread"}
            logger.info("session_roundup_run_adopted roundup=%s", roundup_id)
            code = 202
        if code >= 500:
            logger.warning("session_roundup_submit_error roundup=%s http=%s", roundup_id, code)
            return {"waiting": roundup_id, "for": f"durable_runs:http_{code}"}
        if code >= 400:
            logger.warning("session_roundup_run_refused roundup=%s http=%s body=%s", roundup_id, code, body)
            return await self._fail(pool, roundup_id, f"orion_run_refused:http_{code}")
        logger.info("session_roundup_run_submitted roundup=%s threads=%s", roundup_id, len(selection.sections))
        return await self._poll(pool, roundup_id, resubmit=False)

    async def _poll(self, pool, roundup_id: str, *, resubmit: bool = True) -> Dict[str, Any]:
        """Read the round's durable run. Completed with text: Orion's turn per thread. Anything
        else that is final fails the round."""
        try:
            run = await self._durable.get_run(roundup_id)
        except Exception as exc:  # noqa: BLE001 -- unreachable: ask again next tick
            return {"waiting": roundup_id, "for": f"durable_runs:{type(exc).__name__}"}
        if run is None:
            if resubmit:   # marked but never submitted (a restart in between): submit it now
                return await self._submit(pool, roundup_id)
            return {"waiting": roundup_id, "for": "orion_run_not_visible_yet"}
        status = str(run.get("status") or "")
        if status in ("failed", "cancelled", "abandoned"):
            error = core.one_line(run.get("error") or status, 200)
            if error == "workflow_deadline" and run.get("harness_turn_meta"):
                # The deadline came after Orion's turn was attempted: the round's failure, not the body's.
                error = "workflow_deadline_after_turn"
            return await self._fail(pool, roundup_id, f"orion_run_{status}:{error}")
        if status != "completed":
            # durable-runs fails a run at its admission deadline. A run still not ended well past it
            # (paused, or a state Hub does not know) must not keep its threads in `reading`, which
            # would block every later round: Hub stops waiting and holds the threads again.
            overdue = self._overdue_sec(roundup_id)
            if overdue is not None and overdue > 0:
                return await self._fail(pool, roundup_id, f"orion_run_overdue:{status or 'unknown'}")
            return {"waiting": roundup_id, "for": f"orion_run:{status or 'unknown'}"}
        from orion.harness.cut_short import CUT_SHORT_CODES
        from orion.schemas.session_roundup_run import SESSION_ROUNDUP_STEP_KEY

        step = (run.get("harness_steps") or {}).get(SESSION_ROUNDUP_STEP_KEY) or {}
        text = str(step.get("text") or "").strip()
        if not text:
            return await self._fail(pool, roundup_id, "orion_run_completed_empty")
        debug = step.get("debug") or {}
        if step.get("cut_short") or debug.get("harness_grounding_status") in CUT_SHORT_CODES or _CUT_SHORT_RE.match(text):
            return await self._fail(pool, roundup_id, "orion_turn_cut_short")
        return await self._apply(pool, roundup_id, text, model=debug.get("fcc_model_label"))

    def _overdue_sec(self, roundup_id: str) -> Optional[float]:
        """Seconds past the round's deadline plus grace (negative: not yet). None: id not a bucket."""
        try:
            bucket = datetime.strptime(roundup_id.rsplit("-", 1)[1], "%Y%m%dT%H%M").replace(tzinfo=timezone.utc)
        except (IndexError, ValueError):
            return None
        return (self._clock() - bucket).total_seconds() - self.run_deadline_sec - OVERDUE_GRACE_SEC

    async def _apply(self, pool, roundup_id: str, text: str, *, model: Optional[str]) -> Dict[str, Any]:
        threads, cands = await self._round(pool, roundup_id)
        if not threads:
            return {"skipped": "round_already_applied", "roundup_id": roundup_id}
        # The prompt's order (the parser trusts headings only in that order), then any thread of the
        # round that has no turn left to show.
        ordered = [c[0]["thread_id"] for c in cands]
        ordered += [t["thread_id"] for t in threads if t["thread_id"] not in ordered]
        # A thread of the round with no turn left to show has no candidate: judge it on all its turns.
        phase = {c[0]["thread_id"]: c[0]["answer_phase"] for c in cands}
        missing = [tid for tid in ordered if tid not in phase]
        if missing:
            all_turns = await thread_turns(pool, missing)
            for tid in missing:
                phase[tid] = core.in_answer_phase([t for t in all_turns if t["thread_id"] == tid],
                                                  exclude_roundup=roundup_id)
        try:
            answers = core.parse_roundup_reply(text, ordered, [tid for tid in ordered if phase.get(tid)])
        except ValueError:
            return await self._fail(pool, roundup_id, "orion_run_completed_empty")
        by_id = {t["thread_id"]: t for t in threads}
        counts = {"asked": 0, "no_questions": 0, "unparsed": 0, "closed": 0, "dropped_question_lines": 0,
                  "pr_comments": 0}
        next_at = core.next_roundup_at(self._clock(), self.interval_sec)
        for answer in answers:
            meta: Dict[str, Any] = {"model": model} if model else {}
            if answer.unparsed and (not answer.closing or answer.shared):
                # A parse accident is not Orion asking: keep the words, send them nowhere, read again. A
                # reply shared by several threads holds the others' sections (and session text Orion quoted
                # in them): it is NOT stored on this thread, only on the round's durable run.
                kept = not answer.shared
                thread, turn, hub, created = await requeue_unparsed(
                    pool, thread_id=answer.thread_id, roundup_id=roundup_id, body=answer.text if kept else None,
                    meta=meta, reason=UNPARSED_REASON,
                    text_for=lambda n, kept=kept: core.unparsed_text(failures=n, next_at=next_at, tz_name=self.tz_name,
                                                                     kept=kept, roundup_id=roundup_id))
                counts["unparsed"] += 1
                if created and turn is not None:
                    self._push_turn(thread, turn)
                if hub is not None:
                    self._push_turn(thread, hub)
                logger.info("session_thread_reply_unparsed thread=%s roundup=%s failures=%s status=%s stored=%s",
                            answer.thread_id, roundup_id, thread.get("roundup_failures"), thread.get("status"), kept)
                continue
            if answer.unparsed:
                meta["unparsed"] = True
            if answer.no_questions:
                meta["no_questions"] = True
            if answer.closing:
                meta["closing_note"] = True
                meta["dropped_question_lines"] = answer.dropped_question_lines
                status = core.STATUS_CLOSED
            else:
                status = core.STATUS_NO_QUESTIONS if answer.no_questions else core.STATUS_ASKED
            thread, turn, created = await store_answer(pool, thread_id=answer.thread_id, roundup_id=roundup_id,
                                                       body=answer.text, meta=meta, status=status)
            if answer.closing:
                counts["closed"] += 1
                counts["dropped_question_lines"] += answer.dropped_question_lines
                if answer.dropped_question_lines:
                    logger.info("session_thread_questions_dropped thread=%s roundup=%s lines=%s why=answer_phase",
                                answer.thread_id, roundup_id, answer.dropped_question_lines)
            else:
                counts["unparsed" if answer.unparsed else status] += 1
            if created:
                self._push_turn(thread, turn)
            if await self._comment(pool, by_id[answer.thread_id], turn, answer):
                counts["pr_comments"] += 1
        logger.info("session_roundup_applied roundup=%s %s", roundup_id, counts)
        return {"applied": roundup_id, **counts}

    async def _comment(self, pool, thread: Dict[str, Any], turn: Dict[str, Any], answer: core.ThreadAnswer) -> bool:
        """One PR comment for real, parsed questions. Never fails the round."""
        if (not self.pr_comments or self._commenter is None or not thread.get("pr_number")
                or answer.no_questions or answer.unparsed or answer.closing or turn.get("pr_comment_url")):
            return False
        try:
            url = await self._commenter.post(
                int(thread["pr_number"]), core.pr_comment_body(thread_id=thread["thread_id"], questions=answer.text),
                branch=thread.get("branch"))
            if not url:
                return False
            await set_pr_comment_url(pool, turn["turn_id"], url)
            logger.info("session_thread_pr_comment thread=%s pr=%s url=%s", thread["thread_id"], thread["pr_number"], url)
            return True
        except Exception as exc:  # noqa: BLE001 -- the turn is stored; a comment is extra
            logger.warning("session_thread_pr_comment_error thread=%s err=%s", thread["thread_id"], exc)
            return False

    async def _fail(self, pool, roundup_id: str, reason: str) -> Dict[str, Any]:
        next_at = core.next_roundup_at(self._clock(), self.interval_sec)
        strike = not core.is_body_refusal(reason)
        refusal_text_for = lambda n: core.body_refusal_text(reason=reason, next_at=next_at, tz_name=self.tz_name)  # noqa: E731
        if core.is_sizing_refusal(reason) and self.budget_chars > MIN_PROMPT_CHARS:
            # Hub built a prompt too big for the lane: Hub's sizing, not the thread's. Shrink and retry; a
            # refusal at the floor budget is a strike (Hub cannot shrink further).
            self.budget_chars = core.shrunk_budget(self.budget_chars, reason, floor=MIN_PROMPT_CHARS)
            self._budget_cut_at = self._clock()
            logger.warning("session_roundup_budget_shrunk roundup=%s budget=%s", roundup_id, self.budget_chars)
            strike = False
            budget = self.budget_chars
            refusal_text_for = lambda n: core.sizing_refusal_text(  # noqa: E731
                reason=reason, budget_chars=budget, next_at=next_at, tz_name=self.tz_name)
        failed = await fail_round(
            pool, roundup_id, reason=reason, strike=strike,
            text_for=lambda n: core.failure_text(reason=reason, failures=n, next_at=next_at, tz_name=self.tz_name),
            refusal_text_for=refusal_text_for)
        for thread, turn in failed:
            self._push_turn(thread, turn)
        logger.warning("session_roundup_failed roundup=%s reason=%s strike=%s lines=%s", roundup_id, reason,
                       strike, len(failed))
        return {"failed": roundup_id, "why": reason, "strike": strike, "threads": len(failed)}

    # -- Hub chat -------------------------------------------------------------------------------

    def _push_turn(self, thread: Dict[str, Any], turn: Dict[str, Any]) -> None:
        created = turn.get("created_at")
        self._push({
            "kind": FRAME_KIND, "thread_id": thread["thread_id"], "turn_id": turn["turn_id"],
            "speaker": turn["speaker"], "text": turn["body"], "title": thread.get("title"),
            "status": thread.get("status"), "session_name": thread.get("session_name") or thread.get("session_id"),
            "pr_number": thread.get("pr_number"), "meta": {k: v for k, v in (turn.get("meta") or {}).items() if k != "spec"},
            "spec_paths": [s.get("path") for s in ((turn.get("meta") or {}).get("spec") or [])],
            "created_at": created.isoformat() if hasattr(created, "isoformat") else created,
        })

    def _push(self, payload: Dict[str, Any]) -> None:
        """Every live chat window sees a session thread (it is not one chat session's)."""
        for connection_id, entry in list((self._connections_provider() or {}).items()):
            queue = entry.get("queue") if isinstance(entry, dict) else None
            if queue is None:
                continue
            try:
                queue.put_nowait(dict(payload))
            except Exception as exc:  # noqa: BLE001
                logger.warning("session_thread_push_failed connection=%s err=%s", connection_id, exc)

    def status(self) -> Dict[str, Any]:
        now = self._clock()
        return {"enabled": self.enabled, "interval_sec": self.interval_sec, "pr_comments": self.pr_comments,
                "prompt_max_chars": self.prompt_max_chars, "budget_chars": self.budget_chars,
                "next_roundup_at": core.next_roundup_at(now, self.interval_sec).isoformat(), "last_tick": self.last_tick}
