"""Curiosity focus: one question Orion chose in a chat with Juniper (or, since 2026-10-30, alone
in Prime's between-conversations thinking), offered once to the next regular investigate run.

Design: docs/superpowers/specs/2026-10-16-orion-queue-curiosity-focus-design.md. Table:
services/orion-sql-db/manual_migration_curiosity_focus_request_v1.sql.

Lifecycle, all Hub:

1. Queue (chat turn, the `queue_curiosity_focus` introspect tool): one slot. A new focus REPLACES
   the pending one and the reply names what it replaced -- except that a focus Juniper asked for
   (origin "juniper") is never replaced by Orion's own (origin "self") choice.
2. Take (investigate kickoff, after every gate): the pending row becomes "taken" by that run.
   RELEASE gives it back only when the run was cancelled before Orion saw the prompt.
3. Outcome (turn end): "ran" or "failed", plus the disposition the run wrote on its
   ``:TurnOutcome`` (``focus_disposition``): "explored" or "set_aside:<reason>". No disposition
   is unknown, never explored. A failed run never re-queues the focus.

The run is SHOWN the focus, not forced: it goes through every normal gate, cooldown and cap.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional, Sequence
from uuid import NAMESPACE_URL, UUID, uuid5

logger = logging.getLogger("orion.curiosity.focus")

MIGRATION = "services/orion-sql-db/manual_migration_curiosity_focus_request_v1.sql"
# The ``refs`` column (Orion Prime step 1, 2026-10-30): ids the question cites, each openable with
# ``python -P -m orion.dig <kind> <id>`` (orion/refs.py). NULL on rows written before it and on any
# question that cites nothing.
REFS_MIGRATION = "services/orion-sql-db/manual_migration_curiosity_focus_request_refs_v1.sql"

# How long a pending focus waits for a run. Measured 2026-10-16 over the 30 days before
# (206 regular investigate runs, durable_admission_runs): the line runs ~7 times right after
# local midnight (daily cap 7, 30 min cooldown), so the longest gap between two runs was 21.1 h
# and no day was skipped. 48 h covers that worst wait plus one whole missed day.
FOCUS_TTL = timedelta(hours=48)

ORIGIN_SELF = "self"
ORIGIN_JUNIPER = "juniper"

PENDING, REPLACED, CLEARED, EXPIRED, TAKEN, RAN, FAILED = (
    "pending", "replaced", "cleared", "expired", "taken", "ran", "failed",
)
DISPOSITION_EXPLORED = "explored"
DISPOSITION_SET_ASIDE = "set_aside"
DISPOSITION_CAP = 300

JUNIPER_PROTECTED = (
    "Juniper asked for this focus; it stays until a run takes it or it expires"
)
PENDING_KEPT = "a focus is already waiting for the next run; it is not replaced by a question chosen alone"

_ID_NS = uuid5(NAMESPACE_URL, "orion.curiosity.focus.v1")
# Serializes the multi-statement writers (queue, release, clear) so two of them cannot both
# see "no pending row" and collide on the one-slot index.
LOCK_KEY = 0x0C0F0C05


def focus_id_for(question: str, why: str, origin: str, chat_correlation_id: str | None) -> UUID:
    """Stable id: a retried tool call returns the same row, not a second queue."""
    key = "\x1f".join((question, why, origin, chat_correlation_id or ""))
    return uuid5(_ID_NS, key)


_COLS = (
    "focus_id, question, why, origin, chat_correlation_id, created_at, expires_at, status, "
    "status_at, replaced_by, offered_at, offered_run_id, outcome_stamped_at, turn_ok, "
    "disposition, disposition_raw, refs"
)
# What a reader should call the row now: a pending row past its expiry is expired even before
# the sweep marks it.
_EFFECTIVE = (
    "CASE WHEN status = 'pending' AND expires_at <= now() THEN 'expired' ELSE status END "
    "AS effective_status"
)

LOCK_SQL = "SELECT pg_advisory_xact_lock($1)"

EXPIRE_SQL = """
UPDATE curiosity_focus_request
   SET status = 'expired', status_at = now()
 WHERE status = 'pending' AND expires_at <= now()
"""

BY_ID_SQL = f"SELECT {_COLS}, {_EFFECTIVE} FROM curiosity_focus_request WHERE focus_id = $1"

PENDING_FOR_UPDATE_SQL = (
    f"SELECT {_COLS}, {_EFFECTIVE} FROM curiosity_focus_request WHERE status = 'pending' FOR UPDATE"
)

REPLACE_SQL = """
UPDATE curiosity_focus_request
   SET status = 'replaced', status_at = now(), replaced_by = $2
 WHERE focus_id = $1 AND status = 'pending'
"""

INSERT_SQL = f"""
INSERT INTO curiosity_focus_request (focus_id, question, why, origin, chat_correlation_id, expires_at, refs)
VALUES ($1, $2, $3, $4, $5, now() + make_interval(secs => $6::double precision), $7::jsonb)
RETURNING {_COLS}, 'pending' AS effective_status
"""

# A focus taken by a run whose end was never recorded (admission never confirmed, Hub died
# mid-turn, an error before dispatch) would read "taken" forever. Past this age it is marked
# failed with turn_ok NULL ("end never recorded"); a late turn end can still stamp it ran.
STALE_TAKEN_AFTER = timedelta(hours=48)

STALE_SQL = """
UPDATE curiosity_focus_request
   SET status = 'failed', status_at = now(), outcome_stamped_at = now()
 WHERE status = 'taken' AND outcome_stamped_at IS NULL
   AND offered_at < now() - make_interval(secs => $1::double precision)
"""

TAKE_SQL = f"""
UPDATE curiosity_focus_request
   SET status = 'taken', status_at = now(), offered_at = now(), offered_run_id = $1
 WHERE focus_id IN (
        SELECT focus_id FROM curiosity_focus_request
         WHERE status = 'pending' AND expires_at > now()
         FOR UPDATE SKIP LOCKED)
RETURNING {_COLS}, 'taken' AS effective_status
"""

# Only a row whose outcome was never stamped: once the turn ended, that run saw it. If Orion
# queued a newer focus meanwhile, the newer decision holds the slot and this one reads replaced.
RELEASE_SQL = """
UPDATE curiosity_focus_request f
   SET status = CASE WHEN p.focus_id IS NULL THEN 'pending' ELSE 'replaced' END,
       replaced_by = p.focus_id,
       status_at = now(), offered_at = NULL, offered_run_id = NULL
  FROM (SELECT (SELECT focus_id FROM curiosity_focus_request WHERE status = 'pending' LIMIT 1) AS focus_id) p
 WHERE f.offered_run_id = $1 AND f.status = 'taken' AND f.outcome_stamped_at IS NULL
RETURNING f.focus_id, f.status
"""

# Turn end. A later failing attempt never downgrades a run that already finished with text.
STAMP_SQL = """
UPDATE curiosity_focus_request
   SET status = CASE WHEN $2 THEN 'ran' ELSE 'failed' END,
       status_at = now(), turn_ok = $2,
       disposition = CASE WHEN $5 THEN $3 ELSE disposition END,
       disposition_raw = CASE WHEN $5 THEN $4 ELSE disposition_raw END,
       outcome_stamped_at = now()
 WHERE offered_run_id = $1 AND status IN ('taken', 'ran', 'failed')
   AND NOT (status = 'ran' AND NOT $2)
RETURNING focus_id
"""

# The durable run ended failed / abandoned / cancelled after its turn started, and no turn end
# stamped the row: it failed. Never back to pending.
FAIL_SQL = """
UPDATE curiosity_focus_request
   SET status = 'failed', status_at = now(), turn_ok = false, outcome_stamped_at = now()
 WHERE offered_run_id = $1 AND status = 'taken'
RETURNING focus_id
"""

CLEAR_SQL = """
UPDATE curiosity_focus_request
   SET status = 'cleared', status_at = now()
 WHERE focus_id = $1 AND status = 'pending' AND expires_at > now()
RETURNING focus_id
"""

PENDING_SQL = (
    f"SELECT {_COLS}, {_EFFECTIVE} FROM curiosity_focus_request "
    "WHERE status = 'pending' AND expires_at > now()"
)

RECENT_SQL = (
    f"SELECT {_COLS}, {_EFFECTIVE}, count(*) OVER () AS total FROM curiosity_focus_request "
    "WHERE ($1::timestamptz IS NULL OR created_at >= $1) "
    "ORDER BY created_at DESC, focus_id DESC LIMIT $2"
)

FOR_RUNS_SQL = (
    f"SELECT {_COLS}, {_EFFECTIVE} FROM curiosity_focus_request WHERE offered_run_id = ANY($1::text[])"
)


class FocusRefused(ValueError):
    """The focus was not queued; the message says why in plain words."""


@dataclass(frozen=True)
class FocusRow:
    focus_id: str
    question: str
    why: str
    origin: str
    status: str
    created_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    status_at: Optional[datetime] = None
    chat_correlation_id: Optional[str] = None
    replaced_by: Optional[str] = None
    offered_at: Optional[datetime] = None
    offered_run_id: Optional[str] = None
    outcome_stamped_at: Optional[datetime] = None
    turn_ok: Optional[bool] = None
    disposition: Optional[str] = None
    disposition_raw: Optional[str] = None
    # [{"kind", "id"}, ...] as stored (orion/refs.py RefV1 shape); empty when none.
    refs: tuple[dict[str, str], ...] = ()


def _refs_value(raw: Any) -> tuple[dict[str, str], ...]:
    """The stored refs column (asyncpg hands jsonb back as text). Unreadable is empty, never a crash."""
    if raw is None:
        return ()
    if isinstance(raw, (str, bytes)):
        try:
            raw = json.loads(raw)
        except ValueError:
            return ()
    if not isinstance(raw, list):
        return ()
    return tuple({"kind": str(r["kind"]), "id": str(r["id"])} for r in raw
                 if isinstance(r, dict) and r.get("kind") and r.get("id"))


def _get(r: Any, key: str) -> Any:
    try:
        return r[key]
    except (KeyError, IndexError):
        return None


def _row(r: Any) -> FocusRow:
    return FocusRow(
        focus_id=str(r["focus_id"]),
        question=str(r["question"] or ""),
        why=str(r["why"] or ""),
        origin=str(r["origin"] or ""),
        status=str(r["effective_status"] or r["status"]),
        created_at=r["created_at"],
        expires_at=r["expires_at"],
        status_at=r["status_at"],
        chat_correlation_id=r["chat_correlation_id"],
        replaced_by=str(r["replaced_by"]) if r["replaced_by"] is not None else None,
        offered_at=r["offered_at"],
        offered_run_id=r["offered_run_id"],
        outcome_stamped_at=r["outcome_stamped_at"],
        turn_ok=r["turn_ok"],
        disposition=r["disposition"],
        disposition_raw=r["disposition_raw"],
        refs=_refs_value(_get(r, "refs")),
    )


@dataclass(frozen=True)
class QueueResult:
    focus: FocusRow
    replaced: Optional[FocusRow]
    already_queued: bool


async def queue_focus(
    pool: Any, *, question: str, why: str, origin: str, chat_correlation_id: Optional[str],
    ttl: timedelta = FOCUS_TTL, keep_pending: bool = False, refs: Sequence[dict[str, str]] = (),
) -> QueueResult:
    """Put ``question`` in the one pending slot. Raises FocusRefused (nothing written) when a
    Juniper-origin focus holds the slot and this is Orion's own choice, or, with ``keep_pending``
    (Prime's between-conversations questions), when ANY focus is pending: a question chosen in a
    conversation outranks one chosen alone. Database errors raise: the caller cannot know whether
    the commit happened."""
    if origin not in (ORIGIN_SELF, ORIGIN_JUNIPER):
        raise FocusRefused(f"origin must be 'self' or 'juniper', not {origin!r}")
    focus_id = focus_id_for(question, why, origin, chat_correlation_id)
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(LOCK_SQL, LOCK_KEY)
            existing = await conn.fetchrow(BY_ID_SQL, focus_id)
            if existing is not None:
                if existing["effective_status"] in (REPLACED, EXPIRED, CLEARED):
                    # The same words from the same chat turn, but that focus is no longer queued:
                    # an ok=true here would let Orion say "queued" when nothing is.
                    raise FocusRefused(
                        f"this exact focus was queued earlier in this conversation and is now "
                        f"{existing['effective_status']}; nothing was queued. Reword it to queue it again"
                    )
                return QueueResult(focus=_row(existing), replaced=None, already_queued=True)
            await conn.execute(EXPIRE_SQL)
            current = await conn.fetchrow(PENDING_FOR_UPDATE_SQL)
            if current is not None and current["origin"] == ORIGIN_JUNIPER and origin == ORIGIN_SELF:
                # Raising inside the transaction rolls back the expiry sweep too: nothing written.
                raise FocusRefused(JUNIPER_PROTECTED + f" (pending: {str(current['question'])[:200]!r})")
            if current is not None and keep_pending:
                raise FocusRefused(PENDING_KEPT + f" (pending: {str(current['question'])[:200]!r})")
            if current is not None:
                await conn.execute(REPLACE_SQL, current["focus_id"], focus_id)
            inserted = await conn.fetchrow(
                INSERT_SQL, focus_id, question, why, origin, chat_correlation_id, ttl.total_seconds(),
                json.dumps([{"kind": r["kind"], "id": r["id"]} for r in refs]) if refs else None,
            )
    replaced = None
    if current is not None:
        replaced = _row({**dict(current), "status": REPLACED, "effective_status": REPLACED,
                         "replaced_by": focus_id})
    return QueueResult(focus=_row(inserted), replaced=replaced, already_queued=False)


async def take_focus(pool: Any, *, run_id: str) -> Optional[FocusRow]:
    """Claim the pending focus for one run. None when there is none, or on any failure: a
    missing table (migration not applied) or a down pool must not break the kickoff."""
    if pool is None or not run_id:
        return None
    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(EXPIRE_SQL)
                await conn.execute(STALE_SQL, STALE_TAKEN_AFTER.total_seconds())
                rows = await conn.fetch(TAKE_SQL, run_id)
    except Exception as exc:  # noqa: BLE001 -- see docstring
        logger.warning("curiosity_focus_take_failed run=%s err=%s", run_id, exc)
        return None
    return _row(rows[0]) if rows else None


async def release_focus(pool: Any, *, run_id: str) -> Optional[str]:
    """Give a run's focus back when the run was cancelled before Orion saw it. Returns the new
    status ('pending' or 'replaced') or None. Never raises."""
    if pool is None or not run_id:
        return None
    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(LOCK_SQL, LOCK_KEY)
                rows = await conn.fetch(RELEASE_SQL, run_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("curiosity_focus_release_failed run=%s err=%s", run_id, exc)
        return None
    return str(rows[0]["status"]) if rows else None


async def stamp_focus_outcome(
    pool: Any, *, run_id: str, turn_ok: bool, disposition: Optional[str], disposition_raw: Optional[str],
    graph_read: bool = True,
) -> Optional[str]:
    """Turn end for the run that took a focus. Returns the focus_id stamped, or None (no focus
    for this run, or a failure -- logged). Never raises. ``graph_read`` False: the disposition
    could not be read this time, so a disposition an earlier attempt recorded is kept."""
    if pool is None or not run_id:
        return None
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(STAMP_SQL, run_id, bool(turn_ok), disposition, disposition_raw, bool(graph_read))
    except Exception as exc:  # noqa: BLE001
        logger.warning("curiosity_focus_stamp_failed run=%s err=%s", run_id, exc)
        return None
    return str(rows[0]["focus_id"]) if rows else None


async def fail_focus(pool: Any, *, run_id: str) -> Optional[str]:
    """The run ended without a turn end stamping its focus, after Orion may have seen it."""
    if pool is None or not run_id:
        return None
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(FAIL_SQL, run_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("curiosity_focus_fail_stamp_failed run=%s err=%s", run_id, exc)
        return None
    return str(rows[0]["focus_id"]) if rows else None


async def clear_focus(pool: Any, *, focus_id: UUID) -> bool:
    """Juniper's Clear button: a pending focus becomes 'cleared'. False when it was not pending."""
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(LOCK_SQL, LOCK_KEY)
            rows = await conn.fetch(CLEAR_SQL, focus_id)
    return bool(rows)


async def read_pending(pool: Any) -> Optional[FocusRow]:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(PENDING_SQL)
    return _row(row) if row is not None else None


async def read_recent(pool: Any, *, since: Optional[datetime], limit: int) -> tuple[list[FocusRow], int]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(RECENT_SQL, since, int(limit))
    total = int(rows[0]["total"]) if rows else 0
    return [_row(r) for r in rows], total


async def read_for_runs(pool: Any, run_ids: Sequence[str]) -> dict[str, FocusRow]:
    if not run_ids:
        return {}
    async with pool.acquire() as conn:
        rows = await conn.fetch(FOR_RUNS_SQL, list(run_ids))
    return {str(r["offered_run_id"]): _row(r) for r in rows}


# --- the disposition, read from the run's own :TurnOutcome ------------------------------------

_SET_ASIDE = re.compile(r"^set[\s_-]?aside\s*[:\-–—]\s*(?P<reason>.+)$", re.IGNORECASE | re.DOTALL)


def parse_disposition(raw: Any) -> tuple[Optional[str], Optional[str]]:
    """(disposition, disposition_raw). ``explored`` or ``set_aside:<reason>``; anything else is
    unknown (None) with the raw text kept, so a malformed answer is visible, not guessed."""
    text = " ".join(str(raw or "").split())
    if not text:
        return None, None
    if text.casefold() == DISPOSITION_EXPLORED:
        return DISPOSITION_EXPLORED, None
    m = _SET_ASIDE.match(text)
    if m and m.group("reason").strip():
        return f"{DISPOSITION_SET_ASIDE}:{m.group('reason').strip()[:DISPOSITION_CAP]}", None
    return None, text[:DISPOSITION_CAP]


def disposition_cypher(run_id: str) -> str:
    # The same run-id shape outcome_for_run_cypher accepts: nothing else reaches Cypher.
    from orion.curiosity.worldview import _RUN_ID_RE, LABEL_TURN_OUTCOME

    if not _RUN_ID_RE.match(str(run_id or "")):
        raise ValueError(f"refusing to build Cypher for a non-hex run_id: {run_id!r}")
    return (
        f"MATCH (t:{LABEL_TURN_OUTCOME}) WHERE t.run_id = '{run_id}' AND t.focus_disposition IS NOT NULL "
        "RETURN t.focus_disposition AS focus_disposition ORDER BY t.written_at DESC LIMIT 1"
    )


def read_disposition(reader: Any, run_id: str) -> tuple[Optional[str], Optional[str], bool]:
    """(disposition, disposition_raw, graph_read). Synchronous; callers use asyncio.to_thread.
    graph_read False: the graph could not answer, so the disposition is unknown for that reason."""
    if reader is None:
        return None, None, False
    try:
        rows = reader.query(disposition_cypher(run_id))
    except Exception as exc:  # noqa: BLE001 -- unknown, never explored
        logger.warning("curiosity_focus_disposition_unreadable run=%s err=%s", run_id, exc)
        return None, None, False
    if not rows:
        return None, None, True
    disposition, raw = parse_disposition(rows[0].get("focus_disposition"))
    return disposition, raw, True


# --- plain words, shared by the kickoff, the read tool and Hub --------------------------------

FOCUS_HEADER = "THE FOCUS YOU QUEUED FOR THIS RUN"


# Prime's resident loop (services/orion-mind/app/prime_loop.py) stores "prime:<thought_id>" as the
# row's chat_correlation_id: the question was chosen between conversations, not in a chat.
PRIME_CORRELATION_PREFIX = "prime:"


def is_prime_focus(focus: FocusRow) -> bool:
    return str(focus.chat_correlation_id or "").startswith(PRIME_CORRELATION_PREFIX)


def format_focus_section(focus: Optional[FocusRow]) -> list[str]:
    if focus is None:
        return []
    if focus.origin == ORIGIN_JUNIPER:
        who = "Juniper asked you to look into this next, and you queued it"
    elif is_prime_focus(focus):
        who = ("Between conversations, in your own slow thinking (Prime), you chose this as what your "
               "next curiosity run should look at")
    else:
        who = "In a conversation with Juniper you chose this as what your next curiosity run should look at"
    return [
        f"{FOCUS_HEADER} (shown to you once).",
        f"{who}:",
        "",
        f"    {' '.join(focus.question.split())}",
        f"    Why, in your words then: {' '.join(focus.why.split())}",
        "",
        *_refs_lines(focus),
        "It is an offer, not an order. You can explore it in this run or set it aside. Either way, "
        "say which on your :TurnOutcome at the end (the node is described under LAST below):",
        '    focus_disposition: "explored"                     // if you looked into it in this run',
        '    focus_disposition: "set_aside: <why not, in plain words>"   // if you did not',
        "Hub reads that back onto the focus, and Juniper and your curiosity tool show it. If you "
        "write no focus_disposition, it is recorded as unknown, not as explored.",
        "",
    ]


def _refs_lines(focus: FocusRow) -> list[str]:
    """What the question cites, each with the one command that opens it (orion/refs.py)."""
    if not focus.refs:
        return []
    from orion.refs import KINDS, RefV1, dig_command

    lines = ["It cites these; open each one with the command shown (the same read that checked it exists):"]
    for r in focus.refs:
        spec = KINDS.get(r["kind"])
        what = f" -- {spec.what}" if spec else ""
        lines.append(f"    {r['kind']} {r['id']}{what}: {dig_command(RefV1.model_construct(kind=r['kind'], id=r['id']))}")
    return [*lines, ""]


def focus_story(focus: FocusRow) -> str:
    """One line: queued -> taken by run X -> disposition / expired / replaced / failed."""

    def at(value: Optional[datetime]) -> str:
        return value.strftime("%Y-%m-%d %H:%M UTC") if value is not None else "?"

    who = ("Juniper asked" if focus.origin == ORIGIN_JUNIPER
           else "you chose it between conversations" if is_prime_focus(focus) else "you chose it")
    parts = [f"queued {at(focus.created_at)} ({who})"]
    status = focus.status
    if status == PENDING:
        parts.append(f"waiting for the next regular investigate run (expires {at(focus.expires_at)})")
    elif status == EXPIRED:
        parts.append(f"expired {at(focus.expires_at)}; no run took it")
    elif status == REPLACED:
        parts.append(f"replaced by a newer focus ({focus.replaced_by or '?'}) before a run took it")
    elif status == CLEARED:
        parts.append(f"cleared by Juniper in Hub {at(focus.status_at)} before a run took it")
    else:
        parts.append(f"taken by run {focus.offered_run_id} {at(focus.offered_at)}")
        if status == TAKEN:
            parts.append("the run has not finished, or its end was not recorded")
        elif status == FAILED and focus.turn_ok is None:
            parts.append("the run's end was never recorded, so it is counted as failed; the focus is not re-queued")
        elif status == FAILED:
            parts.append("the run failed; the focus is not re-queued")
        elif status == RAN:
            parts.append(f"run finished; disposition: {disposition_words(focus)}")
    return " -> ".join(parts)


def disposition_words(focus: FocusRow) -> str:
    if focus.disposition == DISPOSITION_EXPLORED:
        return "explored"
    if focus.disposition and focus.disposition.startswith(DISPOSITION_SET_ASIDE + ":"):
        return "set aside: " + focus.disposition.split(":", 1)[1]
    if focus.disposition_raw:
        return f"unknown (the run wrote {focus.disposition_raw!r}, which is neither explored nor set_aside:<reason>)"
    return "unknown (the run recorded none)"
