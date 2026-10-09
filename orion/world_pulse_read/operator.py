"""Operator view and controls over the reading queue (Hub Reading tab).

Reads are plain SELECTs. Controls reuse the workers' own transitions: cancel
lands on the same ``reading_cancelled_by_operator`` label as ``cancel_claim``,
submit goes through ``enqueue_reading``. Row locks use ``FOR UPDATE`` so a
worker's ``FOR UPDATE SKIP LOCKED`` claim and ``bind_turn`` serialize with us.
"""
from __future__ import annotations

import json
from typing import Any

from orion.schemas.reading import ReadingRequestedV1
from orion.world_pulse_read.documents import DocumentPolicy, is_document_ref, normalize_document_ref
from orion.world_pulse_read.durable import OPERATOR_CANCEL_REASON
from orion.world_pulse_read.queue import (
    ACTIVE_URL_SQL,
    ALREADY_READ,
    READ_URL_SQL,
    STALE_DIGEST_ITEM_LAST_ERROR,
    derive_reading_status,
    enqueue_reading,
)

PHASES = ("all", "active", "done", "failed", "skipped", "with_output")

_PHASE_SQL = {
    "all": "TRUE",
    "active": "(s.status IN ('pending','claimed') OR "
              "(s.status = 'done' AND s.stage2_status IN ('pending','claimed')))",
    "done": "(s.status = 'done' AND s.stage2_status = 'done')",
    "failed": "(s.status = 'failed' OR (s.status = 'done' AND s.stage2_status = 'failed'))",
    "skipped": "(s.status = 'skipped' OR (s.status = 'done' AND s.stage2_status = 'skipped'))",
    "with_output": "(s.handoff_json IS NOT NULL OR s.stage2_result_json IS NOT NULL)",
}

_LIST_SQL = """
SELECT s.seed_id, s.kind, s.url, s.title, s.status, s.stage2_status,
       s.attempts, s.stage2_attempts, s.last_error, s.stage2_error,
       s.created_at, s.claimed_at, s.completed_at, s.handoff_at,
       s.stage2_claimed_at, s.stage2_completed_at, s.landing_at, s.duplicate_of,
       s.request_json->>'requested_by' AS requested_by,
       s.request_json->>'invocation_context' AS invocation_context,
       s.request_json->>'why_now' AS why_now,
       s.handoff_json IS NOT NULL AS has_handoff,
       s.stage2_result_json IS NOT NULL AS has_stage2_result,
       left(COALESCE(s.stage2_result_json->>'summary',
                     CASE WHEN s.status = 'done' THEN s.handoff_json->>'what_i_learned' END,
                     ''), 280) AS preview,
       GREATEST(s.created_at, s.claimed_at, s.completed_at, s.handoff_at,
                s.stage2_claimed_at, s.stage2_completed_at, s.landing_at) AS updated_at,
       (SELECT d.run_id FROM reading_durable_turn d
         WHERE d.seed_id = s.seed_id AND d.consumed_at IS NULL
         ORDER BY d.stage DESC, d.attempt DESC LIMIT 1) AS active_run_id,
       count(*) OVER () AS total
FROM world_pulse_read_seed s
WHERE {phase}
  AND ($1::text IS NULL OR s.kind = $1)
  AND ($2::boolean OR COALESCE(s.last_error, '') <> $3)
ORDER BY updated_at DESC, s.seed_id DESC
LIMIT $4 OFFSET $5
"""

_BINDINGS_SQL = """
SELECT stage, attempt, run_id, consumed_at, created_at
FROM reading_durable_turn WHERE seed_id = $1 ORDER BY stage, attempt
"""

_ALIASES_SQL = """
SELECT seed_id, kind, created_at,
       request_json->>'requested_by' AS requested_by,
       request_json->>'why_now' AS why_now
FROM world_pulse_read_seed WHERE duplicate_of = $1 ORDER BY created_at
"""

_JOURNAL_SQL = """
SELECT entry_id, created_at, title, body, source_ref
FROM journal_entries WHERE source_ref = ANY($1::text[]) ORDER BY created_at
"""

_ACTIVE_BINDING_SQL = """
SELECT run_id FROM reading_durable_turn
WHERE seed_id = $1 AND stage = $2 AND consumed_at IS NULL
ORDER BY attempt DESC LIMIT 1
"""

_OLDER_THAN_SQL = """
SELECT created_at < now() - ($2 * interval '1 second')
FROM world_pulse_read_seed WHERE seed_id = $1
"""


class OperatorActionError(Exception):
    """A refused control. ``code`` is shown to the operator verbatim."""

    def __init__(self, code: str, http_status: int = 409) -> None:
        super().__init__(code)
        self.code = code
        self.http_status = http_status


def _iso(value: Any) -> str | None:
    return value.isoformat() if hasattr(value, "isoformat") else (None if value is None else str(value))


def _json(raw: Any) -> Any:
    return json.loads(raw) if isinstance(raw, str) else raw


def _row_summary(row: Any) -> dict[str, Any]:
    out = {k: row[k] for k in (
        "seed_id", "kind", "url", "title", "status", "stage2_status", "attempts",
        "stage2_attempts", "last_error", "stage2_error", "duplicate_of",
    )}
    for k in ("created_at", "claimed_at", "completed_at", "handoff_at",
              "stage2_claimed_at", "stage2_completed_at", "landing_at"):
        out[k] = _iso(row[k])
    out["reading_status"] = derive_reading_status(row["status"], row["stage2_status"], row["landing_at"])
    return out


async def list_reads(
    conn: Any, *, phase: str = "all", kind: str | None = None,
    include_stale: bool = False, limit: int = 50, offset: int = 0,
) -> dict[str, Any]:
    if phase not in _PHASE_SQL:
        raise OperatorActionError("unknown_phase", 400)
    rows = await conn.fetch(
        _LIST_SQL.format(phase=_PHASE_SQL[phase]),
        kind or None, bool(include_stale), STALE_DIGEST_ITEM_LAST_ERROR,
        max(1, min(int(limit), 200)), max(0, int(offset)),
    )
    items = []
    for row in rows:
        item = _row_summary(row)
        item.update(
            requested_by=row["requested_by"] or "world_pulse",
            invocation_context=row["invocation_context"] or "world_pulse",
            why_now=row["why_now"] or "",
            has_handoff=bool(row["has_handoff"]),
            has_stage2_result=bool(row["has_stage2_result"]),
            preview=row["preview"] or "",
            updated_at=_iso(row["updated_at"]),
            active_run_id=row["active_run_id"],
        )
        items.append(item)
    total = int(rows[0]["total"]) if rows else 0
    return {"items": items, "total": total, "phase": phase}


async def _journal_entries(conn: Any, refs: list[str]) -> list[dict[str, Any]]:
    if not refs:
        return []
    try:
        rows = await conn.fetch(_JOURNAL_SQL, refs)
    except Exception as exc:
        # journal_entries is owned by sql-writer; absent on a fresh database.
        if getattr(exc, "sqlstate", None) == "42P01":
            return []
        raise
    return [
        {"entry_id": r["entry_id"], "created_at": _iso(r["created_at"]),
         "title": r["title"], "body": r["body"], "source_ref": r["source_ref"]}
        for r in rows
    ]


async def read_detail(conn: Any, seed_id: str) -> dict[str, Any]:
    row = await conn.fetchrow("SELECT * FROM world_pulse_read_seed WHERE seed_id = $1", seed_id)
    if row is None:
        raise OperatorActionError("not_found", 404)
    detail = _row_summary(row)
    request = _json(row["request_json"]) or None
    handoff = _json(row["handoff_json"])
    result = _json(row["stage2_result_json"])
    refs = []
    if row["trace_id"]:
        refs.append(f"world_pulse_read:{row['trace_id']}")
    if row["stage2_trace_id"]:
        refs.append(f"world_pulse_read_stage2:{row['stage2_trace_id']}")
    bindings = await conn.fetch(_BINDINGS_SQL, seed_id)
    aliases = await conn.fetch(_ALIASES_SQL, seed_id)
    detail.update(
        request=request,
        requested_by=(request or {}).get("requested_by") or "world_pulse",
        section=row["section"],
        priority=row["priority"],
        trace_id=row["trace_id"],
        stage2_trace_id=row["stage2_trace_id"],
        handoff=handoff,
        # A handoff on a row that did not finish Stage 1 was rejected (e.g. no
        # read evidence) or belongs to an earlier attempt -- never "learned".
        handoff_accepted=bool(handoff) and row["status"] == "done",
        stage2_result=result,
        journal=await _journal_entries(conn, refs),
        durable_turns=[
            {"stage": b["stage"], "attempt": b["attempt"], "run_id": b["run_id"],
             "consumed_at": _iso(b["consumed_at"]), "created_at": _iso(b["created_at"])}
            for b in bindings
        ],
        aliases=[
            {"seed_id": a["seed_id"], "kind": a["kind"], "created_at": _iso(a["created_at"]),
             "requested_by": a["requested_by"] or "world_pulse", "why_now": a["why_now"] or ""}
            for a in aliases
        ],
    )
    return detail


async def _locked_row(conn: Any, seed_id: str) -> Any:
    row = await conn.fetchrow(
        "SELECT * FROM world_pulse_read_seed WHERE seed_id = $1 FOR UPDATE", seed_id,
    )
    if row is None:
        raise OperatorActionError("not_found", 404)
    if row["duplicate_of"]:
        raise OperatorActionError("alias_row_act_on_target")
    return row


async def cancel_read(conn: Any, seed_id: str) -> dict[str, Any]:
    """Plan or perform a cancel. Never charges a wallet or spends an attempt.

    A stage with an unconsumed durable binding must be cancelled at the durable
    runner; the worker then observes ``cancelled`` and calls ``cancel_claim``.
    Returns ``{"action": "cancel_durable_run", "run_id": ...}`` for the caller to
    perform, or ``{"action": "skipped"}`` when done here.
    """
    async with conn.transaction():
        row = await _locked_row(conn, seed_id)
        if row["status"] in ("pending", "claimed"):
            stage, state = 1, row["status"]
        elif row["status"] == "done" and row["stage2_status"] in ("pending", "claimed"):
            stage, state = 2, row["stage2_status"]
        else:
            raise OperatorActionError("not_active")
        run_id = await conn.fetchval(_ACTIVE_BINDING_SQL, seed_id, stage)
        if run_id:
            return {"action": "cancel_durable_run", "stage": stage, "run_id": run_id}
        if state == "claimed":
            # Worker is between claim and bind_turn; its bind will lock this row next.
            raise OperatorActionError("claimed_without_binding_retry_shortly")
        if stage == 1:
            await conn.execute(
                "UPDATE world_pulse_read_seed SET status='skipped', last_error=$2, completed_at=now() "
                "WHERE seed_id=$1", seed_id, OPERATOR_CANCEL_REASON,
            )
        else:
            await conn.execute(
                "UPDATE world_pulse_read_seed SET stage2_status='skipped', stage2_error=$2, "
                "stage2_completed_at=now() WHERE seed_id=$1", seed_id, OPERATOR_CANCEL_REASON,
            )
    return {"action": "skipped", "stage": stage}


def _has_read_evidence(handoff: Any) -> bool:
    handoff = _json(handoff) or {}
    return isinstance(handoff, dict) and bool(handoff.get("read_evidence"))


_OTHER_FOLLOW_UP_SQL = """
SELECT stage2_status FROM world_pulse_read_seed
WHERE url = $1 AND seed_id <> $2 AND duplicate_of IS NULL
  AND status = 'done' AND stage2_status IN ('done', 'pending', 'claimed')
ORDER BY stage2_status = 'done' DESC LIMIT 1
"""


async def retry_read(
    conn: Any, seed_id: str, *, stage: int, digest_item_max_age_sec: float = 0.0,
) -> dict[str, Any]:
    """Return a terminal stage to ``pending`` with a fresh attempt budget.

    ``digest_item_max_age_sec`` must match the Stage 1 worker's stale sweep
    (``skip_stale_digest_items``): a digest item past it would be skipped again
    on the next tick, so the retry is refused instead of reported as queued.
    """
    if stage not in (1, 2):
        raise OperatorActionError("stage_must_be_1_or_2", 400)
    async with conn.transaction():
        row = await _locked_row(conn, seed_id)
        if await conn.fetchval(_ACTIVE_BINDING_SQL, seed_id, stage):
            raise OperatorActionError("active_durable_binding")
        if stage == 1:
            if row["status"] not in ("failed", "skipped"):
                raise OperatorActionError("stage1_not_terminal")
            if row["last_error"] == STALE_DIGEST_ITEM_LAST_ERROR or (
                row["kind"] == "digest_item" and digest_item_max_age_sec > 0
                and await conn.fetchval(_OLDER_THAN_SQL, seed_id, float(digest_item_max_age_sec))
            ):
                raise OperatorActionError("stale_digest_item_would_be_reskipped")
            # Same lock the ingress takes, so a concurrent submit cannot race us.
            await conn.execute("SELECT pg_advisory_xact_lock(hashtextextended($1, 0))", row["url"])
            active = await conn.fetchrow(ACTIVE_URL_SQL, row["url"])
            if active and active["seed_id"] != seed_id:
                raise OperatorActionError("url_already_active")
            if await conn.fetchval(READ_URL_SQL, row["url"], seed_id):
                raise OperatorActionError(ALREADY_READ)
            await conn.execute(
                """UPDATE world_pulse_read_seed
                   SET status='pending', attempts=0, last_error=NULL,
                       claimed_at=NULL, completed_at=NULL,
                       stage2_status='pending', stage2_attempts=0, stage2_error=NULL,
                       stage2_claimed_at=NULL, stage2_completed_at=NULL
                   WHERE seed_id=$1""",
                seed_id,
            )
        else:
            if row["status"] != "done":
                raise OperatorActionError("stage1_not_done")
            if row["stage2_status"] not in ("failed", "skipped"):
                raise OperatorActionError("stage2_not_terminal")
            if not _has_read_evidence(row["handoff_json"]):
                raise OperatorActionError("no_read_evidence")
            await conn.execute("SELECT pg_advisory_xact_lock(hashtextextended($1, 0))", row["url"])
            other = await conn.fetchval(_OTHER_FOLLOW_UP_SQL, row["url"], seed_id)
            if other == "done":
                raise OperatorActionError(ALREADY_READ)
            if other is not None:
                raise OperatorActionError("url_already_active")
            await conn.execute(
                """UPDATE world_pulse_read_seed
                   SET stage2_status='pending', stage2_attempts=0, stage2_error=NULL,
                       stage2_claimed_at=NULL, stage2_completed_at=NULL
                   WHERE seed_id=$1""",
                seed_id,
            )
    return {"action": "requeued", "stage": stage}


def operator_request(*, url: str, why_now: str = "", title: str = "") -> ReadingRequestedV1:
    if is_document_ref(url):
        url = normalize_document_ref(url)
    return ReadingRequestedV1(
        url=url, requested_by="juniper", invocation_context="operator",
        why_now=why_now, title=title,
    )


async def submit_read(
    conn: Any, *, url: str, why_now: str = "", title: str = "",
    bus: Any = None, source: Any = None, documents: DocumentPolicy | None = None,
) -> dict[str, Any]:
    return await enqueue_reading(
        conn, operator_request(url=url, why_now=why_now, title=title), bus=bus, source=source,
        documents=documents,
    )
