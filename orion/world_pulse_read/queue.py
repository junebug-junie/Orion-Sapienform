from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import NAMESPACE_URL, UUID, uuid5
from typing import Any, Sequence

from orion.schemas.reading import ReadingRequestedV1
from orion.world_pulse_read.urls import validate_source_url
from orion.schemas.world_pulse_read import WorldPulseReadHandoffV1, WorldPulseReadSeedV1
from orion.world_pulse_read.seeds import seeds_from_digest_payload

_PRIORITY = {"finding": 0, "reading": 0, "digest_item": 10}
_STAGE1_STATUSES = ("pending", "claimed", "done", "failed", "skipped")
_STAGE2_STATUSES = ("pending", "claimed", "done", "failed", "skipped")

ENSURE_TABLE_SQL = """
create table if not exists world_pulse_read_seed (
    seed_id text primary key,
    kind text not null check (kind in ('finding', 'digest_item')),
    run_id text not null,
    url text not null,
    title text not null default '',
    section text not null default '',
    item_id text null,
    priority int not null default 100,
    status text not null default 'pending'
        check (status in ('pending', 'claimed', 'done', 'failed', 'skipped')),
    trace_id text null,
    last_error text null,
    created_at timestamptz not null default now(),
    claimed_at timestamptz null,
    completed_at timestamptz null,
    handoff_json jsonb null,
    handoff_at timestamptz null,
    stage2_status text not null default 'pending'
        check (stage2_status in ('pending', 'claimed', 'done', 'failed', 'skipped')),
    stage2_claimed_at timestamptz null,
    stage2_completed_at timestamptz null,
    stage2_error text null,
    stage2_trace_id text null
)
"""

ENSURE_CLAIM_INDEX_SQL = """
create index if not exists idx_world_pulse_read_seed_claim
    on world_pulse_read_seed (status, priority, created_at)
"""

ENSURE_RUN_INDEX_SQL = """
create index if not exists idx_world_pulse_read_seed_run
    on world_pulse_read_seed (run_id)
"""

ENSURE_STAGE2_HANDOFF_JSON_SQL = """
alter table world_pulse_read_seed add column if not exists handoff_json jsonb
"""

ENSURE_STAGE2_HANDOFF_AT_SQL = """
alter table world_pulse_read_seed add column if not exists handoff_at timestamptz
"""

ENSURE_STAGE2_STATUS_SQL = """
alter table world_pulse_read_seed
    add column if not exists stage2_status text not null default 'pending'
"""

ENSURE_STAGE2_CLAIMED_AT_SQL = """
alter table world_pulse_read_seed add column if not exists stage2_claimed_at timestamptz
"""

ENSURE_STAGE2_COMPLETED_AT_SQL = """
alter table world_pulse_read_seed add column if not exists stage2_completed_at timestamptz
"""

ENSURE_STAGE2_ERROR_SQL = """
alter table world_pulse_read_seed add column if not exists stage2_error text
"""

ENSURE_STAGE2_TRACE_SQL = """
alter table world_pulse_read_seed add column if not exists stage2_trace_id text
"""

ENSURE_STAGE2_INDEX_SQL = """
create index if not exists idx_world_pulse_read_seed_stage2_claim
    on world_pulse_read_seed (stage2_status, priority, handoff_at)
"""

GENERAL_READING_SQL = "-- Additive general reading ingress; retains the existing queue and workers.\n-- Apply after both world_pulse_read migrations, before deploying Hub.\nALTER TABLE world_pulse_read_seed\n    ADD COLUMN IF NOT EXISTS request_id uuid,\n    ADD COLUMN IF NOT EXISTS request_json jsonb,\n    ADD COLUMN IF NOT EXISTS root_request_id uuid,\n    ADD COLUMN IF NOT EXISTS duplicate_of text,\n    ADD COLUMN IF NOT EXISTS stage2_result_json jsonb,\n    ADD COLUMN IF NOT EXISTS landing_at timestamptz;\nALTER TABLE world_pulse_read_seed DROP CONSTRAINT IF EXISTS world_pulse_read_seed_kind_check;\nALTER TABLE world_pulse_read_seed ADD CONSTRAINT world_pulse_read_seed_kind_check\n    CHECK (kind IN ('finding', 'digest_item', 'reading'));\nCREATE UNIQUE INDEX IF NOT EXISTS idx_reading_request_id ON world_pulse_read_seed(request_id);\nCREATE INDEX IF NOT EXISTS idx_reading_active_url ON world_pulse_read_seed(url)\n    WHERE status IN ('pending', 'claimed', 'done');\nCREATE INDEX IF NOT EXISTS idx_reading_root ON world_pulse_read_seed(root_request_id);\n"

INSERT_SQL = """
INSERT INTO world_pulse_read_seed
    (seed_id, kind, run_id, url, title, section, item_id, priority, status,
     request_id, request_json, root_request_id, duplicate_of, stage2_status)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,
        CASE WHEN $12::text IS NULL THEN 'pending' ELSE 'skipped' END,
        $9,$10::jsonb,$11,$12,
        CASE WHEN $12::text IS NULL THEN 'pending' ELSE 'skipped' END)
ON CONFLICT DO NOTHING
"""

ACTIVE_URL_SQL = """
SELECT seed_id FROM world_pulse_read_seed
WHERE url = $1 AND duplicate_of IS NULL
  AND (status IN ('pending', 'claimed') OR
       (status = 'done' AND stage2_status IN ('pending', 'claimed')))
ORDER BY created_at, seed_id LIMIT 1
"""

REQUEST_ROW_SQL = "SELECT * FROM world_pulse_read_seed WHERE request_id = $1"

CLAIM_SQL = """
UPDATE world_pulse_read_seed
SET status = 'claimed', claimed_at = now()
WHERE seed_id = (
    SELECT seed_id FROM world_pulse_read_seed
    WHERE status = 'pending'
    ORDER BY priority ASC, created_at ASC, seed_id ASC
    FOR UPDATE SKIP LOCKED
    LIMIT 1
)
RETURNING seed_id, kind, run_id, url, title, section, item_id, request_json
"""

CLAIM_STAGE2_SQL = """
UPDATE world_pulse_read_seed
SET stage2_status = 'claimed', stage2_claimed_at = now()
WHERE seed_id = (
    SELECT seed_id FROM world_pulse_read_seed
    WHERE status = 'done'
      AND handoff_json IS NOT NULL
      AND stage2_status = 'pending'
    ORDER BY priority ASC, handoff_at ASC NULLS LAST, seed_id ASC
    FOR UPDATE SKIP LOCKED
    LIMIT 1
)
RETURNING seed_id, kind, run_id, url, title, section, item_id, request_json, handoff_json, trace_id
"""

# Short, grep-friendly reclaim reasons -- same pattern as PR #2166's Stage 2
# fail_reason labels (e.g. `turn_error:fcc_stream_stalled`). Written into
# `last_error`/`stage2_error` by the RECLAIM_* SQL below so a reclaim leaves
# a real trace instead of silently resetting the row to `pending` with no
# indication anything happened (confirmed live 2026-09-10: a Hub restart
# reclaim reset an in-flight seed with zero trace of it). A later successful
# claim + completion still clears the field to null via MARK_DONE_SQL /
# mark_stage2_done, so this is not a permanent-failure marker.
RECLAIM_REASON_PROCESS_RESTART = "interrupted:process_restart"
RECLAIM_REASON_STALE_TIMEOUT = "interrupted:stale_timeout"

RECLAIM_STALE_CLAIMED_SQL = """
UPDATE world_pulse_read_seed
SET status = 'pending', claimed_at = NULL, last_error = $2
WHERE status = 'claimed'
  AND claimed_at IS NOT NULL
  AND claimed_at < now() - ($1 * interval '1 second')
"""

RECLAIM_STALE_STAGE2_CLAIMED_SQL = """
UPDATE world_pulse_read_seed
SET stage2_status = 'pending', stage2_claimed_at = NULL, stage2_error = $2
WHERE stage2_status = 'claimed'
  AND stage2_claimed_at IS NOT NULL
  AND stage2_claimed_at < now() - ($1 * interval '1 second')
"""

MARK_DONE_SQL = """
UPDATE world_pulse_read_seed
SET status = 'done',
    trace_id = $2,
    completed_at = now(),
    last_error = null,
    handoff_json = COALESCE($3::jsonb, handoff_json),
    handoff_at = CASE WHEN $3::jsonb IS NOT NULL THEN now() ELSE handoff_at END
WHERE seed_id = $1
"""

COUNT_BY_STATUS_SQL = """
SELECT status, count(*)::int AS n FROM world_pulse_read_seed GROUP BY status
"""

COUNT_BY_STAGE2_STATUS_SQL = """
SELECT stage2_status, count(*)::int AS n FROM world_pulse_read_seed GROUP BY stage2_status
"""

LAST_TIMESTAMPS_SQL = """
SELECT max(completed_at) AS last_stage1_at,
       max(stage2_completed_at) AS last_stage2_at
FROM world_pulse_read_seed
"""

EXISTING_SEED_IDS_SQL = """
SELECT seed_id FROM world_pulse_read_seed WHERE seed_id = ANY($1::text[])
"""


@dataclass(frozen=True)
class Stage2Claim:
    seed: WorldPulseReadSeedV1
    handoff_json: dict[str, Any]
    stage1_trace_id: str | None


async def ensure_seed_queue_schema(conn: Any) -> None:
    await conn.execute(ENSURE_TABLE_SQL)
    await conn.execute(ENSURE_CLAIM_INDEX_SQL)
    await conn.execute(ENSURE_RUN_INDEX_SQL)
    await conn.execute(ENSURE_STAGE2_HANDOFF_JSON_SQL)
    await conn.execute(ENSURE_STAGE2_HANDOFF_AT_SQL)
    await conn.execute(ENSURE_STAGE2_STATUS_SQL)
    await conn.execute(ENSURE_STAGE2_CLAIMED_AT_SQL)
    await conn.execute(ENSURE_STAGE2_COMPLETED_AT_SQL)
    await conn.execute(ENSURE_STAGE2_ERROR_SQL)
    await conn.execute(ENSURE_STAGE2_TRACE_SQL)
    await conn.execute(ENSURE_STAGE2_INDEX_SQL)
    await conn.execute(GENERAL_READING_SQL)


def request_for_seed(seed: WorldPulseReadSeedV1) -> ReadingRequestedV1:
    if seed.request is not None:
        return seed.request
    return ReadingRequestedV1(
        request_id=uuid5(NAMESPACE_URL, seed.seed_id), url=seed.url,
        requested_by="world_pulse", invocation_context="world_pulse",
        title=seed.title, parent_run_id=seed.run_id,
    )


async def enqueue_seeds(
    conn: Any, seeds: Sequence[WorldPulseReadSeedV1], *,
    bus: Any = None, source: Any = None, max_round_trips: int | None = None,
) -> int:
    """One ingress for World Pulse, tools and reentry. Commit before publication.

    Serialize equal URLs across producers. Keep duplicate deliveries idempotent,
    and persist new requests for active URLs as aliases in this same table so
    their requester/context is preserved without creating duplicate work.
    """
    if conn.is_in_transaction():
        raise RuntimeError("reading ingress must own its commit before publishing")
    inserted = 0
    for seed in seeds:
        request = request_for_seed(seed)
        # An already accepted request remains retryable even if DNS later fails.
        if await conn.fetchrow("SELECT seed_id FROM world_pulse_read_seed WHERE seed_id = $1", seed.seed_id):
            continue
        url = await validate_source_url(str(request.url))
        request = ReadingRequestedV1.model_validate({**request.model_dump(), "url": url})
        async with conn.transaction():
            if max_round_trips is not None and request.root_request_id:
                await conn.execute("SELECT pg_advisory_xact_lock(hashtextextended($1, 0))", str(request.root_request_id))
                count = await conn.fetchval(
                    "SELECT count(*) FROM world_pulse_read_seed WHERE root_request_id = $1 AND request_id <> $1 AND duplicate_of IS NULL",
                    request.root_request_id,
                )
                if count >= max_round_trips:
                    raise ValueError("round_trip_cap")
            await conn.execute("SELECT pg_advisory_xact_lock(hashtextextended($1, 0))", url)
            active = await conn.fetchrow(ACTIVE_URL_SQL, url)
            duplicate_of = active["seed_id"] if active and active["seed_id"] != seed.seed_id else None
            status = await conn.execute(
                INSERT_SQL, seed.seed_id, seed.kind, seed.run_id, url, seed.title,
                seed.section, seed.item_id, _PRIORITY[seed.kind], request.request_id,
                request.model_dump_json(), request.root_request_id or request.request_id, duplicate_of,
            )
        if isinstance(status, str) and status.endswith("1"):
            inserted += 1
            from orion.world_pulse_read.events import publish_accepted
            await publish_accepted(bus, request, source=source)
    return inserted


async def enqueue_reading(conn: Any, request: ReadingRequestedV1, **kwargs: Any) -> dict[str, Any]:
    seed = WorldPulseReadSeedV1(
        seed_id=f"reading:{request.request_id}", kind="reading",
        run_id=request.parent_run_id or str(request.request_id),
        url=str(request.url), title=request.title, request=request,
    )
    await enqueue_seeds(conn, [seed], **kwargs)
    return await reading_status(conn, request.request_id)


async def reading_status(conn: Any, request_id: UUID) -> dict[str, Any]:
    row = await conn.fetchrow(REQUEST_ROW_SQL, request_id)
    if row is None:
        return {"request_id": str(request_id), "status": "not_found"}
    own = row
    if row["duplicate_of"]:
        row = await conn.fetchrow("SELECT * FROM world_pulse_read_seed WHERE seed_id = $1", row["duplicate_of"])
        if row is None:
            raise RuntimeError("reading alias target missing")
    s1, s2 = row["status"], row["stage2_status"]
    status = "queued"
    if s1 == "failed" or (s1 == "done" and s2 == "failed"):
        status = "failed"
    elif s1 == "skipped":
        status = "skipped"
    elif row["landing_at"]:
        status = "completed"
    elif s1 == "done" and s2 == "done":
        status = "landing_pending"
    elif s1 == "done":
        status = "stage2_started" if s2 == "claimed" else "stage1_completed"
    elif s1 == "claimed":
        status = "started"
    handoff = _json_object(row.get("handoff_json"))
    result = _json_object(row.get("stage2_result_json"))
    return {
        "request_id": str(request_id), "status": status,
        "request": _json_object(own["request_json"]),
        "seed_id": own["seed_id"], "duplicate_of": own["duplicate_of"],
        "stage1_status": s1, "stage2_status": s2,
        "stage1_trace_id": row["trace_id"], "stage2_trace_id": row["stage2_trace_id"],
        "summary": (result.get("summary") or handoff.get("what_i_learned") or "")[:6000],
        "error": row["stage2_error"] or row["last_error"],
        "evidence_url": row["url"],
        "landing_at": row["landing_at"].isoformat() if row["landing_at"] else None,
    }


def _json_object(raw: Any) -> dict[str, Any]:
    return json.loads(raw) if isinstance(raw, str) else (raw or {})


def _seed_from_row(row: Any) -> WorldPulseReadSeedV1:
    return WorldPulseReadSeedV1(
        seed_id=row["seed_id"],
        kind=row["kind"],
        run_id=row["run_id"],
        url=row["url"],
        title=row["title"] or "",
        section=row["section"] or "",
        item_id=row["item_id"],
        request=_json_object(row.get("request_json")) or None,
    )


async def claim_next_seed(conn: Any) -> WorldPulseReadSeedV1 | None:
    row = await conn.fetchrow(CLAIM_SQL)
    if not row:
        return None
    return _seed_from_row(row)


def _update_rowcount(status: Any) -> int:
    if isinstance(status, int):
        return status
    if isinstance(status, str) and status.upper().startswith("UPDATE"):
        tail = status.split()[-1]
        if tail.isdigit():
            return int(tail)
    return 0


async def reclaim_stale_claimed(
    conn: Any, *, older_than_sec: float, reason: str
) -> int:
    """Return stuck ``claimed`` rows to ``pending`` after Hub/FCC death.

    ``claimed_at`` older than ``older_than_sec`` is the live-tick guard:
    a turn still running inside ``timeout_sec`` is left alone. Pass
    ``older_than_sec=0`` on the first tick after Hub start so a restart
    immediately frees leftovers from the previous process.

    ``reason`` is stamped into ``last_error`` on every row this reclaims, so
    the caller must pass ``RECLAIM_REASON_PROCESS_RESTART`` for the
    ``older_than_sec=0`` startup case and ``RECLAIM_REASON_STALE_TIMEOUT``
    for the periodic in-loop case -- the caller already knows which one it
    is (it's the same branch that picks ``older_than_sec``).
    """
    status = await conn.execute(RECLAIM_STALE_CLAIMED_SQL, float(older_than_sec), reason)
    return _update_rowcount(status)


async def reclaim_stale_stage2_claimed(
    conn: Any, *, older_than_sec: float, reason: str
) -> int:
    """Stage 2 sibling of :func:`reclaim_stale_claimed` -- see its docstring."""
    status = await conn.execute(
        RECLAIM_STALE_STAGE2_CLAIMED_SQL, float(older_than_sec), reason
    )
    return _update_rowcount(status)


def _handoff_json_text(
    handoff: WorldPulseReadHandoffV1 | dict[str, Any] | None,
) -> str | None:
    if handoff is None:
        return None
    if isinstance(handoff, WorldPulseReadHandoffV1):
        return json.dumps(handoff.model_dump(mode="json"))
    return json.dumps(handoff)


async def mark_seed_done(
    conn: Any,
    seed_id: str,
    *,
    trace_id: str,
    handoff: WorldPulseReadHandoffV1 | dict[str, Any] | None = None,
) -> None:
    await conn.execute(MARK_DONE_SQL, seed_id, trace_id, _handoff_json_text(handoff))


async def mark_seed_failed(conn: Any, seed_id: str, *, error: str) -> None:
    await conn.execute(
        """
        UPDATE world_pulse_read_seed
        SET status = 'failed', last_error = $2, completed_at = now()
        WHERE seed_id = $1
        """,
        seed_id,
        error[:2000],
    )


async def mark_seed_skipped(conn: Any, seed_id: str, *, reason: str) -> None:
    """Mark without a Stage 1 debit — used for section-index URLs etc."""
    await conn.execute(
        """
        UPDATE world_pulse_read_seed
        SET status = 'skipped', last_error = $2, completed_at = now()
        WHERE seed_id = $1
        """,
        seed_id,
        reason[:2000],
    )


async def claim_next_stage2_seed(conn: Any) -> Stage2Claim | None:
    row = await conn.fetchrow(CLAIM_STAGE2_SQL)
    if not row:
        return None
    raw = row["handoff_json"]
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, dict):
        raw = {}
    return Stage2Claim(
        seed=_seed_from_row(row),
        handoff_json=raw,
        stage1_trace_id=row["trace_id"],
    )


async def mark_stage2_done(
    conn: Any, seed_id: str, *, stage2_trace_id: str, result: Any = None
) -> None:
    await conn.execute(
        """
        UPDATE world_pulse_read_seed
        SET stage2_status = 'done',
            stage2_trace_id = $2,
            stage2_completed_at = now(),
            stage2_error = null,
            stage2_result_json = COALESCE($3::jsonb, stage2_result_json)
        WHERE seed_id = $1
        """,
        seed_id,
        stage2_trace_id,
        result.model_dump_json() if result is not None else None,
    )


async def mark_stage2_failed(
    conn: Any,
    seed_id: str,
    *,
    error: str,
    stage2_trace_id: str | None = None,
) -> None:
    await conn.execute(
        """
        UPDATE world_pulse_read_seed
        SET stage2_status = 'failed',
            stage2_error = $2,
            stage2_trace_id = COALESCE($3, stage2_trace_id),
            stage2_completed_at = now()
        WHERE seed_id = $1
        """,
        seed_id,
        error[:2000],
        stage2_trace_id,
    )


def _zero_counts(keys: Sequence[str]) -> dict[str, int]:
    return {k: 0 for k in keys}


async def count_seeds_by_status(conn: Any) -> dict[str, int]:
    out = _zero_counts(_STAGE1_STATUSES)
    rows = await conn.fetch(COUNT_BY_STATUS_SQL)
    for row in rows or []:
        key = str(row["status"] or "")
        if key in out:
            out[key] = int(row["n"])
    return out


async def count_stage2_by_status(conn: Any) -> dict[str, int]:
    out = _zero_counts(_STAGE2_STATUSES)
    rows = await conn.fetch(COUNT_BY_STAGE2_STATUS_SQL)
    for row in rows or []:
        key = str(row["stage2_status"] or "")
        if key in out:
            out[key] = int(row["n"])
    return out


async def last_stage_timestamps(conn: Any) -> dict[str, Any]:
    row = await conn.fetchrow(LAST_TIMESTAMPS_SQL)
    if not row:
        return {"last_stage1_at": None, "last_stage2_at": None}
    return {
        "last_stage1_at": row["last_stage1_at"],
        "last_stage2_at": row["last_stage2_at"],
    }


async def _count_absent_seeds(
    conn: Any, seeds: Sequence[WorldPulseReadSeedV1]
) -> int:
    if not seeds:
        return 0
    ids = [s.seed_id for s in seeds]
    rows = await conn.fetch(EXISTING_SEED_IDS_SQL, ids)
    existing = {r["seed_id"] for r in rows or []}
    return sum(1 for seed_id in ids if seed_id not in existing)


async def enqueue_from_recent_digests(
    conn: Any,
    *,
    limit_digests: int = 5,
    findings_only: bool = False,
    dry_run: bool = False,
    bus: Any = None,
    source: Any = None,
) -> int:
    """Pull recent world_pulse_digest rows and enqueue seeds (findings first via priority)."""
    digests = await conn.fetch(
        """
        SELECT run_id, payload_json
        FROM world_pulse_digest
        ORDER BY created_at DESC NULLS LAST
        LIMIT $1
        """,
        limit_digests,
    )
    run_ids = [d["run_id"] for d in digests]
    article_urls: dict[str, str] = {}
    if run_ids:
        rows = await conn.fetch(
            """
            SELECT article_id, url FROM world_pulse_article
            WHERE run_id = ANY($1::text[])
            """,
            run_ids,
        )
        article_urls = {r["article_id"]: r["url"] for r in rows}
    total = 0
    # chronological enqueue: oldest of the limited set first
    for d in reversed(list(digests)):
        payload = d["payload_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        seeds = seeds_from_digest_payload(payload, article_urls=article_urls)
        if findings_only:
            seeds = [s for s in seeds if s.kind == "finding"]
        if dry_run:
            total += await _count_absent_seeds(conn, seeds)
        else:
            for seed in seeds:
                try:
                    total += await enqueue_seeds(conn, [seed], bus=bus, source=source)
                except ValueError:
                    # A bad discovered URL must not block all other discoveries.
                    import logging
                    logging.getLogger(__name__).warning("reading_discovery_rejected seed=%s", seed.seed_id)
    return total


async def confirm_landings(conn: Any) -> list[WorldPulseReadSeedV1]:
    """A published journal command is not a stored journal entry.

    The SQL writer's existing journal_entries table is the confirmation rail.
    No reading claim is completed merely because an ephemeral publish succeeded.
    """
    rows = await conn.fetch("""
        UPDATE world_pulse_read_seed AS s SET landing_at = now()
        WHERE s.status = 'done' AND s.stage2_status = 'done'
          AND s.landing_at IS NULL AND s.duplicate_of IS NULL
          AND s.handoff_json IS NOT NULL AND s.stage2_result_json IS NOT NULL
          AND EXISTS (SELECT 1 FROM journal_entries j
                      WHERE j.source_ref = 'world_pulse_read:' || s.trace_id)
          AND EXISTS (SELECT 1 FROM journal_entries j
                      WHERE j.source_ref = 'world_pulse_read_stage2:' || s.stage2_trace_id)
        RETURNING s.*
    """)
    return [_seed_from_row(row) for row in rows]


async def pending_journal_landings(conn: Any) -> list[dict[str, Any]]:
    """Bounded replay from existing saved artifacts, never another model run."""
    rows = await conn.fetch("""
        SELECT s.*,
          NOT EXISTS (SELECT 1 FROM journal_entries j
                      WHERE j.source_ref = 'world_pulse_read:' || s.trace_id) AS missing_stage1,
          NOT EXISTS (SELECT 1 FROM journal_entries j
                      WHERE j.source_ref = 'world_pulse_read_stage2:' || s.stage2_trace_id) AS missing_stage2
        FROM world_pulse_read_seed s
        WHERE s.status = 'done' AND s.handoff_json IS NOT NULL
          AND s.landing_at IS NULL AND s.duplicate_of IS NULL
          AND (NOT EXISTS (SELECT 1 FROM journal_entries j
                           WHERE j.source_ref = 'world_pulse_read:' || s.trace_id)
               OR (s.stage2_status = 'done' AND s.stage2_result_json IS NOT NULL
                   AND NOT EXISTS (SELECT 1 FROM journal_entries j
                                   WHERE j.source_ref = 'world_pulse_read_stage2:' || s.stage2_trace_id)))
        ORDER BY s.completed_at LIMIT 10
    """)
    return [dict(row) for row in rows]
