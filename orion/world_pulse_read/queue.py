from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

from orion.schemas.world_pulse_read import WorldPulseReadHandoffV1, WorldPulseReadSeedV1
from orion.world_pulse_read.seeds import seeds_from_digest_payload

_PRIORITY = {"finding": 0, "digest_item": 10}
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

INSERT_SQL = """
INSERT INTO world_pulse_read_seed
    (seed_id, kind, run_id, url, title, section, item_id, priority, status)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,'pending')
ON CONFLICT (seed_id) DO NOTHING
"""

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
RETURNING seed_id, kind, run_id, url, title, section, item_id
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
RETURNING seed_id, kind, run_id, url, title, section, item_id, handoff_json, trace_id
"""

RECLAIM_STALE_CLAIMED_SQL = """
UPDATE world_pulse_read_seed
SET status = 'pending', claimed_at = NULL
WHERE status = 'claimed'
  AND claimed_at IS NOT NULL
  AND claimed_at < now() - ($1 * interval '1 second')
"""

RECLAIM_STALE_STAGE2_CLAIMED_SQL = """
UPDATE world_pulse_read_seed
SET stage2_status = 'pending', stage2_claimed_at = NULL
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


async def enqueue_seeds(conn: Any, seeds: Sequence[WorldPulseReadSeedV1]) -> int:
    inserted = 0
    for seed in seeds:
        priority = _PRIORITY[seed.kind]
        status = await conn.execute(
            INSERT_SQL,
            seed.seed_id,
            seed.kind,
            seed.run_id,
            seed.url,
            seed.title,
            seed.section,
            seed.item_id,
            priority,
        )
        # asyncpg returns "INSERT 0 1" / "INSERT 0 0"
        if isinstance(status, str) and status.endswith("1"):
            inserted += 1
    return inserted


def _seed_from_row(row: Any) -> WorldPulseReadSeedV1:
    return WorldPulseReadSeedV1(
        seed_id=row["seed_id"],
        kind=row["kind"],
        run_id=row["run_id"],
        url=row["url"],
        title=row["title"] or "",
        section=row["section"] or "",
        item_id=row["item_id"],
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


async def reclaim_stale_claimed(conn: Any, *, older_than_sec: float) -> int:
    """Return stuck ``claimed`` rows to ``pending`` after Hub/FCC death.

    ``claimed_at`` older than ``older_than_sec`` is the live-tick guard:
    a turn still running inside ``timeout_sec`` is left alone. Pass
    ``older_than_sec=0`` on the first tick after Hub start so a restart
    immediately frees leftovers from the previous process.
    """
    status = await conn.execute(RECLAIM_STALE_CLAIMED_SQL, float(older_than_sec))
    return _update_rowcount(status)


async def reclaim_stale_stage2_claimed(conn: Any, *, older_than_sec: float) -> int:
    status = await conn.execute(RECLAIM_STALE_STAGE2_CLAIMED_SQL, float(older_than_sec))
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
    conn: Any, seed_id: str, *, stage2_trace_id: str
) -> None:
    await conn.execute(
        """
        UPDATE world_pulse_read_seed
        SET stage2_status = 'done',
            stage2_trace_id = $2,
            stage2_completed_at = now(),
            stage2_error = null
        WHERE seed_id = $1
        """,
        seed_id,
        stage2_trace_id,
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
            total += await enqueue_seeds(conn, seeds)
    return total
