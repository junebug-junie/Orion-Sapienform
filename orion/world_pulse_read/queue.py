from __future__ import annotations

import json
from typing import Any, Sequence

from orion.schemas.world_pulse_read import WorldPulseReadSeedV1
from orion.world_pulse_read.seeds import seeds_from_digest_payload

_PRIORITY = {"finding": 0, "digest_item": 10}

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
    completed_at timestamptz null
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

RECLAIM_STALE_CLAIMED_SQL = """
UPDATE world_pulse_read_seed
SET status = 'pending', claimed_at = NULL
WHERE status = 'claimed'
  AND claimed_at IS NOT NULL
  AND claimed_at < now() - ($1 * interval '1 second')
"""


async def ensure_seed_queue_schema(conn: Any) -> None:
    await conn.execute(ENSURE_TABLE_SQL)
    await conn.execute(ENSURE_CLAIM_INDEX_SQL)
    await conn.execute(ENSURE_RUN_INDEX_SQL)


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


async def claim_next_seed(conn: Any) -> WorldPulseReadSeedV1 | None:
    row = await conn.fetchrow(CLAIM_SQL)
    if not row:
        return None
    return WorldPulseReadSeedV1(
        seed_id=row["seed_id"],
        kind=row["kind"],
        run_id=row["run_id"],
        url=row["url"],
        title=row["title"] or "",
        section=row["section"] or "",
        item_id=row["item_id"],
    )


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


async def mark_seed_done(conn: Any, seed_id: str, *, trace_id: str) -> None:
    await conn.execute(
        """
        UPDATE world_pulse_read_seed
        SET status = 'done', trace_id = $2, completed_at = now(), last_error = null
        WHERE seed_id = $1
        """,
        seed_id,
        trace_id,
    )


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


async def enqueue_from_recent_digests(conn: Any, *, limit_digests: int = 5) -> int:
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
        total += await enqueue_seeds(conn, seeds)
    return total
