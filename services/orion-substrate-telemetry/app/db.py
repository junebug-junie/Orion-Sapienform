from __future__ import annotations

from typing import Any
from uuid import UUID

import asyncpg

# DDL for fixed table `substrate_tier_outcomes_events` (v1).
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS substrate_tier_outcomes_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    correlation_id UUID NOT NULL,
    envelope_kind TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    cold_anchors JSONB NOT NULL DEFAULT '[]'::jsonb,
    tier_outcomes JSONB NOT NULL DEFAULT '{}'::jsonb,
    degraded_producers JSONB NOT NULL DEFAULT '[]'::jsonb,
    source_service TEXT,
    source_node TEXT,
    received_at_utc TIMESTAMPTZ NOT NULL DEFAULT (NOW() AT TIME ZONE 'utc')
);
CREATE INDEX IF NOT EXISTS idx_substrate_tier_corr_received
  ON substrate_tier_outcomes_events (correlation_id, generated_at DESC, received_at_utc DESC);
"""

INSERT_SQL = """
INSERT INTO substrate_tier_outcomes_events (
  correlation_id, envelope_kind, generated_at, cold_anchors, tier_outcomes,
  degraded_producers, source_service, source_node
) VALUES ($1::uuid, $2, $3, $4::jsonb, $5::jsonb, $6::jsonb, $7, $8)
RETURNING id, received_at_utc;
"""

PRUNE_CORRELATION_SQL = """
DELETE FROM substrate_tier_outcomes_events a
USING (
  SELECT id FROM substrate_tier_outcomes_events
  WHERE correlation_id = $1::uuid
  ORDER BY generated_at DESC NULLS LAST, received_at_utc DESC, id DESC
  OFFSET $2::int
) AS doomed
WHERE a.id = doomed.id;
"""

GLOBAL_RETENTION_SQL = """
DELETE FROM substrate_tier_outcomes_events
WHERE received_at_utc < (NOW() AT TIME ZONE 'utc' - $1::interval);
"""

LATEST_SQL = """
SELECT id, correlation_id, envelope_kind, generated_at, cold_anchors, tier_outcomes,
       degraded_producers, source_service, source_node, received_at_utc
FROM substrate_tier_outcomes_events
WHERE correlation_id = $1::uuid
ORDER BY generated_at DESC NULLS LAST, received_at_utc DESC, id DESC
LIMIT 1;
"""

HISTORY_SQL = """
SELECT id, correlation_id, envelope_kind, generated_at, cold_anchors, tier_outcomes,
       degraded_producers, source_service, source_node, received_at_utc
FROM substrate_tier_outcomes_events
WHERE correlation_id = $1::uuid
ORDER BY generated_at DESC NULLS LAST, received_at_utc DESC, id DESC
LIMIT $2::int;
"""


async def ensure_schema(conn: asyncpg.Connection) -> None:
    await conn.execute(CREATE_TABLE_SQL)


async def insert_event(
    conn: asyncpg.Connection,
    *,
    correlation_id: UUID,
    envelope_kind: str,
    generated_at: str,
    cold_anchors: list[Any],
    tier_outcomes: dict[str, Any],
    degraded_producers: list[Any],
    source_service: str | None,
    source_node: str | None,
) -> asyncpg.Record:
    return await conn.fetchrow(
        INSERT_SQL,
        correlation_id,
        envelope_kind,
        generated_at,
        cold_anchors,
        tier_outcomes,
        degraded_producers,
        source_service,
        source_node,
    )


async def prune_correlation(conn: asyncpg.Connection, *, correlation_id: UUID, keep_newest: int) -> None:
    await conn.execute(PRUNE_CORRELATION_SQL, correlation_id, int(keep_newest))


async def global_retention_sweep(conn: asyncpg.Connection, *, max_age_days: int) -> str:
    return await conn.execute(GLOBAL_RETENTION_SQL, f"{int(max_age_days)} days")


async def fetch_latest(conn: asyncpg.Connection, *, correlation_id: UUID) -> asyncpg.Record | None:
    return await conn.fetchrow(LATEST_SQL, correlation_id)


async def fetch_history(conn: asyncpg.Connection, *, correlation_id: UUID, limit: int) -> list[asyncpg.Record]:
    return await conn.fetch(HISTORY_SQL, correlation_id, limit)
