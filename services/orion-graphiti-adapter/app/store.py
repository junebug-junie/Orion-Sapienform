from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import asyncpg
import psycopg2

logger = logging.getLogger(__name__)


def _sql_path() -> Path:
    here = Path(__file__).resolve().parents[3] / "orion" / "core" / "storage" / "sql" / "graphiti_projection.sql"
    return here


def apply_graphiti_schema(dsn: str) -> None:
    sql = _sql_path().read_text(encoding="utf-8")
    with psycopg2.connect(dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(sql)


async def upsert_episode(
    pool: asyncpg.Pool,
    *,
    episode_id: str,
    crystallization_id: str,
    kind: str,
    subject: str,
    summary: str,
    status: str,
    metadata: dict[str, Any],
) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO graphiti_episodes
                (episode_id, crystallization_id, kind, subject, summary, status, metadata, synced_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8)
            ON CONFLICT (episode_id) DO UPDATE SET
                subject = EXCLUDED.subject,
                summary = EXCLUDED.summary,
                status = EXCLUDED.status,
                metadata = EXCLUDED.metadata,
                synced_at = EXCLUDED.synced_at
            """,
            episode_id,
            crystallization_id,
            kind,
            subject,
            summary,
            status,
            json.dumps(metadata),
            datetime.now(timezone.utc),
        )
        entity_id = f"gent_{crystallization_id}"
        await conn.execute(
            """
            INSERT INTO graphiti_entities (entity_id, crystallization_id, name, metadata)
            VALUES ($1, $2, $3, $4::jsonb)
            ON CONFLICT (entity_id) DO UPDATE SET name = EXCLUDED.name
            """,
            entity_id,
            crystallization_id,
            subject,
            json.dumps({"kind": kind}),
        )
        edge_id = f"ged_{crystallization_id}"
        await conn.execute(
            """
            INSERT INTO graphiti_edges (edge_id, from_id, to_id, relation, metadata)
            VALUES ($1, $2, $3, $4, $5::jsonb)
            ON CONFLICT (edge_id) DO NOTHING
            """,
            edge_id,
            entity_id,
            episode_id,
            "has_episode",
            json.dumps({"crystallization_id": crystallization_id}),
        )


async def neighborhood(pool: asyncpg.Pool, crystallization_id: str) -> dict[str, Any]:
    async with pool.acquire() as conn:
        episodes = await conn.fetch(
            "SELECT * FROM graphiti_episodes WHERE crystallization_id = $1",
            crystallization_id,
        )
        entities = await conn.fetch(
            "SELECT * FROM graphiti_entities WHERE crystallization_id = $1",
            crystallization_id,
        )
        edges = await conn.fetch(
            """
            SELECT * FROM graphiti_edges
            WHERE from_id LIKE $1 OR to_id LIKE $1
            """,
            f"%{crystallization_id}%",
        )
    return {
        "crystallization_id": crystallization_id,
        "nodes": [dict(e) for e in entities] + [dict(e) for e in episodes],
        "edges": [dict(e) for e in edges],
    }
