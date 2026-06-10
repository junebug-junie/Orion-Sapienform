from __future__ import annotations

from typing import Any

import asyncpg

from orion.memory.crystallization.schemas import CrystallizationLinkV1


async def insert_link(
    pool: asyncpg.Pool,
    *,
    from_crystallization_id: str,
    link: CrystallizationLinkV1,
) -> None:
    from_id = _normalize_uuid(from_crystallization_id)
    to_id = _normalize_uuid(link.target_crystallization_id)
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO memory_crystallization_links
                (from_crystallization_id, to_crystallization_id, relation, confidence, note)
            VALUES ($1::uuid, $2::uuid, $3, $4, $5)
            ON CONFLICT (from_crystallization_id, to_crystallization_id, relation) DO UPDATE SET
                confidence = EXCLUDED.confidence,
                note = EXCLUDED.note
            """,
            from_id,
            to_id,
            link.relation,
            link.confidence,
            link.note,
        )


async def list_links(pool: asyncpg.Pool, crystallization_id: str) -> list[dict[str, Any]]:
    cid = _normalize_uuid(crystallization_id)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT link_id, from_crystallization_id, to_crystallization_id, relation, confidence, note, created_at
            FROM memory_crystallization_links
            WHERE from_crystallization_id = $1::uuid OR to_crystallization_id = $1::uuid
            ORDER BY created_at
            """,
            cid,
        )
    return [dict(r) for r in rows]


async def neighborhood(pool: asyncpg.Pool, crystallization_id: str, *, depth: int = 1) -> dict[str, Any]:
    """Return crystallization link neighborhood (not Graphiti)."""
    links = await list_links(pool, crystallization_id)
    node_ids = {crystallization_id}
    for link in links:
        node_ids.add(str(link["from_crystallization_id"]))
        node_ids.add(str(link["to_crystallization_id"]))

    nodes: list[dict[str, Any]] = []
    async with pool.acquire() as conn:
        for nid in node_ids:
            row = await conn.fetchrow(
                "SELECT crystallization_id, kind, subject, summary, status, salience FROM memory_crystallizations WHERE crystallization_id = $1::uuid",
                _normalize_uuid(nid),
            )
            if row:
                nodes.append(dict(row))

    return {
        "crystallization_id": crystallization_id,
        "depth": depth,
        "nodes": nodes,
        "edges": links,
    }


def _normalize_uuid(crystallization_id: str) -> str:
    cid = str(crystallization_id)
    if cid.startswith("crys_"):
        hex_id = cid.replace("crys_", "")
        if len(hex_id) == 32:
            return f"{hex_id[:8]}-{hex_id[8:12]}-{hex_id[12:16]}-{hex_id[16:20]}-{hex_id[20:]}"
    return cid
