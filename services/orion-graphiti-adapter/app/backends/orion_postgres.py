from __future__ import annotations

from typing import Any

import asyncpg

from app.store import neighborhood as pg_neighborhood
from app.store import upsert_episode as pg_upsert_episode


async def ingest_episode(pool: asyncpg.Pool, **kwargs: Any) -> dict[str, list[str]]:
    edge_ids = await pg_upsert_episode(pool, **kwargs)
    return {"edge_ids": edge_ids}


async def get_neighborhood(
    pool: asyncpg.Pool, crystallization_id: str, *, depth: int = 1
) -> dict[str, Any]:
    return await pg_neighborhood(pool, crystallization_id, depth=depth)
