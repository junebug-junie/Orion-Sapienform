from __future__ import annotations

import logging
from typing import Optional

import asyncpg

from .settings import settings

logger = logging.getLogger("orion.substrate.telemetry.pg")

_pool: Optional[asyncpg.Pool] = None


async def init_pool() -> asyncpg.Pool:
    """Create the process-wide pool (call once from FastAPI lifespan before the bus chassis)."""
    global _pool
    if _pool is not None:
        return _pool
    _pool = await asyncpg.create_pool(
        dsn=settings.postgres_uri,
        min_size=1,
        max_size=10,
    )
    logger.info("substrate_telemetry_pg_pool_ready max_size=10")
    return _pool


def pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("postgres pool not initialized; lifespan must call init_pool() first")
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
