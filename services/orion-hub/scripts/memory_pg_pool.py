"""Hub memory Postgres pool open with wait/retry.

Hub and ``orion-sql-db`` often restart together. A one-shot ``create_pool``
during Hub startup loses when Postgres is still in recovery (live 2026-09-18:
pool failed ~5s before Postgres accepted connections). Curiosity and memory
routes read ``app.state.memory_pg_pool`` every tick; if that attribute stays
``None``, they stay on ``stores_not_ready`` for the process lifetime.

Retry here so a brief race does not permanently disable the memory pool.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger("orion-hub.memory_pg_pool")

# Observed co-restart race was ~5s; dirty recovery can be longer. 30 × 2s
# covers the live failure with headroom without new env knobs.
DEFAULT_MAX_ATTEMPTS = 30
DEFAULT_DELAY_SEC = 2.0
DEFAULT_MIN_SIZE = 1
DEFAULT_MAX_SIZE = 6

CreatePoolFn = Callable[..., Awaitable[Any]]
SleepFn = Callable[[float], Awaitable[None]]


async def create_memory_pg_pool_with_retry(
    *,
    dsn: str,
    create_pool: CreatePoolFn,
    sleep: SleepFn = asyncio.sleep,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    delay_sec: float = DEFAULT_DELAY_SEC,
    min_size: int = DEFAULT_MIN_SIZE,
    max_size: int = DEFAULT_MAX_SIZE,
) -> Optional[Any]:
    """Open an asyncpg pool, retrying transient connection failures.

    Returns the pool on success, or ``None`` after ``max_attempts`` failures.
    Does not raise — callers mirror the historical Hub startup contract
    (log + leave pool unset) so the rest of boot still completes.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")

    last_exc: Optional[BaseException] = None
    for attempt in range(1, max_attempts + 1):
        try:
            pool = await create_pool(dsn=dsn, min_size=min_size, max_size=max_size)
            if attempt > 1:
                logger.info(
                    "memory_pg_pool_ready dsn_configured=true attempt=%s/%s",
                    attempt,
                    max_attempts,
                )
            else:
                logger.info("memory_pg_pool_ready dsn_configured=true")
            return pool
        except Exception as exc:
            last_exc = exc
            if attempt >= max_attempts:
                break
            logger.warning(
                "memory_pg_pool_retry attempt=%s/%s delay_sec=%s error=%s",
                attempt,
                max_attempts,
                delay_sec,
                exc,
            )
            await sleep(delay_sec)

    logger.error(
        "memory_pg_pool_failed attempts=%s error=%s",
        max_attempts,
        last_exc,
    )
    return None
