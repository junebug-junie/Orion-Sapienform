import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

import asyncpg
from fastapi import FastAPI

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope
from orion.core.bus.bus_service_chassis import ChassisConfig, Hunter

from app.retry_degraded_classifies import run_classify_retry_loop
from app.retry_failed_windows import run_retry_loop
from app.settings import settings
from app.window_state import WindowStore
from app.worker import ConsolidationSuggestRunner, handle_memory_turn_persisted

logger = logging.getLogger(settings.SERVICE_NAME)

bus_hunter: Optional[Hunter] = None
pg_pool: Optional[asyncpg.Pool] = None
bus_client: Optional[OrionBusAsync] = None
_retry_task: Optional[asyncio.Task] = None
_classify_retry_task: Optional[asyncio.Task] = None


def _cfg() -> ChassisConfig:
    return ChassisConfig(
        service_name=settings.SERVICE_NAME,
        service_version=settings.SERVICE_VERSION,
        node_name=settings.NODE_NAME,
        bus_url=settings.ORION_BUS_URL,
        bus_enabled=settings.ORION_BUS_ENABLED,
        health_channel=settings.ORION_HEALTH_CHANNEL,
        error_channel=settings.ERROR_CHANNEL,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bus_hunter, pg_pool, bus_client, _retry_task, _classify_retry_task

    dsn = (settings.POSTGRES_URI or "").strip()
    if dsn:
        pg_pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=4)

    bus_client = OrionBusAsync(url=settings.ORION_BUS_URL, enabled=settings.ORION_BUS_ENABLED)
    await bus_client.connect()

    window_store = WindowStore(pg_pool) if pg_pool is not None else None
    suggest_runner = ConsolidationSuggestRunner(pg_pool, window_store) if pg_pool and window_store else None

    async def _handler(env: BaseEnvelope) -> None:
        if not settings.MEMORY_CONSOLIDATION_ENABLED:
            return
        if env.kind != "memory.turn.persisted.v1":
            return
        if bus_client is None or window_store is None or suggest_runner is None:
            logger.warning("memory_consolidation_not_ready kind=%s", env.kind)
            return
        await handle_memory_turn_persisted(
            env,
            bus=bus_client,
            window_store=window_store,
            suggest_runner=suggest_runner,
        )

    if settings.ORION_BUS_ENABLED:
        bus_hunter = Hunter(
            _cfg(),
            patterns=[settings.CHANNEL_MEMORY_TURN_PERSISTED],
            handler=_handler,
        )
        await bus_hunter.start_background()
        logger.info("memory_consolidation_bus_hunter_started channel=%s", settings.CHANNEL_MEMORY_TURN_PERSISTED)

    if pg_pool is not None and bus_client is not None and suggest_runner is not None and window_store is not None:
        _retry_task = asyncio.create_task(
            run_retry_loop(pool=pg_pool, bus=bus_client, window_store=window_store, suggest_runner=suggest_runner)
        )
        _classify_retry_task = asyncio.create_task(
            run_classify_retry_loop(
                pool=pg_pool,
                bus=bus_client,
                window_store=window_store,
                suggest_runner=suggest_runner,
            )
        )

    app.state.pg_pool = pg_pool
    app.state.bus_hunter = bus_hunter
    yield

    if _retry_task is not None:
        _retry_task.cancel()
    if _classify_retry_task is not None:
        _classify_retry_task.cancel()
    if bus_hunter is not None:
        await bus_hunter.stop()
    if bus_client is not None:
        await bus_client.close()
    if pg_pool is not None:
        await pg_pool.close()


app = FastAPI(title="Orion Memory Consolidation", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "postgres": pg_pool is not None,
        "bus": bus_hunter is not None,
        "enabled": settings.MEMORY_CONSOLIDATION_ENABLED,
    }
