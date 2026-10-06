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
from app.episode_shadow import EpisodeShadowStore
from app.window_state import WindowStore
from app.worker import ConsolidationSuggestRunner, handle_memory_turn_persisted

logger = logging.getLogger(settings.SERVICE_NAME)

bus_hunter: Optional[Hunter] = None
pg_pool: Optional[asyncpg.Pool] = None
grammar_pg_pool: Optional[asyncpg.Pool] = None
bus_client: Optional[OrionBusAsync] = None
_retry_task: Optional[asyncio.Task] = None
_classify_retry_task: Optional[asyncio.Task] = None
_report_task: Optional[asyncio.Task] = None
_referent_task: Optional[asyncio.Task] = None


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
    global bus_hunter, pg_pool, grammar_pg_pool, bus_client, _retry_task, _classify_retry_task, _report_task
    global _referent_task

    dsn = (settings.POSTGRES_URI or "").strip()
    if dsn:
        pg_pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=4)

    grammar_dsn = (settings.MEMORY_CONSOLIDATION_GRAMMAR_DSN or "").strip()
    if grammar_dsn and grammar_dsn != dsn:
        grammar_pg_pool = await asyncpg.create_pool(dsn=grammar_dsn, min_size=1, max_size=2)

    bus_client = OrionBusAsync(url=settings.ORION_BUS_URL, enabled=settings.ORION_BUS_ENABLED)
    await bus_client.connect()

    window_store = WindowStore(pg_pool) if pg_pool is not None else None
    episode_store = EpisodeShadowStore(pg_pool, settings) if pg_pool is not None else None
    suggest_runner = (
        ConsolidationSuggestRunner(pg_pool, window_store, grammar_pool=grammar_pg_pool or pg_pool)
        if pg_pool and window_store
        else None
    )

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
            episode_store=episode_store,
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

    if pg_pool is not None and settings.MEMORY_EPISODE_REPORT_ENABLED:
        from app.episode_report import run_report_loop

        _report_task = asyncio.create_task(run_report_loop(pg_pool, settings))

    if pg_pool is not None and settings.MEMORY_REFERENT_PROJECTOR_ENABLED:
        from app.referent_projector import run_referent_projector_loop

        _referent_task = asyncio.create_task(run_referent_projector_loop(pg_pool, settings))

    app.state.pg_pool = pg_pool
    app.state.bus_hunter = bus_hunter
    yield

    if _retry_task is not None:
        _retry_task.cancel()
    if _classify_retry_task is not None:
        _classify_retry_task.cancel()
    if _report_task is not None:
        _report_task.cancel()
    if _referent_task is not None:
        _referent_task.cancel()
    if bus_hunter is not None:
        await bus_hunter.stop()
    if bus_client is not None:
        await bus_client.close()
    if pg_pool is not None:
        await pg_pool.close()
    if grammar_pg_pool is not None:
        await grammar_pg_pool.close()


app = FastAPI(title="Orion Memory Consolidation", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "postgres": pg_pool is not None,
        "bus": bus_hunter is not None,
        "enabled": settings.MEMORY_CONSOLIDATION_ENABLED,
        "episode_shadow_enabled": settings.MEMORY_EPISODE_SHADOW_ENABLED,
        "referent_projector_enabled": settings.MEMORY_REFERENT_PROJECTOR_ENABLED,
    }
