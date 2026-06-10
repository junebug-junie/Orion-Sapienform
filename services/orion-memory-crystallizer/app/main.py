import logging
from contextlib import asynccontextmanager
from typing import Optional

import asyncpg
from fastapi import FastAPI

from orion.core.bus.bus_service_chassis import ChassisConfig, Hunter
from orion.memory.crystallization.repository import apply_memory_crystallizations_schema

from app.settings import settings

logger = logging.getLogger(settings.SERVICE_NAME)

bus_hunter: Optional[Hunter] = None
pg_pool: Optional[asyncpg.Pool] = None


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
    global bus_hunter, pg_pool

    if settings.ORION_BUS_ENABLED:
        bus_hunter = Hunter(_cfg())
        await bus_hunter.start()
        logger.info("bus_hunter_started")

    dsn = (settings.POSTGRES_URI or "").strip()
    if dsn:
        pg_pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=4)
        if settings.CRYSTALLIZER_AUTO_APPLY_SCHEMA:
            apply_memory_crystallizations_schema(dsn)
            logger.info("memory_crystallizations_schema_applied")

    app.state.pg_pool = pg_pool
    app.state.bus_hunter = bus_hunter
    yield

    if bus_hunter is not None:
        await bus_hunter.stop()
    if pg_pool is not None:
        await pg_pool.close()


app = FastAPI(title="Orion Memory Crystallizer", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "postgres": pg_pool is not None,
        "bus": bus_hunter is not None,
        "graphiti_enabled": settings.GRAPHITI_ENABLED,
    }
