from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI

from app.routes import router
from app.settings import get_settings
from app.worker import ProposalIngestWorker

logging.basicConfig(level=getattr(logging, get_settings().log_level.upper(), logging.INFO))
logger = logging.getLogger("orion.memory-crystallizer")

BOOT_ID = str(uuid.uuid4())

_settings = get_settings()


async def heartbeat_loop(bus_instance: Any) -> None:
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
    from orion.schemas.telemetry.system_health import SystemHealthV1

    logger.info("heartbeat_loop_started boot_id=%s", BOOT_ID)
    while True:
        try:
            payload = SystemHealthV1(
                service=_settings.service_name,
                version=_settings.service_version,
                node=_settings.node_name,
                status="ok",
                boot_id=BOOT_ID,
                last_seen_ts=time.time(),
            ).model_dump(mode="json")
            await bus_instance.publish(
                _settings.health_channel,
                BaseEnvelope(
                    kind="system.health.v1",
                    source=ServiceRef(name=_settings.service_name, version=_settings.service_version),
                    payload=payload,
                ),
            )
        except Exception as exc:
            logger.warning("heartbeat_failed reason=%s", exc)
        await asyncio.sleep(_settings.heartbeat_interval_sec)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from orion.memory.crystallization.repository import (
        CrystallizationRepository,
        apply_memory_crystallizations_schema,
    )

    app.state.settings = _settings
    app.state.bus = None
    app.state.repo = None
    app.state.cards_pool = None
    heartbeat_task: Optional[asyncio.Task] = None
    ingest_worker: Optional[ProposalIngestWorker] = None

    if _settings.auto_apply_schema:
        try:
            apply_memory_crystallizations_schema(_settings.postgres_uri)
        except Exception as exc:
            logger.warning("apply_schema_failed reason=%s", exc)

    try:
        app.state.repo = CrystallizationRepository(_settings.postgres_uri)
    except Exception as exc:
        logger.error("repository_init_failed reason=%s", exc)

    try:
        import asyncpg

        app.state.cards_pool = await asyncpg.create_pool(
            dsn=_settings.postgres_uri, min_size=1, max_size=4
        )
    except Exception as exc:
        logger.warning("cards_pool_unavailable reason=%s — card projection will return payload only", exc)

    if _settings.orion_bus_enabled:
        try:
            from orion.core.bus.async_service import OrionBusAsync

            bus = OrionBusAsync(_settings.orion_bus_url)
            await bus.connect()
            app.state.bus = bus
            heartbeat_task = asyncio.create_task(heartbeat_loop(bus), name="crystallizer-heartbeat")
            ingest_worker = ProposalIngestWorker(
                bus=bus,
                repo=app.state.repo,
                channel=_settings.channel_proposed,
                service_name=_settings.service_name,
            )
            await ingest_worker.start()
        except Exception as exc:
            logger.error("bus_connect_failed reason=%s — running without bus", exc)

    yield

    if ingest_worker:
        await ingest_worker.stop()
    if heartbeat_task and not heartbeat_task.done():
        heartbeat_task.cancel()
    if app.state.cards_pool is not None:
        try:
            await app.state.cards_pool.close()
        except Exception:
            pass
    if app.state.bus is not None:
        try:
            await app.state.bus.disconnect()
        except Exception:
            pass


app = FastAPI(title="orion-memory-crystallizer", lifespan=lifespan)
app.include_router(router)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": _settings.service_name,
        "version": _settings.service_version,
        "boot_id": BOOT_ID,
        "bus_connected": app.state.bus is not None,
        "repository_ready": app.state.repo is not None,
        "graphiti_enabled": _settings.graphiti_enabled,
    }
