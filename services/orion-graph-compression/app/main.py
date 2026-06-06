from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException

from app.settings import get_settings
from app.store import CompressionStore
from app.worker import CompressionWorker
from app.stale_listener import run_stale_listener

logging.basicConfig(level=getattr(logging, get_settings().log_level.upper(), logging.INFO))
logger = logging.getLogger("orion.graph-compression")

BOOT_ID = str(uuid.uuid4())

_settings = get_settings()

bus = None
worker: Optional[CompressionWorker] = None
store: Optional[CompressionStore] = None
heartbeat_task: Optional[asyncio.Task] = None
stale_task: Optional[asyncio.Task] = None


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
    global bus, worker, store, heartbeat_task, stale_task

    store = CompressionStore(_settings.postgres_uri)
    try:
        store.ensure_tables()
    except Exception as exc:
        logger.warning("ensure_tables_failed reason=%s", exc)

    if _settings.orion_bus_enabled:
        try:
            from orion.core.bus.async_service import OrionBusAsync
            bus = OrionBusAsync(_settings.orion_bus_url)
            await bus.connect()
            heartbeat_task = asyncio.create_task(heartbeat_loop(bus), name="gc-heartbeat")
            stale_task = asyncio.create_task(
                run_stale_listener(
                    bus=bus,
                    store=store,
                    channel_rdf_enqueue=_settings.channel_rdf_enqueue,
                    channel_stale=_settings.channel_graph_compression_stale,
                ),
                name="gc-stale-listener",
            )
        except Exception as exc:
            logger.error("bus_connect_failed reason=%s — running without bus", exc)

    worker = CompressionWorker(store=store, bus=bus)
    await worker.start()

    yield

    await worker.stop()
    for task in [heartbeat_task, stale_task]:
        if task and not task.done():
            task.cancel()
    if bus:
        try:
            await bus.disconnect()
        except Exception:
            pass


app = FastAPI(title="orion-graph-compression", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    queue_depth = 0
    artifact_count = 0
    if store:
        try:
            queue_depth = store.stale_queue_depth()
            artifact_count = store.artifact_count()
        except Exception:
            pass
    return {
        "status": "ok",
        "service": _settings.service_name,
        "boot_id": BOOT_ID,
        "compression_runtime_enabled": _settings.enable_compression_runtime,
        "stale_queue_depth": queue_depth,
        "artifact_count": artifact_count,
    }


@app.get("/regions")
async def list_regions(scope: Optional[str] = None) -> list[dict[str, Any]]:
    if not store:
        return []
    return store.list_artifacts(scope=scope)


@app.get("/artifacts/{region_id}")
async def get_artifact(region_id: str) -> dict[str, Any]:
    if not store:
        raise HTTPException(status_code=503, detail="store_unavailable")
    artifact = store.get_artifact(region_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="not_found")
    return artifact
