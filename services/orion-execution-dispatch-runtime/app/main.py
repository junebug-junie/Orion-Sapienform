from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import ServiceRef
from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly
from orion.core.bus.rpc_health_publish import RpcHealthPublisher

from app.settings import get_settings
from app.store import ExecutionDispatchRuntimeStore
from app.worker import RPC_HEALTH_SINK, ExecutionDispatchRuntimeWorker

_settings = get_settings()
logging.basicConfig(level=getattr(logging, _settings.log_level.upper(), logging.INFO))
logger = logging.getLogger("orion.execution.dispatch.runtime.main")

worker = ExecutionDispatchRuntimeWorker()
store = ExecutionDispatchRuntimeStore(_settings.postgres_uri)
heartbeat_chassis: HeartbeatOnly | None = None
rpc_health_bus: OrionBusAsync | None = None
rpc_health_publisher: RpcHealthPublisher | None = None


def build_heartbeat_chassis() -> HeartbeatOnly:
    """Own, independent bus connection publishing SystemHealthV1 to orion:system:health
    every heartbeat_interval_sec. Deliberately separate from the worker's own RPC bus
    connection to cortex-exec. See
    docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md."""
    s = get_settings()
    return HeartbeatOnly(
        ChassisConfig(
            service_name=s.service_name,
            service_version=s.service_version,
            node_name=s.node_name,
            bus_url=s.orion_bus_url,
            bus_enabled=s.orion_bus_enabled,
            heartbeat_interval_sec=s.heartbeat_interval_sec,
        )
    )


def build_rpc_health_publisher(bus: OrionBusAsync) -> RpcHealthPublisher:
    """Publishes RpcHealthSnapshotV1 from ``bus`` (long-lived, publish-only) after draining
    RPC_HEALTH_SINK, where every dispatch tick's short-lived bus folds its cortex-exec
    rpc_request() outcomes (app/worker.py). Without the sink those per-tick stats were
    discarded with the bus."""
    s = get_settings()
    return RpcHealthPublisher(
        enabled=s.rpc_health_publish_enabled and s.orion_bus_enabled,
        bus_getter=lambda: bus,
        service=s.service_name,
        node=s.node_name,
        instance="main",
        source=ServiceRef(name=s.service_name, version=s.service_version, node=s.node_name),
        interval_sec=s.rpc_health_publish_interval_sec,
        include_channel_latency=s.rpc_health_channel_latency_enabled,
        sinks=[RPC_HEALTH_SINK],
    )


async def _start_rpc_health() -> None:
    global rpc_health_bus, rpc_health_publisher
    s = get_settings()
    if not (s.rpc_health_publish_enabled and s.orion_bus_enabled):
        return
    try:
        bus = OrionBusAsync(url=s.orion_bus_url, enabled=s.orion_bus_enabled)
        await bus.connect()
        rpc_health_bus = bus
        rpc_health_publisher = build_rpc_health_publisher(bus)
        rpc_health_publisher.start()
    except Exception as exc:
        logger.warning("rpc_health_publish_start_failed error=%s", exc)


async def _stop_rpc_health() -> None:
    global rpc_health_bus, rpc_health_publisher
    if rpc_health_publisher is not None:
        await rpc_health_publisher.stop()
        rpc_health_publisher = None
    if rpc_health_bus is not None:
        try:
            await rpc_health_bus.close()
        except Exception as exc:
            logger.warning("rpc_health_bus_close_error error=%s", exc)
        rpc_health_bus = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global heartbeat_chassis
    await worker.start()
    try:
        heartbeat_chassis = build_heartbeat_chassis()
        await heartbeat_chassis.start_background()
        logger.info(
            "system_health_heartbeat_started service=%s interval_sec=%s",
            _settings.service_name,
            _settings.heartbeat_interval_sec,
        )
    except Exception as exc:
        logger.warning("system_health_heartbeat_start_failed error=%s", exc)
        heartbeat_chassis = None
    await _start_rpc_health()
    try:
        yield
    finally:
        await _stop_rpc_health()
        if heartbeat_chassis is not None:
            try:
                await heartbeat_chassis.stop()
            except Exception as exc:
                logger.warning("system_health_heartbeat_stop_error error=%s", exc)
            heartbeat_chassis = None
        await worker.stop()


app = FastAPI(title="orion-execution-dispatch-runtime", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": get_settings().service_name}


@app.get("/latest")
async def latest() -> dict[str, Any]:
    frame = store.load_latest_dispatch_frame()
    if frame is None:
        raise HTTPException(status_code=404, detail="not_found")
    payload = frame.model_dump(mode="json")
    payload["theater_tripwire_active"] = worker.theater_tripwire_active
    return payload
