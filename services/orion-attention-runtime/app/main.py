from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly

from app.settings import get_settings
from app.store import AttentionRuntimeStore
from app.worker import AttentionRuntimeWorker

_settings = get_settings()
logging.basicConfig(level=getattr(logging, _settings.log_level.upper(), logging.INFO))
logger = logging.getLogger("orion.attention.runtime.main")

worker = AttentionRuntimeWorker()
store = AttentionRuntimeStore(_settings.postgres_uri)
heartbeat_chassis: HeartbeatOnly | None = None


def build_heartbeat_chassis() -> HeartbeatOnly:
    """Own, independent bus connection publishing SystemHealthV1 to orion:system:health
    every heartbeat_interval_sec. This service has no other bus connection (Postgres-poll
    only worker) -- purely additive. See
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
    try:
        yield
    finally:
        if heartbeat_chassis is not None:
            try:
                await heartbeat_chassis.stop()
            except Exception as exc:
                logger.warning("system_health_heartbeat_stop_error error=%s", exc)
            heartbeat_chassis = None
        await worker.stop()


app = FastAPI(title="orion-attention-runtime", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": get_settings().service_name}


@app.get("/latest")
async def latest() -> dict[str, Any]:
    frame = store.load_latest_attention_frame()
    if frame is None:
        raise HTTPException(status_code=404, detail="not_found")
    return frame.model_dump(mode="json")
