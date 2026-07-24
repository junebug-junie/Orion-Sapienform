from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly

from app.settings import get_settings
from app.worker import EmbodimentWorker

# Configure logging at import (before the worker loop starts) so the loop's
# perception/speech/journal decisions land in `docker logs`. uvicorn installs its
# own handlers for its loggers but leaves the root/app loggers unconfigured, so
# without this the embodiment worker was silent and had to be diagnosed via live
# DB probing.
logging.basicConfig(
    level=getattr(logging, get_settings().log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    force=True,
)

worker = EmbodimentWorker()
logger = logging.getLogger("orion.embodiment.main")
heartbeat_chassis: HeartbeatOnly | None = None


def build_heartbeat_chassis() -> HeartbeatOnly:
    """Own, independent bus connection publishing SystemHealthV1 to orion:system:health
    every heartbeat_interval_sec. Deliberately separate from the worker's own bus connection
    (EmbodimentWorker._bus, which handles perception/intent/outcome/speech traffic). See
    docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md."""
    s = get_settings()
    return HeartbeatOnly(
        ChassisConfig(
            service_name=s.service_name,
            service_version=s.service_version,
            node_name=s.node_name,
            bus_url=s.bus_url,
            bus_enabled=s.bus_enabled,
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
            get_settings().service_name,
            get_settings().heartbeat_interval_sec,
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


app = FastAPI(title="orion-embodiment", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    s = get_settings()
    return {"status": "ok", "service": s.service_name, "enabled": str(s.enabled)}
