from __future__ import annotations

from contextlib import asynccontextmanager

import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly
from orion.schemas.telemetry.system_health import BusConsumerReadinessV1

from .service import MeshGuardianService
from .settings import settings

logger = logging.getLogger("orion.mesh.guardian.main")

guardian = MeshGuardianService(settings)
heartbeat_chassis: HeartbeatOnly | None = None


def build_heartbeat_chassis() -> HeartbeatOnly:
    """Own, independent bus connection publishing SystemHealthV1 to orion:system:health
    every heartbeat_interval_sec. Deliberately separate from `guardian.bus`, which watches
    orion:equilibrium:snapshot and drives remediation. See
    docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md."""
    return HeartbeatOnly(
        ChassisConfig(
            service_name=settings.service_name,
            service_version=settings.service_version,
            node_name=settings.node_name,
            bus_url=settings.orion_bus_url,
            bus_enabled=True,
            heartbeat_interval_sec=settings.heartbeat_interval_sec,
        )
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global heartbeat_chassis
    await guardian.start()
    try:
        heartbeat_chassis = build_heartbeat_chassis()
        await heartbeat_chassis.start_background()
        logger.info(
            "system_health_heartbeat_started service=%s interval_sec=%s",
            settings.service_name,
            settings.heartbeat_interval_sec,
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
        await guardian.stop()


app = FastAPI(title="orion-mesh-guardian", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "service": settings.service_name,
        "version": settings.service_version,
        "enabled": settings.enabled,
        "auto_remediate": settings.auto_remediate,
    }


@app.get("/ready")
async def ready() -> JSONResponse:
    redis = getattr(guardian.bus, "redis", None)
    bus_ok = redis is not None and guardian.bus.enabled
    eq_ok = guardian.equilibrium_subscriber_alive() or not settings.enabled
    ok = bus_ok and eq_ok
    body = BusConsumerReadinessV1(
        ok=ok,
        http_alive=True,
        bus_consumer_ready=bus_ok,
        intake_channel=settings.channel_equilibrium_snapshot,
        subscriber_count=1 if eq_ok else 0,
        dependency_status="available" if ok else "unavailable",
        error=None if ok else "guardian dependencies not ready",
    )
    status_code = 200 if ok else 503
    return JSONResponse(body.model_dump(mode="json"), status_code=status_code)
