from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly

from app.routers.capabilities import router as capabilities_router
from app.routers.datasets import router as datasets_router
from app.routers.drift import router as drift_router
from app.routers.edges import router as edges_router
from app.routers.events import router as events_router
from app.routers.models import router as models_router
from app.routers.runs import router as runs_router
from app.routers.segments import router as segments_router
from app.routers.topics import router as topics_router
from app.settings import settings
from app.services.readiness import readiness_payload
from app.services.drift import drift_daemon_loop
from app.storage.repository import ensure_tables


logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
logger = logging.getLogger("orion-topic-foundry")

heartbeat_chassis: HeartbeatOnly | None = None


def build_heartbeat_chassis() -> HeartbeatOnly:
    """Own, independent bus connection publishing SystemHealthV1 to orion:system:health
    every heartbeat_interval_sec. Deliberately separate from the drift daemon's own bus
    usage (see docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md)."""
    return HeartbeatOnly(
        ChassisConfig(
            service_name=settings.service_name,
            service_version=settings.service_version,
            node_name=settings.node_name,
            bus_url=settings.orion_bus_url,
            bus_enabled=settings.orion_bus_enabled,
            heartbeat_interval_sec=settings.heartbeat_interval_sec,
        )
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global heartbeat_chassis
    ensure_tables()
    logger.info("Topic Foundry service starting")
    # Runs execute as in-process BackgroundTasks and this service has one
    # replica, so a run left `running`/`queued` at startup has no worker that
    # could still advance it -- it is restart residue and will sit there
    # forever. Confirmed live 2026-08-29: six such runs, the oldest 21 hours
    # old, and ZERO complete runs for the Orion model, so the concept-atlas
    # ingest returned `topic_foundry_no_completed_run` and the graph had no
    # source at all. See app/services/run_recovery.py.
    try:
        from app.services.run_recovery import recover_stranded_runs

        recover_stranded_runs()
    except Exception as exc:  # noqa: BLE001 - never block startup on recovery
        logger.warning("run_recovery_startup_failed error=%s", exc)
    drift_task = None
    if settings.topic_foundry_drift_daemon:
        logger.info("Starting drift daemon")
        drift_task = asyncio.create_task(drift_daemon_loop())
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
        if drift_task:
            drift_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await drift_task


app = FastAPI(title="Orion Topic Foundry", version=settings.service_version, lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": settings.service_name,
        "version": settings.service_version,
        "node": settings.node_name,
    }


@app.get("/ready")
def ready():
    return readiness_payload()


app.include_router(datasets_router)
app.include_router(capabilities_router)
app.include_router(drift_router)
app.include_router(edges_router)
app.include_router(events_router)
app.include_router(models_router)
app.include_router(runs_router)
app.include_router(segments_router)
app.include_router(topics_router)
