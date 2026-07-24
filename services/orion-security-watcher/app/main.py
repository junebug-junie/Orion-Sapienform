import asyncio
import logging

from fastapi import FastAPI

from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly

from .bus_worker import bus_worker
from .context import ctx
from .routers import debug, state, test, ui

logger = logging.getLogger("orion-security-watcher")

app = FastAPI(
    title="Orion Security Watcher",
    version=ctx.settings.SERVICE_VERSION,
)

# Routers
app.include_router(ui.router)
app.include_router(state.router)
app.include_router(test.router)
app.include_router(debug.router)


@app.get("/health")
async def health():
    s = ctx.state_store.load()
    return {
        "ok": True,
        "service": ctx.settings.SERVICE_NAME,
        "version": ctx.settings.SERVICE_VERSION,
        "enabled": ctx.settings.SECURITY_ENABLED,
        "armed": s.armed,
        "mode": s.mode,
        "bus_enabled": ctx.bus.enabled,
        "vision_channel": ctx.settings.VISION_EVENTS_SUBSCRIBE_RAW,
    }


def build_heartbeat_chassis() -> HeartbeatOnly:
    """Own, independent bus connection publishing SystemHealthV1 to orion:system:health
    every HEARTBEAT_INTERVAL_SEC. Deliberately separate from `ctx.bus` (Security Watcher's
    vision-events/guard-signal publish connection) so this rollout (see
    docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md) cannot
    interfere with the existing bus_worker lifecycle."""
    return HeartbeatOnly(
        ChassisConfig(
            service_name=ctx.settings.SERVICE_NAME,
            service_version=ctx.settings.SERVICE_VERSION,
            node_name=ctx.settings.NODE_NAME,
            bus_url=ctx.settings.ORION_BUS_URL,
            bus_enabled=ctx.settings.ORION_BUS_ENABLED,
            heartbeat_interval_sec=ctx.settings.HEARTBEAT_INTERVAL_SEC,
        )
    )


@app.on_event("startup")
async def on_startup():
    if ctx.settings.SECURITY_ENABLED and ctx.bus.enabled:
        asyncio.create_task(bus_worker(ctx))
    try:
        app.state.heartbeat_chassis = build_heartbeat_chassis()
        await app.state.heartbeat_chassis.start_background()
        logger.info(
            "system_health_heartbeat_started service=%s interval_sec=%s",
            ctx.settings.SERVICE_NAME,
            ctx.settings.HEARTBEAT_INTERVAL_SEC,
        )
    except Exception as exc:
        logger.warning("system_health_heartbeat_start_failed error=%s", exc)
        app.state.heartbeat_chassis = None


@app.on_event("shutdown")
async def on_shutdown():
    heartbeat_chassis = getattr(app.state, "heartbeat_chassis", None)
    if heartbeat_chassis is not None:
        try:
            await heartbeat_chassis.stop()
        except Exception as exc:
            logger.warning("system_health_heartbeat_stop_error error=%s", exc)
