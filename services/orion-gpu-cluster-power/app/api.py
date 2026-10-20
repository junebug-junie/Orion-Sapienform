import logging
from typing import Optional

from fastapi import FastAPI, APIRouter, Header, HTTPException, Depends, Request

from .settings import settings, Settings
from .service import PsuService
from .bus import start_command_listener_background
from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly
import asyncio


logger = logging.getLogger("orion-psu-proxy.api")


# -------- Dependency wiring --------

def get_settings() -> Settings:
    return settings


def create_service() -> PsuService:
    return PsuService(settings)


def get_service(request: Request) -> PsuService:
    return request.app.state.psu_service

# -------- Router --------

router = APIRouter()

def check_token(x_api_key: Optional[str]):
    if settings.api_token and x_api_key != settings.api_token:
        raise HTTPException(status_code=401, detail="Invalid API token")


@router.post("/on")
async def psu_on(
    x_api_key: Optional[str] = Header(default=None),
    service: PsuService = Depends(get_service),
):
    check_token(x_api_key)
    return await service.on()


@router.post("/off")
async def psu_off(
    x_api_key: Optional[str] = Header(default=None),
    service: PsuService = Depends(get_service),
):
    check_token(x_api_key)
    return await service.off()


@router.post("/cycle")
async def psu_cycle(
    x_api_key: Optional[str] = Header(default=None),
    service: PsuService = Depends(get_service),
):
    check_token(x_api_key)
    return await service.cycle()


# -------- Heartbeat --------

def build_heartbeat_chassis() -> HeartbeatOnly:
    """Standard SystemHealthV1 heartbeat (orion:system:health) on its own bus
    connection, separate from PsuService.bus (the command listener's), so a stuck
    listener cannot silence liveness and vice versa. Replaces a hand-rolled 30s loop
    that reported the fake node name "psu-node". Pattern:
    docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md."""
    return HeartbeatOnly(
        ChassisConfig(
            service_name=settings.service_name,
            service_version=settings.service_version,
            node_name=settings.node_name,
            bus_url=str(settings.orion_bus_url or ""),
            bus_enabled=bool(settings.orion_bus_enabled and settings.orion_bus_url),
            heartbeat_interval_sec=settings.heartbeat_interval_sec,
        )
    )


# -------- App factory (thin main will call this) --------

def create_app() -> FastAPI:
    app = FastAPI(
        title="gpu-cluster-power",
        version="0.1.0",
    )

    # Instantiate core service once and stash it in app.state
    psu_service = create_service()
    app.state.psu_service = psu_service  # type: ignore[attr-defined]

    # Include routes under /psu
    app.include_router(router, prefix="/psu", tags=["psu"])

    @app.on_event("startup")
    async def on_startup():
        # Init bus if enabled, then start listener
        await psu_service.init_bus()
        if psu_service.bus_enabled:
            start_command_listener_background(psu_service)

        # Standard SystemHealthV1 heartbeat on orion:system:health, on its own bus
        # connection (see build_heartbeat_chassis). A failure here is logged, never
        # fatal: the PSU proxy must keep serving /psu/* even when the bus is down.
        app.state.heartbeat_chassis = None
        try:
            chassis = build_heartbeat_chassis()
            await chassis.start_background()
            app.state.heartbeat_chassis = chassis
            logger.info(
                "system_health_heartbeat_started service=%s node=%s interval_sec=%s",
                settings.service_name,
                settings.node_name,
                settings.heartbeat_interval_sec,
            )
        except Exception as exc:
            logger.warning("system_health_heartbeat_start_failed error=%s", exc)

        logger.info("Orion PSU Proxy startup complete.")

    @app.on_event("shutdown")
    async def on_shutdown():
        chassis = getattr(app.state, "heartbeat_chassis", None)
        if chassis is not None:
            try:
                await chassis.stop()
            except Exception as exc:
                logger.warning("system_health_heartbeat_stop_error error=%s", exc)
            app.state.heartbeat_chassis = None

        if psu_service.bus:
            await psu_service.bus.close()

    return app
