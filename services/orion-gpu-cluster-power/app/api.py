import logging
from typing import Optional

from fastapi import FastAPI, APIRouter, Header, HTTPException, Depends, Request

from .settings import settings, Settings
from .service import PsuService
from .bus import start_command_listener_background
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.telemetry.system_health import SystemHealthV1
import asyncio
import uuid
from datetime import datetime, timezone

# SystemHealthV1 requires boot_id and last_seen_ts. Constructing it without them
# raises inside the heartbeat loop's own try/except, which logs a warning and
# sleeps -- so the service looks alive while publishing no heartbeat at all.
# Confirmed live 2026-08-29 on orion-gpu-cluster-power: one failure per 30s tick,
# indefinitely, and nothing downstream noticed the silence.
# BOOT_ID identifies THIS process run, so a consumer can tell a restart from a
# continuous uptime (same convention as services/orion-whisper-tts/app/main.py).
BOOT_ID = str(uuid.uuid4())


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

            # Start Heartbeat
            app.state.heartbeat_task = asyncio.create_task(heartbeat_loop(psu_service))

        logger.info("Orion PSU Proxy startup complete.")

    @app.on_event("shutdown")
    async def on_shutdown():
        if hasattr(app.state, "heartbeat_task") and app.state.heartbeat_task:
            app.state.heartbeat_task.cancel()
            try:
                await app.state.heartbeat_task
            except asyncio.CancelledError:
                pass

        if psu_service.bus:
            await psu_service.bus.close()

    return app

async def heartbeat_loop(service: PsuService):
    """Publishes a heartbeat every 30 seconds."""
    logger.info("Heartbeat loop started.")
    try:
        while True:
            try:
                payload = SystemHealthV1(
                    service=service.settings.service_name,
                    version=service.settings.service_version,
                    node="psu-node",
                    status="ok",
                    boot_id=BOOT_ID,
                    last_seen_ts=datetime.now(timezone.utc),
                ).model_dump(mode="json")

                await service.bus.publish("orion:system:health", BaseEnvelope(
                    kind="system.health.v1",
                    source=ServiceRef(name=service.settings.service_name, version=service.settings.service_version),
                    payload=payload
                ))
            except Exception as e:
                logger.warning(f"Heartbeat failed: {e}")

            await asyncio.sleep(30)
    except asyncio.CancelledError:
        logger.info("Heartbeat loop stopping...")
