import logging
from typing import Optional

from fastapi import FastAPI, APIRouter, Header, HTTPException, Depends, Request

from .settings import settings, Settings
from .service import PsuService
from .bus import start_command_listener_background

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
        logger.info("Orion PSU Proxy startup complete.")

    return app
