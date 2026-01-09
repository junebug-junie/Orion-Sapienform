import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI

from orion.core.bus.bus_service_chassis import ChassisConfig, Hunter
from app.settings import settings
from app.router import router as rdf_router
from app.service import handle_envelope

# Ensure root logger is configured
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(settings.SERVICE_NAME)

def _cfg() -> ChassisConfig:
    return ChassisConfig(
        service_name=settings.SERVICE_NAME,
        service_version=settings.SERVICE_VERSION,
        node_name=settings.NODE_NAME,
        bus_url=settings.ORION_BUS_URL,
        bus_enabled=settings.ORION_BUS_ENABLED,
        health_channel="orion:system:health",
        error_channel="system.error",
    )

# Global reference to keep the hunter alive
hunter: Hunter = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global hunter
    logger.info(f"🚀 Starting {settings.SERVICE_NAME}...")

    channels = settings.get_all_subscribe_channels()
    
    logger.info(f"Subscribing to: {channels}")

    hunter = Hunter(
        _cfg(),
        patterns=channels,
        handler=handle_envelope
    )

    await hunter.start_background()

    yield

    logger.info("🛑 Stopping service...")
    if hunter:
        await hunter.stop()

app = FastAPI(
    title=settings.SERVICE_NAME,
    version=settings.SERVICE_VERSION,
    lifespan=lifespan
)

app.include_router(rdf_router)

@app.get("/health")
def health():
    return {
        "status": "ok", 
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "bus_connected": hunter.bus.is_connected if hunter and hunter.bus else False
    }
