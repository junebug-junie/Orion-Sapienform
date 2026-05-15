import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from orion.core.bus.bus_service_chassis import ChassisConfig, Hunter

from app.router import router as rdf_router
from app.service import (
    handle_envelope,
    init_rdf_write_pipeline,
    register_rdf_write_publisher,
    rdf_write_health_snapshot,
    shutdown_rdf_write_pipeline,
)
from app.settings import settings

# Ensure root logger is configured
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
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
        error_channel="orion:system:error",
    )


hunter: Hunter | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global hunter
    logger.info("Starting %s...", settings.SERVICE_NAME)

    await init_rdf_write_pipeline()

    channels = settings.get_all_subscribe_channels()
    logger.info("Subscribing to: %s", channels)

    hunter = Hunter(
        _cfg(),
        patterns=channels,
        handler=handle_envelope,
    )

    await hunter.start_background()

    async def _pub(channel: str, payload: dict) -> None:
        assert hunter is not None
        await hunter.bus.publish(channel, payload)

    register_rdf_write_publisher(_pub)

    yield

    logger.info("Stopping service...")
    await shutdown_rdf_write_pipeline(drain_timeout_sec=8.0)
    register_rdf_write_publisher(None)
    if hunter:
        await hunter.stop()


app = FastAPI(
    title=settings.SERVICE_NAME,
    version=settings.SERVICE_VERSION,
    lifespan=lifespan,
)

app.include_router(rdf_router)


@app.get("/health")
def health():
    body: dict = {
        "status": "ok",
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "bus_connected": hunter.bus.is_connected if hunter and hunter.bus else False,
    }
    body.update(rdf_write_health_snapshot())
    return body
