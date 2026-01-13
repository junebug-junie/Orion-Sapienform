# app/main.py
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
import logging

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.telemetry.system_health import SystemHealthV1

from .context import settings, camera, bus
from .detector_worker import run_detector_loop
from .capture_worker import capture_loop
from .routes import router as vision_router

logger = logging.getLogger("orion-vision-edge.main")
logging.basicConfig(level=logging.INFO)

async def heartbeat_loop(bus_instance):
    """Publishes a heartbeat every 30 seconds."""
    logger.info("Heartbeat loop started.")
    try:
        while True:
            try:
                payload = SystemHealthV1(
                    service=settings.SERVICE_NAME,
                    version=settings.SERVICE_VERSION,
                    node="vision-edge-node",
                    status="ok"
                ).model_dump(mode="json")

                await bus_instance.publish("orion:system:health", BaseEnvelope(
                    kind="system.health.v1",
                    source=ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION),
                    payload=payload
                ))
            except Exception as e:
                logger.warning(f"Heartbeat failed: {e}")

            await asyncio.sleep(30)
    except asyncio.CancelledError:
        logger.info("Heartbeat loop stopping...")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Vision Edge Service...")
    camera.start()

    if bus.enabled:
        await bus.connect()
        logger.info("Bus connected.")

    # Start workers
    capture_task = asyncio.create_task(capture_loop())
    detector_task = asyncio.create_task(run_detector_loop())
    heartbeat_task = asyncio.create_task(heartbeat_loop(bus)) if bus.enabled else None

    yield

    # Shutdown
    logger.info("Shutting down...")
    capture_task.cancel()
    detector_task.cancel()
    if heartbeat_task:
        heartbeat_task.cancel()

    camera.stop()
    if bus.enabled:
        await bus.close()

def create_app() -> FastAPI:
    app = FastAPI(
        title="Orion Vision Edge Service",
        version=settings.SERVICE_VERSION,
        lifespan=lifespan
    )

    app.include_router(vision_router)

    return app


app = create_app()
