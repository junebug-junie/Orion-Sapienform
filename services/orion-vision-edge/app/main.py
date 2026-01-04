# app/main.py
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
import logging

from .context import settings, camera, bus
from .detector_worker import run_detector_loop
from .capture_worker import capture_loop
from .routes import router as vision_router

logger = logging.getLogger("orion-vision-edge.main")
logging.basicConfig(level=logging.INFO)

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

    yield

    # Shutdown
    logger.info("Shutting down...")
    capture_task.cancel()
    detector_task.cancel()
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
