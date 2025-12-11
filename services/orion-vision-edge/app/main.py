# app/main.py
import asyncio

from fastapi import FastAPI

from .context import settings, camera
from .detector_worker import run_detector_loop
from .routes import router as vision_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Orion Vision Edge Service",
        version=settings.SERVICE_VERSION,
    )

    app.include_router(vision_router)

    @app.on_event("startup")
    async def on_startup():
        camera.start()
        asyncio.create_task(run_detector_loop())

    @app.on_event("shutdown")
    async def on_shutdown():
        camera.stop()

    return app


app = create_app()
