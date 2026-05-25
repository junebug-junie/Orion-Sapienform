from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .settings import get_settings
from .worker import BiometricsSubstrateWorker

_settings = get_settings()
logging.basicConfig(level=getattr(logging, _settings.log_level.upper(), logging.INFO))

worker = BiometricsSubstrateWorker()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await worker.start()
    try:
        yield
    finally:
        await worker.stop()


app = FastAPI(title="orion-substrate-runtime", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": get_settings().service_name}
