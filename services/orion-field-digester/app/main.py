from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.settings import get_settings
from app.worker import FieldDigesterWorker

_settings = get_settings()
logging.basicConfig(level=getattr(logging, _settings.log_level.upper(), logging.INFO))

worker = FieldDigesterWorker()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await worker.start()
    try:
        yield
    finally:
        await worker.stop()


app = FastAPI(title="orion-field-digester", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": get_settings().service_name}
