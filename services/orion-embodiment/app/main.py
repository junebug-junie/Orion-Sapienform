from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.settings import get_settings
from app.worker import EmbodimentWorker

worker = EmbodimentWorker()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await worker.start()
    try:
        yield
    finally:
        await worker.stop()


app = FastAPI(title="orion-embodiment", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    s = get_settings()
    return {"status": "ok", "service": s.service_name, "enabled": str(s.enabled)}
