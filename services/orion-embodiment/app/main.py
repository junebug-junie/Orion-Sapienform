from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.settings import get_settings
from app.worker import EmbodimentWorker

# Configure logging at import (before the worker loop starts) so the loop's
# perception/speech/journal decisions land in `docker logs`. uvicorn installs its
# own handlers for its loggers but leaves the root/app loggers unconfigured, so
# without this the embodiment worker was silent and had to be diagnosed via live
# DB probing.
logging.basicConfig(
    level=getattr(logging, get_settings().log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    force=True,
)

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
