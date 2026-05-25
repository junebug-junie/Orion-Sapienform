from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from app.settings import get_settings
from app.store import ExecutionDispatchRuntimeStore
from app.worker import ExecutionDispatchRuntimeWorker

_settings = get_settings()
logging.basicConfig(level=getattr(logging, _settings.log_level.upper(), logging.INFO))

worker = ExecutionDispatchRuntimeWorker()
store = ExecutionDispatchRuntimeStore(_settings.postgres_uri)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await worker.start()
    try:
        yield
    finally:
        await worker.stop()


app = FastAPI(title="orion-execution-dispatch-runtime", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": get_settings().service_name}


@app.get("/latest")
async def latest() -> dict[str, Any]:
    frame = store.load_latest_dispatch_frame()
    if frame is None:
        raise HTTPException(status_code=404, detail="not_found")
    return frame.model_dump(mode="json")
