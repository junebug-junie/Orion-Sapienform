from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from app.settings import get_settings
from app.store import SelfStateRuntimeStore
from app.worker import SelfStateRuntimeWorker

_settings = get_settings()
logging.basicConfig(level=getattr(logging, _settings.log_level.upper(), logging.INFO))

worker = SelfStateRuntimeWorker()
store = SelfStateRuntimeStore(_settings.postgres_uri)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await worker.start()
    try:
        yield
    finally:
        await worker.stop()


app = FastAPI(title="orion-self-state-runtime", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": get_settings().service_name}


@app.get("/latest")
async def latest() -> dict[str, Any]:
    state = store.load_latest_self_state()
    if state is None:
        raise HTTPException(status_code=404, detail="not_found")
    return state.model_dump(mode="json")
