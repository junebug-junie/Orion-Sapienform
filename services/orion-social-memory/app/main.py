from __future__ import annotations

import contextlib
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Query

from .service import SocialMemoryService
from .settings import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="[SOCIAL_MEMORY] %(levelname)s - %(name)s - %(message)s",
)

service = SocialMemoryService(settings=settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await service.start()
    try:
        yield
    finally:
        with contextlib.suppress(Exception):
            await service.stop()


app = FastAPI(title=settings.app_name, lifespan=lifespan)


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"ok": True, "service": settings.service_name, "version": settings.service_version}


@app.get("/summary")
async def summary(
    platform: str = Query(...),
    room_id: str = Query(...),
    participant_id: str | None = Query(None),
) -> Dict[str, Any]:
    if not platform.strip() or not room_id.strip():
        raise HTTPException(status_code=400, detail="platform_and_room_id_required")
    return await service.get_summary(platform=platform.strip(), room_id=room_id.strip(), participant_id=participant_id)
