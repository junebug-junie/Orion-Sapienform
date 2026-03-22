from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, Header, HTTPException

from .service import SocialRoomBridgeService
from .settings import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="[SOCIAL_BRIDGE] %(levelname)s - %(name)s - %(message)s",
)

service = SocialRoomBridgeService(settings=settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await service.start()
    try:
        yield
    finally:
        await service.stop()


app = FastAPI(title=settings.app_name, lifespan=lifespan)


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "service": settings.service_name,
        "version": settings.service_version,
        "platform": settings.social_bridge_platform,
    }


@app.post("/webhooks/callsyne/room-message")
async def callsyne_room_message(
    payload: Dict[str, Any],
    x_callsyne_webhook_token: Optional[str] = Header(None),
) -> Dict[str, Any]:
    if settings.callsyne_webhook_token and x_callsyne_webhook_token != settings.callsyne_webhook_token:
        raise HTTPException(status_code=401, detail="invalid_callsyne_webhook_token")
    return await service.process_callsyne_message(payload)
