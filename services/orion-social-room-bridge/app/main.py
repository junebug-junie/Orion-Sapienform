from __future__ import annotations

import hashlib
import hmac
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, Header, HTTPException, Request

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


def _normalize_signature(value: str | None, *, prefix: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    normalized_prefix = str(prefix or "").strip().lower()
    if normalized_prefix and raw.lower().startswith(normalized_prefix):
        return raw[len(normalized_prefix) :].strip()
    return raw


def _verify_webhook_hmac_signature(payload: bytes, *, secret: str, signature: str | None, prefix: str) -> bool:
    secret_value = str(secret or "").strip()
    if not secret_value:
        return True
    provided = _normalize_signature(signature, prefix=prefix)
    if not provided:
        return False
    expected = hmac.new(secret_value.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(provided, expected)


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
    request: Request,
    payload: Dict[str, Any],
    x_callsyne_webhook_token: Optional[str] = Header(None),
) -> Dict[str, Any]:
    if settings.callsyne_webhook_token and x_callsyne_webhook_token != settings.callsyne_webhook_token:
        raise HTTPException(status_code=401, detail="invalid_callsyne_webhook_token")
    signature_header = request.headers.get(settings.callsyne_webhook_signature_header)
    raw_body = await request.body()
    if not _verify_webhook_hmac_signature(
        raw_body,
        secret=settings.callsyne_webhook_hmac_secret,
        signature=signature_header,
        prefix=settings.callsyne_webhook_signature_prefix,
    ):
        raise HTTPException(status_code=401, detail="invalid_callsyne_webhook_signature")
    return await service.process_callsyne_message(payload)
