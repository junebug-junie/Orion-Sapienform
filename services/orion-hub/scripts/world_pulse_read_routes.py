"""World-pulse Stage 1 schedule — read-only operator surface.

Key names are imported from Wallet A, never retyped. A dashboard that copied
`orion:wp_read:wallet_a:...` as a string literal would keep rendering a
confident 0 the day that prefix changes.

Degrades to an honest payload rather than a 500: a broken panel must never
take Hub down.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from orion.world_pulse_read.wallet_a import (
    WALLET_A_COOLDOWN_KEY,
    WALLET_A_COUNT_KEY_PREFIX,
)

logger = logging.getLogger("orion-hub.world_pulse_read_routes")

router = APIRouter(prefix="/world-pulse-read", tags=["world-pulse-read"])

_NO_CACHE = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}


def _settings() -> Any:
    from app.settings import get_settings

    return get_settings()


def _redis() -> Any:
    from . import main as hub_main

    return getattr(getattr(hub_main, "bus", None), "redis", None)


def _local_date(tz_name: str) -> str:
    try:
        tz = ZoneInfo(tz_name or "UTC")
    except Exception:  # noqa: BLE001 -- same fallback the loop takes
        tz = timezone.utc
    return datetime.now(timezone.utc).astimezone(tz).date().isoformat()


def _decode(raw: object) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, (bytes, bytearray)):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


@router.get("/api/schedule")
async def world_pulse_read_schedule() -> JSONResponse:
    payload: dict[str, Any] = {
        "available": False,
        "enabled": False,
        "done_today": 0,
        "daily_cap": 6,
        "cooldown_key": WALLET_A_COOLDOWN_KEY,
        "count_key_prefix": WALLET_A_COUNT_KEY_PREFIX,
        "cooldown_sec": None,
        "local_date": None,
        "tz": None,
        "last_read_at": None,
    }
    try:
        cfg = _settings()
        payload["enabled"] = bool(
            getattr(cfg, "HUB_WORLD_PULSE_READ_ENABLED", False)
        )
        payload["daily_cap"] = int(
            getattr(cfg, "HUB_WORLD_PULSE_READ_DAILY_CAP", 6) or 0
        )
        payload["cooldown_sec"] = float(
            getattr(cfg, "HUB_WORLD_PULSE_READ_MIN_COOLDOWN_SEC", 0) or 0
        )
        tz_name = getattr(cfg, "HUB_ENDOGENOUS_OUTREACH_TZ", "UTC") or "UTC"
        local_date = _local_date(tz_name)
        payload["local_date"] = local_date
        payload["tz"] = tz_name

        redis = _redis()
        if redis is None:
            return JSONResponse(content=payload, headers=_NO_CACHE)

        last = _decode(await redis.get(WALLET_A_COOLDOWN_KEY))
        count = _decode(await redis.get(f"{WALLET_A_COUNT_KEY_PREFIX}{local_date}"))
        payload["available"] = True
        payload["last_read_at"] = last
        payload["done_today"] = int(count) if count else 0
    except Exception as exc:  # noqa: BLE001 -- a dashboard never 500s
        logger.warning("world_pulse_read_schedule_unavailable err=%s", exc)
    return JSONResponse(content=payload, headers=_NO_CACHE)
