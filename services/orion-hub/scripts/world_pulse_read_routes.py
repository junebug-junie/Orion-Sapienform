"""World-pulse Stage 1/2 schedule + status — read-only operator surface.

Key names are imported from Wallet A/B, never retyped. A dashboard that copied
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

from orion.world_pulse_read.queue import (
    count_seeds_by_status,
    count_stage2_by_status,
    last_stage_timestamps,
)
from orion.world_pulse_read.wallet_a import (
    WALLET_A_COOLDOWN_KEY,
    WALLET_A_COUNT_KEY_PREFIX,
)
from orion.world_pulse_read.wallet_b import (
    WALLET_B_COOLDOWN_KEY,
    WALLET_B_COUNT_KEY_PREFIX,
)

logger = logging.getLogger("orion-hub.world_pulse_read_routes")

router = APIRouter(prefix="/world-pulse-read", tags=["world-pulse-read"])

_NO_CACHE = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}

_EMPTY_QUEUE = {
    "pending": 0,
    "claimed": 0,
    "done": 0,
    "failed": 0,
    "skipped": 0,
}


def _settings() -> Any:
    from app.settings import get_settings

    return get_settings()


def _redis() -> Any:
    from . import main as hub_main

    return getattr(getattr(hub_main, "bus", None), "redis", None)


def _pool() -> Any:
    from . import main as hub_main

    app = getattr(hub_main, "app", None)
    return getattr(getattr(app, "state", None), "memory_pg_pool", None)


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


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


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


def _wallet_block(
    *,
    enabled: bool,
    daily_cap: int,
    cooldown_key: str,
    count_key_prefix: str,
    last_at: str | None,
    done_today: int,
) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "done_today": done_today,
        "daily_cap": daily_cap,
        "cooldown_key": cooldown_key,
        "count_key_prefix": count_key_prefix,
        "last_at": last_at,
    }


@router.get("/api/status")
async def world_pulse_read_status() -> JSONResponse:
    payload: dict[str, Any] = {
        "available": False,
        "local_date": None,
        "tz": None,
        "wallet_a": _wallet_block(
            enabled=False,
            daily_cap=6,
            cooldown_key=WALLET_A_COOLDOWN_KEY,
            count_key_prefix=WALLET_A_COUNT_KEY_PREFIX,
            last_at=None,
            done_today=0,
        ),
        "wallet_b": _wallet_block(
            enabled=False,
            daily_cap=6,
            cooldown_key=WALLET_B_COOLDOWN_KEY,
            count_key_prefix=WALLET_B_COUNT_KEY_PREFIX,
            last_at=None,
            done_today=0,
        ),
        "queue": dict(_EMPTY_QUEUE),
        "stage2_queue": dict(_EMPTY_QUEUE),
        "last_stage1_at": None,
        "last_stage2_at": None,
        "stage2_max_round_trips": 5,
    }
    try:
        cfg = _settings()
        tz_name = getattr(cfg, "HUB_ENDOGENOUS_OUTREACH_TZ", "UTC") or "UTC"
        local_date = _local_date(tz_name)
        payload["local_date"] = local_date
        payload["tz"] = tz_name
        payload["wallet_a"]["enabled"] = bool(
            getattr(cfg, "HUB_WORLD_PULSE_READ_ENABLED", False)
        )
        payload["wallet_a"]["daily_cap"] = int(
            getattr(cfg, "HUB_WORLD_PULSE_READ_DAILY_CAP", 6) or 0
        )
        payload["wallet_b"]["enabled"] = bool(
            getattr(cfg, "HUB_WORLD_PULSE_READ_STAGE2_ENABLED", False)
        )
        payload["wallet_b"]["daily_cap"] = int(
            getattr(cfg, "HUB_WORLD_PULSE_READ_WALLET_B_DAILY_CAP", 6) or 0
        )
        payload["stage2_max_round_trips"] = int(
            getattr(cfg, "HUB_WORLD_PULSE_READ_STAGE2_MAX_ROUND_TRIPS", 5) or 0
        )

        redis = _redis()
        if redis is not None:
            payload["available"] = True
            payload["wallet_a"]["last_at"] = _decode(await redis.get(WALLET_A_COOLDOWN_KEY))
            a_count = _decode(await redis.get(f"{WALLET_A_COUNT_KEY_PREFIX}{local_date}"))
            payload["wallet_a"]["done_today"] = int(a_count) if a_count else 0
            payload["wallet_b"]["last_at"] = _decode(await redis.get(WALLET_B_COOLDOWN_KEY))
            b_count = _decode(await redis.get(f"{WALLET_B_COUNT_KEY_PREFIX}{local_date}"))
            payload["wallet_b"]["done_today"] = int(b_count) if b_count else 0

        pool = _pool()
        if pool is not None:
            async with pool.acquire() as conn:
                payload["queue"] = await count_seeds_by_status(conn)
                payload["stage2_queue"] = await count_stage2_by_status(conn)
                ts = await last_stage_timestamps(conn)
                payload["last_stage1_at"] = _iso(ts.get("last_stage1_at"))
                payload["last_stage2_at"] = _iso(ts.get("last_stage2_at"))
            payload["available"] = True
    except Exception as exc:  # noqa: BLE001 -- a dashboard never 500s
        logger.warning("world_pulse_read_status_unavailable err=%s", exc)
    return JSONResponse(content=payload, headers=_NO_CACHE)
