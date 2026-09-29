"""Reading operator surface: schedule/status, per-read outputs, and controls.

Controls (submit/cancel/retry) live in ``orion.world_pulse_read.operator``;
this module only guards and maps them. The ``/reading`` page is the Hub
Reading tab body (iframe, same pattern as ``/gpu-pool``).

Key names are imported from Wallet A/B, never retyped. A dashboard that copied
`orion:wp_read:wallet_a:...` as a string literal would keep rendering a
confident 0 the day that prefix changes.

Degrades to an honest payload rather than a 500: a broken panel must never
take Hub down.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Literal, Optional
from zoneinfo import ZoneInfo

import httpx
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, ValidationError

from orion.world_pulse_read import operator as reading_operator
from orion.world_pulse_read.queue import (
    count_retry_state,
    count_seeds_by_status,
    count_stage2_by_status,
    last_stage_timestamps,
)
from orion.world_pulse_read.wallet_a import (
    WALLET_A_COOLDOWN_KEY,
    WALLET_A_COUNT_KEY_PREFIX,
    WALLET_A_RETRY_NOT_BEFORE_KEY,
)
from orion.world_pulse_read.wallet_b import (
    WALLET_B_COOLDOWN_KEY,
    WALLET_B_COUNT_KEY_PREFIX,
    WALLET_B_RETRY_NOT_BEFORE_KEY,
)

logger = logging.getLogger("orion-hub.world_pulse_read_routes")

router = APIRouter(prefix="/world-pulse-read", tags=["world-pulse-read"])
page_router = APIRouter(tags=["world-pulse-read"])

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

_EMPTY_RETRIES = {
    "max_attempts": 3,
    "stage1_pending_retry": 0,
    "stage2_pending_retry": 0,
    "stage1_exhausted": 0,
    "stage2_exhausted": 0,
}


def _settings() -> Any:
    from app.settings import get_settings

    return get_settings()


def _document_policy() -> Any:
    make = getattr(_settings(), "reading_document_policy", None)
    return make() if callable(make) else None


def _bus() -> Any:
    from . import main as hub_main

    return getattr(hub_main, "bus", None)


def _redis() -> Any:
    return getattr(_bus(), "redis", None)


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
        "cooldown_key": WALLET_A_COOLDOWN_KEY,
        "count_key_prefix": WALLET_A_COUNT_KEY_PREFIX,
        "local_date": None,
        "tz": None,
        "last_read_at": None,
    }
    try:
        cfg = _settings()
        payload["enabled"] = bool(
            getattr(cfg, "HUB_WORLD_PULSE_READ_ENABLED", False)
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
    cooldown_key: str,
    count_key_prefix: str,
    last_at: str | None,
    done_today: int,
) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "done_today": done_today,
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
            cooldown_key=WALLET_A_COOLDOWN_KEY,
            count_key_prefix=WALLET_A_COUNT_KEY_PREFIX,
            last_at=None,
            done_today=0,
        ),
        "wallet_b": _wallet_block(
            enabled=False,
            cooldown_key=WALLET_B_COOLDOWN_KEY,
            count_key_prefix=WALLET_B_COUNT_KEY_PREFIX,
            last_at=None,
            done_today=0,
        ),
        "queue": dict(_EMPTY_QUEUE),
        "stage2_queue": dict(_EMPTY_QUEUE),
        "retries": dict(_EMPTY_RETRIES),
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
        payload["wallet_b"]["enabled"] = bool(
            getattr(cfg, "HUB_WORLD_PULSE_READ_STAGE2_ENABLED", False)
        )
        payload["stage2_max_round_trips"] = int(
            getattr(cfg, "HUB_WORLD_PULSE_READ_STAGE2_MAX_ROUND_TRIPS", 5) or 0
        )
        max_attempts = int(getattr(cfg, "HUB_WORLD_PULSE_READ_MAX_ATTEMPTS", 3) or 1)
        payload["retries"]["max_attempts"] = max_attempts

        redis = _redis()
        if redis is not None:
            payload["available"] = True
            payload["wallet_a"]["last_at"] = _decode(await redis.get(WALLET_A_COOLDOWN_KEY))
            # Set only when a turn was refused before reading and its slot refunded.
            payload["wallet_a"]["retry_not_before"] = _decode(
                await redis.get(WALLET_A_RETRY_NOT_BEFORE_KEY)
            )
            a_count = _decode(await redis.get(f"{WALLET_A_COUNT_KEY_PREFIX}{local_date}"))
            payload["wallet_a"]["done_today"] = int(a_count) if a_count else 0
            payload["wallet_b"]["last_at"] = _decode(await redis.get(WALLET_B_COOLDOWN_KEY))
            # Set only when a turn was refused before reading and its slot refunded.
            payload["wallet_b"]["retry_not_before"] = _decode(
                await redis.get(WALLET_B_RETRY_NOT_BEFORE_KEY)
            )
            b_count = _decode(await redis.get(f"{WALLET_B_COUNT_KEY_PREFIX}{local_date}"))
            payload["wallet_b"]["done_today"] = int(b_count) if b_count else 0

        pool = _pool()
        if pool is not None:
            async with pool.acquire() as conn:
                payload["queue"] = await count_seeds_by_status(conn)
                payload["stage2_queue"] = await count_stage2_by_status(conn)
                payload["retries"] = await count_retry_state(conn, max_attempts=max_attempts)
                ts = await last_stage_timestamps(conn)
                payload["last_stage1_at"] = _iso(ts.get("last_stage1_at"))
                payload["last_stage2_at"] = _iso(ts.get("last_stage2_at"))
            payload["available"] = True
    except Exception as exc:  # noqa: BLE001 -- a dashboard never 500s
        logger.warning("world_pulse_read_status_unavailable err=%s", exc)
    return JSONResponse(content=payload, headers=_NO_CACHE)


# --- Reading tab: page, per-read outputs, controls ---------------------------------------------

CSRF_HEADER_VALUE = "orion-hub"


@page_router.get("/reading")
async def reading_page() -> HTMLResponse:
    from .main import TEMPLATES_DIR, build_hub_ui_asset_version

    template = (TEMPLATES_DIR / "reading.html").read_text(encoding="utf-8")
    return HTMLResponse(template.replace("{{HUB_UI_ASSET_VERSION}}", build_hub_ui_asset_version()),
                        headers=_NO_CACHE)


def _require_pool() -> Any:
    pool = _pool()
    if pool is None:
        raise HTTPException(503, "reading_db_unavailable")
    return pool


def _operator_error(exc: reading_operator.OperatorActionError) -> HTTPException:
    return HTTPException(exc.http_status, exc.code)


def _require_operator(request: Request, x_requested_with: str | None) -> None:
    # Same cross-site request forgery rule as the GPU pool panel: a custom header
    # forces a CORS preflight Hub never grants, and JSON rules out simple POSTs.
    if x_requested_with != CSRF_HEADER_VALUE or "application/json" not in request.headers.get("content-type", ""):
        raise HTTPException(403, "reading_control_requires_hub_page")


@router.get("/api/reads")
async def list_reads(
    phase: str = "all", kind: Optional[str] = None, include_stale: bool = False,
    limit: int = 50, offset: int = 0,
) -> JSONResponse:
    pool = _require_pool()
    try:
        async with pool.acquire() as conn:
            payload = await reading_operator.list_reads(
                conn, phase=phase, kind=kind, include_stale=include_stale, limit=limit, offset=offset,
            )
    except reading_operator.OperatorActionError as exc:
        raise _operator_error(exc) from exc
    return JSONResponse(content=payload, headers=_NO_CACHE)


@router.get("/api/reads/{seed_id}")
async def read_detail(seed_id: str) -> JSONResponse:
    pool = _require_pool()
    try:
        async with pool.acquire() as conn:
            payload = await reading_operator.read_detail(conn, seed_id)
    except reading_operator.OperatorActionError as exc:
        raise _operator_error(exc) from exc
    return JSONResponse(content=payload, headers=_NO_CACHE)


class SubmitReadBody(BaseModel):
    url: str = Field(min_length=1, max_length=4000)
    why_now: str = Field(default="", max_length=4000)
    title: str = Field(default="", max_length=1000)


class RetryReadBody(BaseModel):
    stage: Literal[1, 2]


def _source_ref() -> Any:
    from orion.core.bus.bus_schemas import ServiceRef

    from . import main as hub_main

    pipeline = getattr(hub_main, "world_pulse_read_pipeline", None)
    return getattr(pipeline, "_source_ref", None) or ServiceRef(name="orion-hub")


@router.post("/api/reads")
async def submit_read(
    body: SubmitReadBody, request: Request,
    x_requested_with: str | None = Header(default=None),
) -> JSONResponse:
    _require_operator(request, x_requested_with)
    pool = _require_pool()
    try:
        async with pool.acquire() as conn:
            receipt = await reading_operator.submit_read(
                conn, url=body.url.strip(), why_now=body.why_now.strip(), title=body.title.strip(),
                bus=_bus(), source=_source_ref(), documents=_document_policy(),
            )
    except ValidationError as exc:
        raise HTTPException(400, "invalid_source_url") from exc
    except ValueError as exc:
        # Source policy codes (urls.py / documents.py); no SQL, DSN or file text.
        raise HTTPException(400, str(exc)[:200]) from exc
    logger.info("reading_operator_submit seed_id=%s status=%s", receipt.get("seed_id"), receipt.get("status"))
    return JSONResponse(content=receipt, headers=_NO_CACHE)


_DURABLE_FINISHED = frozenset({"completed", "failed"})


async def _cancel_durable_run(run_id: str) -> dict[str, Any]:
    base = str(getattr(_settings(), "HUB_READING_DURABLE_URL", "") or "http://127.0.0.1:8124")
    try:
        async with httpx.AsyncClient(base_url=base.rstrip("/"), timeout=10) as client:
            resp = await client.post(f"/runs/{run_id}/cancel")
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"durable_unavailable:{type(exc).__name__}") from exc
    if resp.status_code == 404:
        # Binding saved but never accepted by the runner; the worker resubmits it next tick.
        raise HTTPException(409, "durable_run_not_found_retry_shortly")
    if resp.status_code >= 400:
        raise HTTPException(502, f"durable_cancel_failed:{resp.status_code}")
    return resp.json()


@router.post("/api/reads/{seed_id}/cancel")
async def cancel_read(
    seed_id: str, request: Request,
    x_requested_with: str | None = Header(default=None),
) -> JSONResponse:
    _require_operator(request, x_requested_with)
    pool = _require_pool()
    try:
        async with pool.acquire() as conn:
            plan = await reading_operator.cancel_read(conn, seed_id)
    except reading_operator.OperatorActionError as exc:
        raise _operator_error(exc) from exc
    if plan["action"] == "cancel_durable_run":
        state = await _cancel_durable_run(plan["run_id"])
        plan["durable_status"] = state.get("status")
        # The runner answers a cancel on a finished run with that run's final status.
        plan["run_already_finished"] = plan["durable_status"] in _DURABLE_FINISHED
    logger.info("reading_operator_cancel seed_id=%s action=%s stage=%s run_id=%s",
                seed_id, plan["action"], plan["stage"], plan.get("run_id"))
    return JSONResponse(content=plan, headers=_NO_CACHE)


@router.post("/api/reads/{seed_id}/retry")
async def retry_read(
    seed_id: str, body: RetryReadBody, request: Request,
    x_requested_with: str | None = Header(default=None),
) -> JSONResponse:
    _require_operator(request, x_requested_with)
    pool = _require_pool()
    try:
        async with pool.acquire() as conn:
            result = await reading_operator.retry_read(
                conn, seed_id, stage=body.stage,
                digest_item_max_age_sec=float(
                    getattr(_settings(), "HUB_WORLD_PULSE_READ_DIGEST_ITEM_MAX_AGE_DAYS", 0) or 0
                ) * 86400.0,
            )
    except reading_operator.OperatorActionError as exc:
        raise _operator_error(exc) from exc
    logger.info("reading_operator_retry seed_id=%s stage=%s", seed_id, body.stage)
    return JSONResponse(content=result, headers=_NO_CACHE)
