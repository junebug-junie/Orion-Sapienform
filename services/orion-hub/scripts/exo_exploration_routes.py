"""Exo Exploration Hub tab -- a read-only proxy over orion-exo-exploration's
`/finds` and `/crawl-runs`.

Settings-driven (`HUB_EXO_EXPLORATION_BASE_URL`), degrading to an honest
"unavailable" payload rather than a 500 if the backing service is
unreachable or not configured -- mirrors `curiosity_routes.py`'s
error-handling style exactly, because this is exactly that kind of surface:
a broken panel must never take Hub down.

Unlike `curiosity_routes.py` / `concept_atlas_routes.py`, this tab is NOT an
iframe to a standalone page -- there is no `/exo-exploration` page route
here. The Hub tab panel (`templates/index.html`'s `data-panel="exo-exploration"`
section) renders directly, fetched and drawn by
`static/js/exo-exploration.js`, matching the shape the design doc asked for
("its own JS file that fetches and renders"). Nothing here writes to
orion-exo-exploration; it is a pure read proxy.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import aiohttp
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

logger = logging.getLogger("orion-hub.exo_exploration_routes")

router = APIRouter(prefix="/api/exo-exploration", tags=["exo-exploration"])

_NO_CACHE = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}


def _settings():
    from app.settings import get_settings

    return get_settings()


async def _proxy_get(path: str, params: dict[str, Any]) -> JSONResponse:
    cfg = _settings()
    base_url = (getattr(cfg, "HUB_EXO_EXPLORATION_BASE_URL", "") or "").strip()
    if not base_url:
        return JSONResponse(
            content={"available": False, "reason": "exo_exploration_not_configured"},
            headers=_NO_CACHE,
        )
    timeout_sec = float(getattr(cfg, "HUB_EXO_EXPLORATION_TIMEOUT_SEC", 10.0) or 10.0)
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    try:
        timeout = aiohttp.ClientTimeout(total=timeout_sec)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    logger.warning(
                        "exo_exploration_proxy_bad_status url=%s status=%s", url, response.status
                    )
                    return JSONResponse(
                        content={"available": False, "reason": "exo_exploration_bad_status", "status": response.status},
                        headers=_NO_CACHE,
                    )
                payload = await response.json()
    except Exception as exc:  # noqa: BLE001 -- a dashboard never 500s
        logger.warning("exo_exploration_proxy_unreachable url=%s err=%s", url, exc)
        return JSONResponse(
            content={"available": False, "reason": "exo_exploration_unreachable", "error": str(exc)[:200]},
            headers=_NO_CACHE,
        )
    payload["available"] = True
    return JSONResponse(content=payload, headers=_NO_CACHE)


@router.get("/finds")
async def exo_exploration_finds(
    category: Optional[str] = Query(default=None),
    min_interest: Optional[float] = Query(default=None),
    status: Optional[str] = Query(default=None),
) -> JSONResponse:
    params: dict[str, Any] = {}
    if category:
        params["category"] = category
    if min_interest is not None:
        params["min_interest"] = min_interest
    if status:
        params["status"] = status
    return await _proxy_get("finds", params)


@router.get("/crawl-runs")
async def exo_exploration_crawl_runs() -> JSONResponse:
    return await _proxy_get("crawl-runs", {})
