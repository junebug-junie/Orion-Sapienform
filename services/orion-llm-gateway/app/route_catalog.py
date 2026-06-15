"""Route catalog and upstream health cache for GET /routes."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from .llm_backend import RouteTarget, get_route_targets
from .settings import settings

CATALOG_ROUTE_IDS = ("chat", "quick", "agent", "metacog")
_CACHE_TTL_SEC = 15.0


@dataclass(frozen=True)
class RouteHealthEntry:
    route_id: str
    served_by: Optional[str]
    backend: Optional[str]
    status: str
    latency_ms: Optional[int]
    last_checked_at: Optional[str]


_cache: Dict[str, RouteHealthEntry] = {}
_cache_lock = asyncio.Lock()
_last_refresh_mono: float = 0.0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _probe_one(route_id: str, target: RouteTarget) -> RouteHealthEntry:
    url = f"{target.url.rstrip('/')}/health"
    start = time.monotonic()
    status = "down"
    latency_ms: Optional[int] = None
    try:
        timeout = float(settings.llm_route_health_timeout_sec or 1.5)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            latency_ms = int((time.monotonic() - start) * 1000)
            status = "up" if response.status_code < 400 else "down"
    except Exception:
        if latency_ms is None:
            latency_ms = int((time.monotonic() - start) * 1000)
        status = "down"
    return RouteHealthEntry(
        route_id=route_id,
        served_by=target.served_by,
        backend=target.backend,
        status=status,
        latency_ms=latency_ms,
        last_checked_at=_utc_now_iso(),
    )


async def refresh_route_health_cache(*, force: bool = False) -> None:
    global _last_refresh_mono
    async with _cache_lock:
        now = time.monotonic()
        if not force and _cache and (now - _last_refresh_mono) < _CACHE_TTL_SEC:
            return
        targets = get_route_targets()
        entries: Dict[str, RouteHealthEntry] = {}
        for route_id in CATALOG_ROUTE_IDS:
            target = targets.get(route_id)
            if target is None:
                entries[route_id] = RouteHealthEntry(
                    route_id=route_id,
                    served_by=None,
                    backend=None,
                    status="not_configured",
                    latency_ms=None,
                    last_checked_at=_utc_now_iso(),
                )
            else:
                entries[route_id] = await _probe_one(route_id, target)
        _cache.clear()
        _cache.update(entries)
        _last_refresh_mono = now


def _entry_to_dict(entry: RouteHealthEntry) -> Dict[str, Any]:
    return {
        "id": entry.route_id,
        "served_by": entry.served_by,
        "backend": entry.backend,
        "status": entry.status,
        "latency_ms": entry.latency_ms,
        "last_checked_at": entry.last_checked_at,
    }


def build_routes_response() -> Dict[str, Any]:
    targets = get_route_targets()
    routes: List[Dict[str, Any]] = []
    for route_id in CATALOG_ROUTE_IDS:
        cached = _cache.get(route_id)
        if cached is not None:
            routes.append(_entry_to_dict(cached))
            continue
        target = targets.get(route_id)
        if target is None:
            routes.append(
                {
                    "id": route_id,
                    "served_by": None,
                    "backend": None,
                    "status": "not_configured",
                    "latency_ms": None,
                    "last_checked_at": None,
                }
            )
        else:
            routes.append(
                {
                    "id": route_id,
                    "served_by": target.served_by,
                    "backend": target.backend,
                    "status": "unknown",
                    "latency_ms": None,
                    "last_checked_at": None,
                }
            )
    return {
        "default_route": str(settings.llm_route_default or "chat"),
        "routes": routes,
    }


async def get_routes_payload() -> Dict[str, Any]:
    await refresh_route_health_cache()
    return build_routes_response()


def reset_route_health_cache_for_tests() -> None:
    global _last_refresh_mono
    _cache.clear()
    _last_refresh_mono = 0.0
