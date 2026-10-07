"""Hub Compute-lane catalog (``GET /api/llm-routes``), built from GPU pool state.

GPU pool stage 6.3: this used to proxy orion-llm-gateway's ``GET /routes``, a compatibility view the
gateway itself generated from pool state. The Hub now asks the pool directly
(``orion:gpu_pool:state`` RPC on the Hub's forked RPC bus, with the pool's config so routes can
be mapped to roles) and builds the same view with ``orion.gpu_pool.route_view``. The payload the
browser sees is unchanged (``default_route``, ``routes[]`` with status/priority/vision/gate_open),
plus ``source``: ``gpu_pool``, or ``gpu_pool_unavailable`` when the pool could not be asked -- in
which case every route is ``unknown``, never a guessed ``up``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from orion.gpu_pool.route_view import SOURCE_UNAVAILABLE, build_route_view, fetch_route_view
from orion.llm.routes import (
    ACCEPTED_LLM_ROUTES,
    BACKGROUND_LLM_ROUTES,
    LLM_ROUTE_DISPLAY_ORDER,
    SYSTEM_LLM_ROUTES,
    normalize_llm_route,
)

from scripts.settings import settings

logger = logging.getLogger("orion-hub.llm-routes")

# Derived, not re-typed -- see orion/llm/routes.py. Hardcoding this set is what dropped
# `quick_background` on the floor here: the gateway would report it and this client silently
# filtered it out of every payload the Hub ever saw.
VALID_ROUTE_IDS = ACCEPTED_LLM_ROUTES

# The Hub's own Compute default, the same value as app.js's HUB_COMPUTE_DEFAULT. Before stage 6.3
# `default_route` echoed the gateway's LLM_ROUTE_DEFAULT; pool state carries no such thing, so the
# payload now states the Hub's constant explicitly rather than pretending the gateway reported it.
HUB_DEFAULT_ROUTE = "quick"

# The picker polls every 30 s per open tab; one pool read serves every tab for this long.
_CACHE_SEC = 10.0
_FAILURE_CACHE_SEC = 2.0
_cache: dict[str, Any] = {"at": 0.0, "ttl": 0.0, "view": None}
_cache_lock = asyncio.Lock()


def reset_route_view_cache() -> None:
    _cache.update(at=0.0, ttl=0.0, view=None)


def _rpc_bus() -> Any:
    """The Hub's forked RPC bus (same one the GPU pool panel uses), or None before startup."""
    from scripts import main

    bus = getattr(main, "rpc_bus", None) or getattr(main, "bus", None)
    return bus if bus is not None and getattr(bus, "enabled", False) else None


async def _route_view() -> dict[str, Any]:
    async with _cache_lock:
        now = time.monotonic()
        if _cache["view"] is not None and now - _cache["at"] < _cache["ttl"]:
            return _cache["view"]
        bus = _rpc_bus()
        if bus is None:
            view = build_route_view(None)
        else:
            view = await fetch_route_view(bus, source=settings.SERVICE_NAME,
                                          timeout_sec=float(settings.HUB_GPU_POOL_RPC_TIMEOUT_SEC))
        ttl = _FAILURE_CACHE_SEC if view.get("source") == SOURCE_UNAVAILABLE else _CACHE_SEC
        _cache.update(at=now, ttl=ttl, view=view)
        return view


async def fetch_routes() -> dict[str, Any]:
    """The Compute picker's catalog. Never raises: an unreachable pool is every route ``unknown``."""
    view = await _route_view()
    payload = _normalize_routes_payload({"default_route": HUB_DEFAULT_ROUTE, "routes": view.get("routes") or []})
    payload["source"] = view.get("source") or SOURCE_UNAVAILABLE
    return payload


def _priority_for(route_id: str, reported: Any) -> str | None:
    """Reported priority, or the route's definitional one -- whichever says 'background'/'system'.

    Fail-safe rather than fail-open: `priority` is what the composer filters its picker on, and
    a background lane that arrives without it (older gateway, route absent from the route
    table) would otherwise be offered as an ordinary interactive lane. `system` (harness,
    2026-08-20) is the same fail-safe for a route that must dispatch immediately -- not a
    yielding lane -- but is still never a human's Compute choice.
    """
    value = str(reported or "").strip().lower() or None
    if value == "background" or route_id in BACKGROUND_LLM_ROUTES:
        return "background"
    if value == "system" or route_id in SYSTEM_LLM_ROUTES:
        return "system"
    return value


def _normalize_routes_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """The pure half of `fetch_routes`: route view in, Hub payload out.

    Split out so the route-set behaviour is testable without HTTP. That matters here
    specifically: the bug this module carried was not in the fetch, it was in the reassembly
    below, and a test that could only reach it through a transport would not have been written.
    """
    # normalize_llm_route resolves aliases too, so a gateway reporting a legacy alias as its
    # default no longer silently collapses to "chat" here.
    # Fallback is "quick", not "chat": an unrecognised gateway default must not be shown as
    # the most contended lane (chat is Juniper's own always-resident worker).
    default_route = normalize_llm_route(payload.get("default_route")) or "quick"
    routes_raw = payload.get("routes") or []
    routes: list[dict[str, Any]] = []
    if isinstance(routes_raw, list):
        for item in routes_raw:
            if not isinstance(item, dict):
                continue
            route_id = str(item.get("id") or "").strip().lower()
            if route_id not in VALID_ROUTE_IDS:
                continue
            routes.append(
                {
                    "id": route_id,
                    "served_by": item.get("served_by"),
                    "backend": item.get("backend"),
                    "status": str(item.get("status") or "unknown"),
                    "latency_ms": item.get("latency_ms"),
                    "last_checked_at": item.get("last_checked_at"),
                    # Passed through so the composer can filter its picker on what a route IS
                    # rather than on a name list that drifts. "background" means the lane
                    # yields, so a human choosing it would just get worse latency.
                    "priority": _priority_for(route_id, item.get("priority")),
                    "reserved_free_slots": item.get("reserved_free_slots"),
                    # True/False from the worker's own /props; None means the
                    # probe could not answer. The composer greys out its attach
                    # button on anything that is not True.
                    "vision": item.get("vision"),
                    # True/False for an operator-gated route (chat-burst: is Juniper
                    # lending her chat lane right now?), None for every other route.
                    "gate_open": item.get("gate_open"),
                }
            )
    by_id = {r["id"]: r for r in routes}
    # These were TWO more hardcoded ("chat", "quick", "agent", "metacog") tuples, and they --
    # not VALID_ROUTE_IDS -- were the ones actually dropping `quick_background`: the filter
    # above let it through and this backfill-then-reorder quietly reassembled the payload from
    # a four-name list, so widening the filter alone changed nothing visible. Verified live
    # 2026-08-19: the gateway returned 5 routes and the Hub still served 4.
    for route_id in LLM_ROUTE_DISPLAY_ORDER:
        by_id.setdefault(
            route_id,
            {
                "id": route_id,
                "served_by": None,
                "backend": None,
                "status": "unknown",
                "latency_ms": None,
                "last_checked_at": None,
                "vision": None,
                # NOT None. This backfill runs when the gateway did not report the route at
                # all -- an older gateway mid-rolling-deploy, or a route dropped from the route
                # table. Asserting "no priority" there is what makes the composer offer a
                # yielding lane to a human, so the definitional answer is used instead.
                "priority": _priority_for(route_id, None),
                "reserved_free_slots": None,
                "gate_open": None,
            },
        )
    ordered = [by_id[rid] for rid in LLM_ROUTE_DISPLAY_ORDER]
    return {"default_route": default_route, "routes": ordered}
