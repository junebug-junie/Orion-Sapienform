"""Hub client for LLM gateway GET /routes catalog."""

from __future__ import annotations

import logging
from typing import Any

import aiohttp

from orion.llm.routes import ACCEPTED_LLM_ROUTES, LLM_ROUTE_DISPLAY_ORDER, normalize_llm_route

from scripts.settings import settings

logger = logging.getLogger("orion-hub.llm-gateway")

# Derived, not re-typed -- see orion/llm/routes.py. Hardcoding this set is what dropped
# `quick_background` on the floor here: the gateway would report it and this client silently
# filtered it out of every payload the Hub ever saw.
VALID_ROUTE_IDS = ACCEPTED_LLM_ROUTES


class LlmGatewayClientError(Exception):
    """Controlled LLM gateway client failure."""


def _base_url() -> str:
    return str(settings.HUB_LLM_GATEWAY_URL or "").strip().rstrip("/")


def _timeout() -> aiohttp.ClientTimeout:
    return aiohttp.ClientTimeout(total=float(settings.HUB_LLM_GATEWAY_TIMEOUT_SEC))


async def fetch_routes() -> dict[str, Any]:
    base = _base_url()
    if not base:
        raise LlmGatewayClientError("HUB_LLM_GATEWAY_URL is not configured")
    url = f"{base}/routes"
    try:
        async with aiohttp.ClientSession(timeout=_timeout()) as session:
            async with session.get(url) as response:
                if response.status >= 400:
                    body = await response.text()
                    raise LlmGatewayClientError(
                        f"LLM gateway /routes HTTP {response.status}: {body[:240]}"
                    )
                payload = await response.json()
    except aiohttp.ClientError as exc:
        logger.warning("LLM gateway /routes unreachable: %s", exc)
        raise LlmGatewayClientError("LLM gateway /routes unreachable") from exc
    if not isinstance(payload, dict):
        raise LlmGatewayClientError("LLM gateway /routes returned non-object payload")
    return _normalize_routes_payload(payload)


def _normalize_routes_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """The pure half of `fetch_routes`: gateway payload in, Hub payload out.

    Split out so the route-set behaviour is testable without HTTP. That matters here
    specifically: the bug this module carried was not in the fetch, it was in the reassembly
    below, and a test that could only reach it through aiohttp would not have been written.
    """
    # normalize_llm_route resolves aliases too, so a gateway reporting a legacy alias as its
    # default no longer silently collapses to "chat" here.
    default_route = normalize_llm_route(payload.get("default_route")) or "chat"
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
                    "priority": item.get("priority"),
                    "reserved_free_slots": item.get("reserved_free_slots"),
                    # True/False from the worker's own /props; None means the
                    # probe could not answer. The composer greys out its attach
                    # button on anything that is not True.
                    "vision": item.get("vision"),
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
                "priority": None,
                "reserved_free_slots": None,
            },
        )
    ordered = [by_id[rid] for rid in LLM_ROUTE_DISPLAY_ORDER]
    return {"default_route": default_route, "routes": ordered}
