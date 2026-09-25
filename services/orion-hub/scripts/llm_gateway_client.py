"""Hub client for LLM gateway GET /routes catalog."""

from __future__ import annotations

import logging
from typing import Any

import aiohttp

from orion.llm.routes import (
    ACCEPTED_LLM_ROUTES,
    BACKGROUND_LLM_ROUTES,
    LLM_ROUTE_DISPLAY_ORDER,
    SYSTEM_LLM_ROUTES,
    normalize_llm_route,
)

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


async def _gateway_json(method: str, path: str, *, json_body: Any = None) -> dict[str, Any]:
    """One HTTP round-trip to the gateway, returning its JSON object or raising the
    controlled client error. Shared by the catalog fetch and the gate helpers below so
    all three fail the same way (unreachable -> LlmGatewayClientError, never a raw aiohttp
    exception leaking into a route handler)."""
    base = _base_url()
    if not base:
        raise LlmGatewayClientError("HUB_LLM_GATEWAY_URL is not configured")
    url = f"{base}{path}"
    # session.get / session.put by name rather than session.request(method, ...): the
    # existing tests fake ClientSession with just those verbs, and GET must not send a body.
    kwargs = {} if json_body is None else {"json": json_body}
    try:
        async with aiohttp.ClientSession(timeout=_timeout()) as session:
            async with getattr(session, method.lower())(url, **kwargs) as response:
                if response.status >= 400:
                    body = await response.text()
                    raise LlmGatewayClientError(
                        f"LLM gateway {path} HTTP {response.status}: {body[:240]}"
                    )
                payload = await response.json()
    except aiohttp.ClientError as exc:
        logger.warning("LLM gateway %s unreachable: %s", path, exc)
        raise LlmGatewayClientError(f"LLM gateway {path} unreachable") from exc
    if not isinstance(payload, dict):
        raise LlmGatewayClientError(f"LLM gateway {path} returned non-object payload")
    return payload


async def fetch_routes() -> dict[str, Any]:
    payload = await _gateway_json("GET", "/routes")
    return _normalize_routes_payload(payload)


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
    """The pure half of `fetch_routes`: gateway payload in, Hub payload out.

    Split out so the route-set behaviour is testable without HTTP. That matters here
    specifically: the bug this module carried was not in the fetch, it was in the reassembly
    below, and a test that could only reach it through aiohttp would not have been written.
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
