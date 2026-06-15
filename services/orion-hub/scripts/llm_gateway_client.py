"""Hub client for LLM gateway GET /routes catalog."""

from __future__ import annotations

import logging
from typing import Any

import aiohttp

from scripts.settings import settings

logger = logging.getLogger("orion-hub.llm-gateway")

VALID_ROUTE_IDS = frozenset({"chat", "quick", "agent", "metacog"})


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
    default_route = str(payload.get("default_route") or "chat").strip().lower()
    if default_route not in VALID_ROUTE_IDS:
        default_route = "chat"
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
                }
            )
    by_id = {r["id"]: r for r in routes}
    for route_id in ("chat", "quick", "agent", "metacog"):
        by_id.setdefault(
            route_id,
            {
                "id": route_id,
                "served_by": None,
                "backend": None,
                "status": "unknown",
                "latency_ms": None,
                "last_checked_at": None,
            },
        )
    ordered = [by_id[rid] for rid in ("chat", "quick", "agent", "metacog")]
    return {"default_route": default_route, "routes": ordered}
