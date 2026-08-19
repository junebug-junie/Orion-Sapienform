import argparse
import asyncio
import json
import os
import uuid
from typing import Dict

import httpx

from orion.core.bus.async_service import OrionBusAsync
from orion.llm.routes import ACCEPTED_LLM_ROUTES, LLM_ROUTE_DISPLAY_ORDER, normalize_llm_route
from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, LLMMessage, ServiceRef


DEFAULT_ROUTE_SERVERS = {
    "chat": "atlas-worker-1",
    "agent": "atlas-worker-1",
    "metacog": "atlas-worker-2",
    "quick": "atlas-worker-fast-1",
    # Same llama.cpp process as `quick` under a background admission policy -- one worker, two
    # routes. Without this entry, widening the dispatch loop below to the full route set raised
    # a bare KeyError in `_expected_served_by`.
    "quick_background": "atlas-worker-fast-1",
}

# NOT fixed here, but do not trust the two entries above: `chat` and `agent` actually run on
# CIRCE (`circe-worker-1` / `circe-worker-agent-1`), not atlas. These defaults only apply when
# LLM_ROUTE_<ROUTE>_SERVED_BY is unset AND the live route table omits served_by, which is not
# the current deployment -- see the "latent trap" note in the scarcity roadmap's A2 section.


def _load_route_urls() -> Dict[str, str]:
    raw_json = os.getenv("LLM_GATEWAY_ROUTE_TABLE_JSON")
    if raw_json:
        raw = json.loads(raw_json)
        if not isinstance(raw, dict):
            raise ValueError("LLM_GATEWAY_ROUTE_TABLE_JSON must be a JSON object")
        route_urls: Dict[str, str] = {}
        for route, value in raw.items():
            if isinstance(value, str):
                route_urls[str(route)] = value
            elif isinstance(value, dict):
                url = value.get("url") or value.get("base_url")
                if url:
                    route_urls[str(route)] = url
        return route_urls

    return {
        "chat": os.getenv("LLM_ROUTE_CHAT_URL", ""),
        "agent": os.getenv("LLM_ROUTE_AGENT_URL", ""),
        "metacog": os.getenv("LLM_ROUTE_METACOG_URL", ""),
        "quick": os.getenv("LLM_ROUTE_QUICK_URL", ""),
    }


def _expected_served_by(route: str) -> str:
    override = os.getenv(f"LLM_ROUTE_{route.upper()}_SERVED_BY")
    if override:
        return override
    try:
        return DEFAULT_ROUTE_SERVERS[route]
    except KeyError:
        # A bare KeyError here reads as a smoke crash; it is actually "a route exists that this
        # file has never heard of", which is the drift this whole patch is about.
        raise AssertionError(
            f"route {route!r} has no expected served_by: add it to DEFAULT_ROUTE_SERVERS "
            f"or set LLM_ROUTE_{route.upper()}_SERVED_BY"
        ) from None


async def _rpc_chat(
    bus: OrionBusAsync,
    *,
    route: str,
    expected_served_by: str,
    request_channel: str,
    timeout_sec: float,
) -> None:
    corr_id = str(uuid.uuid4())
    reply_channel = f"orion:exec:result:LLMGatewayService:{corr_id}"
    req = ChatRequestPayload(
        messages=[LLMMessage(role="user", content=f"smoke test ({route})")],
        raw_user_text=f"smoke test ({route})",
        route=route,
        options={"max_tokens": 32, "temperature": 0.2},
    )
    env = BaseEnvelope(
        kind="llm.chat.request",
        source=ServiceRef(name="llm-gateway-route-smoke"),
        correlation_id=corr_id,
        reply_to=reply_channel,
        payload=req.model_dump(mode="json"),
    )
    msg = await bus.rpc_request(
        request_channel,
        env,
        reply_channel=reply_channel,
        timeout_sec=timeout_sec,
    )
    decoded = bus.codec.decode(msg.get("data"))
    if not decoded.ok:
        raise RuntimeError(f"Decode failed: {decoded.error}")

    payload = decoded.envelope.payload if isinstance(decoded.envelope.payload, dict) else {}
    if payload.get("error"):
        raise RuntimeError(f"Gateway error for route={route}: {payload.get('error')}")
    meta = payload.get("meta") or {}
    served_by = meta.get("served_by")
    if served_by != expected_served_by:
        raise AssertionError(f"route={route} served_by={served_by} expected={expected_served_by}")

    print(f"[ok] route={route} served_by={served_by}")


async def _verify_routes_http(gateway_url: str, timeout_sec: float) -> None:
    url = f"{gateway_url.rstrip('/')}/routes"
    async with httpx.AsyncClient(timeout=timeout_sec) as client:
        response = await client.get(url)
        response.raise_for_status()
        payload = response.json()
    # Assert the default is a REAL route, not a specific name. This hardcoded `!= "chat"` until
    # 2026-08-19 while both the live gateway and `services/orion-llm-gateway/.env_example` have
    # said `LLM_ROUTE_DEFAULT=quick` -- so the smoke could only ever pass against a
    # configuration nobody runs. Which lane is default is an operator choice; a default that is
    # not a route at all is the actual bug, and that is what this now guards.
    default_route = str(payload.get("default_route") or "")
    if normalize_llm_route(default_route) is None:
        raise AssertionError(
            f"default_route={default_route!r} is not a recognised route "
            f"(accepted: {sorted(ACCEPTED_LLM_ROUTES)})"
        )
    routes = payload.get("routes") or []
    ids = [str(r.get("id")) for r in routes if isinstance(r, dict)]
    # Every accepted route, not a hardcoded four. Until 2026-08-19 this asserted a list that
    # omitted `quick_background` -- so the smoke passed green for months while the lane Orion's
    # own journalling runs on was absent from the catalog entirely.
    for route_id in LLM_ROUTE_DISPLAY_ORDER:
        if route_id not in ids:
            raise AssertionError(f"GET /routes missing route id={route_id}")
    # And assert the background lane declares itself as one. `priority` is what the Hub filters
    # its picker on; a background route reporting no priority is indistinguishable from an
    # interactive one and would be offered to a human as a normal lane.
    bg = next((r for r in routes if isinstance(r, dict) and r.get("id") == "quick_background"), None)
    if bg is not None and bg.get("status") != "not_configured":
        if bg.get("priority") != "background":
            raise AssertionError(
                f"quick_background priority={bg.get('priority')!r} expected 'background'"
            )
        # `reserved_free_slots` is genuinely OPTIONAL: priority_admission falls back to
        # _DEFAULT_RESERVED_FREE_SLOTS when it is unset, so a route table entry carrying only
        # `"priority": "background"` is a correctly-working background lane. Assert the type
        # only when a value is present -- and exclude bool, which isinstance(x, int) accepts.
        reserved = bg.get("reserved_free_slots")
        if reserved is not None and (isinstance(reserved, bool) or not isinstance(reserved, int)):
            raise AssertionError(
                f"quick_background reserved_free_slots={reserved!r} expected an int or absent"
            )
    for entry in routes:
        if not isinstance(entry, dict):
            continue
        for key in ("id", "served_by", "backend", "status", "latency_ms", "last_checked_at"):
            if key not in entry:
                raise AssertionError(f"route entry missing key={key}: {entry}")
    # Print the value that was actually read. This said "default_route=chat" literally,
    # regardless of the response -- so the smoke reported the fact it was asserting rather
    # than the fact it observed, which is how the stale assertion above stayed invisible.
    print(f"[ok] GET /routes default_route={default_route} routes={ids}")


async def _main_async(args: argparse.Namespace) -> None:
    bus = OrionBusAsync(args.redis)
    await bus.connect()

    route_urls = _load_route_urls()
    for route in ("chat", "agent", "metacog", "quick"):
        if not route_urls.get(route):
            raise RuntimeError(f"Route '{route}' is not configured (missing URL)")

    # Every route, not a hardcoded four. The GET /routes check above proves a route is
    # CATALOGUED; only this loop proves it actually SERVES traffic, and `quick_background` --
    # the lane Orion's own journalling runs on -- was exercised by neither.
    routes_to_test = list(LLM_ROUTE_DISPLAY_ORDER)

    for route in routes_to_test:
        await _rpc_chat(
            bus,
            route=route,
            expected_served_by=_expected_served_by(route),
            request_channel=args.request_channel,
            timeout_sec=args.timeout,
        )

    if args.gateway_url:
        await _verify_routes_http(args.gateway_url, min(args.timeout, 15.0))

    await bus.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for LLM gateway route table.")
    parser.add_argument(
        "--redis",
        default=os.getenv("ORION_BUS_URL", "redis://localhost:6379/0"),
        help="Redis/Orion bus URL.",
    )
    parser.add_argument(
        "--request-channel",
        default=os.getenv("CHANNEL_LLM_INTAKE", "orion:exec:request:LLMGatewayService"),
        help="LLM gateway request channel.",
    )
    parser.add_argument("--timeout", type=float, default=90.0, help="RPC timeout seconds.")
    parser.add_argument(
        "--gateway-url",
        default=os.getenv("LLM_GATEWAY_URL", os.getenv("HUB_LLM_GATEWAY_URL", "")),
        help="Optional LLM gateway base URL for GET /routes verification.",
    )
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
