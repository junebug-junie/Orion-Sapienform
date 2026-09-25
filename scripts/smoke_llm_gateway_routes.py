import argparse
import asyncio
import os
import uuid
from typing import Dict

import httpx

from orion.core.bus.async_service import OrionBusAsync
from orion.llm.routes import ACCEPTED_LLM_ROUTES, LLM_ROUTE_DISPLAY_ORDER, normalize_llm_route
from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, LLMMessage, ServiceRef




def _pool_routes():
    """Routes and the roles that may serve them, from config/gpu_pool.yaml (the gateway's own source
    since the 2026-09-24 GPU pool cutover; the old LLM_GATEWAY_ROUTE_TABLE_JSON is gone)."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from orion.gpu_pool.config import load_pool_config

    return load_pool_config()


def _load_route_urls() -> Dict[str, str]:
    """Each route's HOME role URL (first role of its class) -- the pool may still place a call elsewhere."""
    cfg = _pool_routes()
    return {route: cfg.url(cfg.classes[spec.work_class].roles[0]) for route, spec in cfg.routes.items()}


def _expected_served_by(route: str) -> set:
    """Every served_by the pool may legitimately answer with for this route ("circe-worker-<role>")."""
    cfg = _pool_routes()
    if route not in cfg.routes:
        raise AssertionError(f"route {route!r} is not in config/gpu_pool.yaml routes")
    return {f"{cfg.host.name}-worker-{role}" for role in cfg.classes[cfg.routes[route].work_class].roles}


async def _rpc_chat(
    bus: OrionBusAsync,
    *,
    route: str,
    expected_served_by: set,
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
    if served_by not in expected_served_by:
        raise AssertionError(f"route={route} served_by={served_by} expected one of {sorted(expected_served_by)}")

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
    # Same check, `harness` (2026-08-20): it is never a human's Compute choice either, and the
    # failure mode is the same one this whole block exists to catch -- an operator's route-table
    # entry carrying no `priority` key at all (easy: neighbouring `chat`/`agent` entries in the
    # same JSON blob carry none) would report `priority=None` and slip past Hub's picker filter
    # as an ordinary interactive lane. Unlike `quick_background`, the expected value is
    # `"system"`, not `"background"` -- `harness` must dispatch immediately, never wait for slot
    # slack, so the two priority values are deliberately not interchangeable here.
    harness = next((r for r in routes if isinstance(r, dict) and r.get("id") == "harness"), None)
    if harness is not None and harness.get("status") != "not_configured":
        if harness.get("priority") != "system":
            raise AssertionError(
                f"harness priority={harness.get('priority')!r} expected 'system'"
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
