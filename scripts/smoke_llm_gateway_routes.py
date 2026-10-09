import argparse
import asyncio
import os
import uuid
from typing import Dict

from orion.core.bus.async_service import OrionBusAsync
from orion.llm.routes import LLM_ROUTE_DISPLAY_ORDER
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
    """Every served_by the pool may legitimately answer with for this route ("<node>-worker-<role>")."""
    cfg = _pool_routes()
    if route not in cfg.routes:
        raise AssertionError(f"route {route!r} is not in config/gpu_pool.yaml routes")
    return {f"{cfg.role_host(role)}-worker-{role}" for role in cfg.classes[cfg.routes[route].work_class].roles}


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


async def _verify_route_view(bus: OrionBusAsync, timeout_sec: float) -> None:
    """The per-route catalog every former GET /routes reader now builds from GPU pool state
    (stage 6.3, orion/gpu_pool/route_view.py). Proves the pool answers the state RPC with its config
    and that the view it yields still carries every route and its definitional priority. (The
    gateway's own default route was only visible through GET /routes; it is not re-checked here.)"""
    from orion.gpu_pool.route_view import SOURCE_POOL, fetch_route_view

    view = await fetch_route_view(bus, source="llm-gateway-route-smoke", timeout_sec=timeout_sec)
    if view.get("source") != SOURCE_POOL:
        raise AssertionError(f"GPU pool state unavailable (source={view.get('source')!r}); every route is unknown")
    routes = view.get("routes") or []
    ids = [str(r.get("id")) for r in routes if isinstance(r, dict)]
    # Every accepted route, not a hardcoded four. Until 2026-08-19 this asserted a list that
    # omitted `quick_background` -- so the smoke passed green for months while the lane Orion's
    # own journalling runs on was absent from the catalog entirely.
    for route_id in LLM_ROUTE_DISPLAY_ORDER:
        if route_id not in ids:
            raise AssertionError(f"pool route view missing route id={route_id}")
    # `priority` is what the Hub filters its picker on; a background/system route reporting no
    # priority would be offered to a human as an ordinary interactive lane.
    by_id = {r.get("id"): r for r in routes if isinstance(r, dict)}
    for route_id, expected in (("quick_background", "background"), ("harness", "system")):
        entry = by_id.get(route_id)
        if entry is not None and entry.get("priority") != expected:
            raise AssertionError(f"{route_id} priority={entry.get('priority')!r} expected {expected!r}")
    for entry in routes:
        for key in ("id", "served_by", "backend", "status", "model", "n_ctx", "vision"):
            if key not in entry:
                raise AssertionError(f"route entry missing key={key}: {entry}")
    status = {r["id"]: r["status"] for r in routes}
    print(f"[ok] pool route view routes={ids} status={status}")


async def _main_async(args: argparse.Namespace) -> None:
    bus = OrionBusAsync(args.redis)
    await bus.connect()

    route_urls = _load_route_urls()
    for route in ("chat", "agent", "metacog", "quick"):
        if not route_urls.get(route):
            raise RuntimeError(f"Route '{route}' is not configured (missing URL)")

    # Every route, not a hardcoded four. The pool route-view check below proves a route is
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

    await _verify_route_view(bus, min(args.timeout, 15.0))

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
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
