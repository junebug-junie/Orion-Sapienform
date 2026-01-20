import argparse
import asyncio
import json
import os
import uuid
from typing import Dict

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, LLMMessage, ServiceRef


DEFAULT_ROUTE_SERVERS = {
    "chat": "atlas-worker-1",
    "metacog": "athena-worker-1",
    "latents": "atlas-worker-2",
    "specialist": "atlas-worker-3",
}


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
        "metacog": os.getenv("LLM_ROUTE_METACOG_URL", ""),
        "latents": os.getenv("LLM_ROUTE_LATENTS_URL", ""),
        "specialist": os.getenv("LLM_ROUTE_SPECIALIST_URL", ""),
    }


def _expected_served_by(route: str) -> str:
    override = os.getenv(f"LLM_ROUTE_{route.upper()}_SERVED_BY")
    return override or DEFAULT_ROUTE_SERVERS[route]


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


async def _main_async(args: argparse.Namespace) -> None:
    bus = OrionBusAsync(args.redis)
    await bus.connect()

    route_urls = _load_route_urls()
    for route in ("chat", "metacog"):
        if not route_urls.get(route):
            raise RuntimeError(f"Route '{route}' is not configured (missing URL)")

    routes_to_test = ["chat", "metacog"]
    if route_urls.get("latents"):
        routes_to_test.append("latents")
    if route_urls.get("specialist"):
        routes_to_test.append("specialist")

    for route in routes_to_test:
        await _rpc_chat(
            bus,
            route=route,
            expected_served_by=_expected_served_by(route),
            request_channel=args.request_channel,
            timeout_sec=args.timeout,
        )

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
