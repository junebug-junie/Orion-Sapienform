from __future__ import annotations

"""
Minimal bus harness to prove Client → Bus → Orch → Exec → Workers → Bus → Client.

Usage:
  python scripts/bus_harness.py brain "hello world"
  python scripts/bus_harness.py agent "plan a trip"
  python scripts/bus_harness.py tap  (pattern-subscribe to orion:*)
"""

import argparse
import asyncio
import json
import sys
from typing import List
from uuid import uuid4

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, LLMMessage, ServiceRef
from orion.schemas.cortex.contracts import (
    CortexClientContext,
    CortexClientRequest,
    RecallDirective,
)


def _service_ref(name: str, version: str, node: str | None) -> ServiceRef:
    return ServiceRef(name=name, version=version, node=node)


async def _rpc_request(
    *,
    bus_url: str,
    channel: str,
    req: CortexClientRequest,
    service_name: str,
    service_version: str,
    node: str | None,
    timeout: float,
) -> dict:
    bus = OrionBusAsync(url=bus_url)
    await bus.connect()
    try:
        corr = uuid4()
        reply = f"orion-cortex-orch:result:{corr}"
        env = BaseEnvelope(
            kind="cortex.orch.request",
            source=_service_ref(service_name, service_version, node),
            correlation_id=corr,
            reply_to=reply,
            payload=req.model_dump(mode="json"),
        )
        msg = await bus.rpc_request(channel, env, reply_channel=reply, timeout_sec=timeout)
        decoded = bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            raise RuntimeError(f"Decode failed: {decoded.error}")
        payload = decoded.envelope.payload
        return payload if isinstance(payload, dict) else decoded.envelope.model_dump(mode="json")
    finally:
        await bus.close()


async def _tap(bus_url: str) -> None:
    bus = OrionBusAsync(url=bus_url)
    await bus.connect()
    print(f"[tap] subscribing to orion:* on {bus_url}")
    async with bus.subscribe("orion:*", patterns=True) as pubsub:
        async for msg in bus.iter_messages(pubsub):
            decoded = bus.codec.decode(msg.get("data"))
            env = decoded.envelope
            print(
                json.dumps(
                    {
                        "channel": msg.get("channel") or msg.get("pattern"),
                        "kind": env.kind,
                        "correlation_id": str(env.correlation_id),
                        "reply_to": env.reply_to,
                    }
                )
            )


def _build_request(mode: str, text: str, args: argparse.Namespace) -> CortexClientRequest:
    recall = RecallDirective(
        enabled=not args.disable_recall,
        required=args.require_recall,
        mode=args.recall_mode,
        time_window_days=args.time_window_days,
        max_items=args.max_items,
    )
    ctx = CortexClientContext(
        messages=[LLMMessage(role="user", content=text)],
        session_id=args.session_id,
        user_id=args.user_id,
        trace_id=args.trace_id,
        metadata={"harness": True},
    )
    return CortexClientRequest(
        mode=mode,
        verb=args.verb,
        packs=args.packs,
        options={"temperature": args.temperature, "max_tokens": args.max_tokens},
        recall=recall,
        context=ctx,
    )


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Orion bus harness")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--bus-url", default="redis://100.92.216.81:6379/0")
        p.add_argument("--channel", default="orion-cortex-orch:request")
        p.add_argument("--service-name", default="bus-harness")
        p.add_argument("--service-version", default="0.0.1")
        p.add_argument("--node", default="harness")
        p.add_argument("--timeout", type=float, default=30.0)
        p.add_argument("--verb", default="chat_general")
        p.add_argument("--packs", nargs="*", default=["executive_pack"])
        p.add_argument("--temperature", type=float, default=0.7)
        p.add_argument("--max-tokens", type=int, default=256)
        p.add_argument("--session-id", default="harness-session")
        p.add_argument("--user-id", default="harness-user")
        p.add_argument("--trace-id", default=None)
        p.add_argument("--disable-recall", action="store_true")
        p.add_argument("--require-recall", action="store_true")
        p.add_argument("--recall-mode", default="hybrid")
        p.add_argument("--time-window-days", type=int, default=90)
        p.add_argument("--max-items", type=int, default=8)

    brain = sub.add_parser("brain", help="Brain chat through Orch/Exec/LLM")
    add_common(brain)
    brain.add_argument("text")

    agent = sub.add_parser("agent", help="Agentic path via AgentChainService")
    add_common(agent)
    agent.add_argument("text")

    tap = sub.add_parser("tap", help="Subscribe to orion:* traffic")
    tap.add_argument("--bus-url", default="redis://100.92.216.81:6379/0")

    return parser.parse_args(argv)


async def _main(argv: List[str]) -> int:
    args = _parse_args(argv)
    if args.cmd == "tap":
        await _tap(args.bus_url)
        return 0

    req = _build_request(args.cmd, args.text, args)
    payload = await _rpc_request(
        bus_url=args.bus_url,
        channel=args.channel,
        req=req,
        service_name=args.service_name,
        service_version=args.service_version,
        node=args.node,
        timeout=args.timeout,
    )
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main(sys.argv[1:])))
