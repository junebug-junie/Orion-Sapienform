from __future__ import annotations

import argparse
import asyncio
import json
from uuid import uuid4

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.contracts import CHANNELS, KINDS, reply_channel


def _default_messages(text: str):
    return [{"role": "user", "content": text}]


async def run_once(*, mode: str, text: str, bus_url: str, timeout: float):
    bus = OrionBusAsync(url=bus_url)
    await bus.connect()

    corr = uuid4()
    reply = reply_channel(CHANNELS.cortex_reply_prefix, corr)

    env = BaseEnvelope(
        kind=KINDS.cortex_orch_request,
        source=ServiceRef(name="bus-harness", version="dev", node="cli"),
        correlation_id=corr,
        reply_to=reply,
        payload={
            "verb_name": mode,
            "args": {"request_id": str(corr), "mode": mode},
            "context": {"messages": _default_messages(text), "mode": mode},
        },
    )

    msg = await bus.rpc_request(
        CHANNELS.cortex_request,
        env,
        reply_channel=reply,
        timeout_sec=timeout,
    )
    decoded = bus.codec.decode(msg.get("data"))
    if not decoded.ok:
        raise RuntimeError(f"Decode failed: {decoded.error}")
    return decoded.envelope.payload


def main():
    parser = argparse.ArgumentParser(description="Minimal bus harness for cortex pipeline.")
    parser.add_argument("--mode", default="brain", choices=["brain", "agent", "council", "chat_general"])
    parser.add_argument("--text", required=True, help="User message to send.")
    parser.add_argument("--bus-url", default="redis://orion-redis:6379/0")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    payload = asyncio.run(
        run_once(mode=args.mode, text=args.text, bus_url=args.bus_url, timeout=args.timeout)
    )
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()

