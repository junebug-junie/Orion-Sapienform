from __future__ import annotations

import argparse
import asyncio
import os
import time
from typing import Optional

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef


async def _await_fanout(bus: OrionBusAsync, corr_id: str, timeout_sec: float) -> tuple[bool, bool]:
    triage_seen = False
    sql_seen = False
    deadline = time.time() + timeout_sec
    async with bus.subscribe("orion:collapse:triage", "orion:collapse:sql-write") as pubsub:
        async for msg in bus.iter_messages(pubsub):
            channel = msg.get("channel")
            if hasattr(channel, "decode"):
                channel = channel.decode("utf-8")
            decoded = bus.codec.decode(msg.get("data"))
            if not decoded.ok:
                continue
            env = decoded.envelope
            if str(env.correlation_id) != corr_id:
                if time.time() > deadline:
                    break
                continue
            if channel == "orion:collapse:triage":
                triage_seen = True
            if channel == "orion:collapse:sql-write":
                sql_seen = True
            if triage_seen and sql_seen:
                break
            if time.time() > deadline:
                break
    return triage_seen, sql_seen


async def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test: Juniper collapse intake fanout.")
    parser.add_argument("--redis", default=os.getenv("ORION_BUS_URL", "redis://localhost:6379/0"))
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    parser.add_argument("--intake", default=os.getenv("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake"))
    args = parser.parse_args()

    bus = OrionBusAsync(url=args.redis)
    await bus.connect()
    try:
        corr_id = f"smoke-{int(time.time())}"
        payload = {
            "observer": "Juniper",
            "trigger": "smoke-test",
            "observer_state": ["focused"],
            "field_resonance": "steady",
            "type": "flow",
            "emergent_entity": "Juniper",
            "summary": "Smoke test collapse",
            "mantra": "steady",
            "causal_echo": None,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        env = BaseEnvelope(
            kind="collapse.submit",
            source=ServiceRef(name="smoke-juniper"),
            correlation_id=corr_id,
            payload=payload,
        )
        await bus.publish(args.intake, env)
        triage_seen, sql_seen = await _await_fanout(bus, corr_id, args.timeout_sec)
    finally:
        await bus.close()

    if triage_seen and sql_seen:
        print("PASS triage=ok sql-write=ok")
        return 0
    print(f"FAIL triage={triage_seen} sql-write={sql_seen}")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
