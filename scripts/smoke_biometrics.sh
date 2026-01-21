#!/usr/bin/env bash
set -euo pipefail

BUS_URL="${ORION_BUS_URL:-redis://orion-redis:6379/0}"
STATE_CHANNEL="${STATE_REQUEST_CHANNEL:-orion:state:request}"

python - <<'PY'
import asyncio
import json
import os
from uuid import uuid4

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.state.contracts import StateGetLatestRequest

BUS_URL = os.getenv("ORION_BUS_URL", "redis://orion-redis:6379/0")
STATE_CHANNEL = os.getenv("STATE_REQUEST_CHANNEL", "orion:state:request")

TARGET_CHANNELS = {
    "orion:biometrics:sample",
    "orion:biometrics:summary",
    "orion:biometrics:induction",
}


async def tap_once():
    bus = OrionBusAsync(url=BUS_URL)
    await bus.connect()
    seen = set()
    try:
        async with bus.subscribe("orion:biometrics:*", patterns=True) as pubsub:
            async for msg in bus.iter_messages(pubsub):
                channel = msg.get("channel") or msg.get("pattern")
                decoded = bus.codec.decode(msg.get("data"))
                if not decoded.ok:
                    continue
                env = decoded.envelope
                if channel in TARGET_CHANNELS and channel not in seen:
                    print(json.dumps({"channel": channel, "kind": env.kind}))
                    seen.add(channel)
                if seen == TARGET_CHANNELS:
                    break
    finally:
        await bus.close()


async def fetch_state():
    bus = OrionBusAsync(url=BUS_URL)
    await bus.connect()
    try:
        reply_channel = f"orion:state:reply:{uuid4()}"
        req = StateGetLatestRequest(scope="global")
        env = BaseEnvelope(
            kind="state.get_latest.v1",
            source=ServiceRef(name="biometrics-smoke", version="0.0.0", node="local"),
            correlation_id=uuid4(),
            reply_to=reply_channel,
            payload=req.model_dump(mode="json"),
        )
        msg = await bus.rpc_request(STATE_CHANNEL, env, reply_channel=reply_channel, timeout_sec=15.0)
        decoded = bus.codec.decode(msg.get("data"))
        print(json.dumps({"state_reply_ok": decoded.ok, "kind": decoded.envelope.kind}))
    finally:
        await bus.close()


async def main():
    await tap_once()
    await fetch_state()


asyncio.run(main())
PY
