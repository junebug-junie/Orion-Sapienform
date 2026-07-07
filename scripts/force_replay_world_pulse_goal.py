#!/usr/bin/env python3
"""Clear world-pulse gap goal cooldown and replay with goal capture."""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from urllib.request import urlopen

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.world_pulse import WorldPulseRunResultV1

RUN_ID = "8f07e526-1604-4674-9820-dc9d147eb326"
GAP_SIG = "3f9d0b1d9aaa80ca2ab8c70b"
CHANNEL = "orion:world_pulse:run:result"


async def main() -> None:
    store_path = os.environ["CONCEPT_STORE_PATH"]
    goal_ch = os.environ["BUS_GOAL_PROPOSAL_OUT"]
    data = json.load(open(store_path, encoding="utf-8"))
    data.get("goal_cooldowns", {}).pop(GAP_SIG, None)
    with open(store_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    print("cleared cooldown for", GAP_SIG)

    raw = json.loads(urlopen(f"http://orion-world-pulse:8628/api/world-pulse/runs/{RUN_ID}", timeout=15).read())
    wp = WorldPulseRunResultV1.model_validate(raw)
    captured: list[dict] = []

    bus = OrionBusAsync(os.environ["ORION_BUS_URL"], enforce_catalog=True)
    await bus.connect()

    async def listen() -> None:
        async with bus.subscribe(goal_ch) as pubsub:
            deadline = asyncio.get_event_loop().time() + 10.0
            async for msg in bus.iter_messages(pubsub):
                if asyncio.get_event_loop().time() > deadline:
                    break
                decoded = bus.codec.decode(msg.get("data"))
                if not decoded.ok or decoded.envelope is None:
                    continue
                payload = decoded.envelope.payload
                if hasattr(payload, "model_dump"):
                    payload = payload.model_dump()
                if not isinstance(payload, dict) or payload.get("drive_origin") != "predictive":
                    continue
                prov = payload.get("provenance") or {}
                captured.append(
                    {
                        "artifact_id": payload.get("artifact_id"),
                        "statement": (payload.get("goal_statement") or "")[:240],
                        "spawned_correlation_id": prov.get("spawned_correlation_id"),
                        "tension_kinds": payload.get("tension_kinds"),
                        "proposal_status": payload.get("proposal_status"),
                    }
                )

    listener = asyncio.create_task(listen())
    await asyncio.sleep(0.5)
    corr = uuid.uuid4()
    env = BaseEnvelope(
        kind="world.pulse.run.result.v1",
        source=ServiceRef(name="orion-replay-test", node="athena", version="0.0.1"),
        payload=wp.model_dump(mode="json"),
        correlation_id=corr,
    )
    print("publishing replay correlation_id=", corr)
    await bus.publish(CHANNEL, env)
    await asyncio.sleep(4.0)
    listener.cancel()
    try:
        await listener
    except asyncio.CancelledError:
        pass
    await bus.close()

    after = json.load(open(store_path, encoding="utf-8"))
    print("captured goals:", json.dumps(captured, indent=2))
    print("orion:predictive slot:", json.dumps(after.get("goal_slots", {}).get("orion:predictive")))
    print("gap cooldown:", json.dumps(after.get("goal_cooldowns", {}).get(GAP_SIG)))
    print("FORCED_REPLAY_OK")


if __name__ == "__main__":
    asyncio.run(main())
