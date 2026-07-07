#!/usr/bin/env python3
"""Replay a world-pulse run result onto the bus for metabolism verification."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from urllib.request import urlopen

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.world_pulse import WorldPulseRunResultV1


def load_store(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def snapshot(label: str, store_path: str) -> None:
    data = load_store(store_path)
    slot = data.get("goal_slots", {}).get("orion:predictive")
    predictive = data.get("drive_states", {}).get("orion", {}).get("pressures", {}).get("predictive")
    print(f"=== {label} ===")
    print("orion:predictive slot:", json.dumps(slot))
    print("predictive pressure:", predictive)


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="8f07e526-1604-4674-9820-dc9d147eb326")
    parser.add_argument("--world-pulse-url", default=os.environ.get("WORLD_PULSE_BASE_URL", "http://orion-world-pulse:8628"))
    parser.add_argument("--wait-sec", type=float, default=3.0)
    args = parser.parse_args()

    bus_url = os.environ.get("ORION_BUS_URL", "redis://127.0.0.1:6379/0")
    enforce = os.environ.get("ORION_BUS_ENFORCE_CATALOG", "true").lower() == "true"
    channel = "orion:world_pulse:run:result"
    goal_channel = os.environ.get("BUS_GOAL_PROPOSAL_OUT", "orion:memory:goals:proposed")
    store_path = os.environ.get("CONCEPT_STORE_PATH", "/data/concept-induction-state.json")

    snapshot("BEFORE", store_path)

    raw = json.loads(
        urlopen(f"{args.world_pulse_url.rstrip('/')}/api/world-pulse/runs/{args.run_id}", timeout=15).read()
    )
    wp = WorldPulseRunResultV1.model_validate(raw)
    captured: list[dict] = []

    async def listen_goals(bus: OrionBusAsync) -> None:
        async with bus.subscribe(goal_channel) as pubsub:
            deadline = asyncio.get_event_loop().time() + max(args.wait_sec + 2.0, 5.0)
            async for msg in bus.iter_messages(pubsub):
                if asyncio.get_event_loop().time() > deadline:
                    break
                data = msg.get("data")
                if not data:
                    continue
                decoded = bus.codec.decode(data)
                if not decoded.ok or decoded.envelope is None:
                    continue
                payload = decoded.envelope.payload
                if hasattr(payload, "model_dump"):
                    payload = payload.model_dump()
                if not isinstance(payload, dict):
                    continue
                prov = payload.get("provenance") if isinstance(prov := payload.get("provenance"), dict) else {}
                captured.append(
                    {
                        "artifact_id": payload.get("artifact_id"),
                        "drive_origin": payload.get("drive_origin"),
                        "goal_statement": (payload.get("goal_statement") or "")[:240],
                        "spawned_correlation_id": prov.get("spawned_correlation_id"),
                    }
                )

    bus = OrionBusAsync(bus_url, enforce_catalog=enforce)
    await bus.connect()
    listener = asyncio.create_task(listen_goals(bus))

    corr = uuid.uuid4()
    env = BaseEnvelope(
        kind="world.pulse.run.result.v1",
        source=ServiceRef(name="orion-replay-test", node="athena", version="0.0.1"),
        payload=wp.model_dump(mode="json"),
        correlation_id=corr,
        created_at=datetime.now(timezone.utc),
    )
    print(f"Publishing replay channel={channel} correlation_id={corr} run_id={args.run_id}")
    await bus.publish(channel, env)
    await asyncio.sleep(args.wait_sec)
    listener.cancel()
    try:
        await listener
    except asyncio.CancelledError:
        pass
    await bus.close()

    snapshot("AFTER", store_path)
    print("goal messages captured:", json.dumps(captured, indent=2))
    print("REPLAY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
