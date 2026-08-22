"""Subscribe to the AffectGPT multimodal-assessment broadcast and print each one.

Mirrors services/orion-vision-host/scripts/tap_artifacts.py and
services/orion-world-model/scripts/tap_predictions.py. This is the only real
consumer of `orion:affectgpt:assessment` in this patch -- there is no
downstream cognition consumer yet (README.md "No real downstream cognition
consumer yet"). Documented honestly per CLAUDE.md rather than hidden, and
this script is what keeps the metric-lineage orphan gate honest about that:
it makes "no downstream consumer" a real, live-subscribable fact rather than
a channel nothing in the repo ever reads.
"""

from __future__ import annotations

import argparse
import asyncio

from orion.core.bus.async_service import OrionBusAsync
from orion.schemas.affectgpt import JuniperMultimodalAffectV1


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="redis://localhost:6379/0", help="Redis URL")
    parser.add_argument(
        "--channel", default="orion:affectgpt:assessment", help="Assessment channel"
    )
    args = parser.parse_args()

    bus = OrionBusAsync(url=args.url)
    await bus.connect()

    print(f"Listening on {args.channel}...")

    async with bus.subscribe(args.channel) as pubsub:
        async for msg in bus.iter_messages(pubsub):
            data = msg.get("data")
            decoded = bus.codec.decode(data)
            if not decoded.ok:
                print(f"Decode error: {decoded.error}")
                continue
            env = decoded.envelope
            print(f"\n[ASSESSMENT] {env.kind} ({env.correlation_id})")
            try:
                payload = (
                    JuniperMultimodalAffectV1(**env.payload)
                    if isinstance(env.payload, dict)
                    else env.payload
                )
                print(f"  ok: {payload.ok}")
                print(f"  observed_at: {payload.observed_at}")
                if payload.ok:
                    print(f"  raw_response: {payload.raw_response}")
                    print(f"  face_detection: {payload.face_detection}")
                else:
                    print(f"  error: {payload.error} ({payload.error_code})")
            except Exception as exc:
                print(f"  Payload validation error: {exc}")
                print(f"  Raw payload: {env.payload}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
