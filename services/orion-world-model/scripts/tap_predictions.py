"""Subscribe to the world-model prediction broadcast and print each one.

Mirrors services/orion-vision-host/scripts/tap_artifacts.py. This is the only
real consumer of `orion:worldmodel:predictions` in this patch -- there is no
downstream cognition consumer yet (README.md "No real downstream consumer
yet"). Documented honestly per CLAUDE.md rather than hidden.
"""

from __future__ import annotations

import argparse
import asyncio

from orion.core.bus.async_service import OrionBusAsync
from orion.schemas.world_model import WorldModelPredictionPayload


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="redis://localhost:6379/0", help="Redis URL")
    parser.add_argument("--channel", default="orion:worldmodel:predictions", help="Predictions channel")
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
            print(f"\n[PREDICTION] {env.kind} ({env.correlation_id})")
            try:
                payload = (
                    WorldModelPredictionPayload(**env.payload)
                    if isinstance(env.payload, dict)
                    else env.payload
                )
                print(f"  ok: {payload.ok}")
                print(f"  device: {payload.device}")
                print(f"  model_untrained: {payload.model_untrained}")
                print(f"  state_dim: {payload.state_dim}  window_len: {payload.window_len}")
                print(f"  model_param_count: {payload.model_param_count}")
                if payload.error:
                    print(f"  error: {payload.error} ({payload.error_code})")
            except Exception as exc:
                print(f"  Payload validation error: {exc}")
                print(f"  Raw payload: {env.payload}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
