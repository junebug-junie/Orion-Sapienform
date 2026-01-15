from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone

from orion.core.bus.async_service import OrionBusAsync


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


async def run() -> int:
    parser = argparse.ArgumentParser(description="PSUBSCRIBE to Orion bus patterns and print envelopes.")
    parser.add_argument(
        "--redis",
        default=os.getenv("ORION_BUS_URL", "redis://localhost:6379/0"),
        help="Redis/Orion bus URL (default: ORION_BUS_URL or redis://localhost:6379/0).",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="PSUBSCRIBE pattern (repeatable). Default: orion:*",
    )
    parser.add_argument(
        "--trace-id",
        default=None,
        help="Optional trace_id filter (matches envelope.trace.trace_id or correlation_id).",
    )

    args = parser.parse_args()
    patterns = args.pattern or ["orion:*"]

    bus = OrionBusAsync(url=args.redis)
    await bus.connect()

    print(f"[{_ts()}] bus_probe connected url={args.redis} patterns={patterns}")

    try:
        async with bus.subscribe(*patterns, patterns=True) as pubsub:
            async for msg in bus.iter_messages(pubsub):
                channel = msg.get("channel")
                if hasattr(channel, "decode"):
                    channel = channel.decode("utf-8")
                decoded = bus.codec.decode(msg.get("data"))

                if not decoded.ok:
                    print(
                        json.dumps(
                            {
                                "ts": _ts(),
                                "channel": channel,
                                "ok": False,
                                "error": decoded.error,
                                "raw": decoded.raw,
                            },
                            default=str,
                        )
                    )
                    continue

                env = decoded.envelope
                trace_id = (env.trace or {}).get("trace_id") or str(env.correlation_id)

                if args.trace_id and trace_id != args.trace_id:
                    continue

                print(
                    json.dumps(
                        {
                            "ts": _ts(),
                            "channel": channel,
                            "trace_id": trace_id,
                            "kind": env.kind,
                            "schema_id": env.schema_id,
                            "envelope": env.model_dump(mode="json"),
                        },
                        default=str,
                    )
                )
    except KeyboardInterrupt:
        print(f"[{_ts()}] bus_probe stopped (KeyboardInterrupt)")
    finally:
        await bus.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
