from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from datetime import datetime, timezone
from uuid import uuid4

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.telemetry.metacog_trigger import MetacogTriggerV1


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _trace_meta(*, trace_id: str, event_id: str, source_service: str) -> dict:
    return {
        "trace_id": trace_id,
        "event_id": event_id,
        "parent_event_id": None,
        "source_service": source_service,
        "created_at": _ts(),
    }


async def run() -> int:
    parser = argparse.ArgumentParser(description="End-to-end metacog trace publisher + bus tapper.")
    parser.add_argument(
        "--redis",
        default=os.getenv("ORION_BUS_URL", "redis://localhost:6379/0"),
        help="Redis/Orion bus URL (default: ORION_BUS_URL or redis://localhost:6379/0).",
    )
    parser.add_argument(
        "--trigger-channel",
        default=os.getenv("CHANNEL_EQUILIBRIUM_METACOG_TRIGGER", "orion:equilibrium:metacog:trigger"),
        help="Cortex-orch intake channel for metacog triggers.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=45.0,
        help="Seconds to wait for downstream messages.",
    )
    parser.add_argument(
        "--trigger-kind",
        default="baseline",
        help="Trigger kind (baseline|dense|manual|pulse).",
    )
    parser.add_argument(
        "--reason",
        default="trace_probe",
        help="Trigger reason for the test event.",
    )
    parser.add_argument(
        "--pressure",
        type=float,
        default=0.25,
        help="Pressure value (0..1) for the trigger.",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="PSUBSCRIBE pattern to watch (repeatable).",
    )
    parser.add_argument(
        "--source-service",
        default="scripts.trace_metacog",
        help="Source service name stamped into envelopes.",
    )
    args = parser.parse_args()

    exec_request_channel = os.getenv("CHANNEL_EXEC_REQUEST", "orion:cortex:exec:request")
    exec_result_prefix = os.getenv("EXEC_RESULT_PREFIX", "orion:exec:result")
    collapse_sql_channel = os.getenv("CHANNEL_COLLAPSE_SQL_WRITE", "orion:collapse:sql-write")
    collapse_intake_channel = os.getenv("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake")

    patterns = args.pattern or [
        args.trigger_channel,
        exec_request_channel,
        f"{exec_result_prefix}:*",
        collapse_sql_channel,
        collapse_intake_channel,
        "orion:verb:request",
        "orion:verb:result",
        "orion:cognition:trace",
        "orion:system:error",
    ]

    bus = OrionBusAsync(url=args.redis)
    await bus.connect()

    trace_id = str(uuid4())
    event_id = str(uuid4())
    trace_meta = _trace_meta(
        trace_id=trace_id,
        event_id=event_id,
        source_service=args.source_service,
    )

    trigger = MetacogTriggerV1(
        trigger_kind=args.trigger_kind,
        reason=args.reason,
        zen_state="unknown",
        pressure=args.pressure,
        recall_enabled=True,
    )

    env = BaseEnvelope(
        kind="orion.metacog.trigger.v1",
        source=ServiceRef(name=args.source_service),
        correlation_id=trace_id,
        id=event_id,
        trace=trace_meta,
        payload=trigger.model_dump(mode="json"),
    )

    print(
        json.dumps(
            {
                "ts": _ts(),
                "event": "publish",
                "channel": args.trigger_channel,
                "trace_id": trace_id,
                "event_id": event_id,
                "envelope": env.model_dump(mode="json"),
            },
            default=str,
        )
    )

    validation_errors: list[dict] = []
    downstream_seen = False
    deadline = time.monotonic() + args.timeout

    async with bus.subscribe(*patterns, patterns=True) as pubsub:
        await bus.publish(args.trigger_channel, env)

        while time.monotonic() < deadline:
            try:
                msg = await asyncio.wait_for(bus.iter_messages(pubsub).__anext__(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            channel = msg.get("channel")
            if hasattr(channel, "decode"):
                channel = channel.decode("utf-8")

            decoded = bus.codec.decode(msg.get("data"))
            if not decoded.ok:
                validation_errors.append(
                    {"channel": channel, "error": decoded.error, "raw": decoded.raw}
                )
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
            msg_trace_id = (env.trace or {}).get("trace_id") or str(env.correlation_id)
            if msg_trace_id != trace_id:
                continue

            if channel != args.trigger_channel:
                downstream_seen = True

            print(
                json.dumps(
                    {
                        "ts": _ts(),
                        "channel": channel,
                        "trace_id": msg_trace_id,
                        "kind": env.kind,
                        "schema_id": env.schema_id,
                        "envelope": env.model_dump(mode="json"),
                    },
                    default=str,
                )
            )

    await bus.close()

    if validation_errors:
        print(f"[{_ts()}] validation errors observed: {validation_errors}")
        return 2

    if not downstream_seen:
        print(f"[{_ts()}] no downstream messages observed for trace_id={trace_id}")
        return 1

    print(f"[{_ts()}] trace complete trace_id={trace_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
