from __future__ import annotations

import argparse
import json
import os
import sys
import time
from uuid import uuid4

import requests

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef


def publish_trigger(*, bus_url: str, channel: str, kind: str, date: str | None) -> str:
    corr = str(uuid4())
    env = BaseEnvelope(
        kind=kind,
        source=ServiceRef(name="scripts.smoke_daily_self_experiments"),
        correlation_id=corr,
        payload={"date": date} if date else {},
    )

    async def _run() -> None:
        bus = OrionBusAsync(bus_url)
        await bus.connect()
        await bus.publish(channel, env)
        await bus.close()

    import asyncio

    asyncio.run(_run())
    return corr


def poll_experiments(*, base_url: str, correlation_id: str, timeout_s: int) -> dict | None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = requests.get(
            f"{base_url.rstrip('/')}/v1/experiments",
            params={"limit": 50, "correlation_id": correlation_id},
            timeout=10,
        )
        resp.raise_for_status()
        parsed = resp.json()
        payload = parsed if isinstance(parsed, dict) else {}
        items = payload.get("items") if isinstance(payload, dict) else []
        if isinstance(items, list):
            for item in items:
                prov = item.get("provenance") if isinstance(item, dict) else {}
                if str((prov or {}).get("correlation_id") or "") == correlation_id:
                    return item if isinstance(item, dict) else None
        time.sleep(2)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke daily pulse/metacog to self-experiments")
    parser.add_argument("--action", choices=["pulse", "metacog"], required=True)
    parser.add_argument("--bus-url", default=os.getenv("ORION_BUS_URL", "redis://localhost:6379/0"))
    parser.add_argument("--self-experiments-url", default=os.getenv("ACTIONS_SELF_EXPERIMENTS_URL", "http://localhost:7172"))
    parser.add_argument("--date", default=os.getenv("ACTIONS_DAILY_RUN_ONCE_DATE"))
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()

    if args.action == "pulse":
        channel = "orion:actions:trigger:daily_pulse.v1"
        kind = "orion.actions.trigger.daily_pulse.v1"
    else:
        channel = "orion:actions:trigger:daily_metacog.v1"
        kind = "orion.actions.trigger.daily_metacog.v1"

    corr = publish_trigger(bus_url=args.bus_url, channel=channel, kind=kind, date=args.date)
    print(json.dumps({"published": True, "action": args.action, "correlation_id": corr, "channel": channel}))

    row = poll_experiments(
        base_url=args.self_experiments_url,
        correlation_id=corr,
        timeout_s=int(args.timeout),
    )
    if row:
        print(json.dumps({"ok": True, "correlation_id": corr, "experiment": row}))
        return 0
    print(f"FAIL action={args.action} correlation_id={corr} no_experiment_record", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
