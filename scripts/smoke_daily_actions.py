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
        source=ServiceRef(name="scripts.smoke_daily_actions"),
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


def poll_notify(*, base_url: str, token: str | None, event_kind: str, correlation_id: str, timeout_s: int) -> bool:
    headers = {"X-Orion-Notify-Token": token} if token else {}
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = requests.get(
            f"{base_url.rstrip('/')}/notifications",
            params={"limit": 50, "event_kind": event_kind},
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        rows = resp.json()
        for row in rows:
            if str(row.get("correlation_id")) == correlation_id:
                print(json.dumps({"ok": True, "notification_id": row.get("notification_id"), "correlation_id": correlation_id}))
                return True
        time.sleep(2)
    return False


def main() -> int:
    p = argparse.ArgumentParser(description="Smoke trigger for daily pulse/metacog actions")
    p.add_argument("--action", choices=["pulse", "metacog"], required=True)
    p.add_argument("--bus-url", default=os.getenv("ORION_BUS_URL", "redis://localhost:6379/0"))
    p.add_argument("--notify-base-url", default=os.getenv("NOTIFY_BASE_URL", "http://localhost:7140"))
    p.add_argument("--notify-api-token", default=os.getenv("NOTIFY_API_TOKEN"))
    p.add_argument("--date", default=os.getenv("ACTIONS_DAILY_RUN_ONCE_DATE"))
    p.add_argument("--timeout", type=int, default=60)
    args = p.parse_args()

    if args.action == "pulse":
        channel = "orion:actions:trigger:daily_pulse.v1"
        kind = "orion.actions.trigger.daily_pulse.v1"
        event_kind = "orion.daily.pulse"
    else:
        channel = "orion:actions:trigger:daily_metacog.v1"
        kind = "orion.actions.trigger.daily_metacog.v1"
        event_kind = "orion.daily.metacog"

    corr = publish_trigger(bus_url=args.bus_url, channel=channel, kind=kind, date=args.date)
    print(json.dumps({"published": True, "action": args.action, "correlation_id": corr, "channel": channel}))

    ok = poll_notify(
        base_url=args.notify_base_url,
        token=args.notify_api_token,
        event_kind=event_kind,
        correlation_id=corr,
        timeout_s=args.timeout,
    )
    if ok:
        print(f"SUCCESS action={args.action} correlation_id={corr}")
        return 0

    print(f"FAIL action={args.action} correlation_id={corr}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
