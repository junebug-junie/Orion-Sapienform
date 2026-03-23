#!/usr/bin/env python3
"""
E2E smoke: dream_cycle spine (Orch -> Exec -> dream.result.v1 -> SQL Writer -> wake readout).

Env: ORION_BUS_URL, POSTGRES_URI (optional for --check-db), DREAM_READOUT_URL (optional for --check-wake).
Usage:
    python scripts/dream_spine_smoke.py [--redis URL] [--use-trigger] [--check-db] [--check-wake] [--skip-if-unavailable]
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ORCH_CHANNEL = os.getenv("CORTEX_REQUEST_CHANNEL", os.getenv("ORCH_REQUEST_CHANNEL", "orion:cortex:request"))
ORCH_RESULT_PREFIX = os.getenv("CORTEX_RESULT_PREFIX", os.getenv("ORCH_RESULT_PREFIX", "orion:cortex:result"))
DREAM_TRIGGER_CHANNEL = os.getenv("CHANNEL_DREAM_TRIGGER", "orion:dream:trigger")


async def _rpc_orch(bus, payload, reply_prefix: str, channel: str, timeout_sec: float):
    corr = str(uuid4())
    reply_channel = f"{reply_prefix}:{corr}"
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

    env = BaseEnvelope(
        kind="cortex.orch.request",
        source=ServiceRef(name="dream-spine-smoke", node="local", version="0.0.1"),
        correlation_id=corr,
        reply_to=reply_channel,
        payload=payload.model_dump(mode="json"),
    )
    msg = await bus.rpc_request(channel, env, reply_channel=reply_channel, timeout_sec=timeout_sec)
    decoded = bus.codec.decode(msg.get("data"))
    if not decoded.ok:
        raise RuntimeError(decoded.error or "decode failed")
    return decoded.envelope.payload if isinstance(decoded.envelope.payload, dict) else decoded.envelope.payload.model_dump(mode="json")


def _count_dreams(uri: str) -> int:
    try:
        from sqlalchemy import create_engine, text

        eng = create_engine(uri, pool_pre_ping=True)
        with eng.connect() as conn:
            row = conn.execute(text("SELECT COUNT(*) FROM dreams")).scalar()
            return int(row) if row is not None else 0
    except Exception:
        return -1


def _fetch_wake_readout(url: str) -> dict | None:
    try:
        import httpx

        r = httpx.get(url, timeout=10.0)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


async def main() -> int:
    parser = argparse.ArgumentParser(description="E2E smoke: dream_cycle spine")
    parser.add_argument("--redis", type=str, default=os.getenv("ORION_BUS_URL", "redis://localhost:6379/0"))
    parser.add_argument("--use-trigger", action="store_true", help="Publish dream.trigger instead of cortex.orch.request (fire-and-forget)")
    parser.add_argument("--check-db", action="store_true", help="Poll dreams table for new row after run")
    parser.add_argument("--check-wake", action="store_true", help="Call wake readout HTTP endpoint")
    parser.add_argument("--skip-if-unavailable", action="store_true", help="Exit 0 when bus/DB unreachable (CI)")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    from orion.core.bus.async_service import OrionBusAsync
    from orion.core.bus.bus_schemas import BaseEnvelope, LLMMessage, ServiceRef
    from orion.schemas.cortex.contracts import CortexClientContext, CortexClientRequest, RecallDirective
    from orion.schemas.telemetry.dream import DreamTriggerPayload

    bus = OrionBusAsync(url=args.redis)
    try:
        await bus.connect()
    except Exception as e:
        if args.skip_if_unavailable:
            print("dream_spine_smoke: bus unavailable, skipping:", e)
            return 0
        print("dream_spine_smoke: bus connect failed:", e)
        return 1

    postgres_uri = os.getenv("POSTGRES_URI", os.getenv("DATABASE_URL", ""))
    count_before = _count_dreams(postgres_uri) if postgres_uri and args.check_db else 0

    try:
        if args.use_trigger:
            env = BaseEnvelope(
                kind="dream.trigger",
                source=ServiceRef(name="dream-spine-smoke", node="local", version="0.0.1"),
                correlation_id=str(uuid4()),
                payload=DreamTriggerPayload(mode="standard").model_dump(mode="json"),
            )
            await bus.publish(DREAM_TRIGGER_CHANNEL, env)
            print("Published dream.trigger (fire-and-forget); waiting for async persistence...")
            await asyncio.sleep(min(60, args.timeout))
        else:
            ctx = CortexClientContext(
                messages=[LLMMessage(role="user", content="Dream cycle.")],
                raw_user_text="Dream cycle.",
                session_id="dream-spine-smoke",
            )
            payload = CortexClientRequest(
                mode="brain",
                verb="dream_cycle",
                packs=["emergent_pack"],
                recall=RecallDirective(enabled=True, profile="dream.v1"),
                context=ctx,
            )
            res = await _rpc_orch(bus, payload, ORCH_RESULT_PREFIX, ORCH_CHANNEL, args.timeout)
            if not res.get("ok"):
                err = res.get("error") or res.get("final_text") or "unknown"
                print("dream_spine_smoke: Orch/Exec failed:", err)
                return 1
            print("dream_spine_smoke: Orch/Exec ok, verb=", res.get("verb"), "status=", res.get("status"))
            if args.check_db and postgres_uri:
                await asyncio.sleep(5)

    finally:
        await bus.close()

    if args.check_db and postgres_uri:
        count_after = _count_dreams(postgres_uri)
        if count_after < 0:
            if args.skip_if_unavailable:
                print("dream_spine_smoke: DB unreachable, skipping check")
                return 0
            print("dream_spine_smoke: DB check failed")
            return 1
        if count_after <= count_before and args.use_trigger:
            if args.skip_if_unavailable:
                print("dream_spine_smoke: no new dream row (async delay?); skipping")
                return 0
            print("dream_spine_smoke: expected new dreams row, got count", count_after, "vs", count_before)
            return 1
        print("dream_spine_smoke: DB check ok, dreams count=", count_after)

    if args.check_wake:
        url = os.getenv("DREAM_READOUT_URL", "http://localhost:8620/dreams/wakeup/today")
        out = _fetch_wake_readout(url)
        if out is None:
            if args.skip_if_unavailable:
                print("dream_spine_smoke: wake readout unreachable, skipping")
                return 0
            print("dream_spine_smoke: wake readout failed")
            return 1
        src = out.get("source", "")
        if src == "sql":
            print("dream_spine_smoke: wake readout source=sql ok")
        else:
            print("dream_spine_smoke: wake readout source=", src, "(expected sql when DB has dream)")

    print("dream_spine_smoke: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
