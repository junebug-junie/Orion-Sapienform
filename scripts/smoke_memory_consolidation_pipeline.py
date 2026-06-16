#!/usr/bin/env python3
"""Smoke test for memory consolidation pipeline (requires live stack)."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from uuid import uuid4

import asyncpg

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef


async def _wait_for_spark_meta(pool: asyncpg.Pool, corr: str, timeout_sec: float) -> dict | None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        row = await pool.fetchrow(
            "SELECT spark_meta FROM chat_history_log WHERE correlation_id = $1",
            corr,
        )
        if row and row["spark_meta"]:
            meta = row["spark_meta"]
            if isinstance(meta, str):
                import json

                meta = json.loads(meta)
            if meta.get("memory_significance_score") is not None or meta.get("memory_classify_status") == "degraded":
                return meta
        await asyncio.sleep(1.0)
    return None


async def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke memory consolidation pipeline")
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    args = parser.parse_args()

    bus_url = os.getenv("ORION_BUS_URL", "redis://127.0.0.1:6379/0")
    postgres_uri = os.getenv("POSTGRES_URI", "")
    turn_channel = os.getenv("CHANNEL_CHAT_HISTORY_TURN", "orion:chat:history:turn")
    if not postgres_uri:
        print("POSTGRES_URI required", file=sys.stderr)
        return 2

    corr = str(uuid4())
    bus = OrionBusAsync(url=bus_url, enabled=True)
    await bus.connect()
    pool = await asyncpg.create_pool(postgres_uri, min_size=1, max_size=2)

    env = BaseEnvelope(
        kind="chat.history",
        correlation_id=corr,
        source=ServiceRef(name="smoke-memory-consolidation", version="0.1.0", node="local"),
        payload={
            "id": corr,
            "correlation_id": corr,
            "prompt": "smoke test turn",
            "response": "smoke test response",
            "session_id": "smoke-session",
            "spark_meta": {"conversation_phase": {"phase_change": "same_breath"}},
        },
    )

    print(f"publishing chat.history corr={corr} channel={turn_channel}")
    await bus.publish(turn_channel, env)

    meta = await _wait_for_spark_meta(pool, corr, args.timeout_sec)
    if meta is None:
        print(f"FAIL: no classification in spark_meta within {args.timeout_sec}s corr={corr}")
        return 1

    print(f"PASS: spark_meta updated corr={corr} meta_keys={sorted(meta.keys())}")
    await pool.close()
    await bus.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
