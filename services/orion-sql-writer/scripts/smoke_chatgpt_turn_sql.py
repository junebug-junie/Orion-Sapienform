#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import os
import sys
import time
import uuid
from pathlib import Path

from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef


async def publish_turn(bus_url: str, channel: str, turn_id: str) -> None:
    bus = OrionBusAsync(bus_url)
    await bus.connect()
    try:
        source = ServiceRef(name="chatgpt-import-smoke", version="0.1.0", node="smoke")
        env = BaseEnvelope(
            schema_id="ChatGptLogTurnV1",
            kind="chat.gpt.turn.v1",
            source=source,
            correlation_id=uuid.UUID(turn_id),
            payload={
                "id": turn_id,
                "correlation_id": turn_id,
                "source": "chatgpt_import",
                "prompt": "smoke prompt",
                "response": "smoke response",
                "session_id": "chatgpt:smoke",
                "user_id": "smoke-user",
                "spark_meta": {"smoke": True, "published_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
            },
        )
        await bus.publish(channel, env)
    finally:
        await bus.close()


def check_row(database_url: str, turn_id: str) -> bool:
    engine = create_engine(database_url)
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT id, prompt, response FROM chat_gpt_log WHERE id = :id LIMIT 1"),
            {"id": turn_id},
        ).first()
    return row is not None


async def main() -> int:
    bus_url = os.getenv("ORION_BUS_URL", "redis://localhost:6379/0")
    database_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/conjourney")
    channel = os.getenv("GPT_TURN_CHANNEL", "orion:chat:gpt:turn")
    max_wait_sec = float(os.getenv("SMOKE_MAX_WAIT_SEC", "15"))
    poll_interval_sec = float(os.getenv("SMOKE_POLL_INTERVAL_SEC", "1"))

    turn_id = str(uuid.uuid4())
    await publish_turn(bus_url, channel, turn_id)

    deadline = time.time() + max_wait_sec
    found = False
    while time.time() < deadline:
        if check_row(database_url, turn_id):
            found = True
            break
        await asyncio.sleep(poll_interval_sec)

    print(
        {
            "turn_id": turn_id,
            "channel": channel,
            "found_in_chat_gpt_log": found,
            "expected_log": "Written ChatGptLogTurnV1 -> chat_gpt_log",
        }
    )
    return 0 if found else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
