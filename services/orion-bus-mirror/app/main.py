import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
from loguru import logger
from redis.asyncio import Redis

from app.settings import settings


def _serialize_envelope(payload: str | bytes) -> str:
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        parsed = {"raw": payload}
    return json.dumps(parsed, ensure_ascii=False)


async def _ensure_schema(conn: aiosqlite.Connection) -> None:
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bus_events (
            timestamp TEXT,
            channel TEXT,
            envelope_json TEXT
        )
        """
    )
    await conn.commit()


async def mirror_bus() -> None:
    redis = Redis.from_url(settings.ORION_BUS_URL, decode_responses=True)
    pubsub = redis.pubsub()

    Path(settings.MIRROR_PARQUET_DIR).mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(settings.MIRROR_SQLITE_PATH) as conn:
        await _ensure_schema(conn)
        await pubsub.psubscribe(settings.MIRROR_PATTERN)
        logger.info(
            "Bus mirror connected: {} pattern={}",
            settings.ORION_BUS_URL,
            settings.MIRROR_PATTERN,
        )

        async for message in pubsub.listen():
            if message is None:
                continue
            if message.get("type") != "pmessage":
                continue
            timestamp = datetime.now(timezone.utc).isoformat()
            channel = message.get("channel")
            envelope_json = _serialize_envelope(message.get("data", ""))

            await conn.execute(
                "INSERT INTO bus_events(timestamp, channel, envelope_json) VALUES (?, ?, ?)",
                (timestamp, channel, envelope_json),
            )
            await conn.commit()


def main() -> None:
    asyncio.run(mirror_bus())


if __name__ == "__main__":
    main()
