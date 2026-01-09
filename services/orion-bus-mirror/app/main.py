import asyncio
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
from loguru import logger

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.codec import DecodeResult, OrionCodec

from app.settings import settings


def _serialize_envelope(decoded: DecodeResult) -> str:
    if decoded.ok:
        parsed = decoded.envelope.model_dump(by_alias=True, mode="json")
    else:
        parsed = {"error": decoded.error, "raw": decoded.raw}
    return OrionCodec().encode(parsed).decode("utf-8")


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
    bus = OrionBusAsync(settings.ORION_BUS_URL)
    await bus.connect()

    Path(settings.MIRROR_PARQUET_DIR).mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(settings.MIRROR_SQLITE_PATH) as conn:
        await _ensure_schema(conn)
        logger.info(
            "Bus mirror connected: {} pattern={}",
            settings.ORION_BUS_URL,
            settings.MIRROR_PATTERN,
        )

        async with bus.subscribe(settings.MIRROR_PATTERN, patterns=True) as pubsub:
            async for message in bus.iter_messages(pubsub):
                timestamp = datetime.now(timezone.utc).isoformat()
                channel = message.get("channel")
                if isinstance(channel, bytes):
                    channel = channel.decode("utf-8", errors="replace")
                decoded = bus.codec.decode(message.get("data", b""))
                envelope_json = _serialize_envelope(decoded)

                await conn.execute(
                    "INSERT INTO bus_events(timestamp, channel, envelope_json) VALUES (?, ?, ?)",
                    (timestamp, channel, envelope_json),
                )
                await conn.commit()

    await bus.close()


def main() -> None:
    asyncio.run(mirror_bus())


if __name__ == "__main__":
    main()
