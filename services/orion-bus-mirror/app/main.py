import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiosqlite
from loguru import logger

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.codec import DecodeResult, OrionCodec
from orion.graph.falkor_client import RedisGraphQueryClient

from app.graph_writer import (
    BusSynapticGraphWriter,
    ChainTracker,
    extract_bus_event_fact,
)
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


def _build_graph_writer() -> Optional[BusSynapticGraphWriter]:
    if not settings.MIRROR_GRAPH_ENABLED:
        return None
    # No fallback to ORION_BUS_URL: FALKORDB_URI is hard-defaulted in
    # settings.py to the standard FalkorDB hostname. Falling back to the bus
    # URL here would silently point GRAPH.QUERY at the pub/sub Redis instead
    # of FalkorDB if this ever went empty -- fail loud instead.
    if not settings.FALKORDB_URI:
        logger.error("MIRROR_GRAPH_ENABLED=true but FALKORDB_URI is empty -- graph writer disabled")
        return None
    client = RedisGraphQueryClient(uri=settings.FALKORDB_URI, graph_name=settings.FALKORDB_BUS_GRAPH)
    return BusSynapticGraphWriter(client, alpha=settings.MIRROR_GRAPH_EWMA_ALPHA)


async def _record_graph_event(
    writer: BusSynapticGraphWriter,
    chain_tracker: ChainTracker,
    decoded: DecodeResult,
    channel: str,
    now_epoch: float,
) -> None:
    try:
        fact = extract_bus_event_fact(decoded.envelope, channel=channel, now=now_epoch)
        if fact is None:
            return
        # Chain-tracker bookkeeping runs before the (possibly-failing) Falkor
        # write, not after: it's a pure in-memory operation, and if it only
        # ran after a successful write, a transient Falkor outage would drop
        # this correlation_id's "first sighting" entirely -- silently losing
        # the later leg's CAUSALLY_FOLLOWED_BY edge even once Falkor recovers.
        prior = chain_tracker.observe(fact)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, writer.record_publish, fact)
        if prior is not None:
            prior_organ_id, prior_epoch = prior
            await loop.run_in_executor(
                None,
                lambda: writer.record_causal_hop(
                    prior_organ_id=prior_organ_id,
                    prior_epoch=prior_epoch,
                    fact=fact,
                ),
            )
    except Exception as exc:  # fail-open: graph writes never block/crash the mirror
        logger.warning("bus synaptic graph write failed channel={} err={}", channel, exc)


async def mirror_bus() -> None:
    bus = OrionBusAsync(settings.ORION_BUS_URL)
    await bus.connect()

    Path(settings.MIRROR_PARQUET_DIR).mkdir(parents=True, exist_ok=True)

    graph_writer = _build_graph_writer()
    chain_tracker = ChainTracker(ttl_sec=settings.MIRROR_GRAPH_CHAIN_TTL_SEC) if graph_writer else None
    if graph_writer:
        logger.info(
            "Bus synaptic graph writer enabled: graph={} alpha={}",
            settings.FALKORDB_BUS_GRAPH,
            settings.MIRROR_GRAPH_EWMA_ALPHA,
        )

    async with aiosqlite.connect(settings.MIRROR_SQLITE_PATH) as conn:
        await _ensure_schema(conn)
        logger.info(
            "Bus mirror connected: {} pattern={}",
            settings.ORION_BUS_URL,
            settings.MIRROR_PATTERN,
        )

        async with bus.subscribe(settings.MIRROR_PATTERN, patterns=True) as pubsub:
            async for message in bus.iter_messages(pubsub):
                now = datetime.now(timezone.utc)
                timestamp = now.isoformat()
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

                if graph_writer is not None and decoded.ok:
                    await _record_graph_event(
                        graph_writer, chain_tracker, decoded, channel, now.timestamp()
                    )

    await bus.close()


def main() -> None:
    asyncio.run(mirror_bus())


if __name__ == "__main__":
    main()
