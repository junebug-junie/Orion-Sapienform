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
    extract_verb_step_facts,
    summarize_open_chains,
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

        # Phase 2: per-verb latency, mined separately from the envelope-level
        # PUBLISHES/CAUSALLY_FOLLOWED_BY facts above -- most envelopes have no
        # steps[] payload at all, so this is usually a no-op list. Capped via
        # max_steps so a buggy/adversarial producer sending a huge steps
        # array can't stall this loop for the whole envelope (found in review).
        for verb_fact in extract_verb_step_facts(
            decoded.envelope, now=now_epoch, max_steps=settings.MIRROR_GRAPH_MAX_VERB_STEPS
        ):
            await loop.run_in_executor(None, writer.record_verb_step, verb_fact)
    except Exception as exc:  # fail-open: graph writes never block/crash the mirror
        logger.warning("bus synaptic graph write failed channel={} err={}", channel, exc)


async def _run_inflight_chain_summary_loop(chain_tracker: ChainTracker) -> None:
    """Phase 2: periodic, read-only visibility into currently-tracked chains
    -- no bus/graph I/O, just logs. Not wired to any consumer yet (README's
    own /stats stub still doesn't exist); this is the "smallest real" step
    that makes the signal observable before deciding how it should be
    consumed. Runs independently of message processing so it keeps reporting
    even during a quiet period.
    """
    while True:
        await asyncio.sleep(settings.MIRROR_GRAPH_INFLIGHT_LOG_INTERVAL_SEC)
        try:
            now_epoch = datetime.now(timezone.utc).timestamp()
            open_chains = chain_tracker.snapshot_open_chains(now_epoch)
            summary = summarize_open_chains(
                open_chains, long_running_threshold_sec=settings.MIRROR_GRAPH_LONG_RUNNING_THRESHOLD_SEC
            )
            if summary.open_count > 0:
                logger.info(
                    "bus synaptic graph in-flight chains: open={} long_running={} max_duration_sec={:.1f}",
                    summary.open_count,
                    summary.long_running_count,
                    summary.max_duration_sec,
                )
        except Exception as exc:  # fail-open, matching _record_graph_event's discipline:
            # an unhandled exception here would otherwise die silently (nothing
            # awaits this task directly) instead of just skipping one tick.
            logger.warning("bus synaptic graph in-flight chain summary failed: {}", exc)


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

    inflight_task: Optional[asyncio.Task] = None
    if chain_tracker is not None:
        inflight_task = asyncio.create_task(_run_inflight_chain_summary_loop(chain_tracker))

    try:
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
    finally:
        if inflight_task is not None:
            inflight_task.cancel()
        await bus.close()


def main() -> None:
    asyncio.run(mirror_bus())


if __name__ == "__main__":
    main()
