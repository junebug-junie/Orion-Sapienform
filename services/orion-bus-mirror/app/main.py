import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiosqlite
from loguru import logger

from orion.bus.census import load_channel_catalog_names
from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly
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
    catalog_names: set[str],
) -> None:
    try:
        fact = extract_bus_event_fact(
            decoded.envelope, channel=channel, now=now_epoch, catalog_names=catalog_names
        )
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
            prior_organ_id, prior_epoch, prior_node = prior
            await loop.run_in_executor(
                None,
                lambda: writer.record_causal_hop(
                    prior_organ_id=prior_organ_id,
                    prior_epoch=prior_epoch,
                    prior_node=prior_node,
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


def build_heartbeat_chassis() -> HeartbeatOnly:
    """Own, independent bus connection publishing SystemHealthV1 to orion:system:health
    every HEARTBEAT_INTERVAL_SEC. Deliberately separate from `bus` above (the mirror's own
    subscribe connection) so this rollout (see
    docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md) cannot
    interfere with the mirror's message loop."""
    return HeartbeatOnly(
        ChassisConfig(
            service_name=settings.SERVICE_NAME,
            service_version=settings.SERVICE_VERSION,
            node_name=settings.NODE_NAME,
            bus_url=settings.ORION_BUS_URL,
            bus_enabled=True,
            heartbeat_interval_sec=settings.HEARTBEAT_INTERVAL_SEC,
        )
    )


async def mirror_bus() -> None:
    bus = OrionBusAsync(settings.ORION_BUS_URL)
    await bus.connect()

    # Awaited (not fired concurrently) before the subscribe loop starts, matching PR #1350's
    # pilot-5 shape -- an unreachable bus can add up to connect_timeout_sec (default 10s) to
    # startup, bounded and non-fatal (caught below), not unbounded blocking.
    heartbeat_chassis: Optional[HeartbeatOnly] = None
    try:
        heartbeat_chassis = build_heartbeat_chassis()
        await heartbeat_chassis.start_background()
        logger.info(
            "system_health_heartbeat_started service={} interval_sec={}",
            settings.SERVICE_NAME,
            settings.HEARTBEAT_INTERVAL_SEC,
        )
    except Exception as exc:
        logger.warning("system_health_heartbeat_start_failed error={}", exc)
        heartbeat_chassis = None

    Path(settings.MIRROR_PARQUET_DIR).mkdir(parents=True, exist_ok=True)

    graph_writer = _build_graph_writer()
    chain_tracker = ChainTracker(ttl_sec=settings.MIRROR_GRAPH_CHAIN_TTL_SEC) if graph_writer else None
    # Loaded once at startup, not per-message: channels.yaml is static for the
    # life of this process. Used to collapse dynamic reply channels (e.g.
    # "orion:exec:result:LLMGatewayService:<uuid>") to their catalog wildcard
    # entry before they become graph nodes -- see extract_bus_event_fact's
    # docstring for why this matters (a real, live-found ~9K-node inflation).
    catalog_names = load_channel_catalog_names() if graph_writer else set()
    if graph_writer:
        logger.info(
            "Bus synaptic graph writer enabled: graph={} alpha={} catalog_channels={}",
            settings.FALKORDB_BUS_GRAPH,
            settings.MIRROR_GRAPH_EWMA_ALPHA,
            len(catalog_names),
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
                            graph_writer, chain_tracker, decoded, channel, now.timestamp(), catalog_names
                        )
    finally:
        if inflight_task is not None:
            inflight_task.cancel()
        if heartbeat_chassis is not None:
            try:
                await heartbeat_chassis.stop()
            except Exception as exc:
                logger.warning("system_health_heartbeat_stop_error error={}", exc)
        await bus.close()


def main() -> None:
    asyncio.run(mirror_bus())


if __name__ == "__main__":
    main()
