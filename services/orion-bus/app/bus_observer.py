from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from redis import asyncio as aioredis

from orion.bus.census import compute_census
from orion.bus.ewma import EwmaUpdate, compute_ewma_update
from orion.bus.velocity import scan_active_channels
from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly

from app.grammar_emit import BusTransportGrammarCollector, build_bus_transport_grammar_events
from app.grammar_publish import publish_bus_transport_grammar_trace
from app.settings import Settings, settings

# The bounded XREVRANGE schema sample that fed contract_pressure was retired
# 2026-10-07 (fix/transport-lattice-names-and-contract). It sampled only the
# BUS_OBSERVER_STREAMS keys (two world_pulse streams) and read exactly 0 on
# 123,412 of 123,412 field ticks. Widening it to the whole mesh would not help:
# OrionBusAsync.publish() already validates every payload against its catalog
# schema_id and raises before sending, so a receiver-side sample reads calm by
# construction (live 2026-10-07: 9,498 cataloged messages on 125 channels over
# 120 s, 0 mismatches, 0 decode failures). See docs/superpowers/specs/
# 2026-10-07-transport-lattice-names-and-contract.md.


def _sample_window_id(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _resolve_catalog_path(catalog_path: str) -> Path:
    path = Path(catalog_path)
    if path.is_file():
        return path
    service_root = Path(__file__).resolve().parents[1]
    for base in (Path.cwd(), service_root, service_root.parent, service_root.parent.parent):
        candidate = base / catalog_path
        if candidate.is_file():
            return candidate
    return path


def load_channel_catalog_names(catalog_path: str) -> set[str]:
    path = _resolve_catalog_path(catalog_path)
    if not path.is_file():
        return set()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    names: set[str] = set()
    for ch in data.get("channels") or []:
        if isinstance(ch, dict) and ch.get("name"):
            names.add(str(ch["name"]))
    return names


@dataclass
class ObserverRollup:
    node_id: str
    sample_window_id: str
    observed_at: datetime
    ping_ok: bool
    # How many BUS_OBSERVER_STREAMS keys this tick was configured to check for
    # catalog membership. Not a depth reading: per-stream XLEN
    # depth and the backpressure threshold were retired 2026-09-25 (XLEN is
    # retained length, not backlog; see grammar_emit.py's retirement note).
    streams_observed: int = 0
    uncataloged_streams: list[str] = field(default_factory=list)
    # Mesh-wide census diff (orion.bus.census.compute_census()) -- see
    # TransportBusStateV1.undeclared_active_count's docstring for why this
    # is a genuinely different measurement than uncataloged_streams above.
    # None (not 0) when the census step itself failed/was skipped, so the
    # consumer can tell "measured zero" from "not measured" -- same "no
    # empty-shell cognition" discipline this repo applies elsewhere.
    undeclared_active_count: int | None = None
    catalog_size: int = 0
    # Idea 6 (bus_activity_zscore) -- the raw current-tick total, not yet
    # z-scored. Z-scoring needs cross-tick EWMA state (ActivityEwmaTracker),
    # which this dataclass deliberately does not hold -- run_observer_tick()
    # does that stateful step after to_collector() returns, not here.
    total_mesh_publish_rate: float | None = None

    def to_collector(self, *, code_version: str | None) -> BusTransportGrammarCollector:
        c = BusTransportGrammarCollector(
            node_id=self.node_id,
            sample_window_id=self.sample_window_id,
            observed_at=self.observed_at,
            code_version=code_version,
        )
        c.record_tick_started()
        c.record_health_observed(redis_ping_ok=self.ping_ok)
        for stream_key in self.uncataloged_streams:
            c.record_uncataloged_stream(stream_key=stream_key)
        if self.undeclared_active_count is not None:
            c.record_bus_census_computed(
                undeclared_active_count=self.undeclared_active_count,
                catalog_size=self.catalog_size,
            )
        c.record_tick_completed(streams_observed=self.streams_observed)
        return c


def build_rollup_from_redis_snapshot(
    *,
    settings: Settings,
    snapshot: dict[str, Any],
    observed_at: datetime,
    sample_window_id: str,
) -> ObserverRollup:
    ping_ok = bool(snapshot.get("ping_ok"))
    catalog_names: set[str] = set(snapshot.get("catalog_names") or [])
    uncataloged = [
        sk for sk in settings.observer_stream_list if sk not in catalog_names
    ]

    # None (not 0) when the snapshot step didn't run the census scan at all
    # (e.g. Redis unreachable) -- see ObserverRollup.undeclared_active_count's
    # docstring for why that distinction matters.
    undeclared_active_count = snapshot.get("undeclared_active_count")

    return ObserverRollup(
        node_id=settings.bus_observer_node_id,
        sample_window_id=sample_window_id,
        observed_at=observed_at,
        ping_ok=ping_ok,
        streams_observed=len(settings.observer_stream_list),
        uncataloged_streams=uncataloged,
        undeclared_active_count=undeclared_active_count,
        catalog_size=len(catalog_names),
        total_mesh_publish_rate=snapshot.get("total_mesh_publish_rate"),
    )


async def _fetch_redis_snapshot(settings: Settings) -> dict[str, Any]:
    client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    try:
        ping_ok = (await client.ping()) is True
        catalog_names = load_channel_catalog_names(settings.channels_catalog_path)

        # Mesh-wide census diff, gated (see settings.py's
        # bus_observer_census_enabled docstring). undeclared_active_count
        # stays None (not 0) if disabled or if the scan itself fails --
        # scan_active_channels() is already fail-open internally (returns {}
        # on a SCAN/read error), but "the mesh had zero undeclared channels"
        # and "we didn't check" must never collapse into the same value.
        # Idea 6 (bus_activity_zscore, docs/superpowers/specs/2026-07-24-bus-
        # vitality-field-signal-brainstorm.md): total_mesh_publish_rate reuses
        # the same active_channels scan the census diff above already pays
        # for -- zero extra Redis cost. None (not 0.0) if the census step
        # didn't run this tick, same "not measured != measured zero"
        # discipline as undeclared_active_count.
        undeclared_active_count: int | None = None
        total_mesh_publish_rate: float | None = None
        if settings.bus_observer_census_enabled:
            try:
                active_channels = await scan_active_channels(
                    client, window_minutes=settings.bus_observer_census_window_minutes
                )
                census = compute_census(catalog_names, active_channels)
                undeclared_active_count = len(census.undeclared_active)
                total_mesh_publish_rate = sum(active_channels.values())
            except Exception:
                undeclared_active_count = None
                total_mesh_publish_rate = None

        return {
            "ping_ok": ping_ok,
            "catalog_names": catalog_names,
            "undeclared_active_count": undeclared_active_count,
            "total_mesh_publish_rate": total_mesh_publish_rate,
        }
    finally:
        await client.aclose()


class ActivityEwmaTracker:
    """In-process EWMA state for Idea 6 (``bus_activity_zscore``) -- total
    mesh publish rate, not per-channel. Lives for the lifetime of the
    ``run_bus_observer_loop()`` process; there is exactly one bus-observer
    instance per node today (same single-instance assumption
    ``BusSynapticGraphWriter`` documents for the analogous bus-mirror case),
    so no CAS/version check is needed here either.
    """

    def __init__(self, *, alpha: float) -> None:
        self._alpha = alpha
        self._ewma = 0.0
        self._variance = 0.0
        self._count = 0

    def observe(self, total_rate: float) -> EwmaUpdate:
        update = compute_ewma_update(
            prev_ewma=self._ewma,
            prev_variance=self._variance,
            prev_count=self._count,
            value=total_rate,
            alpha=self._alpha,
        )
        self._ewma = update.ewma
        self._variance = update.variance
        self._count += 1
        return update


async def run_observer_tick(
    *, bus: Any, settings: Settings, activity_tracker: ActivityEwmaTracker | None = None
) -> None:
    observed_at = datetime.now(timezone.utc)
    window = _sample_window_id(observed_at)
    try:
        snapshot = await _fetch_redis_snapshot(settings)
        rollup = build_rollup_from_redis_snapshot(
            settings=settings,
            snapshot=snapshot,
            observed_at=observed_at,
            sample_window_id=window,
        )
        collector = rollup.to_collector(code_version=settings.SERVICE_VERSION)
        # Idea 6: z-scoring needs cross-tick EWMA state, which ObserverRollup
        # deliberately doesn't hold -- done here, imperatively, after
        # to_collector() returns a mutable collector one more atom can still
        # be added to before publish. activity_tracker is None in tests that
        # call run_observer_tick() directly without one (backward compatible
        # -- same "not measured" semantics as the flag being off).
        if activity_tracker is not None and rollup.total_mesh_publish_rate is not None:
            update = activity_tracker.observe(rollup.total_mesh_publish_rate)
            collector.record_bus_activity_zscore_computed(
                total_rate=rollup.total_mesh_publish_rate,
                ewma=update.ewma,
                zscore=update.zscore,
            )
        events = build_bus_transport_grammar_events(collector)
        await publish_bus_transport_grammar_trace(
            bus,
            events,
            channel=settings.grammar_event_channel,
            source_name=settings.SERVICE_NAME,
            enabled=settings.publish_orion_bus_grammar,
        )
        logger.debug(
            "bus observer tick ok window={} streams={}",
            window,
            rollup.streams_observed,
        )
    except Exception as exc:
        # 2026-10-07 (#2534 decision 4, fix/field-decisions-d3-credit-novelty-
        # observer): a failed tick no longer publishes a bus_observer_tick_failed
        # trace. That trace fed observer_failure_pressure (0.0 on 123,099 of
        # 123,099 field ticks; zero failed ticks in 72 h of retained atoms) and,
        # because it carried no ping and no census, reduced to ping-unknown 0.5
        # reliability and a 0.0 catalog drift -- readings nobody took. Now a
        # failed tick publishes nothing: node:athena's transport channels go
        # unrefreshed, and the observer's own liveness is this log line plus its
        # SystemHealthV1 heartbeat (build_heartbeat_chassis, orion:system:health).
        logger.warning("bus observer tick failed: {}", exc, exc_info=True)


def build_heartbeat_chassis(settings: Settings) -> HeartbeatOnly:
    """Own, independent bus connection publishing SystemHealthV1 to orion:system:health
    every HEARTBEAT_INTERVAL_SEC. Deliberately separate from the `bus` connection above
    (the observer's own grammar-publish connection) so this rollout (see
    docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md) cannot
    interfere with the observer's existing tick loop."""
    return HeartbeatOnly(
        ChassisConfig(
            service_name=settings.SERVICE_NAME,
            service_version=settings.SERVICE_VERSION,
            node_name=settings.NODE_NAME,
            bus_url=settings.REDIS_URL,
            bus_enabled=True,
            heartbeat_interval_sec=settings.HEARTBEAT_INTERVAL_SEC,
        )
    )


async def run_bus_observer_loop() -> None:
    bus = OrionBusAsync(settings.REDIS_URL)
    await bus.connect()
    logger.info(
        "bus-observer started node={} interval={}s publish={}",
        settings.bus_observer_node_id,
        settings.bus_observer_poll_interval_sec,
        settings.publish_orion_bus_grammar,
    )

    # Awaited (not fired concurrently) before the tick loop starts, matching PR #1350's pilot-5
    # shape -- an unreachable bus can add up to connect_timeout_sec (default 10s) to startup,
    # bounded and non-fatal (caught below), not unbounded blocking.
    heartbeat_chassis: HeartbeatOnly | None = None
    try:
        heartbeat_chassis = build_heartbeat_chassis(settings)
        await heartbeat_chassis.start_background()
        logger.info(
            "system_health_heartbeat_started service={} interval_sec={}",
            settings.SERVICE_NAME,
            settings.HEARTBEAT_INTERVAL_SEC,
        )
    except Exception as exc:
        logger.warning("system_health_heartbeat_start_failed error={}", exc)
        heartbeat_chassis = None

    # One tracker for the process lifetime -- Idea 6's whole point is a
    # rolling baseline across ticks, not a fresh one each time.
    activity_tracker = ActivityEwmaTracker(alpha=settings.bus_activity_ewma_alpha)

    try:
        while True:
            await run_observer_tick(bus=bus, settings=settings, activity_tracker=activity_tracker)
            await asyncio.sleep(settings.bus_observer_poll_interval_sec)
    finally:
        if heartbeat_chassis is not None:
            try:
                await heartbeat_chassis.stop()
            except Exception as exc:
                logger.warning("system_health_heartbeat_stop_error error={}", exc)
        await bus.close()
