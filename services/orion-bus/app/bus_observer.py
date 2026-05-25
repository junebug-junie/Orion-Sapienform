from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from redis import asyncio as aioredis

from orion.core.bus.async_service import OrionBusAsync

from app.grammar_emit import BusTransportGrammarCollector, build_bus_transport_grammar_events
from app.grammar_publish import publish_bus_transport_grammar_trace
from app.settings import Settings, settings


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
    stream_lengths: dict[str, int] = field(default_factory=dict)
    uncataloged_streams: list[str] = field(default_factory=list)
    backpressure: list[tuple[str, int, int, str]] = field(default_factory=list)

    def to_collector(self, *, code_version: str | None) -> BusTransportGrammarCollector:
        c = BusTransportGrammarCollector(
            node_id=self.node_id,
            sample_window_id=self.sample_window_id,
            observed_at=self.observed_at,
            code_version=code_version,
        )
        c.record_tick_started()
        c.record_health_observed(redis_ping_ok=self.ping_ok)
        for stream_key, length in sorted(self.stream_lengths.items()):
            c.record_stream_depth(stream_key=stream_key, stream_length=length)
        for stream_key, length, threshold, severity in self.backpressure:
            c.record_backpressure(
                stream_key=stream_key,
                stream_length=length,
                threshold=threshold,
                severity=severity,
            )
        for stream_key in self.uncataloged_streams:
            c.record_uncataloged_stream(stream_key=stream_key)
        c.record_tick_completed(streams_observed=len(self.stream_lengths))
        return c


def build_rollup_from_redis_snapshot(
    *,
    settings: Settings,
    snapshot: dict[str, Any],
    observed_at: datetime,
    sample_window_id: str,
) -> ObserverRollup:
    ping_ok = bool(snapshot.get("ping_ok"))
    stream_lengths: dict[str, int] = dict(snapshot.get("stream_lengths") or {})
    catalog_names: set[str] = set(snapshot.get("catalog_names") or [])
    uncataloged = [
        sk for sk in settings.observer_stream_list if sk not in catalog_names
    ]
    backpressure: list[tuple[str, int, int, str]] = []
    for stream_key, length in stream_lengths.items():
        if length >= settings.bus_stream_depth_critical:
            backpressure.append(
                (stream_key, length, settings.bus_stream_depth_critical, "critical")
            )
        elif length >= settings.bus_stream_depth_warning:
            backpressure.append(
                (stream_key, length, settings.bus_stream_depth_warning, "warning")
            )
    return ObserverRollup(
        node_id=settings.bus_observer_node_id,
        sample_window_id=sample_window_id,
        observed_at=observed_at,
        ping_ok=ping_ok,
        stream_lengths=stream_lengths,
        uncataloged_streams=uncataloged,
        backpressure=backpressure,
    )


async def _fetch_redis_snapshot(settings: Settings) -> dict[str, Any]:
    client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    try:
        ping_ok = (await client.ping()) is True
        stream_lengths: dict[str, int] = {}
        for stream_key in settings.observer_stream_list:
            try:
                stream_lengths[stream_key] = int(await client.xlen(stream_key))
            except Exception:
                stream_lengths[stream_key] = 0
        catalog_names = load_channel_catalog_names(settings.channels_catalog_path)
        return {
            "ping_ok": ping_ok,
            "stream_lengths": stream_lengths,
            "catalog_names": catalog_names,
        }
    finally:
        await client.aclose()


async def run_observer_tick(*, bus: Any, settings: Settings) -> None:
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
            len(rollup.stream_lengths),
        )
    except Exception as exc:
        logger.warning("bus observer tick failed: {}", exc, exc_info=True)
        fail_collector = BusTransportGrammarCollector(
            node_id=settings.bus_observer_node_id,
            sample_window_id=window,
            observed_at=observed_at,
            code_version=settings.SERVICE_VERSION,
        )
        fail_collector.record_tick_started()
        fail_collector.record_tick_failed(error_kind=type(exc).__name__)
        events = build_bus_transport_grammar_events(fail_collector)
        await publish_bus_transport_grammar_trace(
            bus,
            events,
            channel=settings.grammar_event_channel,
            source_name=settings.SERVICE_NAME,
            enabled=settings.publish_orion_bus_grammar,
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
    try:
        while True:
            await run_observer_tick(bus=bus, settings=settings)
            await asyncio.sleep(settings.bus_observer_poll_interval_sec)
    finally:
        await bus.close()
