from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from pydantic import ValidationError
from redis import asyncio as aioredis

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.codec import OrionCodec
from orion.schemas.registry import resolve as resolve_schema_id

from app.grammar_emit import BusTransportGrammarCollector, build_bus_transport_grammar_events
from app.grammar_publish import publish_bus_transport_grammar_trace
from app.settings import Settings, settings

# Field names, checked in priority order, that carry the encoded envelope/
# payload blob on a Redis Stream entry. Every live producer we traced writes
# to exactly one of these (RedisStreamWorkQueue.enqueue() -> "envelope",
# orion-landing-pad's redis_store.py -> "data", orion-dream's
# memory_listener.py -> "payload"). No single convention is universal across
# every stream in the mesh, so this is a best-effort, documented assumption,
# not a guarantee -- see BUS_OBSERVER_SCHEMA_SAMPLE_COUNT docs.
_ENVELOPE_FIELD_CANDIDATES: tuple[str, ...] = ("envelope", "data", "payload")


def _extract_envelope_raw(fields: dict[str, Any]) -> str | None:
    for name in _ENVELOPE_FIELD_CANDIDATES:
        val = fields.get(name)
        if val:
            return val if isinstance(val, str) else str(val)
    return None


def count_schema_mismatches(
    entries: list[tuple[str, dict[str, Any]]],
    *,
    schema_id: str,
) -> tuple[int, int]:
    """Validate a bounded sample of raw XREVRANGE entries against a stream's
    declared schema_id (orion/bus/channels.yaml).

    Returns (mismatch_count, sampled_count). Reuses OrionCodec.decode()'s
    DecodeResult/error mechanism for the envelope-decode step -- note this is
    NOT literally the same call path as OrionBusAsync._validate_payload():
    that method validates an in-memory payload before encoding/publish and
    never calls codec.decode() at all. What IS shared with it is the
    downstream step: resolve_schema_id() + model.model_validate(payload).
    After decode, this validates the decoded envelope's payload against the
    resolved schema model. An entry that fails to decode at all, or whose
    envelope carries a payload that does not fit the declared schema, counts
    as a mismatch -- both are real contract violations for this stream, just
    at different layers.

    Never returns raw payload content -- counts only (see
    SUBSTRATE_TRACE_MAP.md: bus-observer emits bounded counts, never
    per-message content)."""
    try:
        model = resolve_schema_id(schema_id)
    except ValueError:
        return 0, 0

    codec = OrionCodec()
    sampled = 0
    mismatches = 0
    for _entry_id, raw_fields in entries:
        sampled += 1
        raw = _extract_envelope_raw(raw_fields)
        if raw is None:
            mismatches += 1
            continue
        result = codec.decode(raw)
        if not result.ok:
            mismatches += 1
            continue
        try:
            model.model_validate(result.envelope.payload)
        except ValidationError:
            mismatches += 1
    return mismatches, sampled


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


def load_channel_catalog_schema_ids(catalog_path: str) -> dict[str, str]:
    """name -> schema_id for every cataloged channel that declares one.

    Same exact-name matching convention as load_channel_catalog_names() (no
    wildcard expansion for entries like "orion:effect:*") -- a stream key
    only gets a schema_id here if it appears verbatim in the catalog."""
    path = _resolve_catalog_path(catalog_path)
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    schema_ids: dict[str, str] = {}
    for ch in data.get("channels") or []:
        if isinstance(ch, dict) and ch.get("name") and ch.get("schema_id"):
            schema_ids[str(ch["name"])] = str(ch["schema_id"])
    return schema_ids


@dataclass
class ObserverRollup:
    node_id: str
    sample_window_id: str
    observed_at: datetime
    ping_ok: bool
    stream_lengths: dict[str, int] = field(default_factory=dict)
    uncataloged_streams: list[str] = field(default_factory=list)
    backpressure: list[tuple[str, int, int, str]] = field(default_factory=list)
    # Only streams where mismatch_count > 0 land here (same "only the
    # anomaly, not every check" convention as uncataloged_streams).
    schema_mismatches: list[tuple[str, int, int]] = field(default_factory=list)

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
        for stream_key, mismatch_count, sampled_count in self.schema_mismatches:
            c.record_schema_mismatch(
                stream_key=stream_key,
                mismatch_count=mismatch_count,
                sampled_count=sampled_count,
            )
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

    # Schema-mismatch check: only meaningful for streams that are BOTH
    # cataloged (we know what schema to check against) and have a sample to
    # check. Uncataloged streams are already covered by uncataloged_streams
    # above -- don't double-flag them here too.
    catalog_schema_ids: dict[str, str] = dict(snapshot.get("catalog_schema_ids") or {})
    stream_samples: dict[str, list[tuple[str, dict[str, Any]]]] = dict(
        snapshot.get("stream_samples") or {}
    )
    schema_mismatches: list[tuple[str, int, int]] = []
    for stream_key, entries in stream_samples.items():
        schema_id = catalog_schema_ids.get(stream_key)
        if not schema_id or not entries:
            continue
        mismatch_count, sampled_count = count_schema_mismatches(entries, schema_id=schema_id)
        if mismatch_count > 0:
            schema_mismatches.append((stream_key, mismatch_count, sampled_count))

    return ObserverRollup(
        node_id=settings.bus_observer_node_id,
        sample_window_id=sample_window_id,
        observed_at=observed_at,
        ping_ok=ping_ok,
        stream_lengths=stream_lengths,
        uncataloged_streams=uncataloged,
        backpressure=backpressure,
        schema_mismatches=schema_mismatches,
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
        catalog_schema_ids = load_channel_catalog_schema_ids(settings.channels_catalog_path)

        # Bounded per-stream sample (XREVRANGE ... COUNT n) for the
        # schema-mismatch check, cataloged streams only -- an uncataloged
        # stream has no declared schema_id to check against, and is already
        # covered by catalog_names/uncataloged_streams above. Cost: at most
        # len(observer_stream_list) * bus_observer_schema_sample_count extra
        # Redis reads per tick, on top of the len(observer_stream_list) XLEN
        # calls already made above.
        stream_samples: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        sample_count = max(settings.bus_observer_schema_sample_count, 0)
        if sample_count > 0:
            for stream_key in settings.observer_stream_list:
                if stream_key not in catalog_schema_ids:
                    continue
                try:
                    entries = await client.xrevrange(stream_key, count=sample_count)
                except Exception:
                    continue
                stream_samples[stream_key] = [
                    (entry_id, dict(fields)) for entry_id, fields in entries
                ]

        return {
            "ping_ok": ping_ok,
            "stream_lengths": stream_lengths,
            "catalog_names": catalog_names,
            "catalog_schema_ids": catalog_schema_ids,
            "stream_samples": stream_samples,
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
