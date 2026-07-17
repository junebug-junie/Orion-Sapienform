from __future__ import annotations

import re
from datetime import datetime, timezone

from orion.schemas.grammar import GrammarEventV1
from orion.schemas.transport_projection import TransportBusStateV1

from .constants import DEFAULT_STREAM_DEPTH_CRITICAL, TRANSPORT_SOURCE_SERVICE, TRANSPORT_TRACE_PREFIX

_KV_RE = re.compile(r"(\w+)=([^,;\s]+)")

_IGNORED_ROLES = frozenset(
    {
        "trace_started",
        "trace_ended",
        "edge_emitted",
        "bus_observer_tick_started",
    }
)

_ATOM_ROLES = frozenset(
    {
        "bus_health_observed",
        "bus_stream_depth_observed",
        "bus_backpressure_observed",
        "bus_configured_stream_uncataloged",
        "bus_schema_validation_failed",
        "bus_observer_tick_failed",
        "bus_observer_tick_completed",
    }
)


def _utc_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    return now if now.tzinfo else now.replace(tzinfo=timezone.utc)


def parse_bus_transport_trace_id(trace_id: str) -> tuple[str, str] | None:
    if not trace_id or not trace_id.startswith(TRANSPORT_TRACE_PREFIX):
        return None
    parts = trace_id.split(":", 2)
    if len(parts) != 3:
        return None
    node_id, sample_window_id = parts[1], parts[2]
    if not node_id or not sample_window_id:
        return None
    return node_id, sample_window_id


def _parse_summary_kv(summary: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, val in _KV_RE.findall(summary or ""):
        out[key.lower()] = val.strip()
    return out


def _boolish(val: str | None) -> bool | None:
    if val is None:
        return None
    lowered = val.strip().lower()
    if lowered in {"true", "1", "yes", "on"}:
        return True
    if lowered in {"false", "0", "no", "off"}:
        return False
    return None


def compute_transport_pressures(
    state: TransportBusStateV1,
    *,
    stream_depth_critical: int = DEFAULT_STREAM_DEPTH_CRITICAL,
) -> dict[str, float]:
    if state.redis_ping_ok is True:
        bus_health = 1.0
    elif state.redis_ping_ok is False:
        bus_health = 0.0
    else:
        bus_health = 0.5

    observer_failure_pressure = 1.0 if state.observer_failure_count > 0 else 0.0
    denom = max(state.streams_observed, 1)
    critical = max(stream_depth_critical, 1)
    stream_depth_pressure = min(state.max_stream_depth / critical, 1.0)
    backpressure = min(state.backpressure_count / denom, 1.0)
    catalog_drift_pressure = min(state.uncataloged_stream_count / denom, 1.0)
    # Genuinely independent of catalog_drift_pressure: this counts cataloged
    # streams whose sampled traffic failed schema validation, not streams
    # missing from the catalog. Same "count of affected streams / denom"
    # shape as catalog_drift_pressure on purpose -- keeps both channels'
    # dynamic range comparable under the shared watch_at thresholds in
    # config/substrate-lattice/transport_lattice_policy.v1.yaml.

    if observer_failure_pressure > 0.0:
        delivery_confidence = 0.0
    elif bus_health >= 1.0:
        delivery_confidence = 1.0
    elif bus_health == 0.5:
        delivery_confidence = 0.5
    else:
        delivery_confidence = 0.0

    transport_pressure = max(stream_depth_pressure, backpressure)
    contract_pressure = min(state.schema_mismatch_stream_count / denom, 1.0)
    reliability_pressure = max(observer_failure_pressure, 1.0 - delivery_confidence)

    return {
        "bus_health": bus_health,
        "delivery_confidence": delivery_confidence,
        "stream_depth_pressure": stream_depth_pressure,
        "backpressure": backpressure,
        "catalog_drift_pressure": catalog_drift_pressure,
        "observer_failure_pressure": observer_failure_pressure,
        "transport_pressure": transport_pressure,
        "contract_pressure": contract_pressure,
        "reliability_pressure": reliability_pressure,
    }


def extract_transport_bus_state_from_events(
    events: list[GrammarEventV1],
    *,
    now: datetime | None = None,
    stream_depth_critical: int = DEFAULT_STREAM_DEPTH_CRITICAL,
) -> TransportBusStateV1:
    clock = _utc_now(now)
    if not events:
        raise ValueError("events must not be empty")

    trace_id = events[0].trace_id or ""
    parsed = parse_bus_transport_trace_id(trace_id)
    if not parsed:
        raise ValueError(f"invalid transport trace_id: {trace_id}")

    node_id, sample_window_id = parsed
    target_id = f"bus:{node_id}"

    streams_observed = 0
    total_stream_depth = 0
    max_stream_depth = 0
    uncataloged_stream_count = 0
    backpressure_count = 0
    observer_failure_count = 0
    schema_mismatch_stream_count = 0
    redis_ping_ok: bool | None = None
    evidence_event_ids: list[str] = []

    for event in events:
        if event.provenance.source_service != TRANSPORT_SOURCE_SERVICE:
            continue
        atom = event.atom
        if not atom:
            continue
        role = (atom.semantic_role or "").strip()
        if not role or role in _IGNORED_ROLES:
            continue
        if role not in _ATOM_ROLES:
            continue

        evidence_event_ids.append(event.event_id)
        kv = _parse_summary_kv(atom.summary or "")

        if role == "bus_health_observed":
            redis_ping_ok = _boolish(kv.get("redis_ping_ok"))
        elif role == "bus_stream_depth_observed":
            try:
                length = int(kv.get("stream_length", "0") or 0)
            except ValueError:
                length = 0
            streams_observed += 1
            total_stream_depth += length
            max_stream_depth = max(max_stream_depth, length)
        elif role == "bus_backpressure_observed":
            backpressure_count += 1
            try:
                length = int(kv.get("stream_length", "0") or 0)
            except ValueError:
                length = 0
            streams_observed = max(streams_observed, 1)
            total_stream_depth += length
            max_stream_depth = max(max_stream_depth, length)
        elif role == "bus_configured_stream_uncataloged":
            uncataloged_stream_count += 1
        elif role == "bus_schema_validation_failed":
            schema_mismatch_stream_count += 1
        elif role == "bus_observer_tick_failed":
            observer_failure_count += 1
        elif role == "bus_observer_tick_completed":
            try:
                streams_observed = int(kv.get("streams_observed", streams_observed) or streams_observed)
            except ValueError:
                pass

    state = TransportBusStateV1(
        target_id=target_id,
        node_id=node_id,
        sample_window_id=sample_window_id,
        source_trace_id=trace_id,
        redis_ping_ok=redis_ping_ok,
        streams_observed=streams_observed,
        total_stream_depth=total_stream_depth,
        max_stream_depth=max_stream_depth,
        uncataloged_stream_count=uncataloged_stream_count,
        backpressure_count=backpressure_count,
        observer_failure_count=observer_failure_count,
        schema_mismatch_stream_count=schema_mismatch_stream_count,
        evidence_event_ids=evidence_event_ids,
        observed_at=clock,
    )
    pressures = compute_transport_pressures(state, stream_depth_critical=stream_depth_critical)
    return state.model_copy(update=pressures)
