from __future__ import annotations

import re
from datetime import datetime, timezone

from orion.schemas.grammar import GrammarEventV1
from orion.schemas.transport_projection import TransportBusStateV1

from .constants import (
    NON_BUS_TRANSPORT_NODE_IDS,
    TRANSPORT_SOURCE_SERVICE,
    TRANSPORT_TRACE_PREFIX,
)

_KV_RE = re.compile(r"(\w+)=([^,;\s]+)")

_IGNORED_ROLES = frozenset(
    {
        "trace_started",
        "trace_ended",
        "edge_emitted",
        "bus_observer_tick_started",
        # Retired 2026-09-25 (fix/bus-observer-scope): XLEN depth/backpressure.
        # Named here so a pre-deploy trace still in the reducer backlog is
        # skipped on purpose, not by falling through ATOM_ROLES.
        "bus_stream_depth_observed",
        "bus_backpressure_observed",
    }
)

ATOM_ROLES = frozenset(
    {
        "bus_health_observed",
        "bus_configured_stream_uncataloged",
        "bus_schema_validation_failed",
        "bus_observer_tick_failed",
        "bus_observer_tick_completed",
        "bus_census_computed",
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
    # `bus.transport:rpc_timeout:<corr>` shares the lane prefix but is not a
    # bus node (see constants.NON_BUS_TRANSPORT_NODE_IDS).
    if node_id in NON_BUS_TRANSPORT_NODE_IDS:
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


def compute_transport_pressures(state: TransportBusStateV1) -> dict[str, float]:
    # stream_backlog_health / delivery_confidence / stream_depth_pressure /
    # backpressure / stream_backlog_pressure were retired 2026-09-25
    # (fix/bus-observer-scope, docs/superpowers/specs/2026-09-25-bus-observer-
    # stream-depth-retirement.md). ping_pressure below is the one piece of
    # that family reliability_pressure actually depended on, kept with the
    # exact same values (ok=0.0, unknown=0.5, failed=1.0) so
    # reliability_pressure does not change meaning.
    if state.redis_ping_ok is True:
        ping_pressure = 0.0
    elif state.redis_ping_ok is False:
        ping_pressure = 1.0
    else:
        ping_pressure = 0.5

    observer_failure_pressure = 1.0 if state.observer_failure_count > 0 else 0.0
    denom = max(state.streams_observed, 1)
    # Fixed 2026-07-25 (docs/superpowers/specs/2026-07-25-catalog-drift-
    # pressure-mesh-wide-fix.md): was uncataloged_stream_count/denom, capped
    # at whatever this observer's own small configured stream list covers
    # (denom maxes at 2 in practice) -- structurally incapable of
    # representing general bus health regardless of how correctly wired.
    # undeclared_active_count/catalog_size is the real, mesh-wide diff
    # (orion.bus.census.compute_census(), ~264 real declared channels) --
    # gated behind BUS_OBSERVER_CENSUS_ENABLED, so it's None when the census
    # step didn't run this tick (flag off, or the scan itself failed).
    if state.undeclared_active_count is not None and state.catalog_size > 0:
        catalog_drift_pressure = min(state.undeclared_active_count / state.catalog_size, 1.0)
    else:
        # No honest mesh-wide measurement available this tick -- fall back to
        # the old, narrower formula rather than silently reporting 0.0, which
        # would misrepresent "not measured" as "confirmed no drift" (this
        # repo's own "no empty-shell cognition" rule). Once the census flag
        # is the live default, this branch only fires on a real scan failure.
        catalog_drift_pressure = min(state.uncataloged_stream_count / denom, 1.0)
    # Genuinely independent of catalog_drift_pressure: this counts cataloged
    # streams whose sampled traffic failed schema validation, not streams
    # missing from the catalog. Same "count of affected streams / denom"
    # shape as catalog_drift_pressure on purpose -- keeps both channels'
    # dynamic range comparable under the shared watch_at thresholds in
    # config/substrate-lattice/transport_lattice_policy.v1.yaml.
    contract_pressure = min(state.schema_mismatch_stream_count / denom, 1.0)
    # Same values as the old max(observer_failure, 1 - delivery_confidence).
    reliability_pressure = max(observer_failure_pressure, ping_pressure)

    return {
        "catalog_drift_pressure": catalog_drift_pressure,
        "observer_failure_pressure": observer_failure_pressure,
        "contract_pressure": contract_pressure,
        "reliability_pressure": reliability_pressure,
    }


def extract_transport_bus_state_from_events(
    events: list[GrammarEventV1],
    *,
    now: datetime | None = None,
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
    uncataloged_stream_count = 0
    observer_failure_count = 0
    schema_mismatch_stream_count = 0
    undeclared_active_count: int | None = None
    catalog_size = 0
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
        if role not in ATOM_ROLES:
            continue

        evidence_event_ids.append(event.event_id)
        kv = _parse_summary_kv(atom.summary or "")

        if role == "bus_health_observed":
            redis_ping_ok = _boolish(kv.get("redis_ping_ok"))
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
        elif role == "bus_census_computed":
            try:
                undeclared_active_count = int(kv.get("undeclared_active_count", "0") or 0)
                catalog_size = int(kv.get("catalog_size", "0") or 0)
            except ValueError:
                undeclared_active_count = None
                catalog_size = 0

    state = TransportBusStateV1(
        target_id=target_id,
        node_id=node_id,
        sample_window_id=sample_window_id,
        source_trace_id=trace_id,
        redis_ping_ok=redis_ping_ok,
        streams_observed=streams_observed,
        uncataloged_stream_count=uncataloged_stream_count,
        observer_failure_count=observer_failure_count,
        schema_mismatch_stream_count=schema_mismatch_stream_count,
        undeclared_active_count=undeclared_active_count,
        catalog_size=catalog_size,
        evidence_event_ids=evidence_event_ids,
        observed_at=clock,
    )
    pressures = compute_transport_pressures(state)
    return state.model_copy(update=pressures)
