from __future__ import annotations

from datetime import datetime, timezone

from orion.schemas.grammar import GrammarAtomV1, GrammarEventV1, GrammarProvenanceV1
from orion.schemas.transport_projection import TransportBusProjectionV1
from orion.substrate.transport_loop.extract import (
    compute_transport_pressures,
    extract_transport_bus_state_from_events,
    parse_bus_transport_trace_id,
)
from orion.substrate.transport_loop.reducer import reduce_transport_trace_events

NOW = datetime(2026, 5, 25, 23, 30, 10, tzinfo=timezone.utc)
TRACE = "bus.transport:athena:20260525T233010Z"


def _prov() -> GrammarProvenanceV1:
    return GrammarProvenanceV1(
        source_service="orion-bus",
        source_component="bus_transport_grammar_emit",
        source_event_id="20260525T233010Z",
    )


def _atom(role: str, summary: str) -> GrammarAtomV1:
    return GrammarAtomV1(
        atom_id=f"{TRACE}:{role}",
        trace_id=TRACE,
        atom_type="observation",
        semantic_role=role,
        layer="transport",
        dimensions=["bus"],
        summary=summary,
        confidence=1.0,
        salience=0.5,
        source_event_id="20260525T233010Z",
        payload_ref=f"bus.transport:{role}",
    )


def _event(event_id: str, role: str, summary: str) -> GrammarEventV1:
    return GrammarEventV1(
        event_id=event_id,
        event_kind="atom_emitted",
        trace_id=TRACE,
        session_id="bus-session",
        correlation_id=TRACE,
        emitted_at=NOW,
        observed_at=NOW,
        provenance=_prov(),
        atom=_atom(role, summary),
    )


def _live_events() -> list[GrammarEventV1]:
    return [
        _event("gev_h", "bus_health_observed", "redis_ping_ok=true node_id=athena sample_window_id=20260525T233010Z"),
        _event(
            "gev_d1",
            "bus_stream_depth_observed",
            "stream_key=orion:evt:gateway stream_length=0 sample_window_id=20260525T233010Z",
        ),
        _event(
            "gev_d2",
            "bus_stream_depth_observed",
            "stream_key=orion:bus:out stream_length=0 sample_window_id=20260525T233010Z",
        ),
        _event(
            "gev_u1",
            "bus_configured_stream_uncataloged",
            "stream_key=orion:evt:gateway sample_window_id=20260525T233010Z",
        ),
        _event(
            "gev_u2",
            "bus_configured_stream_uncataloged",
            "stream_key=orion:bus:out sample_window_id=20260525T233010Z",
        ),
        _event("gev_done", "bus_observer_tick_completed", "streams_observed=2 sample_window_id=20260525T233010Z"),
    ]


def test_parse_bus_transport_trace_id() -> None:
    assert parse_bus_transport_trace_id("bus.transport:athena:20260525T233010Z") == (
        "athena",
        "20260525T233010Z",
    )


def test_extract_live_athena_rollup_pressures() -> None:
    state = extract_transport_bus_state_from_events(_live_events(), now=NOW)
    pressures = compute_transport_pressures(state, stream_depth_critical=100_000)
    assert pressures["bus_health"] == 1.0
    assert pressures["catalog_drift_pressure"] == 1.0
    assert pressures["transport_pressure"] == 0.0
    # contract_pressure is now genuinely independent of catalog_drift_pressure
    # (see test_contract_pressure_diverges_from_catalog_drift_pressure below):
    # _live_events() has no bus_schema_validation_failed atoms, so
    # schema_mismatch_stream_count stays 0 and contract_pressure is 0.0 here
    # even though catalog_drift_pressure is 1.0 -- before the fix these two
    # were a literal alias (contract_pressure = catalog_drift_pressure) and
    # this assertion would have read 1.0.
    assert pressures["contract_pressure"] == 0.0
    assert state.target_id == "bus:athena"


def test_contract_pressure_diverges_from_catalog_drift_pressure() -> None:
    """Regression test for the original bug this task started from:
    contract_pressure and catalog_drift_pressure were a literal alias
    (orion/substrate/transport_loop/extract.py used to read
    `contract_pressure = catalog_drift_pressure`), byte-identical across
    122,509+ live corpus rows with 0 mismatches. They must now be able to
    differ given genuinely different inputs: a stream that IS cataloged but
    fails schema validation (bus_schema_validation_failed) vs. a stream that
    is uncataloged entirely (bus_configured_stream_uncataloged) are different
    failure modes and must not collapse to the same pressure value."""
    events = [
        _event(
            "gev_h",
            "bus_health_observed",
            "redis_ping_ok=true node_id=athena sample_window_id=20260525T233010Z",
        ),
        _event(
            "gev_d1",
            "bus_stream_depth_observed",
            "stream_key=orion:core:events stream_length=0 sample_window_id=20260525T233010Z",
        ),
        _event(
            "gev_d2",
            "bus_stream_depth_observed",
            "stream_key=orion:no:schema stream_length=0 sample_window_id=20260525T233010Z",
        ),
        # Only ONE of the two streams is uncataloged...
        _event(
            "gev_u1",
            "bus_configured_stream_uncataloged",
            "stream_key=orion:no:schema sample_window_id=20260525T233010Z",
        ),
        # ...and only the OTHER (cataloged) stream fails schema validation --
        # genuinely different streams, genuinely different failure modes.
        _event(
            "gev_s1",
            "bus_schema_validation_failed",
            "stream_key=orion:core:events mismatch_count=1 sampled_count=5 "
            "sample_window_id=20260525T233010Z",
        ),
        _event("gev_done", "bus_observer_tick_completed", "streams_observed=2 sample_window_id=20260525T233010Z"),
    ]
    state = extract_transport_bus_state_from_events(events, now=NOW)
    pressures = compute_transport_pressures(state, stream_depth_critical=100_000)
    assert state.uncataloged_stream_count == 1
    assert state.schema_mismatch_stream_count == 1
    # Same magnitude here (1/2 each) by coincidence of this fixture, but they
    # are computed from two entirely independent counters now -- prove that
    # by changing just one of the two inputs and checking only that pressure
    # moves.
    assert pressures["catalog_drift_pressure"] == 0.5
    assert pressures["contract_pressure"] == 0.5

    events_more_mismatch = events + [
        _event(
            "gev_s2",
            "bus_schema_validation_failed",
            "stream_key=orion:another:stream mismatch_count=1 sampled_count=5 "
            "sample_window_id=20260525T233010Z",
        ),
    ]
    state2 = extract_transport_bus_state_from_events(events_more_mismatch, now=NOW)
    pressures2 = compute_transport_pressures(state2, stream_depth_critical=100_000)
    # catalog_drift_pressure is untouched by the extra schema-mismatch atom...
    assert pressures2["catalog_drift_pressure"] == 0.5
    # ...while contract_pressure moves independently.
    assert pressures2["contract_pressure"] == 1.0
    assert pressures2["contract_pressure"] != pressures2["catalog_drift_pressure"]


def test_reducer_emits_transport_bus_delta_with_pressure_hints() -> None:
    projection = TransportBusProjectionV1(updated_at=NOW)
    projection, receipt = reduce_transport_trace_events(events=_live_events(), projection=projection, now=NOW)
    assert receipt.state_deltas
    delta = receipt.state_deltas[0]
    assert delta.target_kind == "transport_bus"
    assert delta.target_id == "bus:athena"
    hints = (delta.after or {}).get("pressure_hints") or {}
    assert hints["catalog_drift_pressure"] == 1.0
    assert hints["transport_pressure"] == 0.0
