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
    assert pressures["contract_pressure"] == 1.0
    assert state.target_id == "bus:athena"


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
