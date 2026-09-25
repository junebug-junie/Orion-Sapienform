from __future__ import annotations

from datetime import datetime, timezone

import pytest

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
    # Real observer ticks open with bus_observer_tick_started; the reducer
    # only writes whole ticks (started .. completed/failed).
    return [
        _event("gev_s", "bus_observer_tick_started", "node_id=athena sample_window_id=20260525T233010Z"),
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
    pressures = compute_transport_pressures(state)
    assert pressures["catalog_drift_pressure"] == 1.0
    # ping ok, no observer failure -> reliability 0.0 (same value the retired
    # 1 - delivery_confidence formula gave).
    assert pressures["reliability_pressure"] == 0.0
    # _live_events() still carries two pre-retirement bus_stream_depth_observed
    # atoms: they must not count as evidence or produce any depth field.
    assert "gev_d1" not in state.evidence_event_ids
    assert "gev_d2" not in state.evidence_event_ids
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
    pressures = compute_transport_pressures(state)
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
    pressures2 = compute_transport_pressures(state2)
    # catalog_drift_pressure is untouched by the extra schema-mismatch atom...
    assert pressures2["catalog_drift_pressure"] == 0.5
    # ...while contract_pressure moves independently.
    assert pressures2["contract_pressure"] == 1.0
    assert pressures2["contract_pressure"] != pressures2["catalog_drift_pressure"]


class TestCatalogDriftPressureMeshWideFix:
    """docs/superpowers/specs/2026-07-25-catalog-drift-pressure-mesh-wide-fix.md:
    catalog_drift_pressure now prefers the mesh-wide census diff
    (undeclared_active_count/catalog_size) when available, falling back to
    the old narrow formula (uncataloged_stream_count/streams_observed) when
    the census step didn't run this tick (gated off, or the scan failed)."""

    def _events_with_census(self, *, undeclared_active_count: int, catalog_size: int) -> list[GrammarEventV1]:
        return [
            _event(
                "gev_h",
                "bus_health_observed",
                "redis_ping_ok=true node_id=athena sample_window_id=20260525T233010Z",
            ),
            _event(
                "gev_d1",
                "bus_stream_depth_observed",
                "stream_key=orion:evt:gateway stream_length=0 sample_window_id=20260525T233010Z",
            ),
            # Old-formula inputs deliberately present and would read
            # differently (1/1 = 1.0) -- proves the mesh-wide value wins
            # when both are available, not silently ignored.
            _event(
                "gev_u1",
                "bus_configured_stream_uncataloged",
                "stream_key=orion:evt:gateway sample_window_id=20260525T233010Z",
            ),
            _event(
                "gev_c1",
                "bus_census_computed",
                f"undeclared_active_count={undeclared_active_count} catalog_size={catalog_size} "
                "sample_window_id=20260525T233010Z",
            ),
            _event("gev_done", "bus_observer_tick_completed", "streams_observed=1 sample_window_id=20260525T233010Z"),
        ]

    def test_mesh_wide_census_wins_over_old_formula_when_present(self) -> None:
        events = self._events_with_census(undeclared_active_count=2, catalog_size=264)
        state = extract_transport_bus_state_from_events(events, now=NOW)
        pressures = compute_transport_pressures(state)

        assert state.undeclared_active_count == 2
        assert state.catalog_size == 264
        # Old formula would read 1.0 (1 uncataloged / 1 stream observed) --
        # the mesh-wide value (2/264) must be what actually wins.
        assert pressures["catalog_drift_pressure"] == pytest.approx(2 / 264)
        assert pressures["catalog_drift_pressure"] != 1.0

    def test_falls_back_to_old_formula_when_census_not_available(self) -> None:
        # No bus_census_computed atom at all -- same fixture as
        # test_extract_live_athena_rollup_pressures, confirming the fallback
        # path is exactly the pre-existing behavior, not a new formula.
        state = extract_transport_bus_state_from_events(_live_events(), now=NOW)
        pressures = compute_transport_pressures(state)

        assert state.undeclared_active_count is None
        assert pressures["catalog_drift_pressure"] == 1.0

    def test_zero_undeclared_active_is_a_real_measured_zero_not_fallback(self) -> None:
        events = self._events_with_census(undeclared_active_count=0, catalog_size=264)
        state = extract_transport_bus_state_from_events(events, now=NOW)
        pressures = compute_transport_pressures(state)

        assert state.undeclared_active_count == 0
        # A real, honest zero -- not the old formula's 1.0.
        assert pressures["catalog_drift_pressure"] == 0.0


def test_reducer_emits_transport_bus_delta_with_pressure_hints() -> None:
    projection = TransportBusProjectionV1(updated_at=NOW)
    projection, receipt = reduce_transport_trace_events(events=_live_events(), projection=projection, now=NOW)
    assert receipt.state_deltas
    delta = receipt.state_deltas[0]
    assert delta.target_kind == "transport_bus"
    assert delta.target_id == "bus:athena"
    hints = (delta.after or {}).get("pressure_hints") or {}
    assert hints["catalog_drift_pressure"] == 1.0
    # Exactly these four survive the 2026-09-25 retirement of the XLEN
    # depth family (stream_backlog_*/delivery_confidence/stream_depth_pressure/
    # backpressure). A retired name reappearing here would re-feed the field.
    assert set(hints) == {
        "catalog_drift_pressure",
        "observer_failure_pressure",
        "contract_pressure",
        "reliability_pressure",
    }


@pytest.mark.parametrize("ping_ok", [True, None, False])
@pytest.mark.parametrize("observer_failures", [0, 1])
def test_reliability_pressure_unchanged_by_delivery_confidence_retirement(
    ping_ok: bool | None, observer_failures: int
) -> None:
    """reliability_pressure used to be max(observer_failure, 1 -
    delivery_confidence), with delivery_confidence derived from the ping. The
    2026-09-25 retirement removed delivery_confidence; reliability_pressure
    must read exactly what it read before in every ping/failure combination,
    including the non-calm ones (ping failed, observer failed)."""
    from orion.schemas.transport_projection import TransportBusStateV1

    state = TransportBusStateV1(
        target_id="bus:athena",
        node_id="athena",
        sample_window_id="w",
        source_trace_id="t",
        redis_ping_ok=ping_ok,
        observer_failure_count=observer_failures,
    )
    # The pre-retirement formula, verbatim.
    health = 1.0 if ping_ok is True else (0.0 if ping_ok is False else 0.5)
    obs = 1.0 if observer_failures > 0 else 0.0
    if obs > 0.0:
        dc = 0.0
    elif health >= 1.0:
        dc = 1.0
    elif health == 0.5:
        dc = 0.5
    else:
        dc = 0.0
    expected = max(obs, 1.0 - dc)

    assert compute_transport_pressures(state)["reliability_pressure"] == expected
