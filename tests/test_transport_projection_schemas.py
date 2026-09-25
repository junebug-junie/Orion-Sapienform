from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from orion.schemas.transport_projection import TransportBusProjectionV1, TransportBusStateV1

NOW = datetime(2026, 5, 25, 23, 30, 10, tzinfo=timezone.utc)


def test_transport_bus_state_defaults() -> None:
    state = TransportBusStateV1(
        target_id="bus:athena",
        node_id="athena",
        sample_window_id="20260525T233010Z",
        source_trace_id="bus.transport:athena:20260525T233010Z",
        catalog_drift_pressure=1.0,
        contract_pressure=1.0,
    )
    assert state.schema_version == "transport_bus.state.v1"


def test_transport_bus_state_rejects_out_of_range_pressure() -> None:
    with pytest.raises(ValidationError):
        TransportBusStateV1(
            target_id="bus:athena",
            node_id="athena",
            sample_window_id="w",
            source_trace_id="t",
            contract_pressure=1.5,
        )


# Verbatim node entry from the live substrate_transport_bus_projection row,
# 2026-09-25T05:56:24Z, before the stream-depth family was retired. The model is
# extra="forbid"; without the retired-field drop the first load after deploy
# would raise and the reducer would crash-loop (the 2026-07-24 incident in
# scripts/check_substrate_projection_schema_drift.py's docstring).
_LIVE_PRE_RETIREMENT_BUS = {
    "node_id": "athena",
    "target_id": "bus:athena",
    "observed_at": "2026-09-25T05:56:24.376927Z",
    "backpressure": 0.0,
    "catalog_size": 276,
    "redis_ping_ok": True,
    "schema_version": "transport_bus.state.v1",
    "source_trace_id": "bus.transport:athena:20260925T055623Z",
    "max_stream_depth": 160,
    "sample_window_id": "20260925T055623Z",
    "streams_observed": 2,
    "contract_pressure": 0.0,
    "backpressure_count": 0,
    "evidence_event_ids": ["gev_b400f921965f3f2f"],
    "total_stream_depth": 160,
    "delivery_confidence": 1.0,
    "reliability_pressure": 0.0,
    "stream_backlog_health": 1.0,
    "stream_depth_pressure": 0.0016,
    "catalog_drift_pressure": 0.010869565217391304,
    "observer_failure_count": 0,
    "stream_backlog_pressure": 0.0016,
    "undeclared_active_count": 3,
    "uncataloged_stream_count": 0,
    "observer_failure_pressure": 0.0,
    "schema_mismatch_stream_count": 0,
}


def test_pre_retirement_persisted_row_still_loads_and_drops_retired_fields() -> None:
    proj = TransportBusProjectionV1.model_validate(
        {
            "buses": {"bus:athena": _LIVE_PRE_RETIREMENT_BUS},
            "updated_at": "2026-09-25T05:56:24.376927Z",
            "projection_id": "active_transport_bus_projection",
            "schema_version": "transport_bus.projection.v1",
        }
    )
    bus = proj.buses["bus:athena"]
    assert bus.catalog_drift_pressure == pytest.approx(0.010869565217391304)
    dumped = bus.model_dump()
    for retired in (
        "stream_backlog_pressure",
        "stream_backlog_health",
        "delivery_confidence",
        "stream_depth_pressure",
        "backpressure",
        "max_stream_depth",
        "total_stream_depth",
        "backpressure_count",
    ):
        assert retired not in dumped


def test_retired_field_drop_does_not_open_extra_forbid_to_other_keys() -> None:
    with pytest.raises(ValidationError):
        TransportBusStateV1.model_validate({**_LIVE_PRE_RETIREMENT_BUS, "some_new_field": 1})


def test_transport_bus_projection_roundtrip() -> None:
    state = TransportBusStateV1(
        target_id="bus:athena",
        node_id="athena",
        sample_window_id="20260525T233010Z",
        source_trace_id="bus.transport:athena:20260525T233010Z",
    )
    proj = TransportBusProjectionV1(updated_at=NOW, buses={"bus:athena": state})
    raw = proj.model_dump(mode="json")
    assert TransportBusProjectionV1.model_validate(raw).buses["bus:athena"].node_id == "athena"
