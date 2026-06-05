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
        bus_health=1.0,
        delivery_confidence=1.0,
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
            transport_pressure=1.5,
        )


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
