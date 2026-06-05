from __future__ import annotations

from datetime import datetime, timezone

from orion.schemas.grammar import GrammarAtomV1, GrammarEventV1, GrammarProvenanceV1
from orion.schemas.transport_projection import TransportBusProjectionV1
from orion.substrate.transport_loop.constants import TRANSPORT_BUS_PROJECTION_ID
from orion.substrate.transport_loop.pipeline import process_transport_grammar_events

FIXED_TS = datetime(2026, 5, 25, 23, 30, 10, tzinfo=timezone.utc)
TRACE = "bus.transport:athena:20260525T233010Z"


def _event(event_id: str, role: str) -> GrammarEventV1:
    return GrammarEventV1(
        event_id=event_id,
        event_kind="atom_emitted",
        trace_id=TRACE,
        emitted_at=FIXED_TS,
        observed_at=FIXED_TS,
        atom=GrammarAtomV1(
            atom_id=f"{TRACE}:{role}",
            trace_id=TRACE,
            atom_type="observation",
            semantic_role=role,
            layer="transport",
            summary=f"redis_ping_ok=true streams_observed=1 sample_window_id=20260525T233010Z role={role}",
        ),
        provenance=GrammarProvenanceV1(
            source_service="orion-bus",
            source_component="bus_transport_grammar_emit",
        ),
        correlation_id=TRACE,
    )


def test_pipeline_groups_by_trace_and_persists_receipts() -> None:
    state = {
        "projection": TransportBusProjectionV1(
            projection_id=TRANSPORT_BUS_PROJECTION_ID,
            updated_at=FIXED_TS,
            buses={},
        ),
        "receipts": [],
    }

    stats = process_transport_grammar_events(
        events=[
            _event("gev_h", "bus_health_observed"),
            _event("gev_done", "bus_observer_tick_completed"),
        ],
        load_projection=lambda: state["projection"],
        save_projection=lambda p: state.update(projection=p),
        save_receipt=lambda r: state["receipts"].append(r),
        now=FIXED_TS,
    )

    assert stats["events"] == 2
    assert stats["receipts"] == 1
    assert "bus:athena" in state["projection"].buses
    assert state["receipts"][0].state_deltas[0].target_kind == "transport_bus"
