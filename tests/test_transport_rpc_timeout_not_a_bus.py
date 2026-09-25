"""The RPC-timeout grammar atom must never become a transport "bus".

orion/core/bus/async_service.py::_emit_rpc_timeout_grammar publishes every
rpc_request() timeout on trace `bus.transport:rpc_timeout:<corr>`, which the
transport reducer used to parse as bus node "rpc_timeout" -- minting a fake
`bus:rpc_timeout` projection entry (zero evidence, redis_ping_ok=None ->
fabricated 0.5 delivery_confidence/reliability_pressure/stream_backlog_health)
that the field digester turned into node:rpc_timeout, a dominant attention
target in 42% of attention frames on 2026-09-19.

_LIVE_EVENT is the verbatim grammar_events.event_json row for
correlation f3510a9a-..., 2026-09-22T05:39:25Z.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from orion.core.bus.async_service import OrionBusAsync
from orion.schemas.grammar import GrammarAtomV1, GrammarEventV1, GrammarProvenanceV1
from orion.schemas.transport_projection import TransportBusProjectionV1, TransportBusStateV1
from orion.substrate.transport_loop.constants import TRANSPORT_BUS_PROJECTION_ID
from orion.substrate.transport_loop.extract import parse_bus_transport_trace_id
from orion.substrate.transport_loop.pipeline import (
    process_transport_grammar_events,
    prune_non_bus_entries,
)
from orion.substrate.transport_loop.reducer import reduce_transport_trace_events

NOW = datetime(2026, 9, 22, 5, 39, 30, tzinfo=timezone.utc)
CORR = "f3510a9a-7b03-45db-8896-33cdc29bc59f"
TRACE = f"bus.transport:rpc_timeout:{CORR}"

_LIVE_EVENT = {
    "atom": {
        "layer": "transport",
        "atom_id": f"{TRACE}:90693daa4c0e",
        "summary": (
            "RPC timeout: orion:exec:request:LLMGatewayService -> "
            f"orion:exec:result:LLMGatewayService:{CORR} after 3.0s (elapsed 4673.4ms)"
        ),
        "salience": 0.9,
        "trace_id": TRACE,
        "atom_type": "uncertainty_marker",
        "confidence": 1.0,
        "dimensions": ["transport", "bus", "rpc", "liveness"],
        "text_value": "orion:exec:request:LLMGatewayService",
        "semantic_role": "rpc_transport_timeout",
        "schema_version": "grammar_atom.v1",
        "source_event_id": CORR,
    },
    "layer": "transport",
    "event_id": f"{TRACE}:90693daa4c0e",
    "trace_id": TRACE,
    "dimensions": ["transport", "bus", "rpc", "liveness"],
    "emitted_at": "2026-09-22T05:39:25.112625Z",
    "event_kind": "atom_emitted",
    "provenance": {
        "source_service": "orion-bus",
        "source_event_id": CORR,
        "source_trace_id": TRACE,
        "source_component": "rpc_request_timeout",
    },
    "correlation_id": CORR,
    "schema_version": "grammar_event.v1",
}

# Verbatim live substrate_transport_bus_projection entry, 2026-09-24T20:02:50Z.
_LIVE_PHANTOM = {
    "node_id": "rpc_timeout",
    "target_id": "bus:rpc_timeout",
    "observed_at": "2026-09-24T20:02:50.407009Z",
    "redis_ping_ok": None,
    "source_trace_id": "bus.transport:rpc_timeout:9bef2d9e-d470-42c1-b15c-96fb5658adaf",
    "sample_window_id": "9bef2d9e-d470-42c1-b15c-96fb5658adaf",
    "evidence_event_ids": [],
    "delivery_confidence": 0.5,
    "reliability_pressure": 0.5,
    "stream_backlog_health": 0.5,
}


def _live_event() -> GrammarEventV1:
    return GrammarEventV1.model_validate(_LIVE_EVENT)


def _athena_event(role: str) -> GrammarEventV1:
    trace = "bus.transport:athena:20260922T053930Z"
    return GrammarEventV1(
        event_id=f"{trace}:{role}",
        event_kind="atom_emitted",
        trace_id=trace,
        emitted_at=NOW,
        atom=GrammarAtomV1(
            atom_id=f"{trace}:{role}",
            trace_id=trace,
            atom_type="observation",
            semantic_role=role,
            layer="transport",
            summary="redis_ping_ok=true streams_observed=1",
        ),
        provenance=GrammarProvenanceV1(source_service="orion-bus", source_component="bus_transport_grammar_emit"),
    )


def _empty() -> TransportBusProjectionV1:
    return TransportBusProjectionV1(projection_id=TRANSPORT_BUS_PROJECTION_ID, updated_at=NOW, buses={})


def test_rpc_timeout_trace_is_not_a_bus_trace() -> None:
    assert parse_bus_transport_trace_id(TRACE) is None
    # Real observer traces still parse.
    assert parse_bus_transport_trace_id("bus.transport:athena:20260922T053930Z") == (
        "athena",
        "20260922T053930Z",
    )


def test_live_rpc_timeout_event_is_a_noop_for_the_reducer() -> None:
    projection = _empty()
    out, receipt = reduce_transport_trace_events(events=[_live_event()], projection=projection, now=NOW)
    assert out.buses == {}
    assert receipt.state_deltas == []
    assert receipt.noop_event_ids == [_LIVE_EVENT["event_id"]]


@pytest.mark.asyncio
async def test_whatever_the_real_emitter_publishes_is_not_a_bus() -> None:
    """Pins the consumer to the emitter's actual output, not a hand copy."""
    bus = OrionBusAsync("redis://localhost:6379/0", enabled=False)
    bus.publish = AsyncMock()
    await bus._emit_rpc_timeout_grammar(
        request_channel="orion:exec:request:LLMGatewayService",
        reply_channel=f"orion:exec:result:LLMGatewayService:{CORR}",
        corr=CORR,
        timeout_sec=3.0,
        timeout_elapsed_ms=4673.4,
    )
    _, envelope = bus.publish.await_args.args
    event = GrammarEventV1.model_validate(envelope.payload)
    out, receipt = reduce_transport_trace_events(events=[event], projection=_empty(), now=NOW)
    assert out.buses == {}
    assert receipt.state_deltas == []


def test_zero_evidence_bus_trace_does_not_mint_fabricated_half_health() -> None:
    """A real-node trace carrying only trace_started/ended (no observer atom)
    must not write the extractor's 0.5 defaults over a real bus reading."""
    real = TransportBusStateV1.model_validate(
        {
            "target_id": "bus:athena",
            "node_id": "athena",
            "sample_window_id": "prev",
            "source_trace_id": "bus.transport:athena:prev",
            "redis_ping_ok": True,
            "evidence_event_ids": ["e1"],
            "delivery_confidence": 1.0,
            "stream_backlog_health": 1.0,
            "observed_at": NOW,
        }
    )
    projection = TransportBusProjectionV1(
        projection_id=TRANSPORT_BUS_PROJECTION_ID, updated_at=NOW, buses={"bus:athena": real}
    )
    out, receipt = reduce_transport_trace_events(
        events=[_athena_event("trace_ended")], projection=projection, now=NOW
    )
    assert out.buses["bus:athena"].delivery_confidence == 1.0
    assert receipt.state_deltas == []
    assert receipt.warnings and "no bus observer evidence" in receipt.warnings[0]


def test_persisted_phantom_is_pruned_on_next_batch_and_real_bus_kept() -> None:
    phantom = TransportBusStateV1.model_validate(_LIVE_PHANTOM)
    stored = {
        "projection": TransportBusProjectionV1(
            projection_id=TRANSPORT_BUS_PROJECTION_ID,
            updated_at=NOW,
            buses={"bus:rpc_timeout": phantom},
        )
    }
    receipts: list = []
    process_transport_grammar_events(
        events=[
            _live_event(),
            _athena_event("bus_health_observed"),
            _athena_event("bus_observer_tick_completed"),
        ],
        load_projection=lambda: stored["projection"],
        save_projection=lambda p: stored.update(projection=p),
        save_receipt=receipts.append,
        now=NOW,
    )
    assert set(stored["projection"].buses) == {"bus:athena"}
    kinds = [d.target_id for r in receipts for d in r.state_deltas]
    assert kinds == ["bus:athena"]


def test_prune_non_bus_entries_reports_what_it_dropped() -> None:
    projection = TransportBusProjectionV1(
        projection_id=TRANSPORT_BUS_PROJECTION_ID,
        updated_at=NOW,
        buses={"bus:rpc_timeout": TransportBusStateV1.model_validate(_LIVE_PHANTOM)},
    )
    assert prune_non_bus_entries(projection) == ["bus:rpc_timeout"]
    assert projection.buses == {}
