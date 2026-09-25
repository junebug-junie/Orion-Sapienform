"""A bus observer window cut across two reducer batches must not fabricate
bus health.

The transport reducer cursor pages grammar_events by (created_at, event_id)
with a row limit (TRANSPORT_GRAMMAR_BATCH_LIMIT) and polls while sql-writer is
still inserting a trace, so one 13-event observer tick can arrive in two
pieces. Each piece used to be reduced alone into a full TransportBusStateV1
that REPLACES buses["bus:athena"]:

  * a tail without bus_health_observed -> redis_ping_ok=None ->
    stream_backlog_health / delivery_confidence = 0.5, reliability_pressure
    = 0.5 (fabricated half-health), and
  * a head without bus_census_computed -> catalog_drift_pressure falls back to
    0/denom = 0.0 ("no drift" that nobody measured), streams_observed short.

Fixture: one real, unedited bus.transport:athena trace pulled from
grammar_events on 2026-09-25 (13 events in cursor order).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from orion.schemas.grammar import GrammarEventV1
from orion.schemas.transport_projection import TransportBusProjectionV1, TransportBusStateV1
from orion.substrate.transport_loop.constants import TRANSPORT_BUS_PROJECTION_ID
from orion.substrate.transport_loop.pipeline import process_transport_grammar_events
from orion.substrate.transport_loop.reducer import reduce_transport_trace_events

FIXTURE = Path(__file__).parent / "fixtures" / "transport_bus_observer_trace_live_2026-09-25.jsonl"
NOW = datetime(2026, 9, 25, 6, 1, tzinfo=timezone.utc)


def _live_trace() -> list[GrammarEventV1]:
    return [GrammarEventV1.model_validate(json.loads(line)) for line in FIXTURE.read_text().splitlines() if line]


def _prior_window() -> TransportBusStateV1:
    # A real, healthy previous-window reading that must survive a held piece.
    return TransportBusStateV1(
        target_id="bus:athena",
        node_id="athena",
        sample_window_id="20260925T060046Z",
        source_trace_id="bus.transport:athena:20260925T060046Z",
        redis_ping_ok=True,
        streams_observed=2,
        total_stream_depth=160,
        max_stream_depth=160,
        undeclared_active_count=3,
        catalog_size=276,
        stream_backlog_health=1.0,
        delivery_confidence=1.0,
        stream_depth_pressure=0.0016,
        catalog_drift_pressure=3 / 276,
        stream_backlog_pressure=0.0016,
        reliability_pressure=0.0,
        evidence_event_ids=["gev_prev"],
        observed_at=NOW,
    )


def _projection() -> TransportBusProjectionV1:
    return TransportBusProjectionV1(
        projection_id=TRANSPORT_BUS_PROJECTION_ID, updated_at=NOW, buses={"bus:athena": _prior_window()}
    )


def test_fixture_is_the_real_observer_shape() -> None:
    roles = [(e.atom.semantic_role if e.atom else e.event_kind) for e in _live_trace()]
    assert roles[0] == "trace_started" and roles[-1] == "trace_ended"
    assert roles.index("bus_observer_tick_started") < roles.index("bus_health_observed")
    assert roles.index("bus_health_observed") < roles.index("bus_census_computed")
    assert roles.index("bus_census_computed") < roles.index("bus_observer_tick_completed")
    assert "bus_activity_zscore_computed" in roles
    assert len(roles) == 13


# The honest reading of the fixture tick (redis ping ok, 2 streams, max depth
# 160, census 3/276).
_EXPECTED = {
    "redis_ping_ok": True,
    "streams_observed": 2,
    "max_stream_depth": 160,
    "total_stream_depth": 160,
    "undeclared_active_count": 3,
    "catalog_size": 276,
    "stream_backlog_health": 1.0,
    "delivery_confidence": 1.0,
    "reliability_pressure": 0.0,
    "catalog_drift_pressure": 3 / 276,
}


def _assert_honest(after: dict) -> None:
    for key, want in _EXPECTED.items():
        assert after[key] == pytest.approx(want), (key, after[key], want)


def _run_split(cut: int, *, with_loader: bool) -> tuple[TransportBusProjectionV1, list]:
    trace = _live_trace()
    stored = {"projection": _projection()}
    receipts: list = []
    persisted: list[GrammarEventV1] = []

    def loader(trace_id: str) -> list[GrammarEventV1]:
        # grammar_events only holds what sql-writer inserted so far; the
        # cursor has already passed everything in earlier batches.
        return [e for e in persisted if e.trace_id == trace_id]

    for batch in (trace[:cut], trace[cut:]):
        persisted.extend(batch)
        # Only pass the loader kwarg when used, so the no-loader case also runs
        # (and fails on the fabricated values) against pre-fix code.
        extra = {"load_trace_events": loader} if with_loader else {}
        process_transport_grammar_events(
            events=batch,
            load_projection=lambda: stored["projection"],
            save_projection=lambda p: stored.update(projection=p),
            save_receipt=receipts.append,
            now=NOW,
            **extra,
        )
    return stored["projection"], receipts


@pytest.mark.parametrize("cut", range(1, 13))
def test_every_cut_writes_only_the_honest_reading(cut: int) -> None:
    projection, receipts = _run_split(cut, with_loader=True)
    deltas = [d for r in receipts for d in r.state_deltas if d.target_id == "bus:athena"]
    # Every cut still lands the tick exactly once...
    assert len(deltas) == 1, [r.warnings for r in receipts]
    # ...and nothing written, at any point, is fabricated.
    for delta in deltas:
        _assert_honest(delta.after)
        assert delta.after["pressure_hints"]["stream_backlog_health"] == 1.0
        assert delta.after["pressure_hints"]["reliability_pressure"] == 0.0
    _assert_honest(projection.buses["bus:athena"].model_dump(mode="json"))
    assert projection.buses["bus:athena"].sample_window_id == "20260925T060056Z"


@pytest.mark.parametrize("cut", range(1, 13))
def test_without_a_loader_a_piece_is_held_not_fabricated(cut: int) -> None:
    projection, receipts = _run_split(cut, with_loader=False)
    for delta in (d for r in receipts for d in r.state_deltas):
        _assert_honest(delta.after)
    bus = projection.buses["bus:athena"]
    # Either the whole tick landed (cut outside the observer atoms) or the
    # previous window's real reading still stands -- never 0.5.
    assert bus.stream_backlog_health == 1.0
    assert bus.reliability_pressure == 0.0
    assert bus.sample_window_id in {"20260925T060046Z", "20260925T060056Z"}


def test_tail_without_health_alone_is_the_reported_bug() -> None:
    """Direct repro of PR #2323's named risk: the piece after
    bus_health_observed, reduced on its own, used to overwrite bus:athena
    with stream_backlog_health=0.5 / reliability_pressure=0.5."""
    trace = _live_trace()
    health_at = next(i for i, e in enumerate(trace) if e.atom and e.atom.semantic_role == "bus_health_observed")
    tail = trace[health_at + 1 :]
    out, receipt = reduce_transport_trace_events(events=tail, projection=_projection(), now=NOW)
    assert receipt.state_deltas == []
    assert out.buses["bus:athena"].stream_backlog_health == 1.0
    assert out.buses["bus:athena"].reliability_pressure == 0.0
    assert any("incomplete observer window held" in w for w in receipt.warnings)


def test_head_without_census_does_not_claim_no_drift() -> None:
    trace = _live_trace()
    census_at = next(i for i, e in enumerate(trace) if e.atom and e.atom.semantic_role == "bus_census_computed")
    out, receipt = reduce_transport_trace_events(events=trace[:census_at], projection=_projection(), now=NOW)
    assert receipt.state_deltas == []
    assert out.buses["bus:athena"].catalog_drift_pressure == pytest.approx(3 / 276)


def test_loader_failure_holds_instead_of_writing_the_piece() -> None:
    trace = _live_trace()

    def boom(trace_id: str) -> list[GrammarEventV1]:
        raise RuntimeError("db down")

    out, receipt = reduce_transport_trace_events(
        events=trace[5:], projection=_projection(), now=NOW, load_trace_events=boom
    )
    assert receipt.state_deltas == []
    assert out.buses["bus:athena"].sample_window_id == "20260925T060046Z"
    assert any("trace reload failed" in w for w in receipt.warnings)


def test_rebuilt_tail_cites_the_whole_tick_as_evidence() -> None:
    trace = _live_trace()
    _, receipt = reduce_transport_trace_events(
        events=trace[6:], projection=_projection(), now=NOW, load_trace_events=lambda _t: trace
    )
    (delta,) = receipt.state_deltas
    health_id = next(e.event_id for e in trace if e.atom and e.atom.semantic_role == "bus_health_observed")
    assert health_id in delta.caused_by_event_ids
    assert health_id in delta.after["evidence_event_ids"]
