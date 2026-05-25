from __future__ import annotations

from datetime import datetime, timezone
from typing import get_args

import pytest

from app.grammar_emit import (
    BusTransportGrammarCollector,
    build_bus_transport_grammar_events,
    bus_transport_trace_id,
)
from orion.schemas.grammar import AtomType, GrammarEventKind, RelationType

FIXED_OBS = datetime(2026, 5, 25, 17, 0, 0, tzinfo=timezone.utc)
WINDOW = "20260525T170000Z"
NODE = "athena"


def test_trace_id_format() -> None:
    assert bus_transport_trace_id(NODE, WINDOW) == f"bus.transport:{NODE}:{WINDOW}"


def test_builds_transport_rollup_trace() -> None:
    collector = BusTransportGrammarCollector(
        node_id=NODE,
        sample_window_id=WINDOW,
        observed_at=FIXED_OBS,
        code_version="0.1.0",
    )
    collector.record_tick_started()
    collector.record_health_observed(redis_ping_ok=True)
    collector.record_stream_depth(stream_key="orion:evt:gateway", stream_length=123)
    collector.record_backpressure(
        stream_key="orion:evt:gateway",
        stream_length=50000,
        threshold=25000,
        severity="warning",
    )
    collector.record_uncataloged_stream(stream_key="orion:evt:gateway")
    collector.record_tick_completed(streams_observed=3)

    events = build_bus_transport_grammar_events(collector)
    assert events
    kinds = {e.event_kind for e in events}
    assert kinds <= set(get_args(GrammarEventKind))
    assert "trace_started" in kinds
    assert "trace_ended" in kinds

    roles = {e.atom.semantic_role for e in events if e.atom}
    assert roles >= {
        "bus_observer_tick_started",
        "bus_health_observed",
        "bus_stream_depth_observed",
        "bus_backpressure_observed",
        "bus_configured_stream_uncataloged",
        "bus_observer_tick_completed",
    }

    for event in events:
        if event.atom:
            assert event.atom.atom_type in get_args(AtomType)
        if event.edge:
            assert event.edge.relation_type in get_args(RelationType)

    health = next(
        e.atom for e in events if e.atom and e.atom.semantic_role == "bus_health_observed"
    )
    assert "redis_ping_ok=true" in health.summary
    assert "node_id=athena" in health.summary

    uncataloged = next(
        e.atom
        for e in events
        if e.atom and e.atom.semantic_role == "bus_configured_stream_uncataloged"
    )
    assert "not declared in channel catalog" in uncataloged.summary.lower()


def test_no_payload_blobs_in_summaries() -> None:
    collector = BusTransportGrammarCollector(
        node_id=NODE,
        sample_window_id=WINDOW,
        observed_at=FIXED_OBS,
    )
    collector.record_tick_started()
    collector.record_stream_depth(stream_key="orion:bus:out", stream_length=1)
    collector.record_tick_completed(streams_observed=1)
    events = build_bus_transport_grammar_events(collector)
    for event in events:
        if event.atom:
            assert "envelope" not in event.atom.summary.lower()
            assert "payload" not in event.atom.summary.lower()


def test_summaries_never_include_redis_values_or_envelope_material() -> None:
    collector = BusTransportGrammarCollector(
        node_id=NODE,
        sample_window_id=WINDOW,
        observed_at=FIXED_OBS,
    )
    collector.record_tick_started()
    collector.record_stream_depth(stream_key="orion:bus:out", stream_length=42)
    collector.record_tick_completed(streams_observed=1)
    events = build_bus_transport_grammar_events(collector)
    forbidden_fragments = (
        "{",
        "}",
        '"kind"',
        "grammar.event",
        "BaseEnvelope",
        "XREAD",
        "XRANGE",
        "redis://",
        "password",
    )
    for event in events:
        if not event.atom:
            continue
        summary = event.atom.summary
        assert event.atom.text_value is None
        for frag in forbidden_fragments:
            assert frag not in summary, f"forbidden fragment {frag!r} in {summary!r}"
        if event.atom.semantic_role == "bus_stream_depth_observed":
            assert "stream_length=42" in summary
