from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
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


def _install_fake_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch grammar_emit's datetime.now() to return strictly increasing,
    deterministic timestamps -- one tick per call -- so tests can assert
    real ordering/distinctness without depending on wall-clock speed. Same
    pattern as HarnessGrammarCollector/CortexExecGrammarCollector's test
    suites."""
    import app.grammar_emit as ge

    counter = {"n": 0}
    base = datetime(2026, 5, 25, 17, 0, 0, tzinfo=timezone.utc)

    def _fake_now(tz: object = None) -> datetime:
        counter["n"] += 1
        return base + timedelta(milliseconds=counter["n"])

    monkeypatch.setattr(ge, "datetime", SimpleNamespace(now=_fake_now))


def _record_full_tick(collector: BusTransportGrammarCollector) -> None:
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
    collector.record_tick_completed(streams_observed=1)


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


def test_atoms_get_distinct_real_observed_at_not_shared_flush_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Before this fix: every atom in a bus-transport trace shared one
    trace-START timestamp (BusTransportGrammarCollector.observed_at,
    captured once at construction). Each record_*() call must now stamp
    its own real observed_at, in recording order. Same fix shape as
    HarnessGrammarCollector/CortexExecGrammarCollector."""
    _install_fake_clock(monkeypatch)
    collector = BusTransportGrammarCollector(
        node_id=NODE, sample_window_id=WINDOW, observed_at=FIXED_OBS,
    )
    _record_full_tick(collector)

    events = build_bus_transport_grammar_events(collector)
    atom_events = [e for e in events if e.atom is not None]
    observed_ats = [e.observed_at for e in atom_events]

    assert len(set(observed_ats)) == len(observed_ats), (
        f"expected distinct observed_at per atom, got duplicates: {observed_ats}"
    )
    assert observed_ats == sorted(observed_ats), "expected observed_at in recording order"


def test_emitted_at_stays_shared_flush_time_across_all_atoms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """emitted_at was never wrong -- every event in a trace genuinely is
    published to the bus in the same flush batch. Only observed_at (real
    occurrence time) should change."""
    _install_fake_clock(monkeypatch)
    collector = BusTransportGrammarCollector(
        node_id=NODE, sample_window_id=WINDOW, observed_at=FIXED_OBS,
    )
    _record_full_tick(collector)

    events = build_bus_transport_grammar_events(collector)
    atom_events = [e for e in events if e.atom is not None]
    emitted_ats = {e.emitted_at for e in atom_events}
    assert len(emitted_ats) == 1, f"expected one shared emitted_at, got {emitted_ats}"


def test_edge_observed_at_matches_target_atom_real_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_clock(monkeypatch)
    collector = BusTransportGrammarCollector(
        node_id=NODE, sample_window_id=WINDOW, observed_at=FIXED_OBS,
    )
    _record_full_tick(collector)

    events = build_bus_transport_grammar_events(collector)
    atom_observed_at = {e.atom.atom_id: e.observed_at for e in events if e.atom is not None}
    edge_events = [e for e in events if e.edge is not None]
    assert edge_events, "expected at least one edge in a full tick"
    for edge_event in edge_events:
        target_id = edge_event.edge.to_atom_id
        assert target_id in atom_observed_at
        assert edge_event.observed_at == atom_observed_at[target_id]
        # Self-consistency alone (edge == its own target atom) passes
        # trivially even if the whole per-atom fix were reverted back to one
        # shared value -- also assert against the pre-fix shared constant
        # (trace-start observed_at) directly, so a revert fails this too.
        assert edge_event.observed_at != FIXED_OBS


def test_atom_missing_from_observed_at_map_falls_back_to_trace_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defensive fallback: if an atom somehow bypassed _put_atom() and has no
    captured observed_at, use the collector's trace-start observed_at rather
    than crashing or emitting None."""
    _install_fake_clock(monkeypatch)
    collector = BusTransportGrammarCollector(
        node_id=NODE, sample_window_id=WINDOW, observed_at=FIXED_OBS,
    )
    collector.record_tick_started()
    missing_atom_id = next(iter(collector._atoms.values())).atom_id
    del collector._atom_observed_at[missing_atom_id]

    events = build_bus_transport_grammar_events(collector)
    atom_event = next(e for e in events if e.atom is not None and e.atom.atom_id == missing_atom_id)
    assert atom_event.observed_at == FIXED_OBS


def test_atom_time_range_populated_with_real_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GrammarAtomV1.time_range previously stayed None for every bus-transport
    atom -- orion/grammar/ledger.py wires it through to persisted
    grammar_atoms.time_start/time_end, read by the live Grammar Atlas API,
    so this was a second place the same 'timing is fake' symptom survived
    even after observed_at is fixed on the sibling GrammarEventV1 envelope.
    _put_atom() must now stamp it too, from the same captured moment as
    _atom_observed_at."""
    _install_fake_clock(monkeypatch)
    collector = BusTransportGrammarCollector(
        node_id=NODE, sample_window_id=WINDOW, observed_at=FIXED_OBS,
    )
    _record_full_tick(collector)

    events = build_bus_transport_grammar_events(collector)
    atom_events = [e for e in events if e.atom is not None]
    assert atom_events
    for event in atom_events:
        assert event.atom.time_range is not None
        assert event.atom.time_range.start == event.observed_at
        assert event.atom.time_range.end == event.observed_at


def test_trace_ended_observed_at_reflects_last_atom_not_trace_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Before this fix: trace_ended.observed_at reused collector.observed_at
    (trace-START time), identical to trace_started's -- the exact mechanism
    behind orion/grammar/ledger.py's grammar_traces.started_at/ended_at
    collapsing to zero duration. trace_ended must now reflect the real last
    recorded atom's timestamp."""
    _install_fake_clock(monkeypatch)
    collector = BusTransportGrammarCollector(
        node_id=NODE, sample_window_id=WINDOW, observed_at=FIXED_OBS,
    )
    _record_full_tick(collector)

    events = build_bus_transport_grammar_events(collector)
    trace_started = next(e for e in events if e.event_kind == "trace_started")
    trace_ended = next(e for e in events if e.event_kind == "trace_ended")
    atom_events = [e for e in events if e.atom is not None]

    assert trace_started.observed_at == FIXED_OBS
    assert trace_ended.observed_at != trace_started.observed_at
    assert trace_ended.observed_at == max(e.observed_at for e in atom_events)


def test_trace_ended_falls_back_to_trace_start_with_zero_atoms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A trace that fails before any record_*() call has no atoms to derive
    a real end time from -- trace_ended must fall back to trace-start
    observed_at rather than raising (max() on an empty sequence)."""
    _install_fake_clock(monkeypatch)
    collector = BusTransportGrammarCollector(
        node_id=NODE, sample_window_id=WINDOW, observed_at=FIXED_OBS,
    )
    events = build_bus_transport_grammar_events(collector)
    trace_ended = next(e for e in events if e.event_kind == "trace_ended")
    assert trace_ended.observed_at == FIXED_OBS
