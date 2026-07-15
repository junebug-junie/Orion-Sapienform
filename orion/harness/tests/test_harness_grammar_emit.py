from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from orion.harness.grammar_emit import (
    HarnessGrammarCollector,
    build_harness_grammar_events,
    build_harness_grammar_finalize_events,
    compute_harness_reasoning_present,
    publish_harness_lifecycle_grammar,
)
from orion.substrate.execution_loop.ids import cortex_exec_trace_id

NODE = "athena"
CORR = "corr-harness-1"
FIXED = datetime(2026, 7, 9, 12, 0, tzinfo=timezone.utc)


def _install_fake_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch grammar_emit's datetime.now() to return strictly increasing,
    deterministic timestamps -- one tick per call -- so tests can assert
    real ordering/distinctness without depending on wall-clock speed. Same
    pattern as CortexExecGrammarCollector's test suite
    (services/orion-cortex-exec/tests/test_exec_grammar_emit.py)."""
    import orion.harness.grammar_emit as ge

    counter = {"n": 0}
    base = datetime(2026, 7, 9, 12, 0, 0, tzinfo=timezone.utc)

    def _fake_now(tz: object = None) -> datetime:
        counter["n"] += 1
        return base + timedelta(milliseconds=counter["n"])

    monkeypatch.setattr(ge, "datetime", SimpleNamespace(now=_fake_now))


def _record_two_step_trace(collector: HarnessGrammarCollector) -> None:
    collector.record_request_received()
    collector.record_plan_started(step_count=2)
    collector.record_recall_gate_observed(run_recall=False, profile="p", reason="r")
    for i in (1, 2):
        collector.record_step_started(order=i, summary=f"step_{i}")
        collector.record_step_completed(order=i)
    collector.record_result_assembled(
        status="success", final_text_present=True, step_count=2,
        grammar_receipt_count=1, reflection_ran=False, quick_lane_skipped_5b=True,
    )
    collector.record_result_emitted(reply_present=True, status="success")


def test_trace_id_matches_cortex_exec_shape() -> None:
    c = HarnessGrammarCollector(node_name=NODE, correlation_id=CORR, observed_at=FIXED)
    assert c.trace_id == cortex_exec_trace_id(NODE, CORR)


def test_compute_reasoning_present_rules() -> None:
    assert compute_harness_reasoning_present(step_count=2, reflection_ran=False, quick_lane_skipped_5b=True, grammar_receipt_count=0) is True
    assert compute_harness_reasoning_present(step_count=0, reflection_ran=True, quick_lane_skipped_5b=False, grammar_receipt_count=0) is True
    assert compute_harness_reasoning_present(step_count=0, reflection_ran=True, quick_lane_skipped_5b=True, grammar_receipt_count=0) is False
    assert compute_harness_reasoning_present(step_count=0, reflection_ran=False, quick_lane_skipped_5b=False, grammar_receipt_count=3) is True


def test_build_events_include_lifecycle_roles() -> None:
    c = HarnessGrammarCollector(node_name=NODE, correlation_id=CORR, observed_at=FIXED)
    c.record_request_received()
    c.record_plan_started(step_count=0)
    c.record_step_started(order=1, summary="fcc step")
    c.record_step_completed(order=1)
    c.record_result_assembled(
        status="success",
        final_text_present=True,
        step_count=1,
        grammar_receipt_count=1,
        reflection_ran=False,
        quick_lane_skipped_5b=True,
    )
    events = build_harness_grammar_events(c)
    roles = {e.atom.semantic_role for e in events if e.atom}
    assert {
        "exec_request_received",
        "exec_plan_started",
        "exec_step_started",
        "exec_step_completed",
        "exec_result_assembled",
    } <= roles
    assembled = next(e for e in events if e.atom and e.atom.semantic_role == "exec_result_assembled")
    assert "reasoning_present=True" in assembled.atom.summary
    assert "thinking_source=harness_fcc" in assembled.atom.summary


def test_record_tool_provenance_mismatch_emits_uncertainty_marker_atom() -> None:
    c = HarnessGrammarCollector(node_name=NODE, correlation_id=CORR, observed_at=FIXED)
    c.record_request_received()
    c.record_step_started(order=1, summary="fcc step")
    c.record_step_completed(order=1)
    c.record_tool_provenance_mismatch(mismatch="tool_provenance_mismatch: draft uses live-immediacy language...")
    events = build_harness_grammar_events(c)
    atom = next(e for e in events if e.atom and e.atom.semantic_role == "exec_tool_provenance_mismatch")
    assert atom.atom.atom_type == "uncertainty_marker"
    assert atom.atom.summary.startswith("tool_provenance_mismatch:")


def test_record_tool_provenance_mismatch_becomes_last_completed_atom() -> None:
    """Regression: record_tool_provenance_mismatch must update
    _last_completed_atom_id like its sibling terminal methods do, or the
    subsequent record_result_assembled's own derived_from edge points past
    it at the last *step* atom, leaving the mismatch atom a graph leaf with
    no outgoing edge into the result-assembled chain."""
    c = HarnessGrammarCollector(node_name=NODE, correlation_id=CORR, observed_at=FIXED)
    c.record_request_received()
    c.record_step_started(order=1, summary="fcc step")
    c.record_step_completed(order=1)
    c.record_tool_provenance_mismatch(mismatch="tool_provenance_mismatch: ...")
    c.record_result_assembled(
        status="success",
        final_text_present=True,
        step_count=1,
        grammar_receipt_count=1,
        reflection_ran=False,
        quick_lane_skipped_5b=True,
    )
    events = build_harness_grammar_events(c)
    mismatch_atom = next(e.atom for e in events if e.atom and e.atom.semantic_role == "exec_tool_provenance_mismatch")
    assembled_atom = next(e.atom for e in events if e.atom and e.atom.semantic_role == "exec_result_assembled")
    edges = [e.edge for e in events if e.edge]
    assert any(
        edge.from_atom_id == mismatch_atom.atom_id and edge.to_atom_id == assembled_atom.atom_id
        for edge in edges
    )


def test_atoms_get_distinct_real_observed_at_not_shared_flush_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Before this fix: every atom in a harness trace shared one trace-START
    timestamp (confirmed live: harness traces with 7-55 atoms and up to
    2m15s real wall-clock span all showed exactly 1 distinct observed_at
    across every atom). Each record_*() call must now stamp its own real
    observed_at, in recording order. Same fix shape as
    CortexExecGrammarCollector (services/orion-cortex-exec/app/grammar_emit.py)."""
    _install_fake_clock(monkeypatch)
    c = HarnessGrammarCollector(node_name=NODE, correlation_id=CORR, observed_at=FIXED)
    _record_two_step_trace(c)

    events = build_harness_grammar_events(c)
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
    c = HarnessGrammarCollector(node_name=NODE, correlation_id=CORR, observed_at=FIXED)
    _record_two_step_trace(c)

    events = build_harness_grammar_events(c)
    atom_events = [e for e in events if e.atom is not None]
    emitted_ats = {e.emitted_at for e in atom_events}
    assert len(emitted_ats) == 1, f"expected one shared emitted_at, got {emitted_ats}"


def test_edge_observed_at_matches_target_atom_real_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_clock(monkeypatch)
    c = HarnessGrammarCollector(node_name=NODE, correlation_id=CORR, observed_at=FIXED)
    _record_two_step_trace(c)

    events = build_harness_grammar_events(c)
    atom_observed_at = {e.atom.atom_id: e.observed_at for e in events if e.atom is not None}
    edge_events = [e for e in events if e.edge is not None]
    assert edge_events, "expected at least one edge in a two-step trace"
    for edge_event in edge_events:
        target_id = edge_event.edge.to_atom_id
        assert target_id in atom_observed_at
        assert edge_event.observed_at == atom_observed_at[target_id]
        # Self-consistency alone (edge == its own target atom) passes
        # trivially even if the whole per-atom fix were reverted back to one
        # shared value -- also assert against the pre-fix shared constant
        # (trace-start observed_at) directly, so a revert fails this too.
        assert edge_event.observed_at != FIXED


def test_atom_missing_from_observed_at_map_falls_back_to_trace_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defensive fallback: if an atom somehow bypassed _put_atom() and has no
    captured observed_at, use the collector's trace-start observed_at rather
    than crashing or emitting None."""
    _install_fake_clock(monkeypatch)
    c = HarnessGrammarCollector(node_name=NODE, correlation_id=CORR, observed_at=FIXED)
    c.record_request_received()
    missing_atom_id = next(iter(c._atoms.values())).atom_id
    del c._atom_observed_at[missing_atom_id]

    events = build_harness_grammar_events(c)
    atom_event = next(e for e in events if e.atom is not None and e.atom.atom_id == missing_atom_id)
    assert atom_event.observed_at == FIXED


def test_atom_time_range_populated_with_real_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GrammarAtomV1.time_range previously stayed None for every harness
    atom -- orion/grammar/ledger.py wires it through to persisted
    grammar_atoms.time_start/time_end, read by the live Grammar Atlas API,
    so this was a second place the same 'timing is fake' symptom survived
    even after observed_at is fixed on the sibling GrammarEventV1 envelope.
    _put_atom() must now stamp it too, from the same captured moment as
    _atom_observed_at."""
    _install_fake_clock(monkeypatch)
    c = HarnessGrammarCollector(node_name=NODE, correlation_id=CORR, observed_at=FIXED)
    _record_two_step_trace(c)

    events = build_harness_grammar_events(c)
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
    c = HarnessGrammarCollector(node_name=NODE, correlation_id=CORR, observed_at=FIXED)
    _record_two_step_trace(c)

    events = build_harness_grammar_events(c)
    trace_started = next(e for e in events if e.event_kind == "trace_started")
    trace_ended = next(e for e in events if e.event_kind == "trace_ended")
    atom_events = [e for e in events if e.atom is not None]

    assert trace_started.observed_at == FIXED
    assert trace_ended.observed_at != trace_started.observed_at
    assert trace_ended.observed_at == max(e.observed_at for e in atom_events)


def test_trace_ended_falls_back_to_trace_start_with_zero_atoms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A trace that fails before any record_*() call has no atoms to derive
    a real end time from -- trace_ended must fall back to trace-start
    observed_at rather than raising (max() on an empty sequence)."""
    _install_fake_clock(monkeypatch)
    c = HarnessGrammarCollector(node_name=NODE, correlation_id=CORR, observed_at=FIXED)
    events = build_harness_grammar_events(c)
    trace_ended = next(e for e in events if e.event_kind == "trace_ended")
    assert trace_ended.observed_at == FIXED


def test_finalize_atoms_use_real_observed_at(monkeypatch: pytest.MonkeyPatch) -> None:
    """build_harness_grammar_finalize_events must apply the same per-atom
    real-timestamp fix as build_harness_grammar_events -- it re-emits
    exec_result_assembled/exec_result_emitted from the same collector state,
    so it shares the exact bug shape if left on the old shared observed_at."""
    _install_fake_clock(monkeypatch)
    c = HarnessGrammarCollector(node_name=NODE, correlation_id=CORR, observed_at=FIXED)
    _record_two_step_trace(c)

    events = build_harness_grammar_finalize_events(c)
    atom_events = [e for e in events if e.atom is not None]
    assert len(atom_events) == 2
    observed_ats = {e.observed_at for e in atom_events}
    assert len(observed_ats) == 2, f"expected distinct observed_at per atom, got {observed_ats}"
    assert FIXED not in observed_ats

    edge_events = [e for e in events if e.edge is not None]
    assert edge_events
    emitted_atom = next(e for e in atom_events if e.atom.semantic_role == "exec_result_emitted")
    assert edge_events[0].observed_at == emitted_atom.observed_at


def test_finalize_events_emit_only_assembled_and_egress() -> None:
    c = HarnessGrammarCollector(node_name=NODE, correlation_id=CORR, observed_at=FIXED)
    c.record_request_received()
    c.record_plan_started(step_count=0)
    c.record_result_assembled(
        status="success",
        final_text_present=True,
        step_count=1,
        grammar_receipt_count=1,
        reflection_ran=True,
        quick_lane_skipped_5b=False,
    )
    c.record_result_emitted(reply_present=True, status="success")
    events = build_harness_grammar_finalize_events(c)
    kinds = {e.event_kind for e in events}
    roles = {e.atom.semantic_role for e in events if e.atom}
    assert "trace_started" not in kinds
    assert roles == {"exec_result_assembled", "exec_result_emitted"}
    assert any(e.event_kind == "edge_emitted" for e in events)
