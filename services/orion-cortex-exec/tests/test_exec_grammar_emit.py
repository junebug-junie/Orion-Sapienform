from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import get_args

import pytest

from app.grammar_emit import (
    CortexExecGrammarCollector,
    build_cortex_exec_grammar_events,
    cortex_exec_trace_id,
)
from orion.schemas.cortex.schemas import ExecutionPlan, PlanExecutionRequest
from orion.schemas.cortex.types import ExecutionStep
from orion.schemas.grammar import AtomType, GrammarEventKind, RelationType

FIXED_OBS = datetime(2026, 5, 24, 12, 0, 0, tzinfo=timezone.utc)
CORR = "corr-abc-123"
NODE = "athena"


def _minimal_plan(*, steps: int = 2) -> PlanExecutionRequest:
    plan_steps = [
        ExecutionStep(
            step_name=f"step_{i}",
            verb_name="chat_general" if i == 1 else "noop",
            order=i,
            services=["LLMGatewayService"] if i == 1 else [],
        )
        for i in range(1, steps + 1)
    ]
    return PlanExecutionRequest(
        plan=ExecutionPlan(verb_name="chat_general", steps=plan_steps),
        args={"request_id": "req-1", "extra": {"mode": "brain"}},
    )


def test_trace_id_stable_for_node_and_correlation() -> None:
    assert cortex_exec_trace_id(NODE, CORR) == f"cortex.exec:{NODE}:{CORR}"


def test_trace_lane_isolates_harness_finalize_verbs() -> None:
    from app.grammar_emit import trace_lane_for_verb

    assert trace_lane_for_verb("stance_react") == "stance_react"
    assert trace_lane_for_verb("harness_finalize_reflect") == "harness_finalize_reflect"
    assert trace_lane_for_verb("orion_voice_finalize") == "orion_voice_finalize"
    assert trace_lane_for_verb("chat_general") is None

    stance = CortexExecGrammarCollector(
        node_name=NODE,
        correlation_id=CORR,
        code_version="0.2.0",
        observed_at=FIXED_OBS,
        trace_lane="stance_react",
    )
    assert stance.trace_id == f"cortex.exec:{NODE}:{CORR}:stance_react"

    collector = CortexExecGrammarCollector(
        node_name=NODE,
        correlation_id=CORR,
        code_version="0.2.0",
        observed_at=FIXED_OBS,
        trace_lane="harness_finalize_reflect",
    )
    assert collector.trace_id == f"cortex.exec:{NODE}:{CORR}:harness_finalize_reflect"


def test_builds_valid_grammar_events_with_required_semantic_roles() -> None:
    collector = CortexExecGrammarCollector(
        node_name=NODE,
        correlation_id=CORR,
        code_version="0.2.0",
        observed_at=FIXED_OBS,
        session_id="sess-1",
        turn_id="turn-9",
    )
    req = _minimal_plan()
    collector.record_request_received(req=req, mode="brain")
    collector.record_plan_started(req=req, depth=None, step_count=2)
    collector.record_recall_gate_observed(
        run_recall=False,
        profile="assist.light.v1",
        reason="gating_disabled",
    )
    collector.record_step_started(
        order=1,
        step_name="step_1",
        verb_name="chat_general",
        services=["LLMGatewayService"],
    )
    collector.record_step_completed(
        order=1,
        step_name="step_1",
        latency_ms=120,
        result_service_keys=["LLMGatewayService"],
    )
    collector.record_step_started(
        order=2,
        step_name="step_2",
        verb_name="noop",
        services=[],
    )
    collector.record_step_completed(
        order=2,
        step_name="step_2",
        latency_ms=5,
        result_service_keys=[],
    )
    collector.record_result_assembled(
        status="success",
        final_text_present=True,
        reasoning_present=False,
        thinking_source="none",
    )
    collector.record_result_emitted(reply_present=True, status="success")

    events = build_cortex_exec_grammar_events(collector)
    assert events
    kinds = {e.event_kind for e in events}
    assert kinds <= set(get_args(GrammarEventKind))
    assert "trace_started" in kinds
    assert "trace_ended" in kinds
    roles = {e.atom.semantic_role for e in events if e.atom}
    required = {
        "exec_request_received",
        "exec_plan_started",
        "exec_recall_gate_observed",
        "exec_step_started",
        "exec_step_completed",
        "exec_result_assembled",
        "exec_result_emitted",
    }
    assert required <= roles
    assert all(e.trace_id == cortex_exec_trace_id(NODE, CORR) for e in events)
    assert events[0].session_id == "sess-1"
    assert events[0].turn_id == "turn-9"


def test_step_failure_emits_exec_step_failed() -> None:
    collector = CortexExecGrammarCollector(
        node_name=NODE,
        correlation_id=CORR,
        code_version="0.2.0",
        observed_at=FIXED_OBS,
    )
    req = _minimal_plan(steps=1)
    collector.record_request_received(req=req, mode="brain")
    collector.record_plan_started(req=req, depth=None, step_count=1)
    collector.record_recall_gate_observed(run_recall=False, profile=None, reason="skipped")
    collector.record_step_started(
        order=1, step_name="step_1", verb_name="chat_general", services=["LLMGatewayService"]
    )
    collector.record_step_failed(order=1, step_name="step_1", error_kind="timeout")
    collector.record_result_assembled(
        status="fail",
        final_text_present=False,
        reasoning_present=False,
        thinking_source="none",
    )
    collector.record_result_emitted(reply_present=True, status="fail")
    roles = {e.atom.semantic_role for e in build_cortex_exec_grammar_events(collector) if e.atom}
    assert "exec_step_failed" in roles
    assert "exec_step_completed" not in roles


def test_no_raw_prompt_or_llm_blobs_in_atoms() -> None:
    collector = CortexExecGrammarCollector(
        node_name=NODE,
        correlation_id=CORR,
        code_version="0.2.0",
        observed_at=FIXED_OBS,
    )
    req = _minimal_plan(steps=1)
    collector.record_request_received(req=req, mode="brain")
    collector.record_plan_started(req=req, depth=None, step_count=1)
    collector.record_recall_gate_observed(run_recall=False, profile=None, reason="skipped")
    collector.record_step_started(
        order=1, step_name="step_1", verb_name="chat_general", services=["LLMGatewayService"]
    )
    collector.record_step_completed(
        order=1,
        step_name="step_1",
        latency_ms=1,
        result_service_keys=["LLMGatewayService"],
    )
    collector.record_result_assembled(
        status="success",
        final_text_present=True,
        reasoning_present=True,
        thinking_source="provider_reasoning",
    )
    collector.record_result_emitted(reply_present=True, status="success")
    for event in build_cortex_exec_grammar_events(collector):
        if event.atom:
            assert event.atom.text_value is None
            summary = event.atom.summary or ""
            assert "hello user" not in summary.lower()
            assert len(summary) < 500
            assert event.atom.payload_ref
            assert "prompt" not in (event.atom.payload_ref or "")


def test_edges_use_allowed_relation_types_only() -> None:
    collector = CortexExecGrammarCollector(
        node_name=NODE,
        correlation_id=CORR,
        code_version="0.2.0",
        observed_at=FIXED_OBS,
    )
    req = _minimal_plan(steps=2)
    collector.record_request_received(req=req, mode="brain")
    collector.record_plan_started(req=req, depth=None, step_count=2)
    collector.record_recall_gate_observed(run_recall=False, profile="p", reason="r")
    for i in (1, 2):
        collector.record_step_started(
            order=i,
            step_name=f"step_{i}",
            verb_name="chat_general",
            services=["LLMGatewayService"],
        )
        collector.record_step_completed(
            order=i,
            step_name=f"step_{i}",
            latency_ms=10,
            result_service_keys=["LLMGatewayService"],
        )
    collector.record_result_assembled(
        status="success",
        final_text_present=True,
        reasoning_present=False,
        thinking_source="none",
    )
    collector.record_result_emitted(reply_present=True, status="success")
    allowed = set(get_args(RelationType))
    for event in build_cortex_exec_grammar_events(collector):
        if event.edge:
            assert event.edge.relation_type in allowed


def test_two_steps_emit_temporal_successor_edge() -> None:
    collector = CortexExecGrammarCollector(
        node_name=NODE,
        correlation_id=CORR,
        code_version="0.2.0",
        observed_at=FIXED_OBS,
    )
    req = _minimal_plan(steps=2)
    collector.record_request_received(req=req, mode="brain")
    collector.record_plan_started(req=req, depth=None, step_count=2)
    collector.record_recall_gate_observed(run_recall=False, profile="p", reason="r")
    for i in (1, 2):
        collector.record_step_started(
            order=i,
            step_name=f"step_{i}",
            verb_name="chat_general",
            services=["LLMGatewayService"],
        )
        collector.record_step_completed(
            order=i,
            step_name=f"step_{i}",
            latency_ms=10,
            result_service_keys=["LLMGatewayService"],
        )
    collector.record_result_assembled(
        status="success",
        final_text_present=True,
        reasoning_present=False,
        thinking_source="none",
    )
    collector.record_result_emitted(reply_present=True, status="success")
    relations = [
        e.edge.relation_type
        for e in build_cortex_exec_grammar_events(collector)
        if e.edge
    ]
    assert "temporal_successor" in relations


def test_atom_types_are_allowed_literals() -> None:
    collector = CortexExecGrammarCollector(
        node_name=NODE,
        correlation_id=CORR,
        code_version="0.2.0",
        observed_at=FIXED_OBS,
    )
    req = _minimal_plan(steps=1)
    collector.record_request_received(req=req, mode="brain")
    collector.record_plan_started(req=req, depth=None, step_count=1)
    collector.record_recall_gate_observed(run_recall=True, profile="assist.light.v1", reason="run")
    collector.record_step_started(
        order=1, step_name="s1", verb_name="v", services=["RecallService"]
    )
    collector.record_step_completed(
        order=1, step_name="s1", latency_ms=3, result_service_keys=["RecallService"]
    )
    collector.record_result_assembled(
        status="success",
        final_text_present=False,
        reasoning_present=False,
        thinking_source="none",
    )
    collector.record_result_emitted(reply_present=False, status="success")
    allowed = set(get_args(AtomType))
    for event in build_cortex_exec_grammar_events(collector):
        if event.atom:
            assert event.atom.atom_type in allowed


def _install_fake_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch app.grammar_emit's datetime.now() to return strictly increasing,
    deterministic timestamps -- one tick per call -- so tests can assert real
    ordering/distinctness without depending on wall-clock speed. A plain
    SimpleNamespace stand-in, not a datetime subclass: grammar_emit.py never
    constructs datetime(...) directly (only .now()), so nothing here needs
    to preserve isinstance/type identity with the real datetime class."""
    import app.grammar_emit as ge

    counter = {"n": 0}
    base = datetime(2026, 5, 24, 12, 0, 0, tzinfo=timezone.utc)

    def _fake_now(tz: object = None) -> datetime:
        counter["n"] += 1
        return base + timedelta(milliseconds=counter["n"])

    monkeypatch.setattr(ge, "datetime", SimpleNamespace(now=_fake_now))


def _record_two_step_trace(collector: CortexExecGrammarCollector) -> None:
    req = _minimal_plan(steps=2)
    collector.record_request_received(req=req, mode="brain")
    collector.record_plan_started(req=req, depth=None, step_count=2)
    collector.record_recall_gate_observed(run_recall=False, profile="p", reason="r")
    for i in (1, 2):
        collector.record_step_started(
            order=i, step_name=f"step_{i}", verb_name="chat_general", services=["LLMGatewayService"]
        )
        collector.record_step_completed(
            order=i, step_name=f"step_{i}", latency_ms=10, result_service_keys=["LLMGatewayService"]
        )
    collector.record_result_assembled(
        status="success", final_text_present=True, reasoning_present=False, thinking_source="none"
    )
    collector.record_result_emitted(reply_present=True, status="success")


def test_atoms_get_distinct_real_observed_at_not_shared_flush_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Before this fix: every atom in a trace shared one flush-time
    timestamp (confirmed live: median intra-trace duration across 35,994
    real cortex-exec traces was 0.00s). Each record_*() call must now stamp
    its own real observed_at, in recording order."""
    _install_fake_clock(monkeypatch)
    collector = CortexExecGrammarCollector(
        node_name=NODE, correlation_id=CORR, code_version="0.2.0", observed_at=FIXED_OBS,
    )
    _record_two_step_trace(collector)

    events = build_cortex_exec_grammar_events(collector)
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
    occurrence time) should change; emitted_at stays uniform, same as the
    pre-existing trace_started/trace_ended behavior."""
    _install_fake_clock(monkeypatch)
    collector = CortexExecGrammarCollector(
        node_name=NODE, correlation_id=CORR, code_version="0.2.0", observed_at=FIXED_OBS,
    )
    _record_two_step_trace(collector)

    events = build_cortex_exec_grammar_events(collector)
    atom_events = [e for e in events if e.atom is not None]
    emitted_ats = {e.emitted_at for e in atom_events}
    assert len(emitted_ats) == 1, f"expected one shared emitted_at, got {emitted_ats}"


def test_edge_observed_at_matches_target_atom_real_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_clock(monkeypatch)
    collector = CortexExecGrammarCollector(
        node_name=NODE, correlation_id=CORR, code_version="0.2.0", observed_at=FIXED_OBS,
    )
    _record_two_step_trace(collector)

    events = build_cortex_exec_grammar_events(collector)
    atom_observed_at = {e.atom.atom_id: e.observed_at for e in events if e.atom is not None}
    edge_events = [e for e in events if e.edge is not None]
    assert edge_events, "expected at least one edge in a two-step trace"
    for edge_event in edge_events:
        target_id = edge_event.edge.to_atom_id
        assert target_id in atom_observed_at
        assert edge_event.observed_at == atom_observed_at[target_id]
        # Self-consistency alone (edge == its own target atom) passes
        # trivially even if the whole per-atom fix were reverted back to
        # one shared value -- also assert against the pre-fix shared
        # constant (trace-start observed_at) directly, so a revert fails
        # this test too, not just its distinctness-focused sibling above.
        assert edge_event.observed_at != FIXED_OBS


def test_atom_missing_from_observed_at_map_falls_back_to_trace_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defensive fallback: if an atom somehow bypassed _put_atom() and has no
    captured observed_at, use the collector's trace-start observed_at rather
    than crashing or emitting None."""
    _install_fake_clock(monkeypatch)
    collector = CortexExecGrammarCollector(
        node_name=NODE, correlation_id=CORR, code_version="0.2.0", observed_at=FIXED_OBS,
    )
    req = _minimal_plan(steps=1)
    collector.record_request_received(req=req, mode="brain")
    # Simulate a bypass: drop the captured timestamp for one real atom.
    missing_atom_id = next(iter(collector._atoms.values())).atom_id
    del collector._atom_observed_at[missing_atom_id]

    events = build_cortex_exec_grammar_events(collector)
    atom_event = next(e for e in events if e.atom is not None and e.atom.atom_id == missing_atom_id)
    assert atom_event.observed_at == FIXED_OBS


def test_atom_time_range_populated_with_real_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GrammarAtomV1.time_range (orion/schemas/grammar.py) previously stayed
    None for every cortex-exec atom -- orion/grammar/ledger.py wires it
    through to persisted grammar_atoms.time_start/time_end, read by the live
    Grammar Atlas API, so this was a second place the same 'timing is fake'
    symptom survived even after observed_at was fixed on the sibling
    GrammarEventV1 envelope. _put_atom() must now stamp it too, from the
    same captured moment as _atom_observed_at."""
    _install_fake_clock(monkeypatch)
    collector = CortexExecGrammarCollector(
        node_name=NODE, correlation_id=CORR, code_version="0.2.0", observed_at=FIXED_OBS,
    )
    _record_two_step_trace(collector)

    events = build_cortex_exec_grammar_events(collector)
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
    collector = CortexExecGrammarCollector(
        node_name=NODE, correlation_id=CORR, code_version="0.2.0", observed_at=FIXED_OBS,
    )
    _record_two_step_trace(collector)

    events = build_cortex_exec_grammar_events(collector)
    trace_started = next(e for e in events if e.event_kind == "trace_started")
    trace_ended = next(e for e in events if e.event_kind == "trace_ended")
    atom_events = [e for e in events if e.atom is not None]

    assert trace_started.observed_at == FIXED_OBS
    assert trace_ended.observed_at != trace_started.observed_at
    assert trace_ended.observed_at == max(e.observed_at for e in atom_events)


def test_result_assembled_reinvoked_keeps_first_observed_at(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """record_result_assembled really can be called more than once on the
    SAME collector in production: get_or_create_collector() caches one
    collector per ctx, and record_assembled_grammar() reads that same
    cached instance from 4 separate call sites in router.py (motor/
    finalize/voice-finalize lanes) -- the exact same-collector-reinvoked-
    across-phases shape found live in the sibling HarnessGrammarCollector
    (orion/harness/grammar_emit.py, 2026-07-15). Before this fix: the
    second call would advance _atom_observed_at, so an earlier-phase
    publish and a later-phase publish would carry the SAME atom_id/
    event_id (deterministic, trace_id + fixed key) but DIFFERENT
    observed_at/time_range -- the 'same identity, contradictory timestamp'
    shape this whole fix exists to eliminate. on_conflict_do_nothing masks
    it at the DB layer, but a raw bus consumer of the event stream
    wouldn't be."""
    _install_fake_clock(monkeypatch)
    collector = CortexExecGrammarCollector(
        node_name=NODE, correlation_id=CORR, code_version="0.2.0", observed_at=FIXED_OBS,
    )
    _record_two_step_trace(collector)
    first_events = build_cortex_exec_grammar_events(collector)
    first_assembled = next(
        e for e in first_events if e.atom and e.atom.semantic_role == "exec_result_assembled"
    )
    first_ts = first_assembled.observed_at

    collector.record_result_assembled(
        status="success", final_text_present=True, reasoning_present=True,
        thinking_source="finalize_reflect",
    )

    second_events = build_cortex_exec_grammar_events(collector)
    second_assembled = next(
        e for e in second_events if e.atom and e.atom.semantic_role == "exec_result_assembled"
    )
    assert second_assembled.atom.atom_id == first_assembled.atom.atom_id
    assert second_assembled.observed_at == first_ts
    assert second_assembled.atom.time_range.start == first_ts
    # Content still refreshes on the second call -- only timing is pinned.
    assert "thinking_source=finalize_reflect" in second_assembled.atom.summary


def test_trace_ended_falls_back_to_trace_start_with_zero_atoms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A trace that fails before any record_*() call has no atoms to derive
    a real end time from -- trace_ended must fall back to trace-start
    observed_at rather than raising (max() on an empty sequence)."""
    _install_fake_clock(monkeypatch)
    collector = CortexExecGrammarCollector(
        node_name=NODE, correlation_id=CORR, code_version="0.2.0", observed_at=FIXED_OBS,
    )
    events = build_cortex_exec_grammar_events(collector)
    trace_ended = next(e for e in events if e.event_kind == "trace_ended")
    assert trace_ended.observed_at == FIXED_OBS
