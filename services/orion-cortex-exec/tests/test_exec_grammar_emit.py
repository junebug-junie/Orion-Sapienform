from __future__ import annotations

from datetime import datetime, timezone
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
