from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion.schemas.execution_projection import ExecutionTrajectoryProjectionV1
from orion.schemas.grammar import GrammarAtomV1, GrammarEventV1, GrammarProvenanceV1
from orion.substrate.execution_loop.constants import EXECUTION_TRAJECTORY_PROJECTION_ID
from orion.substrate.execution_loop.grammar_extract import (
    compute_pressure_hints,
    extract_execution_state_from_events,
)
from orion.substrate.execution_loop.reducer import reduce_execution_trace_events

FIXED_TS = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)
TRACE = "cortex.exec:athena:corr-abc"


def _exec_atom(role: str, summary: str, *, event_id: str = "gev_x") -> GrammarEventV1:
    atom = GrammarAtomV1(
        atom_id=f"{TRACE}:{role}",
        trace_id=TRACE,
        atom_type="observation",
        semantic_role=role,
        layer="execution",
        summary=summary,
    )
    return GrammarEventV1(
        event_id=event_id,
        event_kind="atom_emitted",
        trace_id=TRACE,
        emitted_at=FIXED_TS,
        observed_at=FIXED_TS,
        atom=atom,
        provenance=GrammarProvenanceV1(
            source_service="orion-cortex-exec",
            source_component="cortex_exec_grammar_emit",
        ),
        correlation_id="corr-abc",
        session_id="sess-1",
        turn_id="turn-1",
    )


def _empty_projection() -> ExecutionTrajectoryProjectionV1:
    return ExecutionTrajectoryProjectionV1(
        projection_id=EXECUTION_TRAJECTORY_PROJECTION_ID,
        generated_at=FIXED_TS,
        runs={},
    )


def test_extract_builds_run_state_from_exec_atoms() -> None:
    events = [
        _exec_atom(
            "exec_plan_started",
            "Execution plan started for verb=chat_general; step_count=2; depth=none",
            event_id="gev_1",
        ),
        _exec_atom(
            "exec_step_started",
            "Step started: order=1, step=step_1, verb=chat_general, services=LLMGatewayService",
            event_id="gev_2",
        ),
        _exec_atom(
            "exec_step_completed",
            "Step completed: step=step_1, status=success, latency_ms=120, result_services=LLMGatewayService",
            event_id="gev_3",
        ),
        _exec_atom(
            "exec_result_assembled",
            "Final result assembled: status=success, final_text_present=True, reasoning_present=True, thinking_source=provider_reasoning",
            event_id="gev_4",
        ),
        _exec_atom(
            "exec_result_emitted",
            "Cortex exec result emitted to reply_to=True, status=success",
            event_id="gev_5",
        ),
    ]
    run = extract_execution_state_from_events(events, now=FIXED_TS)
    assert run.trace_id == TRACE
    assert run.node_id == "athena"
    assert run.status == "success"
    assert run.started_step_count == 1
    assert run.completed_step_count == 1
    assert run.final_text_present is True
    assert run.reasoning_present is True
    assert run.thinking_source == "provider_reasoning"
    assert "gev_5" in run.evidence_event_ids


def test_pressure_hints_reasoning_and_egress() -> None:
    run = extract_execution_state_from_events(
        [
            _exec_atom(
                "exec_result_assembled",
                "Final result assembled: status=success, final_text_present=True, reasoning_present=True, thinking_source=provider_reasoning",
            )
        ],
        now=FIXED_TS,
    )
    hints = compute_pressure_hints(run, egress_emitted=False)
    assert hints["reasoning_load"] == 0.35
    assert hints["egress_confidence"] == 0.25


def test_pressure_hints_failure_pressure() -> None:
    run = extract_execution_state_from_events(
        [_exec_atom("exec_step_failed", "Step failed: step=step_1, error_kind=timeout")],
        now=FIXED_TS,
    )
    hints = compute_pressure_hints(run, egress_emitted=True)
    assert hints["failure_pressure"] == 1.0


def test_reducer_emits_execution_run_delta() -> None:
    events = [
        _exec_atom(
            "exec_plan_started",
            "Execution plan started for verb=chat_general; step_count=1; depth=none",
        ),
        _exec_atom(
            "exec_result_emitted",
            "Cortex exec result emitted to reply_to=True, status=success",
        ),
    ]
    proj, receipt = reduce_execution_trace_events(
        events=events,
        projection=_empty_projection(),
        now=FIXED_TS,
    )
    assert receipt.accepted_event_ids
    assert len(receipt.state_deltas) == 1
    delta = receipt.state_deltas[0]
    assert delta.target_kind == "execution_run"
    assert delta.target_id == TRACE
    assert delta.after["node_id"] == "athena"
    assert "execution_load" in delta.after["pressure_hints"]
    assert TRACE in proj.runs


def test_reducer_noops_non_cortex_exec() -> None:
    bio = GrammarEventV1(
        event_id="gev_bio",
        event_kind="atom_emitted",
        trace_id="biometrics.node:atlas:ts",
        emitted_at=FIXED_TS,
        observed_at=FIXED_TS,
        atom=GrammarAtomV1(
            atom_id="a1",
            trace_id="biometrics.node:atlas:ts",
            atom_type="signal",
            semantic_role="body_state",
            layer="biometrics",
            summary="body",
        ),
        provenance=GrammarProvenanceV1(source_service="orion-biometrics", source_component="x"),
    )
    proj, receipt = reduce_execution_trace_events(
        events=[bio], projection=_empty_projection(), now=FIXED_TS
    )
    assert proj.runs == {}
    assert receipt.noop_event_ids == ["gev_bio"]


def test_stable_delta_id_on_replay() -> None:
    events = [
        _exec_atom(
            "exec_result_emitted",
            "Cortex exec result emitted to reply_to=True, status=success",
        )
    ]
    _, r1 = reduce_execution_trace_events(
        events=events, projection=_empty_projection(), now=FIXED_TS
    )
    _, r2 = reduce_execution_trace_events(
        events=events, projection=_empty_projection(), now=FIXED_TS
    )
    assert r1.receipt_id == r2.receipt_id
    assert r1.state_deltas[0].delta_id == r2.state_deltas[0].delta_id
