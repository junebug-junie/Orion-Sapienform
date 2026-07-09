from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.schemas.execution_projection import ExecutionTrajectoryProjectionV1
from orion.schemas.grammar import GrammarAtomV1, GrammarEventV1, GrammarProvenanceV1
from orion.substrate.execution_loop.constants import EXECUTION_TRAJECTORY_PROJECTION_ID
from orion.substrate.execution_loop.grammar_extract import (
    compute_pressure_hints,
    extract_execution_state_from_events,
)
from orion.schemas.execution_projection import ExecutionRunStateV1
from orion.substrate.execution_loop.merge import merge_execution_run_state
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


def _run_with_full_egress() -> ExecutionRunStateV1:
    events = [
        _exec_atom(
            "exec_plan_started",
            "Execution plan started for verb=chat_general; step_count=2; depth=none",
            event_id="gev_full_1",
        ),
        _exec_atom(
            "exec_step_started",
            "Step started: order=1, step=step_1, verb=chat_general, services=LLMGatewayService",
            event_id="gev_full_2",
        ),
        _exec_atom(
            "exec_result_assembled",
            "Final result assembled: status=success, final_text_present=True, reasoning_present=True, thinking_source=provider_reasoning",
            event_id="gev_full_3",
        ),
        _exec_atom(
            "exec_result_emitted",
            "Cortex exec result emitted to reply_to=True, status=success",
            event_id="gev_full_4",
        ),
    ]
    return extract_execution_state_from_events(events, now=FIXED_TS)


def _partial_batch_no_egress() -> list:
    return [
        _exec_atom(
            "exec_step_started",
            "Step started: order=1, step=step_2, verb=chat_general, services=LLMGatewayService",
            event_id="gev_partial_1",
        ),
    ]


def test_merge_does_not_downgrade_egress_confidence() -> None:
    full = _run_with_full_egress()
    assert full.pressure_hints["egress_confidence"] == 1.0
    partial = extract_execution_state_from_events(_partial_batch_no_egress(), now=FIXED_TS)
    merged = merge_execution_run_state(full, partial)
    assert merged.pressure_hints["egress_confidence"] == 1.0


def test_merge_does_not_downgrade_status_or_flags() -> None:
    full = _run_with_full_egress()
    partial = extract_execution_state_from_events(_partial_batch_no_egress(), now=FIXED_TS)
    merged = merge_execution_run_state(full, partial)
    assert merged.status == "success"
    assert merged.final_text_present is True
    assert merged.reasoning_present is True


def test_merge_unions_evidence_event_ids() -> None:
    full = _run_with_full_egress()
    partial = extract_execution_state_from_events(_partial_batch_no_egress(), now=FIXED_TS)
    merged = merge_execution_run_state(full, partial)
    assert "gev_full_4" in merged.evidence_event_ids
    assert "gev_partial_1" in merged.evidence_event_ids


def test_merge_caps_evidence_event_ids_at_200() -> None:
    base = _run_with_full_egress()
    base.evidence_event_ids = [f"gev_old_{i}" for i in range(250)]
    incoming = extract_execution_state_from_events(_partial_batch_no_egress(), now=FIXED_TS)
    incoming.evidence_event_ids = [f"gev_new_{i}" for i in range(100)]
    merged = merge_execution_run_state(base, incoming)
    assert len(merged.evidence_event_ids) <= 200


def test_reducer_partial_batch_after_full_does_not_downgrade() -> None:
    full_events = [
        _exec_atom(
            "exec_plan_started",
            "Execution plan started for verb=chat_general; step_count=1; depth=none",
            event_id="gev_r1",
        ),
        _exec_atom(
            "exec_result_assembled",
            "Final result assembled: status=success, final_text_present=True, reasoning_present=True, thinking_source=provider_reasoning",
            event_id="gev_r2",
        ),
        _exec_atom(
            "exec_result_emitted",
            "Cortex exec result emitted to reply_to=True, status=success",
            event_id="gev_r3",
        ),
    ]
    proj = _empty_projection()
    proj, _ = reduce_execution_trace_events(events=full_events, projection=proj, now=FIXED_TS)
    proj, receipt2 = reduce_execution_trace_events(
        events=_partial_batch_no_egress(),
        projection=proj,
        now=FIXED_TS,
    )
    run = proj.runs[TRACE]
    assert run.pressure_hints["egress_confidence"] == 1.0
    assert run.status == "success"
    assert run.final_text_present is True
    assert run.reasoning_present is True
    delta = receipt2.state_deltas[0]
    assert delta.before["pressure_hints"]["egress_confidence"] == 1.0


def test_merge_failed_step_raises_failure_pressure() -> None:
    full = _run_with_full_egress()
    fail_events = [
        _exec_atom("exec_step_failed", "Step failed: step=step_9, error_kind=timeout", event_id="gev_fail"),
    ]
    incoming = extract_execution_state_from_events(fail_events, now=FIXED_TS)
    merged = merge_execution_run_state(full, incoming)
    assert merged.failed_step_count >= 1
    assert merged.pressure_hints["failure_pressure"] == 1.0


def _harness_atom(role: str, summary: str, *, event_id: str = "gev_h") -> GrammarEventV1:
    atom = GrammarAtomV1(
        atom_id=f"{TRACE}:{role}",
        trace_id=TRACE,
        atom_type="observation",
        semantic_role=role,
        layer="harness",
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
            source_service="orion-harness-governor",
            source_component="harness_grammar_emit",
        ),
        correlation_id="corr-abc",
    )


def test_extract_accepts_harness_governor_lifecycle() -> None:
    events = [
        _harness_atom(
            "exec_request_received",
            "Harness exec received request for verb=orion_unified, mode=orion, steps=0",
            event_id="h1",
        ),
        _harness_atom("exec_step_started", "Step started: order=1, step=fcc, verb=orion_unified, services=none", event_id="h2"),
        _harness_atom(
            "exec_result_assembled",
            "Final result assembled: status=success, final_text_present=True, reasoning_present=True, thinking_source=harness_fcc",
            event_id="h3",
        ),
        _harness_atom(
            "exec_result_emitted",
            "Harness exec result emitted to reply_to=True, status=success",
            event_id="h4",
        ),
    ]
    run = extract_execution_state_from_events(events, now=FIXED_TS)
    assert run.verb == "orion_unified"
    assert run.mode == "orion"
    assert run.started_step_count == 1
    assert run.reasoning_present is True
    assert run.thinking_source == "harness_fcc"


def test_reducer_noops_harness_fcc_step_role() -> None:
    bad = GrammarEventV1(
        event_id="noop1",
        event_kind="atom_emitted",
        trace_id=TRACE,
        emitted_at=FIXED_TS,
        atom=GrammarAtomV1(
            atom_id=f"{TRACE}:harness_fcc_step",
            trace_id=TRACE,
            atom_type="observation",
            semantic_role="harness_fcc_step",
            layer="harness",
            summary="Harness step 0: tool=none, ok",
        ),
        provenance=GrammarProvenanceV1(source_service="orion-harness-governor"),
    )
    proj, receipt = reduce_execution_trace_events(
        events=[bad], projection=_empty_projection(), now=FIXED_TS,
    )
    assert receipt.noop_event_ids == ["noop1"]
    assert proj.runs == {}


def test_reducer_noops_harness_fcc_step_in_mixed_batch() -> None:
    lifecycle = _harness_atom(
        "exec_step_started",
        "Step started: order=1, step=fcc, verb=orion_unified, services=none",
        event_id="h2",
    )
    fcc_step = GrammarEventV1(
        event_id="noop-fcc",
        event_kind="atom_emitted",
        trace_id=TRACE,
        emitted_at=FIXED_TS,
        atom=GrammarAtomV1(
            atom_id=f"{TRACE}:harness_fcc_step",
            trace_id=TRACE,
            atom_type="observation",
            semantic_role="harness_fcc_step",
            layer="harness",
            summary="Harness step 0: tool=none, ok",
        ),
        provenance=GrammarProvenanceV1(source_service="orion-harness-governor"),
    )
    proj, receipt = reduce_execution_trace_events(
        events=[lifecycle, fcc_step],
        projection=_empty_projection(),
        now=FIXED_TS,
    )
    assert receipt.noop_event_ids == ["noop-fcc"]
    assert receipt.accepted_event_ids == ["h2"]
    assert proj.runs[TRACE].started_step_count == 1


def test_stance_react_lane_does_not_merge_with_primary_motor_trace() -> None:
    intake = _harness_atom(
        "exec_request_received",
        "Harness exec received plan request for verb=orion_unified, mode=orion, steps=0",
        event_id="motor-intake",
    )
    assembled = _harness_atom(
        "exec_result_assembled",
        "Result assembled: status=ok, final_text_present=true, reasoning_present=true, thinking_source=harness_fcc",
        event_id="motor-assembled",
    )
    stance_trace = f"{TRACE}:stance_react"
    stance = GrammarEventV1(
        event_id="stance-intake",
        event_kind="atom_emitted",
        trace_id=stance_trace,
        emitted_at=FIXED_TS,
        atom=GrammarAtomV1(
            atom_id=f"{stance_trace}:exec_request_received",
            trace_id=stance_trace,
            atom_type="observation",
            semantic_role="exec_request_received",
            layer="intake",
            summary="Cortex exec received plan request for verb=stance_react, mode=brain, steps=1",
        ),
        provenance=GrammarProvenanceV1(source_service="orion-cortex-exec"),
    )
    proj, _ = reduce_execution_trace_events(
        events=[stance],
        projection=_empty_projection(),
        now=FIXED_TS,
    )
    proj, _ = reduce_execution_trace_events(
        events=[intake, assembled],
        projection=proj,
        now=FIXED_TS,
    )
    assert set(proj.runs) == {TRACE, stance_trace}
    assert proj.runs[TRACE].verb == "orion_unified"
    assert proj.runs[TRACE].reasoning_present is True
    assert proj.runs[stance_trace].verb == "stance_react"


def test_isolated_lane_trace_does_not_merge_with_primary_motor_trace() -> None:
    primary = _harness_atom(
        "exec_result_assembled",
        "Result assembled: status=ok, final_text_present=true, reasoning_present=true, thinking_source=harness_fcc",
        event_id="primary-assembled",
    )
    isolated_trace = f"{TRACE}:harness_finalize_reflect"
    isolated = GrammarEventV1(
        event_id="isolated-step",
        event_kind="atom_emitted",
        trace_id=isolated_trace,
        emitted_at=FIXED_TS,
        atom=GrammarAtomV1(
            atom_id=f"{isolated_trace}:exec_step_started",
            trace_id=isolated_trace,
            atom_type="observation",
            semantic_role="exec_step_started",
            layer="execution",
            summary="Step started: order=1, step=reflect, verb=harness_finalize_reflect, services=none",
        ),
        provenance=GrammarProvenanceV1(source_service="orion-cortex-exec"),
    )
    proj, receipt_primary = reduce_execution_trace_events(
        events=[primary],
        projection=_empty_projection(),
        now=FIXED_TS,
    )
    proj, receipt_isolated = reduce_execution_trace_events(
        events=[isolated],
        projection=proj,
        now=FIXED_TS,
    )
    assert receipt_primary.accepted_event_ids == ["primary-assembled"]
    assert receipt_isolated.accepted_event_ids == ["isolated-step"]
    assert set(proj.runs) == {TRACE, isolated_trace}
    assert proj.runs[TRACE].reasoning_present is True
    assert proj.runs[isolated_trace].started_step_count == 1
    assert proj.runs[isolated_trace].reasoning_present is False


def _event_for_trace(
    trace_id: str,
    role: str,
    summary: str,
    *,
    event_id: str,
    ts: datetime,
) -> GrammarEventV1:
    atom = GrammarAtomV1(
        atom_id=f"{trace_id}:{role}",
        trace_id=trace_id,
        atom_type="observation",
        semantic_role=role,
        layer="execution",
        summary=summary,
    )
    return GrammarEventV1(
        event_id=event_id,
        event_kind="atom_emitted",
        trace_id=trace_id,
        emitted_at=ts,
        observed_at=ts,
        atom=atom,
        provenance=GrammarProvenanceV1(
            source_service="orion-cortex-exec",
            source_component="cortex_exec_grammar_emit",
        ),
        correlation_id=trace_id.split(":")[-1],
    )


def test_reducer_evicts_oldest_runs_when_max_runs_exceeded() -> None:
    proj = _empty_projection()
    trace_ids = [f"cortex.exec:athena:corr-{i}" for i in range(5)]
    for i, trace_id in enumerate(trace_ids):
        ts = FIXED_TS + timedelta(seconds=i)
        event = _event_for_trace(
            trace_id,
            "exec_result_emitted",
            "Cortex exec result emitted to reply_to=True, status=success",
            event_id=f"gev_evict_{i}",
            ts=ts,
        )
        proj, _ = reduce_execution_trace_events(
            events=[event],
            projection=proj,
            now=ts,
            max_runs=3,
        )
        assert len(proj.runs) <= 3

    # Only the 3 most-recently-touched trace_ids should survive; oldest first evicted.
    assert set(proj.runs) == set(trace_ids[-3:])


def test_reducer_never_evicts_run_just_written_even_on_last_updated_at_tie() -> None:
    # Regression for the hard requirement: "the just-written run must never be
    # evicted by this pass in the same tick it was written." pipeline.py shares
    # a single `clock` across every trace_id in a batch, so two distinct runs
    # can legitimately end up with an identical last_updated_at. A naive
    # stable-sort-by-timestamp eviction would then be free to evict whichever
    # tied run happens to sit earlier in dict iteration order -- which, for an
    # existing (updated) run, is its ORIGINAL insertion position, not "now".
    proj = _empty_projection()
    trace_a = "cortex.exec:athena:corr-a"
    trace_b = "cortex.exec:athena:corr-b"

    # trace_a is created first (occupies the earlier dict position).
    proj, _ = reduce_execution_trace_events(
        events=[
            _event_for_trace(
                trace_a,
                "exec_result_emitted",
                "Cortex exec result emitted to reply_to=True, status=success",
                event_id="gev_tie_a1",
                ts=FIXED_TS,
            )
        ],
        projection=proj,
        now=FIXED_TS,
        max_runs=2,
    )
    assert set(proj.runs) == {trace_a}

    # trace_b is created in the *same tick* (same clock/last_updated_at as the
    # update trace_a is about to receive below), landing at a later dict
    # position than trace_a.
    proj, _ = reduce_execution_trace_events(
        events=[
            _event_for_trace(
                trace_b,
                "exec_result_emitted",
                "Cortex exec result emitted to reply_to=True, status=success",
                event_id="gev_tie_b1",
                ts=FIXED_TS,
            )
        ],
        projection=proj,
        now=FIXED_TS,
        max_runs=2,
    )
    assert set(proj.runs) == {trace_a, trace_b}

    # Now re-touch trace_a (an UPDATE, so it keeps its original, earlier dict
    # position) at the exact same clock as trace_b's last_updated_at, with
    # max_runs=1 forcing exactly one eviction. trace_a is "just written" in
    # this call, so it must survive regardless of the tie with trace_b.
    proj, _ = reduce_execution_trace_events(
        events=[
            _event_for_trace(
                trace_a,
                "exec_step_started",
                "Step started: order=1, step=step_1, verb=chat_general, services=LLMGatewayService",
                event_id="gev_tie_a2",
                ts=FIXED_TS,
            )
        ],
        projection=proj,
        now=FIXED_TS,
        max_runs=1,
    )
    assert set(proj.runs) == {trace_a}, (
        "the just-written run (trace_a) must never be evicted in the same "
        "call it was written, even when tied on last_updated_at with another "
        "run from the same batch"
    )


def test_reducer_evicts_runs_older_than_max_age_sec() -> None:
    proj = _empty_projection()
    old_trace = "cortex.exec:athena:corr-old"
    new_trace = "cortex.exec:athena:corr-new"

    old_ts = FIXED_TS
    old_event = _event_for_trace(
        old_trace,
        "exec_result_emitted",
        "Cortex exec result emitted to reply_to=True, status=success",
        event_id="gev_age_old",
        ts=old_ts,
    )
    proj, _ = reduce_execution_trace_events(
        events=[old_event],
        projection=proj,
        now=old_ts,
        max_runs=None,
        max_age_sec=None,
    )
    assert old_trace in proj.runs

    # Advance the clock well past max_age_sec and write a second, unrelated run.
    new_ts = old_ts + timedelta(seconds=3600)
    new_event = _event_for_trace(
        new_trace,
        "exec_result_emitted",
        "Cortex exec result emitted to reply_to=True, status=success",
        event_id="gev_age_new",
        ts=new_ts,
    )
    proj, _ = reduce_execution_trace_events(
        events=[new_event],
        projection=proj,
        now=new_ts,
        max_runs=None,
        max_age_sec=60,
    )
    assert old_trace not in proj.runs
    assert new_trace in proj.runs


def test_reducer_default_caps_do_not_evict_single_trace() -> None:
    # Regression: calling without max_runs/max_age_sec must behave exactly as
    # before (single trace, well within EXECUTION_TRAJECTORY_MAX_RUNS /
    # EXECUTION_TRAJECTORY_MAX_AGE_SEC defaults) -- no eviction.
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
    proj, _ = reduce_execution_trace_events(
        events=events,
        projection=_empty_projection(),
        now=FIXED_TS,
    )
    assert TRACE in proj.runs
    assert len(proj.runs) == 1


def test_reducer_eviction_does_not_run_on_early_return_paths() -> None:
    # Empty events list: untouched projection returned as-is, no eviction logic runs.
    proj = _empty_projection()
    proj.runs["preexisting"] = ExecutionRunStateV1(
        trace_id="preexisting",
        correlation_id="corr-pre",
        node_id="athena",
        last_updated_at=FIXED_TS,
    )
    result_proj, receipt = reduce_execution_trace_events(
        events=[],
        projection=proj,
        now=FIXED_TS,
        max_runs=0,
        max_age_sec=0,
    )
    assert result_proj is proj
    assert "preexisting" in result_proj.runs
    assert receipt.accepted_event_ids == []

    # Non-execution trace_id noop path also leaves the projection untouched.
    bio = GrammarEventV1(
        event_id="gev_bio_noop",
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
    result_proj2, _ = reduce_execution_trace_events(
        events=[bio],
        projection=proj,
        now=FIXED_TS,
        max_runs=0,
        max_age_sec=0,
    )
    assert result_proj2 is proj
    assert "preexisting" in result_proj2.runs
