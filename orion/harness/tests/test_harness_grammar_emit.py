from datetime import datetime, timezone

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
