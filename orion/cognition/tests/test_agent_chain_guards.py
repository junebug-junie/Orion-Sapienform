"""Pass 2: hard triage cap and repeated plan_action guards."""

from __future__ import annotations

from orion.cognition.agent_chain_guards import (
    consecutive_tool_count,
    plan_action_saturated,
    repeated_plan_action_needs_delivery,
    triage_must_finalize,
)


def test_triage_finalize_when_prior_trace():
    assert triage_must_finalize(tool_id="triage", step_idx=0, prior_trace_len=1) is True


def test_triage_allowed_step0_empty_trace():
    assert triage_must_finalize(tool_id="triage", step_idx=0, prior_trace_len=0) is False


def test_triage_blocked_step_idx_positive():
    assert triage_must_finalize(tool_id="triage", step_idx=1, prior_trace_len=0) is True


def test_repeated_plan_action_detection():
    assert repeated_plan_action_needs_delivery(tool_id="plan_action", tools_called=["plan_action"]) is True
    assert repeated_plan_action_needs_delivery(tool_id="plan_action", tools_called=["analyze_text"]) is False


def test_consecutive_tool_count():
    assert consecutive_tool_count(tools_called=["triage", "plan_action", "plan_action"], candidate="plan_action") == 2
    assert consecutive_tool_count(tools_called=["plan_action", "analyze_text"], candidate="plan_action") == 0


def test_plan_action_saturation_on_third_consecutive_attempt():
    assert plan_action_saturated(tool_id="plan_action", tools_called=["plan_action", "plan_action"]) is True
    assert plan_action_saturated(tool_id="plan_action", tools_called=["plan_action"]) is False
