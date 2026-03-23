from __future__ import annotations

from orion.normalizers.agent_trace import build_agent_trace_summary
from orion.schemas.cortex.types import StepExecutionResult


def _planner_step() -> StepExecutionResult:
    return StepExecutionResult(
        status="success",
        verb_name="agent_runtime",
        step_name="planner_react",
        order=-1,
        latency_ms=120,
        result={
            "PlannerReactService": {
                "status": "ok",
                "stop_reason": "delegate",
                "trace": [
                    {
                        "step_index": 0,
                        "thought": "Need more structure.",
                        "action": {"tool_id": "plan_action", "input": {"goal": "deploy"}},
                    }
                ],
            }
        },
    )


def _recall_step() -> StepExecutionResult:
    return StepExecutionResult(
        status="success",
        verb_name="recall",
        step_name="recall",
        order=0,
        latency_ms=80,
        result={"RecallService": {"count": 2, "profile": "assist.light.v1"}},
    )


def _agent_chain_step() -> StepExecutionResult:
    return StepExecutionResult(
        status="success",
        verb_name="agent_chain",
        step_name="agent_chain",
        order=100,
        latency_ms=240,
        result={
            "AgentChainService": {
                "text": "Final answer",
                "runtime_debug": {"output_mode": "implementation_guide"},
                "planner_raw": {
                    "trace": [
                        {
                            "step_index": 0,
                            "thought": "Need analysis first",
                            "action": {"tool_id": "analyze_text", "input": {"text": "hello"}},
                            "observation": {"llm_output": "Inspected the request."},
                        },
                        {
                            "step_index": 1,
                            "thought": "Now produce the answer",
                            "action": {"tool_id": "finalize_response", "input": {"request": "hello"}},
                            "observation": {"llm_output": "Final answer"},
                        },
                    ]
                },
            }
        },
    )


def test_build_agent_trace_summary_normalizes_agent_steps_and_nested_tool_calls() -> None:
    summary = build_agent_trace_summary(
        correlation_id="corr-agent-1",
        message_id="msg-agent-1",
        mode="agent",
        status="success",
        final_text="Final answer",
        steps=[_planner_step(), _recall_step(), _agent_chain_step()],
        metadata={"trace_verb": "finalize_response"},
    )

    assert summary is not None
    assert summary.corr_id == "corr-agent-1"
    assert summary.mode == "agent"
    assert summary.status == "success"
    assert summary.step_count == 5
    assert summary.tool_call_count == 5
    assert summary.unique_tool_count == 5
    assert summary.unique_tool_families == ["planning", "recall", "reasoning", "communication", "orchestration"]
    assert summary.action_counts["retrieve"] == 1
    assert summary.action_counts["delegate"] == 1
    assert summary.action_counts["summarize"] == 1
    assert summary.effect_counts["read_only"] == 5
    assert "without write or side-effect actions" in summary.summary_text
    assert [step.tool_id for step in summary.steps] == [
        "planner_react",
        "recall",
        "agent_chain",
        "analyze_text",
        "finalize_response",
    ]
    assert summary.steps[3].event_type == "agent_delegate_tool"
    assert summary.steps[4].summary == "Final answer"


def test_build_agent_trace_summary_returns_none_for_non_agent_mode() -> None:
    summary = build_agent_trace_summary(
        correlation_id="corr-brain-1",
        message_id="msg-brain-1",
        mode="brain",
        status="success",
        final_text="Hello",
        steps=[_planner_step()],
        metadata={},
    )

    assert summary is None


def test_build_agent_trace_summary_handles_empty_agent_trace_gracefully() -> None:
    summary = build_agent_trace_summary(
        correlation_id="corr-agent-empty",
        message_id="msg-agent-empty",
        mode="agent",
        status="partial",
        final_text=None,
        steps=[],
        metadata={"route": "agent"},
    )

    assert summary is not None
    assert summary.step_count == 0
    assert summary.tool_call_count == 0
    assert summary.unique_tool_count == 0
    assert summary.tools == []
    assert summary.steps == []
    assert summary.summary_text == "Agent processed the request without write or side-effect actions."
