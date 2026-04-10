from orion.normalizers.agent_trace import build_agent_trace_summary
from orion.schemas.cortex.schemas import StepExecutionResult


def test_agent_trace_summary_surfaces_bound_failure_details():
    steps = [
        StepExecutionResult(
            status="success",
            verb_name="planner",
            step_name="planner_react",
            order=0,
            result={"PlannerReactService": {"trace": [{"action": {"tool_id": "assess_runtime_state"}}]}},
            latency_ms=2,
            node="test",
            logs=[],
            error=None,
        ),
        StepExecutionResult(
            status="success",
            verb_name="agent_chain",
            step_name="agent_chain",
            order=1,
            result={
                "AgentChainService": {
                    "text": "Bound capability execution failed: capability execution timed out after 25.00s",
                    "bound_capability": {
                        "status": "fail",
                        "reason": "capability_executor_unavailable",
                        "path": "bound_direct_timeout",
                        "detail": "capability execution timed out after 25.00s",
                    },
                }
            },
            latency_ms=25,
            node="test",
            logs=[],
            error=None,
        ),
    ]

    summary = build_agent_trace_summary(
        correlation_id="corr-1",
        message_id="corr-1",
        mode="agent",
        status="fail",
        final_text="Bound capability execution failed: capability execution timed out after 25.00s",
        steps=steps,
        metadata={},
    )

    assert summary is not None
    assert summary.status == "fail"
    assert "encountered 1 failed step" in summary.summary_text
    step_summaries = [s.summary for s in summary.steps if s.tool_id == "agent_chain"]
    assert step_summaries
    assert "Bound capability execution failed" in step_summaries[0]


def test_agent_trace_summary_surfaces_structured_bound_failure_details():
    steps = [
        StepExecutionResult(
            status="success",
            verb_name="agent_chain",
            step_name="agent_chain",
            order=1,
            result={
                "AgentChainService": {
                    "text": "Bound capability execution failed: capability execution timed out after 25.00s",
                    "structured": {
                        "finalization_reason": "bound_capability_fail_closed",
                        "bound_capability": {
                            "status": "fail",
                            "reason": "capability_executor_unavailable",
                            "path": "bound_direct_timeout",
                            "detail": "capability execution timed out after 25.00s",
                        },
                    },
                }
            },
            latency_ms=25,
            node="test",
            logs=[],
            error=None,
        ),
    ]

    summary = build_agent_trace_summary(
        correlation_id="corr-structured",
        message_id="corr-structured",
        mode="agent",
        status="fail",
        final_text="Bound capability execution failed: capability execution timed out after 25.00s",
        steps=steps,
        metadata={},
    )

    assert summary is not None
    assert summary.status == "fail"
    step_summaries = [s.summary for s in summary.steps if s.tool_id == "agent_chain"]
    assert step_summaries
    assert "Bound capability execution failed" in step_summaries[0]
