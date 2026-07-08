from orion.normalizers.agent_trace import build_agent_trace_summary
from orion.schemas.cortex.schemas import StepExecutionResult


def _bound_capability_step(*, payload: dict) -> StepExecutionResult:
    return StepExecutionResult(
        status="success",
        verb_name="housekeep_runtime",
        step_name="bound_capability_execution",
        order=1,
        result={"ContextExecService": payload},
        latency_ms=25,
        node="test",
        logs=[],
        error=None,
    )


def test_agent_trace_summary_surfaces_bound_failure_details():
    steps = [
        _bound_capability_step(
            payload={
                "text": "Bound capability execution failed: capability execution timed out after 25.00s",
                "bound_capability": {
                    "status": "fail",
                    "reason": "capability_executor_unavailable",
                    "path": "bound_direct_timeout",
                    "detail": "capability execution timed out after 25.00s",
                },
            }
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
    delegate = [s for s in summary.steps if s.tool_id == "bound_capability_execution"]
    assert delegate
    assert delegate[0].status == "fail"
    assert "Bound capability failed" in delegate[0].summary
    assert "timed out" in delegate[0].summary


def test_agent_trace_summary_surfaces_structured_bound_failure_details():
    steps = [
        _bound_capability_step(
            payload={
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
    delegate = [s for s in summary.steps if s.tool_id == "bound_capability_execution"]
    assert delegate
    assert delegate[0].status == "fail"
    assert "Bound capability failed" in delegate[0].summary
    assert "timed out" in delegate[0].summary


def test_agent_trace_delegate_row_fail_on_domain_negative_structured():
    steps = [
        _bound_capability_step(
            payload={
                "text": "Could not check mesh status: tailscale_not_installed",
                "structured": {
                    "finalization_reason": "bound_capability_domain_negative",
                    "bound_capability": {
                        "status": "fail",
                        "reason": "capability_executor_unavailable",
                        "detail": "Could not check mesh status: tailscale_not_installed",
                        "observation": {
                            "raw_payload_ref": {"ok": False, "domain_negative": True},
                        },
                    },
                },
                "runtime_debug": {"bound_capability_terminal_path": "bound_direct_skill_domain_negative"},
            }
        ),
    ]
    summary = build_agent_trace_summary(
        correlation_id="corr-neg",
        message_id="corr-neg",
        mode="agent",
        status="fail",
        final_text="Could not check mesh status: tailscale_not_installed",
        steps=steps,
        metadata={},
    )
    assert summary is not None
    assert summary.status == "fail"
    delegate = [s for s in summary.steps if s.tool_id == "bound_capability_execution"]
    assert delegate
    assert delegate[0].status == "fail"
    assert delegate[0].summary.startswith("Bound capability failed:")
    assert "tailscale_not_installed" in delegate[0].summary


def test_agent_trace_delegate_row_fail_when_non_service_keys_precede_context_exec_payload():
    """Regression: ``next(iter(result.keys()))`` picked ``metadata`` first and skipped delegate failure overrides."""
    ce = {
        "text": "Could not check mesh status: tailscale_not_installed",
        "structured": {
            "finalization_reason": "bound_capability_domain_negative",
            "bound_capability": {
                "status": "fail",
                "reason": "capability_executor_unavailable",
                "detail": "Could not check mesh status: tailscale_not_installed",
                "observation": {"raw_payload_ref": {"ok": False, "domain_negative": True}},
            },
        },
        "runtime_debug": {"bound_capability_terminal_path": "bound_direct_skill_domain_negative"},
    }
    steps = [
        StepExecutionResult(
            status="success",
            verb_name="assess_mesh_presence",
            step_name="bound_capability_execution",
            order=1,
            result={"metadata": {"corr": "x"}, "ContextExecService": ce},
            latency_ms=10,
            node="test",
            logs=[],
            error=None,
        ),
    ]
    summary = build_agent_trace_summary(
        correlation_id="corr-key-order",
        message_id="corr-key-order",
        mode="agent",
        status="fail",
        final_text="Could not check mesh status: tailscale_not_installed",
        steps=steps,
        metadata={},
    )
    assert summary is not None
    delegate = [s for s in summary.steps if s.tool_id == "bound_capability_execution"]
    assert delegate
    assert delegate[0].status == "fail"
    assert delegate[0].summary == (
        "Bound capability failed: Could not check mesh status: tailscale_not_installed."
    )
