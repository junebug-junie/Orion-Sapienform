"""Hub bridge tests for context-exec agent grounding preflight."""

from __future__ import annotations

from scripts.context_exec_agent_bridge import build_agent_trace_summary, format_agent_operator_inline
from orion.schemas.context_exec import (
    ContextExecOperatorSummaryV1,
    ContextExecRunV1,
    ContextExecSafetySummaryV1,
    ContextExecVerbStepV1,
)


def _fake_engine_run() -> ContextExecRunV1:
    return ContextExecRunV1(
        run_id="run-fake",
        status="ok",
        mode="general_investigation",
        text="do you recall where my mom lives?",
        final_text=(
            "Runtime completed, but this was not a grounded investigation. "
            "The selected engine was fake and no real recall or synthesis was performed."
        ),
        artifact_type="GenericInvestigationV1",
        artifact={"summary": "Investigation complete for: do you recall where my mom lives?"},
        verb_trace=[
            ContextExecVerbStepV1(
                step_index=0,
                verb="synthesize",
                callable="rlm_engine.run",
                status="ok",
            )
        ],
        runtime_debug={
            "engine_selected": "fake",
            "model_synthesis_used": False,
            "subcalls": 0,
            "real_recall_enabled": True,
            "real_trace_enabled": True,
            "llm_profile_selected": "quick",
            "answer_evaluation": {
                "runtime_status": "ok",
                "answer_status": "failed_fake_engine_selected",
                "grounding_status": "skipped",
                "synthesis_status": "skipped",
                "evidence_count": 0,
                "grounding_required": True,
                "summary_text": (
                    "Runtime completed, but this was not a grounded investigation. "
                    "The selected engine was fake and no real recall or synthesis was performed."
                ),
            },
        },
        operator_summary=ContextExecOperatorSummaryV1(
            title="Runtime completed (not grounded)",
            summary=(
                "Runtime completed, but this was not a grounded investigation. "
                "The selected engine was fake and no real recall or synthesis was performed."
            ),
            agent_mode="general_investigation",
            route_used="quick",
            model_synthesis_used=False,
            safety=ContextExecSafetySummaryV1(),
        ),
    )


def test_agent_trace_summary_fake_engine_failure_shape() -> None:
    run = _fake_engine_run()
    summary = build_agent_trace_summary(run=run, correlation_id="corr-fake")
    assert summary["status"] == "failed_fake_engine_selected"
    assert summary["tool_call_count"] == 0
    assert summary["unique_tool_count"] == 0
    assert summary["tools"] == []
    assert summary["raw"]["runtime_step_count"] == 1
    assert summary["raw"]["answer_status"] == "failed_fake_engine_selected"
    assert "Investigation complete for:" not in summary["summary_text"]


def test_format_agent_operator_inline_does_not_claim_success() -> None:
    inline = format_agent_operator_inline(_fake_engine_run(), llm_profile="quick")
    assert "Runtime completed (not a grounded investigation)" in inline
    assert "Agent run complete" not in inline
    assert "failed_fake_engine_selected" in inline


def test_agent_trace_summary_grounded_success_shape() -> None:
    run = ContextExecRunV1(
        run_id="run-grounded",
        status="ok",
        mode="general_investigation",
        text="do you recall where my mom lives?",
        final_text="Mom lives in Denver based on prior session memory.",
        verb_trace=[
            ContextExecVerbStepV1(
                step_index=0,
                verb="recall",
                callable="recall.query",
                status="ok",
            )
        ],
        runtime_debug={
            "engine_selected": "alexzhang",
            "model_synthesis_used": True,
            "answer_evaluation": {
                "runtime_status": "ok",
                "answer_status": "answered_grounded",
                "grounding_status": "attempted",
                "synthesis_status": "completed",
                "evidence_count": 1,
                "grounding_required": True,
                "summary_text": "Mom lives in Denver based on prior session memory.",
            },
        },
        operator_summary=ContextExecOperatorSummaryV1(
            title="Agent investigation complete",
            summary="Mom lives in Denver based on prior session memory.",
            agent_mode="general_investigation",
            route_used="agent",
            model_synthesis_used=True,
            safety=ContextExecSafetySummaryV1(),
        ),
    )
    summary = build_agent_trace_summary(run=run, correlation_id="corr-grounded")
    assert summary["status"] == "ok"
    assert summary["tool_call_count"] == 1
    assert summary["unique_tool_count"] == 1
    assert len(summary["tools"]) == 1
    assert summary["raw"]["answer_status"] == "answered_grounded"
