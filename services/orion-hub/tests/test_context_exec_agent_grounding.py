"""Hub bridge tests for context-exec agent grounding preflight."""

from __future__ import annotations

import os
import sys
from pathlib import Path

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(REPO_ROOT), str(HUB_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

for key, value in {
    "CHANNEL_VOICE_TRANSCRIPT": "orion:voice:transcript",
    "CHANNEL_VOICE_LLM": "orion:voice:llm",
    "CHANNEL_VOICE_TTS": "orion:voice:tts",
    "CHANNEL_COLLAPSE_INTAKE": "orion:collapse:intake",
    "CHANNEL_COLLAPSE_TRIAGE": "orion:collapse:triage",
}.items():
    os.environ.setdefault(key, value)

from scripts.context_exec_agent_bridge import (
    FAKE_ENGINE_BLOCKED_SUMMARY,
    agent_answer_headline,
    build_agent_trace_summary,
    format_agent_operator_inline,
)
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
        final_text=FAKE_ENGINE_BLOCKED_SUMMARY,
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
            "organ_status": {
                "recall": {
                    "enabled": True,
                    "attempted": True,
                    "ok": False,
                    "hit_count": 0,
                    "error": "Timeout reading from 100.92.216.81:6379",
                },
                "trace": {
                    "enabled": True,
                    "attempted": True,
                    "ok": True,
                    "hit_count": 0,
                    "error": None,
                },
                "repo": {
                    "enabled": False,
                    "attempted": False,
                    "ok": False,
                    "hit_count": 0,
                    "error": None,
                },
            },
            "answer_evaluation": {
                "runtime_status": "ok",
                "answer_status": "failed_fake_engine_selected",
                "grounding_status": "skipped",
                "synthesis_status": "blocked",
                "evidence_count": 0,
                "grounding_required": True,
                "summary_text": FAKE_ENGINE_BLOCKED_SUMMARY,
            },
        },
        operator_summary=ContextExecOperatorSummaryV1(
            title="Runtime completed (not grounded)",
            summary=FAKE_ENGINE_BLOCKED_SUMMARY,
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
    assert summary["raw"]["organ_status"]["recall"]["attempted"] is True
    assert summary["raw"]["organ_status"]["recall"]["ok"] is False
    assert "Investigation complete for:" not in summary["summary_text"]
    assert summary["summary_text"] == FAKE_ENGINE_BLOCKED_SUMMARY


def test_format_agent_operator_inline_does_not_claim_success() -> None:
    inline = format_agent_operator_inline(_fake_engine_run(), llm_profile="quick")
    assert agent_answer_headline("failed_fake_engine_selected") in inline
    assert "Blocked: fake engine selected" in inline
    assert "Agent run complete" not in inline
    assert "Agent investigation complete" not in inline
    assert "failed_fake_engine_selected" in inline
    assert "Recall failed: Redis timeout" in inline


def test_agent_answer_headline_grounded_vs_blocked() -> None:
    assert agent_answer_headline("answered_grounded") == "Agent investigation complete"
    assert agent_answer_headline("failed_fake_engine_selected") == "Blocked: fake engine selected"
    assert agent_answer_headline("failed_grounding_preflight") == (
        "Runtime completed, but no real investigation occurred"
    )


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


def test_agent_trace_summary_propagates_semantic_tool_callable() -> None:
    run = ContextExecRunV1(
        run_id="run-agent-repl",
        status="ok",
        mode="agent_repl",
        text="find runner.py",
        final_text="Found runner.py in context-exec.",
        verb_trace=[
            ContextExecVerbStepV1(
                step_index=0,
                verb="agent_step",
                callable="repo_find_files",
                input_summary='repo_find_files("*.py")',
                output_summary="runner.py",
                status="ok",
                duration_ms=120,
            )
        ],
        runtime_debug={
            "agent_repl": True,
            "agent_repl_tool_counts": {"repo_find_files": 1},
            "agent_repl_semantic_tool_detected": True,
        },
        operator_summary=ContextExecOperatorSummaryV1(
            title="Agent reasoning loop",
            summary="Found runner.py in context-exec.",
            agent_mode="agent_repl",
            route_used="agent",
            model_synthesis_used=False,
            safety=ContextExecSafetySummaryV1(),
        ),
    )
    summary = build_agent_trace_summary(run=run, correlation_id="corr-repl")
    assert summary["steps"][0]["tool_id"] == "repo_find_files"
    assert summary["tool_call_count"] == 1
    assert summary["tools"][0]["tool_id"] == "repo_find_files"
    assert summary["tools"][0]["count"] == 1
