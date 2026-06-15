"""Tests for grounding preflight and answer-success evaluation."""

from __future__ import annotations

from app.grounding_eval import (
    evaluate_investigation_outcome,
    explicit_fake_run_requested,
    grounding_required,
    is_placeholder_investigation_summary,
)
from orion.schemas.context_exec import ContextExecVerbStepV1


def test_grounding_required_for_recall_questions() -> None:
    assert grounding_required("do you recall where my mom lives?", mode="general_investigation") is True
    assert grounding_required("hello", mode="general_investigation") is False


def test_explicit_fake_run_requested() -> None:
    assert explicit_fake_run_requested("run a smoke test please") is True
    assert explicit_fake_run_requested("do you recall Denver?") is False
    assert explicit_fake_run_requested("question", scopes={"smoke_run": True}) is True


def test_placeholder_summary_detection() -> None:
    assert is_placeholder_investigation_summary("Investigation complete for: where is mom?")
    assert not is_placeholder_investigation_summary("Denver evidence found in memory.")


def test_fake_engine_blocks_answer_success() -> None:
    verb_trace = [
        ContextExecVerbStepV1(
            step_index=0,
            verb="synthesize",
            callable="rlm_engine.run",
            status="ok",
        )
    ]
    runtime_debug = {
        "engine_selected": "fake",
        "model_synthesis_used": False,
        "subcalls": 0,
        "real_recall_enabled": True,
        "real_trace_enabled": True,
        "llm_profile_selected": "quick",
    }
    outcome = evaluate_investigation_outcome(
        runtime_status="ok",
        text="do you recall where my mom lives?",
        mode="general_investigation",
        artifact={"summary": "Investigation complete for: do you recall where my mom lives?"},
        runtime_debug=runtime_debug,
        verb_trace=verb_trace,
        model_synthesis_used=False,
        current_summary="Investigation complete for: do you recall where my mom lives?",
    )
    assert outcome["runtime_status"] == "ok"
    assert outcome["answer_status"] == "failed_fake_engine_selected"
    assert outcome["grounding_status"] == "skipped"
    assert outcome["synthesis_status"] == "skipped"
    assert outcome["evidence_count"] == 0
    assert "not a grounded investigation" in outcome["summary_text"]
    assert not is_placeholder_investigation_summary(outcome["summary_text"])


def test_grounded_path_with_evidence() -> None:
    verb_trace = [
        ContextExecVerbStepV1(
            step_index=0,
            verb="recall",
            callable="recall.query",
            status="ok",
        )
    ]
    runtime_debug = {
        "engine_selected": "alexzhang",
        "model_synthesis_used": True,
        "grounding_attempts": {"recall": True, "trace": False, "repo": False},
    }
    artifact = {
        "findings": [
            {
                "claim": "Mom lives in Denver",
                "evidence_type": "user_statement",
                "verified": True,
                "confidence": 0.9,
                "scope": "fact",
            }
        ]
    }
    outcome = evaluate_investigation_outcome(
        runtime_status="ok",
        text="do you recall where my mom lives?",
        mode="general_investigation",
        artifact=artifact,
        runtime_debug=runtime_debug,
        verb_trace=verb_trace,
        model_synthesis_used=True,
        current_summary="Mom lives in Denver based on prior session memory.",
    )
    assert outcome["answer_status"] == "answered_grounded"
    assert outcome["grounding_status"] == "attempted"
    assert outcome["synthesis_status"] == "completed"
    assert outcome["evidence_count"] == 1


def test_failed_grounding_preflight_when_no_sources_attempted() -> None:
    outcome = evaluate_investigation_outcome(
        runtime_status="ok",
        text="do you recall where my mom lives?",
        mode="general_investigation",
        artifact={},
        runtime_debug={"engine_selected": "alexzhang", "model_synthesis_used": False},
        verb_trace=[],
        model_synthesis_used=False,
        current_summary="Investigation complete for: do you recall where my mom lives?",
    )
    assert outcome["answer_status"] == "failed_grounding_preflight"
    assert outcome["evidence_count"] == 0
    assert "grounding failed" in outcome["summary_text"]
