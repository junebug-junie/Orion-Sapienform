"""Tests for grounding preflight and answer-success evaluation."""

from __future__ import annotations

from app.grounding_eval import (
    FAKE_ENGINE_BLOCKED_SUMMARY,
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
    assert outcome["synthesis_status"] == "blocked"
    assert outcome["evidence_count"] == 0
    assert outcome["summary_text"] == FAKE_ENGINE_BLOCKED_SUMMARY
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
        "model_synthesis_used": False,
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
        organ_cache={"traces": [{"handle": "t1"}]},
        model_synthesis_used=True,
        current_summary="Mom lives in Denver based on prior session memory.",
    )
    assert outcome["answer_status"] == "answered_grounded"
    assert outcome["grounding_status"] == "attempted"
    assert outcome["synthesis_status"] == "completed"
    assert outcome["evidence_count"] == 1


def test_general_investigation_without_synthesis_not_success() -> None:
    verb_trace = [
        ContextExecVerbStepV1(
            step_index=0,
            verb="recall",
            callable="recall.query",
            status="ok",
        )
    ]
    artifact = {
        "summary": "Investigation complete for: do you recall where my mom lives?",
        "findings": [
            {
                "claim": "Mom lives in Denver",
                "evidence_type": "user_statement",
                "verified": True,
                "confidence": 0.9,
                "scope": "fact",
            }
        ],
    }
    outcome = evaluate_investigation_outcome(
        runtime_status="ok",
        text="do you recall where my mom lives?",
        mode="general_investigation",
        artifact=artifact,
        runtime_debug={
            "engine_selected": "alexzhang",
            "grounding_attempts": {"recall": True, "trace": False, "repo": False},
        },
        verb_trace=verb_trace,
        model_synthesis_used=False,
        current_summary="Investigation complete for: do you recall where my mom lives?",
    )
    assert outcome["answer_status"] == "no_reliable_evidence"
    assert outcome["synthesis_status"] == "skipped"
    assert not is_placeholder_investigation_summary(outcome["summary_text"]) or (
        "Investigation complete" not in outcome["summary_text"]
    )


def test_runtime_failure_never_answered_grounded() -> None:
    outcome = evaluate_investigation_outcome(
        runtime_status="error",
        text="do you recall where my mom lives?",
        mode="general_investigation",
        artifact={},
        runtime_debug={
            "engine_selected": "alexzhang",
            "model_synthesis_used": True,
            "grounding_attempts": {"recall": True, "trace": True, "repo": False},
        },
        verb_trace=[],
        organ_cache={"recall": {"hits": []}, "traces": [{"handle": "t1"}]},
        model_synthesis_used=True,
        current_summary="Context-exec investigation could not complete (error).",
    )
    assert outcome["runtime_status"] == "failed"
    assert outcome["answer_status"] == "failed"
    assert outcome["answer_status"] != "answered_grounded"


def test_general_investigation_synthesis_without_artifact_findings_not_success() -> None:
    outcome = evaluate_investigation_outcome(
        runtime_status="ok",
        text="do you recall where my mom lives?",
        mode="general_investigation",
        artifact={"summary": "No recall or trace evidence found for this inquiry.", "findings": []},
        runtime_debug={
            "engine_selected": "alexzhang",
            "grounding_attempts": {"recall": True, "trace": True, "repo": False},
        },
        verb_trace=[],
        organ_cache={"recall": {"hits": []}, "traces": [{"handle": "self-trace"}]},
        model_synthesis_used=True,
        current_summary="No recall or trace evidence was found to answer the question.",
    )
    assert outcome["answer_status"] == "no_reliable_evidence"
    assert outcome["answer_status"] != "answered_grounded"


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
    assert "no real investigation occurred" in outcome["summary_text"]


def test_repo_impact_recall_hits_do_not_count_as_answer_success() -> None:
    artifact = {
        "status": "insufficient_grounding",
        "affected_paths": [],
        "findings": [],
        "risk": "unknown",
    }
    outcome = evaluate_investigation_outcome(
        runtime_status="ok",
        text="What breaks if we change orion-hub repo entrypoint?",
        mode="repo_impact_analysis",
        artifact=artifact,
        runtime_debug={
            "engine_selected": "alexzhang",
            "grounding_attempts": {"recall": True, "trace": True, "repo": False},
        },
        verb_trace=[],
        organ_cache={
            "recall": {"hits": [{"snippet": "unrelated"}] * 8},
            "traces": [{"handle": "t1"}],
        },
        model_synthesis_used=True,
        current_summary="Changing the orion-hub repo entrypoint may break dependencies.",
    )
    assert outcome["answer_status"] == "no_reliable_evidence"
    assert outcome["evidence_count"] == 0
    assert "insufficient_grounding" in outcome["summary_text"]


def test_repo_impact_with_repo_paths_answered_without_synthesis() -> None:
    artifact = {
        "status": "analyzed",
        "affected_paths": ["services/orion-hub/Dockerfile"],
        "findings": [
            {
                "claim": "services/orion-hub/Dockerfile:68 CMD uvicorn",
                "evidence_type": "repo_file",
            }
        ],
        "breaking_surfaces": ["uvicorn scripts.main:app startup"],
        "risk": "medium",
    }
    outcome = evaluate_investigation_outcome(
        runtime_status="ok",
        text="What breaks if we change orion-hub repo entrypoint?",
        mode="repo_impact_analysis",
        artifact=artifact,
        runtime_debug={
            "engine_selected": "alexzhang",
            "grounding_attempts": {"recall": True, "trace": True, "repo": True},
        },
        verb_trace=[],
        organ_cache={"repo_hits": [{"path": "services/orion-hub/Dockerfile"}]},
        model_synthesis_used=False,
        current_summary="Repo impact: analyzed. Risk=medium. Grounded files: Dockerfile.",
    )
    assert outcome["answer_status"] == "answered_grounded"
    assert outcome["evidence_count"] == 1
