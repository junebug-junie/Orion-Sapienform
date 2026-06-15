"""Agent route-bound synthesis pass tests."""

from __future__ import annotations

import json
from typing import Any

import pytest

from app.agent_synthesis import (
    SYNTHESIS_MODES,
    _synthesis_is_grounded,
    build_operator_summary,
    run_agent_synthesis,
)
from app.llm_profile_resolver import LLMProfileSelection
from app.runner import ContextExecRunner
from app.settings import ContextExecSettings
from orion.schemas.context_exec import ContextExecRequestV1


@pytest.fixture(autouse=True)
def _disable_gateway_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.settings import settings as cfg

    monkeypatch.setattr(cfg, "context_exec_llm_gateway_url", "")
    monkeypatch.setattr(cfg, "context_exec_llm_profile_fallback_enabled", False)


def _selection(route: str = "chat") -> LLMProfileSelection:
    return LLMProfileSelection(
        requested=route,
        selected=route,
        route_used=route,
    )


@pytest.mark.asyncio
async def test_synthesis_disabled_preserves_deterministic_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.settings import settings as cfg

    monkeypatch.setattr(cfg, "context_exec_agent_synthesis_enabled", False)
    artifact = {
        "claim": "Denver residency",
        "status": "unsupported",
        "likely_origin": "inferred session mention",
    }
    req = ContextExecRequestV1(text="Where did Denver belief come from?", mode="belief_provenance")
    result = await run_agent_synthesis(
        request=req,
        artifact=artifact,
        profile_selection=_selection("quick"),
        runtime_debug={"route_used": "quick"},
        bus=None,
    )
    assert result.model_synthesis_used is False
    assert result.fallback_used is False
    assert result.operator_summary is not None
    assert result.operator_summary.route_used == "quick"
    assert "Denver" in result.operator_summary.summary or "unsupported" in result.operator_summary.summary


@pytest.mark.asyncio
async def test_synthesis_enabled_calls_llm_tools_with_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.settings import settings as cfg

    monkeypatch.setattr(cfg, "context_exec_agent_synthesis_enabled", True)
    captured: dict[str, Any] = {}

    async def _fake_llm(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        captured["route"] = kwargs.get("route")
        return {
            "ok": True,
            "content": json.dumps(
                {
                    "title": "Belief provenance complete",
                    "summary": "Insufficient evidence for the Denver belief.",
                }
            ),
        }

    monkeypatch.setattr("app.agent_synthesis.llm_chat_route", _fake_llm)
    artifact = {"claim": "Denver belief", "status": "unsupported", "likely_origin": "unknown"}
    req = ContextExecRequestV1(text="Where did the Denver belief come from?", mode="belief_provenance")
    result = await run_agent_synthesis(
        request=req,
        artifact=artifact,
        profile_selection=_selection("agent"),
        runtime_debug={},
        bus=object(),
    )
    assert captured["route"] == "agent"
    assert result.model_synthesis_used is True
    assert result.operator_summary is not None
    assert result.operator_summary.model_synthesis_used is True


@pytest.mark.asyncio
async def test_synthesis_unavailable_records_fallback_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.settings import settings as cfg

    monkeypatch.setattr(cfg, "context_exec_agent_synthesis_enabled", True)

    async def _fail_llm(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"ok": False, "error": "bus_disabled"}

    monkeypatch.setattr("app.agent_synthesis.llm_chat_route", _fail_llm)
    artifact = {"claim": "Denver", "status": "unknown"}
    req = ContextExecRequestV1(text="denver", mode="belief_provenance")
    result = await run_agent_synthesis(
        request=req,
        artifact=artifact,
        profile_selection=_selection("chat"),
        runtime_debug={},
        bus=None,
    )
    assert result.model_synthesis_used is False
    assert result.fallback_used is True
    assert result.fallback_reason is not None
    assert "synthesis unavailable" in result.fallback_reason


@pytest.mark.asyncio
async def test_synthesis_rejects_ungrounded_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.settings import settings as cfg

    monkeypatch.setattr(cfg, "context_exec_agent_synthesis_enabled", True)

    async def _bad_llm(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "ok": True,
            "content": json.dumps(
                {
                    "title": "Bad",
                    "summary": "Found evidence in /secret/not/in/artifact.py",
                }
            ),
        }

    monkeypatch.setattr("app.agent_synthesis.llm_chat_route", _bad_llm)
    artifact = {"claim": "Denver", "status": "unsupported"}
    req = ContextExecRequestV1(text="denver", mode="belief_provenance")
    result = await run_agent_synthesis(
        request=req,
        artifact=artifact,
        profile_selection=_selection("chat"),
        runtime_debug={},
        bus=object(),
    )
    assert result.model_synthesis_used is False
    assert result.fallback_reason == "synthesis rejected: ungrounded"


def test_grounding_allows_artifact_paths() -> None:
    corpus = "affected paths include services/orion-hub/app/main.py"
    assert _synthesis_is_grounded("Impact on services/orion-hub/app/main.py confirmed.", corpus)


def test_operator_summary_includes_safety_fields() -> None:
    summary = build_operator_summary(
        mode="memory_correction_proposal",
        route_used="metacog",
        artifact={"proposal_id": "prop_x", "review_status": "pending_review"},
        runtime_debug={"triage_action": "promote_to_review"},
        model_synthesis_used=True,
        summary_text="Needs review.",
    )
    assert summary.agent_mode == "memory_correction_proposal"
    assert summary.route_used == "metacog"
    assert summary.proposal_id == "prop_x"
    assert summary.proposal_status == "pending_review"
    assert summary.triage_action == "promote_to_review"
    assert summary.safety.mutation_allowed is False
    assert summary.safety.mutation_performed is False
    assert summary.safety.requires_human_approval is True


@pytest.mark.asyncio
async def test_runner_includes_operator_summary_and_synthesis_debug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.settings import settings as cfg

    monkeypatch.setattr(cfg, "context_exec_agent_synthesis_enabled", False)
    monkeypatch.setattr(cfg, "context_exec_fake_organs_enabled", True)
    runner = ContextExecRunner()
    req = ContextExecRequestV1(
        text="Where did the Denver belief come from?",
        mode="belief_provenance",
        llm_profile="chat",
    )
    run = await runner.run(req)
    assert run.status == "ok"
    assert run.operator_summary is not None
    assert run.operator_summary.agent_mode == "belief_provenance"
    assert run.runtime_debug.get("route_used") == "chat"
    assert "model_synthesis_used" in run.runtime_debug
    assert run.runtime_debug["mutation_allowed"] is False


@pytest.mark.asyncio
async def test_synthesis_skipped_when_repo_impact_has_no_repo_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.settings import settings as cfg

    monkeypatch.setattr(cfg, "context_exec_agent_synthesis_enabled", True)

    async def _should_not_run(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("llm_chat_route should not run without repo grounding")

    monkeypatch.setattr("app.agent_synthesis.llm_chat_route", _should_not_run)
    artifact = {
        "status": "insufficient_grounding",
        "affected_paths": [],
        "findings": [],
        "risk": "unknown",
    }
    req = ContextExecRequestV1(
        text="What breaks if we change orion-hub repo entrypoint?",
        mode="repo_impact_analysis",
    )
    result = await run_agent_synthesis(
        request=req,
        artifact=artifact,
        profile_selection=_selection("quick"),
        runtime_debug={},
        bus=object(),
    )
    assert result.model_synthesis_used is False
    assert result.fallback_reason == "synthesis skipped: insufficient_repo_grounding"
    assert "insufficient_grounding" in (result.operator_summary.summary or "").lower()


@pytest.mark.asyncio
async def test_synthesis_rejects_repo_impact_without_path_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.settings import settings as cfg

    monkeypatch.setattr(cfg, "context_exec_agent_synthesis_enabled", True)

    async def _generic_llm(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "ok": True,
            "content": json.dumps(
                {
                    "title": "Repo impact",
                    "summary": "Changing the entrypoint may break dependencies and workflows.",
                }
            ),
        }

    monkeypatch.setattr("app.agent_synthesis.llm_chat_route", _generic_llm)
    artifact = {
        "status": "analyzed",
        "affected_paths": ["services/orion-hub/Dockerfile"],
        "breaking_surfaces": ["uvicorn scripts.main:app startup"],
        "findings": [
            {
                "claim": "services/orion-hub/Dockerfile:68 CMD uvicorn",
                "evidence_type": "repo_file",
            }
        ],
        "risk": "medium",
    }
    req = ContextExecRequestV1(
        text="What breaks if we change orion-hub repo entrypoint?",
        mode="repo_impact_analysis",
    )
    result = await run_agent_synthesis(
        request=req,
        artifact=artifact,
        profile_selection=_selection("quick"),
        runtime_debug={},
        bus=object(),
    )
    assert result.model_synthesis_used is False
    assert result.fallback_reason == "synthesis rejected: ungrounded"
    assert "Dockerfile" in (result.operator_summary.summary or "")


@pytest.mark.parametrize("mode", sorted(SYNTHESIS_MODES))
def test_synthesis_modes_are_supported(mode: str) -> None:
    assert mode in SYNTHESIS_MODES
