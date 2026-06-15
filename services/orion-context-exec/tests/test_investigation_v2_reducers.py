"""PR3 tests: reducers, composer, synthesis fallback, Hub rendering."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

CTX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]


def _ctx_app_modules():
    for key in list(sys.modules):
        if key == "app" or key.startswith("app."):
            del sys.modules[key]
    for p in (str(REPO_ROOT), str(CTX_ROOT)):
        try:
            sys.path.remove(p)
        except ValueError:
            pass
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(CTX_ROOT))


def _agent_request(**overrides):
    from orion.schemas.context_exec import ContextExecRequestV1, context_exec_permissions_for_llm_profile

    base = {
        "text": "what would happen if we changed the cortex-exec runtime?",
        "mode": "investigation_v2",
        "permissions": context_exec_permissions_for_llm_profile("agent"),
        "llm_profile": "agent",
    }
    base.update(overrides)
    return ContextExecRequestV1(**base)


def _finding(path: str, *, snippet: str = "runtime") -> dict:
    return {
        "claim": f"{path}:10 {snippet}",
        "evidence_type": "repo_file",
        "source_ref": f"repo:{path}",
        "verified": True,
        "confidence": 0.85,
        "scope": "fact",
    }


def test_composite_report_mixed_evidence() -> None:
    from app.investigation_v2_reducers import compose_investigation_report
    from orion.schemas.context_exec import EvidenceBundle, SourceResult, SourceStatus

    bundle = EvidenceBundle(
        repo=SourceResult(
            source="repo",
            status=SourceStatus.hit,
            summary="1 hit",
            findings=[_finding("services/orion-context-exec/app/settings.py")],
        ),
        traces=SourceResult(source="traces", status=SourceStatus.no_hit, summary="no traces"),
        recall=SourceResult(source="recall", status=SourceStatus.unavailable, error="timeout"),
        memory=SourceResult(source="memory", status=SourceStatus.no_hit, summary="no memory"),
        health=SourceResult(source="health", status=SourceStatus.no_hit, summary="shallow"),
    )
    report = compose_investigation_report(bundle, _agent_request())

    assert report.answer_status in {"partial_grounding", "answered_grounded"}
    assert report.answer_status != "no_reliable_evidence"
    assert "repo" in report.sections
    assert report.sections["repo"].status == "hit"
    assert report.sections["recall"].status == "unavailable"
    assert "recall" in report.unavailable_sources
    assert "settings.py" in (report.sections["repo"].summary or "")


@pytest.mark.asyncio
async def test_synthesis_failure_preserves_deterministic_report(monkeypatch: pytest.MonkeyPatch) -> None:
    _ctx_app_modules()
    from app.investigation_v2_reducers import apply_synthesis_to_report, compose_investigation_report
    from app.runner import ContextExecRunner
    from app.settings import settings
    from orion.schemas.context_exec import EvidenceBundle, InvestigationReportV2, SourceResult, SourceStatus

    async def _fail_llm(*_args, **_kwargs):
        return {"ok": False, "error": "timeout"}

    monkeypatch.setattr(settings, "context_exec_real_repo_enabled", False)
    monkeypatch.setattr(settings, "context_exec_real_recall_enabled", False)
    monkeypatch.setattr(settings, "context_exec_real_trace_enabled", False)
    monkeypatch.setattr(settings, "context_exec_agent_synthesis_enabled", True)
    monkeypatch.setattr("app.agent_synthesis.llm_chat_route", _fail_llm)

    bundle = EvidenceBundle(
        repo=SourceResult(
            source="repo",
            status=SourceStatus.hit,
            findings=[_finding("services/orion-context-exec/app/runner.py")],
        ),
        recall=SourceResult(source="recall", status=SourceStatus.unavailable, error="timeout"),
    )
    base_report = compose_investigation_report(bundle, _agent_request())
    updated = apply_synthesis_to_report(base_report, synthesis_summary=None, synthesis_failed=True)
    assert any("LLM synthesis unavailable" in lim for lim in updated.limitations)
    assert updated.sections["repo"].status == "hit"

    runner = ContextExecRunner()
    run = await runner.run(_agent_request())
    assert run.status == "ok"
    assert run.artifact.get("sections")
    assert run.runtime_debug.get("synthesis_fallback_used") is True
    limitations = run.artifact.get("limitations") or []
    assert any("LLM synthesis unavailable" in str(item) for item in limitations)


def test_repo_reducer_includes_paths_and_config_anchors() -> None:
    from app.investigation_v2_reducers import reduce_repo_section
    from orion.schemas.context_exec import SourceResult, SourceStatus

    result = SourceResult(
        source="repo",
        status=SourceStatus.hit,
        findings=[
            _finding("services/orion-context-exec/app/settings.py"),
            _finding("services/orion-context-exec/.env_example"),
            _finding("services/orion-context-exec/tests/test_runner.py"),
        ],
        metadata={"hit_count": 3},
    )
    section = reduce_repo_section(result, request_text="change cortex-exec runtime")
    assert section.status == "hit"
    assert "settings.py" in (section.summary or "")
    assert section.metadata.get("affected_paths")
    assert section.metadata.get("config_anchors")
    assert section.metadata.get("tests_likely_affected")


def test_no_evidence_all_no_hit() -> None:
    from app.investigation_v2_reducers import compose_investigation_report
    from orion.schemas.context_exec import EvidenceBundle, SourceResult, SourceStatus

    bundle = EvidenceBundle(
        repo=SourceResult(source="repo", status=SourceStatus.no_hit),
        traces=SourceResult(source="traces", status=SourceStatus.no_hit),
        recall=SourceResult(source="recall", status=SourceStatus.no_hit),
        memory=SourceResult(source="memory", status=SourceStatus.no_hit),
        health=SourceResult(source="health", status=SourceStatus.no_hit),
    )
    report = compose_investigation_report(bundle, _agent_request())
    assert report.answer_status == "no_reliable_evidence"
    assert report.sections["repo"].status == "no_hit"
    assert report.sections["traces"].status == "no_hit"


def test_dependency_unavailable_no_hits() -> None:
    from app.investigation_v2_reducers import compose_investigation_report
    from orion.schemas.context_exec import EvidenceBundle, SourceResult, SourceStatus

    bundle = EvidenceBundle(
        repo=SourceResult(source="repo", status=SourceStatus.no_hit),
        traces=SourceResult(source="traces", status=SourceStatus.unavailable, error="bus down"),
        recall=SourceResult(source="recall", status=SourceStatus.unavailable, error="timeout"),
        memory=SourceResult(source="memory", status=SourceStatus.no_hit),
        health=SourceResult(source="health", status=SourceStatus.no_hit),
    )
    report = compose_investigation_report(bundle, _agent_request())
    assert report.answer_status == "dependency_unavailable"
    assert "traces" in report.unavailable_sources or "recall" in report.unavailable_sources
