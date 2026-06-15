"""Tests for investigation_v2 evidence sweep (PR2)."""

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


@pytest.mark.asyncio
async def test_recall_timeout_isolation_repo_still_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    _ctx_app_modules()
    from app.investigation_v2 import run_investigation_v2
    from app.settings import settings
    from orion.schemas.context_exec import SourceStatus

    monkeypatch.setattr(settings, "context_exec_real_repo_enabled", True)
    monkeypatch.setattr(settings, "context_exec_real_recall_enabled", True)
    monkeypatch.setattr(settings, "context_exec_real_trace_enabled", False)

    async def slow_recall(*_a, **_k):
        await asyncio.sleep(60)
        return {"hits": []}

    runtime = MagicMock()
    runtime.bus = MagicMock()
    runtime.recall_query = AsyncMock(side_effect=slow_recall)
    runtime.repo_grep = MagicMock(
        return_value=[
            {
                "path": "services/orion-context-exec/app/runner.py",
                "line_start": 1,
                "snippet": "ContextExecRunner",
                "source_ref": "repo:runner.py",
            }
        ]
    )
    runtime.traces_search = AsyncMock(return_value=[])

    namespace = MagicMock()
    namespace.memory.search_claims = MagicMock(return_value=[])
    namespace.repo.grep = MagicMock(return_value=[])

    organ_cache: dict = {}
    monkeypatch.setattr(settings, "context_exec_investigation_v2_probe_timeout_sec", 0.2)
    monkeypatch.setattr(settings, "context_exec_recall_timeout_sec", 0.2)

    artifact = await run_investigation_v2(_agent_request(), namespace, runtime, organ_cache)

    assert artifact["evidence"]["recall"]["status"] == SourceStatus.unavailable.value
    assert artifact["evidence"]["repo"]["status"] == SourceStatus.hit.value
    assert artifact["answer_status"] == "partial_grounding"
    assert organ_cache.get("repo_probe_attempted") is True


@pytest.mark.asyncio
async def test_repo_runs_without_magic_prompt_words(monkeypatch: pytest.MonkeyPatch) -> None:
    _ctx_app_modules()
    from app import repo_tools
    from app.runner import ContextExecRunner
    from app.settings import settings

    monkeypatch.setattr(settings, "context_exec_real_repo_enabled", True)
    monkeypatch.setattr(settings, "context_exec_real_recall_enabled", False)
    monkeypatch.setattr(settings, "context_exec_real_trace_enabled", False)

    grep_calls: list[str] = []
    original_grep = repo_tools.repo_grep

    def tracking_grep(pattern: str, **kwargs):
        grep_calls.append(pattern)
        return original_grep(pattern, **kwargs)

    monkeypatch.setattr(repo_tools, "repo_grep", tracking_grep)

    runner = ContextExecRunner()
    prompt = "what would happen if we changed the cortex-exec runtime?"
    req = _agent_request(text=prompt)
    assert "repo" not in prompt.lower()
    assert "impact" not in prompt.lower()

    run = await runner.run(req)
    assert run.status == "ok"
    assert run.runtime_debug.get("read_repo") is True
    assert grep_calls, "repo grep should run when read_repo=True without keyword routing"


@pytest.mark.asyncio
async def test_per_source_statuses_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    _ctx_app_modules()
    from app.investigation_v2 import basic_investigation_v2_result
    from orion.schemas.context_exec import EvidenceBundle, SourceResult, SourceStatus

    bundle = EvidenceBundle(
        repo=SourceResult(source="repo", status=SourceStatus.hit, summary="repo hit"),
        traces=SourceResult(source="traces", status=SourceStatus.no_hit, summary="no traces"),
        recall=SourceResult(source="recall", status=SourceStatus.unavailable, error="timeout"),
        health=SourceResult(source="health", status=SourceStatus.no_hit, summary="shallow"),
    )
    artifact = basic_investigation_v2_result(bundle, _agent_request())
    assert artifact["sources"]["repo"] == "hit"
    assert artifact["sources"]["traces"] == "no_hit"
    assert artifact["sources"]["recall"] == "unavailable"
    assert artifact["answer_status"] == "partial_grounding"


@pytest.mark.asyncio
async def test_permission_blocking_skips_repo_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    _ctx_app_modules()
    from app.investigation_v2 import run_investigation_v2
    from app.settings import settings
    from orion.schemas.context_exec import ContextExecPermissionV1, SourceStatus

    monkeypatch.setattr(settings, "context_exec_real_repo_enabled", True)

    repo_grep_called = False

    runtime = MagicMock()
    runtime.bus = None
    runtime.repo_grep = MagicMock(
        side_effect=lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("repo_grep should not run"))
    )
    runtime.recall_query = AsyncMock(return_value={"hits": []})
    runtime.traces_search = AsyncMock(return_value=[])

    namespace = MagicMock()
    namespace.memory.search_claims = MagicMock(return_value=[])
    namespace.repo.grep = MagicMock(
        side_effect=lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("repo grep should not run"))
    )

    perms = ContextExecPermissionV1(read_repo=False)
    req = _agent_request(permissions=perms)
    organ_cache: dict = {}
    artifact = await run_investigation_v2(req, namespace, runtime, organ_cache)

    assert artifact["evidence"]["repo"]["status"] == SourceStatus.blocked.value
    assert "repo" in artifact["blocked_sources"]
    assert organ_cache.get("repo_probe_attempted") is None


@pytest.mark.asyncio
async def test_runner_investigation_v2_returns_evidence_report(monkeypatch: pytest.MonkeyPatch) -> None:
    _ctx_app_modules()
    from app.runner import ContextExecRunner
    from app.settings import settings
    from orion.schemas.context_exec import SourceStatus

    monkeypatch.setattr(settings, "context_exec_real_repo_enabled", True)
    monkeypatch.setattr(settings, "context_exec_real_recall_enabled", False)
    monkeypatch.setattr(settings, "context_exec_real_trace_enabled", False)

    runner = ContextExecRunner()
    run = await runner.run(_agent_request())
    assert run.status == "ok"
    assert run.artifact_type == "InvestigationReportV2"
    assert run.artifact.get("artifact_type") == "InvestigationReportV2"
    assert "answer_status" in run.artifact
    assert run.runtime_debug.get("investigation_v2") is True
    repo_status = (run.artifact.get("evidence") or {}).get("repo", {}).get("status")
    assert repo_status in {SourceStatus.hit.value, SourceStatus.no_hit.value}


@pytest.mark.asyncio
async def test_agent_compat_v2_skips_keyword_mode_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    from orion.schemas.agents.schemas import AgentChainRequest

    _ctx_app_modules()
    from app.agent_compat import agent_chain_request_to_context_exec
    from app.settings import settings

    monkeypatch.setattr(settings, "context_exec_investigation_v2_enabled", True)
    body = AgentChainRequest(
        text="what breaks if we replace agent-chain-service with context-exec?",
        mode="agent",
        response_profile="agent",
    )
    req = agent_chain_request_to_context_exec(body)
    assert req.mode == "investigation_v2"
    assert req.permissions.read_repo is True


def test_investigation_report_v2_schema_roundtrip() -> None:
    from orion.schemas.context_exec import (
        EvidenceBundle,
        InvestigationReportV2,
        SourceResult,
        SourceStatus,
    )

    report = InvestigationReportV2(
        answer_status="partial_grounding",
        summary="Evidence from: repo",
        sources={"repo": "hit"},
        evidence=EvidenceBundle(
            repo=SourceResult(source="repo", status=SourceStatus.hit, summary="1 hit"),
        ),
    )
    dumped = report.model_dump(mode="json")
    assert InvestigationReportV2.model_validate(dumped).answer_status == "partial_grounding"
