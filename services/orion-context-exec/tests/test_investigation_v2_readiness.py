"""PR4 tests: investigation_v2 bus readiness preflight and synthesis degradation."""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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
async def test_recall_bus_consumer_dead_marks_unavailable_quickly(monkeypatch: pytest.MonkeyPatch) -> None:
    _ctx_app_modules()
    from app.investigation_v2 import run_investigation_v2
    from app.settings import settings
    from orion.bus.consumer_readiness import BusConsumerReadinessResult
    from orion.schemas.context_exec import SourceStatus

    monkeypatch.setattr(settings, "orion_bus_enabled", True)
    monkeypatch.setattr(settings, "context_exec_real_repo_enabled", True)
    monkeypatch.setattr(settings, "context_exec_real_recall_enabled", True)
    monkeypatch.setattr(settings, "context_exec_real_trace_enabled", False)
    monkeypatch.setattr(settings, "context_exec_bus_readiness_timeout_sec", 0.5)

    recall_query = AsyncMock(side_effect=AssertionError("recall_query must not run when preflight fails"))

    async def _dead_recall_ready(_bus, *, timeout_sec: float):
        return BusConsumerReadinessResult(
            ok=False,
            bus_consumer_ready=False,
            intake_channel=settings.channel_recall_intake,
            subscriber_count=0,
            dependency_status="unavailable",
            error=f"no subscribers on intake channel: {settings.channel_recall_intake}",
        )

    monkeypatch.setattr("app.investigation_v2.check_recall_bus_ready", _dead_recall_ready)
    monkeypatch.setattr(
        "app.investigation_v2.check_llm_gateway_bus_ready",
        AsyncMock(
            return_value=BusConsumerReadinessResult(
                ok=True,
                bus_consumer_ready=True,
                intake_channel=settings.channel_llm_intake,
                subscriber_count=1,
                dependency_status="available",
            )
        ),
    )

    runtime = MagicMock()
    runtime.bus = MagicMock()
    runtime.recall_query = recall_query
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
    started = time.perf_counter()
    artifact = await run_investigation_v2(_agent_request(), namespace, runtime, organ_cache)
    elapsed = time.perf_counter() - started

    assert elapsed < 1.0
    recall_query.assert_not_called()
    assert artifact["evidence"]["recall"]["status"] == SourceStatus.unavailable.value
    assert "subscriber_count=0" in str(artifact["evidence"]["recall"].get("error") or "")
    assert artifact["evidence"]["repo"]["status"] == SourceStatus.hit.value
    assert artifact["answer_status"] in {"dependency_unavailable", "partial_grounding"}
    assert organ_cache.get("repo_probe_attempted") is True


@pytest.mark.asyncio
async def test_llm_gateway_bus_dead_synthesis_limitation(monkeypatch: pytest.MonkeyPatch) -> None:
    _ctx_app_modules()
    from app.runner import ContextExecRunner
    from app.settings import settings

    async def _fail_llm(*_args, **_kwargs):
        return {"ok": False, "error": "timeout"}

    monkeypatch.setattr(settings, "context_exec_real_repo_enabled", True)
    monkeypatch.setattr(settings, "context_exec_real_recall_enabled", False)
    monkeypatch.setattr(settings, "context_exec_real_trace_enabled", False)
    monkeypatch.setattr(settings, "context_exec_agent_synthesis_enabled", True)
    monkeypatch.setattr("app.agent_synthesis.llm_chat_route", _fail_llm)

    runner = ContextExecRunner()
    run = await runner.run(_agent_request())
    assert run.status == "ok"
    limitations = run.artifact.get("limitations") or []
    assert any("LLM synthesis unavailable" in str(item) for item in limitations)
    assert run.artifact.get("sections", {}).get("repo", {}).get("status") in {"hit", "no_hit"}
    answer_eval = run.runtime_debug.get("answer_evaluation") or {}
    assert answer_eval.get("synthesis_status") == "synthesis_unavailable"
    assert answer_eval.get("answer_status") != "failed"
    assert run.runtime_debug.get("synthesis_status") == "synthesis_unavailable"


@pytest.mark.asyncio
async def test_agent_v2_permissions_no_mutation_regression(monkeypatch: pytest.MonkeyPatch) -> None:
    _ctx_app_modules()
    from app.runner import ContextExecRunner
    from app.settings import settings
    from orion.schemas.context_exec import context_exec_permissions_for_llm_profile

    monkeypatch.setattr(settings, "context_exec_real_repo_enabled", False)
    monkeypatch.setattr(settings, "context_exec_real_recall_enabled", False)
    monkeypatch.setattr(settings, "context_exec_real_trace_enabled", False)

    perms = context_exec_permissions_for_llm_profile("agent")
    assert perms.write_repo is False
    assert perms.mutate_runtime is False
    assert perms.write_memory is False
    assert perms.write_graph is False

    runner = ContextExecRunner()
    run = await runner.run(_agent_request(permissions=perms))
    assert run.status == "ok"
    received = run.runtime_debug.get("permissions_received") or {}
    assert received.get("write_repo") is False
    assert received.get("mutate_runtime") is False


def test_legacy_compat_keyword_routing_when_v2_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from orion.schemas.agents.schemas import AgentChainRequest

    _ctx_app_modules()
    from app.agent_compat import agent_chain_request_to_context_exec
    from app.settings import settings

    monkeypatch.setattr(settings, "context_exec_investigation_v2_enabled", False)
    body = AgentChainRequest(
        text="what breaks if we replace agent-chain-service with context-exec?",
        mode="agent",
        response_profile="agent",
    )
    req = agent_chain_request_to_context_exec(body)
    assert req.mode == "repo_impact_analysis"
    assert req.permissions.read_repo is True

    body2 = AgentChainRequest(
        text="Where did the Denver claim come from?",
        mode="agent",
    )
    req2 = agent_chain_request_to_context_exec(body2)
    assert req2.mode == "belief_provenance"
