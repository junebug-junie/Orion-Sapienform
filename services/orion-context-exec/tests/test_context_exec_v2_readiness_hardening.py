"""Hardening tests: /health bus readiness and LLM gateway synthesis preflight."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

CTX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_DIR = CTX_ROOT


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


def _load_app(monkeypatch: pytest.MonkeyPatch, *, reload_settings: bool = False) -> object:
    if str(SERVICE_DIR) not in sys.path:
        sys.path.insert(0, str(SERVICE_DIR))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    modules = ["app.main", "app.api", "app.proposal_review_api"]
    if reload_settings:
        modules.insert(0, "app.settings")
    for mod in modules:
        sys.modules.pop(mod, None)
    from app.main import app  # noqa: WPS433

    return app


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


def _dead_readiness(*, intake_channel: str):
    from orion.bus.consumer_readiness import BusConsumerReadinessResult

    return BusConsumerReadinessResult(
        ok=False,
        bus_consumer_ready=False,
        intake_channel=intake_channel,
        subscriber_count=0,
        dependency_status="unavailable",
        error=f"no subscribers on intake channel: {intake_channel}",
    )


def _live_readiness(*, intake_channel: str):
    from orion.bus.consumer_readiness import BusConsumerReadinessResult

    return BusConsumerReadinessResult(
        ok=True,
        bus_consumer_ready=True,
        intake_channel=intake_channel,
        subscriber_count=1,
        dependency_status="available",
    )


@pytest.mark.asyncio
async def test_health_exposes_bus_dependency_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    """A: /health includes dependency readiness fields and aggregate reflects unavailable deps."""
    _ctx_app_modules()
    from app.settings import settings
    from orion.bus.consumer_readiness import BusConsumerReadinessResult

    monkeypatch.setattr(settings, "orion_bus_enabled", True)

    async def _mock_collect(_bus, *, timeout_sec: float):
        return {
            "bus_enabled": True,
            "bus_connected": True,
            "bus_consumer_ready": False,
            "dependencies": {
                "recall": {
                    "bus_consumer_ready": False,
                    "intake_channel": settings.channel_recall_intake,
                    "subscriber_count": 0,
                    "redis_ping_ok": True,
                    "heartbeat_fresh": None,
                    "rpc_smoke_ok": None,
                    "status": "unavailable",
                },
                "llm_gateway": {
                    "bus_consumer_ready": False,
                    "intake_channel": settings.channel_llm_intake,
                    "subscriber_count": 0,
                    "redis_ping_ok": True,
                    "heartbeat_fresh": None,
                    "rpc_smoke_ok": None,
                    "status": "unavailable",
                },
            },
        }

    monkeypatch.setattr("app.bus_dependency_preflight.collect_bus_dependencies_health", _mock_collect)

    app = _load_app(monkeypatch)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["bus_consumer_ready"] is False
    deps = data["dependencies"]
    assert deps["recall"]["subscriber_count"] == 0
    assert deps["llm_gateway"]["subscriber_count"] == 0
    assert deps["recall"]["intake_channel"] == settings.channel_recall_intake
    assert deps["llm_gateway"]["intake_channel"] == settings.channel_llm_intake
    assert deps["recall"]["status"] == "unavailable"
    assert deps["llm_gateway"]["status"] == "unavailable"


@pytest.mark.asyncio
async def test_llm_gateway_dead_consumer_skips_synthesis_rpc(monkeypatch: pytest.MonkeyPatch) -> None:
    """B: dead LLM gateway consumer skips synthesis RPC; deterministic report preserved."""
    _ctx_app_modules()
    from app.agent_synthesis import LLM_GATEWAY_SYNTHESIS_UNAVAILABLE
    from app.runner import ContextExecRunner
    from app.settings import settings

    llm_chat = AsyncMock(side_effect=AssertionError("llm_chat_route must not run when preflight fails"))

    monkeypatch.setattr(settings, "orion_bus_enabled", True)
    monkeypatch.setattr(settings, "context_exec_real_repo_enabled", True)
    monkeypatch.setattr(settings, "context_exec_real_recall_enabled", False)
    monkeypatch.setattr(settings, "context_exec_real_trace_enabled", False)
    monkeypatch.setattr(settings, "context_exec_agent_synthesis_enabled", True)
    monkeypatch.setattr("app.agent_synthesis.llm_chat_route", llm_chat)
    monkeypatch.setattr(
        "app.runner.effective_llm_gateway_ready",
        AsyncMock(
            return_value=(
                _dead_readiness(intake_channel=settings.channel_llm_intake),
                False,
                False,
            )
        ),
    )

    bus = MagicMock()
    bus.enabled = True
    runner = ContextExecRunner(bus=bus)
    started = time.perf_counter()
    run = await runner.run(_agent_request())
    elapsed = time.perf_counter() - started

    assert elapsed < 2.0
    llm_chat.assert_not_called()
    assert run.status == "ok"
    limitations = run.artifact.get("limitations") or []
    assert any(LLM_GATEWAY_SYNTHESIS_UNAVAILABLE in str(item) for item in limitations)
    assert run.artifact.get("sections", {}).get("repo") is not None or run.artifact.get("evidence")
    assert run.runtime_debug.get("synthesis_status") == "synthesis_unavailable"
    assert run.runtime_debug.get("answer_evaluation", {}).get("answer_status") != "no_reliable_evidence"


@pytest.mark.asyncio
async def test_synthesis_runs_when_llm_gateway_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """C: synthesis path still invoked when gateway bus consumer is ready."""
    _ctx_app_modules()
    from app.runner import ContextExecRunner
    from app.settings import settings

    llm_chat = AsyncMock(
        return_value={
            "ok": True,
            "content": '{"title":"Investigation","summary":"services/orion-context-exec/app/runner.py handles cortex-exec runtime."}',
        }
    )

    monkeypatch.setattr(settings, "orion_bus_enabled", True)
    monkeypatch.setattr(settings, "context_exec_real_repo_enabled", True)
    monkeypatch.setattr(settings, "context_exec_real_recall_enabled", False)
    monkeypatch.setattr(settings, "context_exec_real_trace_enabled", False)
    monkeypatch.setattr(settings, "context_exec_agent_synthesis_enabled", True)
    monkeypatch.setattr("app.agent_synthesis.llm_chat_route", llm_chat)
    monkeypatch.setattr(
        "app.runner.effective_llm_gateway_ready",
        AsyncMock(
            return_value=(
                _live_readiness(intake_channel=settings.channel_llm_intake),
                True,
                True,
            )
        ),
    )

    bus = MagicMock()
    bus.enabled = True
    runner = ContextExecRunner(bus=bus)
    run = await runner.run(_agent_request())

    llm_chat.assert_called_once()
    assert run.status == "ok"
    assert run.artifact.get("evidence") or run.artifact.get("sections")


@pytest.mark.asyncio
async def test_recall_dead_consumer_still_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    """D: dead Recall consumer marks recall unavailable; repo probe still runs."""
    _ctx_app_modules()
    from app.investigation_v2 import run_investigation_v2
    from app.settings import settings
    from orion.schemas.context_exec import SourceStatus

    monkeypatch.setattr(settings, "orion_bus_enabled", True)
    monkeypatch.setattr(settings, "context_exec_real_repo_enabled", True)
    monkeypatch.setattr(settings, "context_exec_real_recall_enabled", True)
    monkeypatch.setattr(settings, "context_exec_real_trace_enabled", False)

    recall_query = AsyncMock(side_effect=AssertionError("recall_query must not run when preflight fails"))
    monkeypatch.setattr(
        "app.investigation_v2.check_recall_bus_ready",
        AsyncMock(return_value=_dead_readiness(intake_channel=settings.channel_recall_intake)),
    )
    monkeypatch.setattr(
        "app.investigation_v2.check_llm_gateway_bus_ready",
        AsyncMock(return_value=_live_readiness(intake_channel=settings.channel_llm_intake)),
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

    artifact = await run_investigation_v2(_agent_request(), namespace, runtime, {})
    recall_query.assert_not_called()
    assert artifact["evidence"]["recall"]["status"] == SourceStatus.unavailable.value
    assert artifact["evidence"]["repo"]["status"] == SourceStatus.hit.value
    assert artifact["answer_status"] == "partial_grounding"


@pytest.mark.asyncio
async def test_agent_v2_no_mutation_permission_regression(monkeypatch: pytest.MonkeyPatch) -> None:
    """E: Agent v2 remains read-broad/write-none."""
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
    assert perms.network_enabled is False
    assert perms.shell_enabled is False

    runner = ContextExecRunner()
    run = await runner.run(_agent_request(permissions=perms))
    assert run.status == "ok"
    received = run.runtime_debug.get("permissions_received") or {}
    assert received.get("write_repo") is False
    assert received.get("mutate_runtime") is False
