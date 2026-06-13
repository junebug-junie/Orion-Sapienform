from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.alexzhang_rlm_engine import AlexZhangRLMEngine, UnsupportedModeError
from app.rlm_engine import FakeRLMEngine, build_engine
from app.runner import ContextExecRunner, FAKE_ORGANS
from orion.schemas.context_exec import (
    BeliefProvenanceReportV1,
    ContextExecRequestV1,
    RepoImpactAnalysisReportV1,
    TraceAutopsyReportV1,
)


@pytest.fixture(autouse=True)
def _reset_fake_organs():
    FAKE_ORGANS.memory_hits = None
    FAKE_ORGANS.trace_hits = None
    yield
    FAKE_ORGANS.memory_hits = None
    FAKE_ORGANS.trace_hits = None


def test_build_engine_defaults_to_fake():
    engine = build_engine("fake")
    assert isinstance(engine, FakeRLMEngine)
    assert engine.engine_name == "fake"


def test_build_engine_selects_alexzhang():
    engine = build_engine("alexzhang")
    assert isinstance(engine, AlexZhangRLMEngine)
    assert engine.engine_name == "alexzhang"


def test_build_engine_unknown_falls_back_to_fake():
    engine = build_engine("unknown_engine")
    assert isinstance(engine, FakeRLMEngine)


@pytest.mark.asyncio
async def test_runner_fake_engine_runtime_debug(monkeypatch):
    monkeypatch.setattr("app.runner.settings.rlm_engine", "fake")
    runner = ContextExecRunner()
    FAKE_ORGANS.memory_hits = [
        {"claim": "User is from Denver", "source_ref": "m:1", "verified": True, "confidence": 0.9}
    ]
    req = ContextExecRequestV1(text="Denver claim?", mode="belief_provenance")
    run = await runner.run(req)
    assert run.runtime_debug["engine"] == "fake"
    assert run.runtime_debug["engine_selected"] == "fake"
    assert run.runtime_debug.get("fallback_engine") is None


@pytest.mark.asyncio
async def test_runner_alexzhang_engine_runtime_debug(monkeypatch):
    monkeypatch.setattr("app.runner.settings.rlm_engine", "alexzhang")
    runner = ContextExecRunner()
    FAKE_ORGANS.memory_hits = [
        {"claim": "User is from Denver", "source_ref": "m:1", "verified": True, "confidence": 0.9}
    ]
    req = ContextExecRequestV1(
        text="Where did Orion get the claim that I am from Denver?",
        mode="belief_provenance",
    )
    run = await runner.run(req)
    assert run.runtime_debug["engine"] == "alexzhang"
    assert run.runtime_debug["engine_selected"] == "alexzhang"
    assert run.runtime_debug["schema_valid"] is True
    assert run.artifact_type == "BeliefProvenanceReportV1"


@pytest.mark.asyncio
async def test_alexzhang_belief_provenance_schema_valid():
    FAKE_ORGANS.memory_hits = [
        {"claim": "User is from Denver", "source_ref": "m:1", "verified": True, "confidence": 0.9}
    ]
    engine = AlexZhangRLMEngine()
    from app.callable_namespace import ContextNamespace
    from orion.schemas.context_exec import ContextExecPermissionV1

    ns = ContextNamespace(
        permissions=ContextExecPermissionV1(),
        memory_fn={"search_claims": lambda q, limit=20: FAKE_ORGANS.memory_hits or [], "read": lambda h: {}},
    )
    req = ContextExecRequestV1(
        text="Where did Orion get the claim that I am from Denver?",
        mode="belief_provenance",
    )
    raw = await engine.run(req, ns)
    model = BeliefProvenanceReportV1.model_validate(raw)
    assert model.status in {"supported", "unknown", "unsupported", "contradicted", "stale", "inferred"}


@pytest.mark.asyncio
async def test_alexzhang_trace_autopsy_schema_valid():
    FAKE_ORGANS.trace_hits = [
        {"handle": "t:1", "snippet": "fail open marker", "corr_id": "abc", "kind": "error", "source": "cortex"}
    ]
    engine = AlexZhangRLMEngine()
    from app.callable_namespace import ContextNamespace
    from orion.schemas.context_exec import ContextExecPermissionV1

    ns = ContextNamespace(
        permissions=ContextExecPermissionV1(),
        traces_fn={
            "search": lambda **kw: FAKE_ORGANS.trace_hits or [],
            "read": lambda h: {"handle": h},
        },
    )
    req = ContextExecRequestV1(text="Why did corr abc fail open?", mode="trace_autopsy")
    raw = await engine.run(req, ns)
    model = TraceAutopsyReportV1.model_validate(raw)
    assert model.status in {"explained", "partial", "unknown"}


@pytest.mark.asyncio
async def test_alexzhang_trace_autopsy_insufficient_evidence():
    engine = AlexZhangRLMEngine()
    from app.callable_namespace import ContextNamespace
    from orion.schemas.context_exec import ContextExecPermissionV1

    ns = ContextNamespace(
        permissions=ContextExecPermissionV1(),
        traces_fn={"search": lambda **kw: [], "read": lambda h: {}},
    )
    req = ContextExecRequestV1(text="Why did corr abc fail open?", mode="trace_autopsy")
    raw = await engine.run(req, ns)
    model = TraceAutopsyReportV1.model_validate(raw)
    assert model.root_cause == "insufficient_trace_evidence"


@pytest.mark.asyncio
async def test_alexzhang_repo_impact_schema_valid(monkeypatch):
    monkeypatch.setattr("app.alexzhang_rlm_engine.settings.context_exec_real_repo_enabled", True)
    engine = AlexZhangRLMEngine()
    from app.callable_namespace import ContextNamespace
    from orion.schemas.context_exec import ContextExecPermissionV1

    fake_hits = [
        {
            "path": "services/orion-cortex-exec/app/clients.py",
            "line_start": 10,
            "snippet": "AgentChainClient",
            "source_ref": "repo:clients.py:10",
        }
    ]

    ns = ContextNamespace(
        permissions=ContextExecPermissionV1(read_repo=True),
    )
    with patch.object(ns.repo, "grep", return_value=fake_hits):
        req = ContextExecRequestV1(
            text="What breaks if I replace agent-chain-service with context-exec?",
            mode="repo_impact_analysis",
            permissions=ContextExecPermissionV1(read_repo=True),
        )
        raw = await engine.run(req, ns)
    model = RepoImpactAnalysisReportV1.model_validate(raw)
    assert model.status in {"analyzed", "partial", "insufficient_grounding"}


@pytest.mark.asyncio
async def test_alexzhang_unsupported_mode_raises():
    engine = AlexZhangRLMEngine()
    from app.callable_namespace import ContextNamespace
    from orion.schemas.context_exec import ContextExecPermissionV1

    ns = ContextNamespace(permissions=ContextExecPermissionV1())
    req = ContextExecRequestV1(text="probe", mode="general_investigation")
    with pytest.raises(UnsupportedModeError):
        await engine.run(req, ns)


@pytest.mark.asyncio
async def test_runner_unsupported_mode_fails_closed(monkeypatch):
    monkeypatch.setattr("app.runner.settings.rlm_engine", "alexzhang")
    runner = ContextExecRunner()
    req = ContextExecRequestV1(text="probe", mode="general_investigation")
    run = await runner.run(req)
    assert run.status == "error"
    assert any("unsupported_mode" in fm for fm in run.failure_modes)


@pytest.mark.asyncio
async def test_bus_reply_alexzhang_selected(monkeypatch):
    from app.bus_listener import _handle_request
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
    from uuid import uuid4
    from types import SimpleNamespace

    monkeypatch.setattr("app.runner.settings.rlm_engine", "alexzhang")
    FAKE_ORGANS.memory_hits = [
        {"claim": "User is from Denver", "source_ref": "m:1", "verified": True, "confidence": 0.9}
    ]
    runner = ContextExecRunner()
    env = BaseEnvelope(
        kind="context.exec.request.v1",
        source=ServiceRef(name="cortex-exec", version="0.2.0"),
        correlation_id=uuid4(),
        reply_to="orion:exec:result:ContextExecService:test",
        causality_chain=[],
        payload={
            "text": "Where did Orion get the claim that I am from Denver?",
            "mode": "belief_provenance",
        },
    )
    bus = SimpleNamespace(
        codec=SimpleNamespace(decode=lambda _raw: SimpleNamespace(ok=True, envelope=env, error=None)),
        publish=AsyncMock(),
    )
    await _handle_request(bus, {"data": b"x"}, runner)  # type: ignore[arg-type]
    bus.publish.assert_awaited_once()
    _, reply_env = bus.publish.await_args.args
    assert reply_env.kind == "context.exec.result.v1"
    assert reply_env.causality_chain == []
    payload = reply_env.payload
    assert payload["runtime_debug"]["engine"] == "alexzhang"
    BaseEnvelope(
        kind="context.exec.result.v1",
        source=ServiceRef(name="context-exec", version="0.1.0"),
        correlation_id=env.correlation_id,
        causality_chain=[],
        payload=payload,
    )
