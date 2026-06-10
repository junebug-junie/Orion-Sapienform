from __future__ import annotations

import os
import sys

import pytest

SERVICE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)
REPO_ROOT = os.path.abspath(os.path.join(SERVICE_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.supervisor import Supervisor  # noqa: E402
from orion.core.bus.bus_schemas import ServiceRef  # noqa: E402
from orion.schemas.context_exec import ContextExecRunV1  # noqa: E402
from scripts.context_exec_probe_lib import assert_hub_context_exec_routing  # noqa: E402


class _FakeContextExecClient:
    def __init__(self, *, raise_on_run: Exception | None = None):
        self.raise_on_run = raise_on_run
        self.run_calls = 0

    async def run(self, **kwargs):
        self.run_calls += 1
        if self.raise_on_run is not None:
            raise self.raise_on_run
        return ContextExecRunV1(
            run_id="ctxrun_route",
            status="ok",
            mode="belief_provenance",
            text="Where did claim come from?",
            final_text="Belief provenance summary.",
            runtime_debug={"engine": "context_exec", "schema_valid": True},
        )


class _FakeAgentClient:
    def __init__(self):
        self.run_calls = 0

    async def run_chain(self, **kwargs):
        self.run_calls += 1
        raise AssertionError("agent chain should not be called when context_exec enabled")


class _FallbackAgentClient:
    def __init__(self):
        self.run_calls = 0

    async def run_chain(self, **kwargs):
        self.run_calls += 1
        from orion.schemas.agents.schemas import AgentChainResult

        return AgentChainResult(
            text="legacy fallback answer",
            final_text="legacy fallback answer",
            mode="agent",
        )


def _ctx_with_context_exec(*, mode: str = "belief_provenance") -> dict:
    return {
        "mode": "agent",
        "messages": [{"role": "user", "content": "Where did the Denver claim come from?"}],
        "options": {
            "agent_runtime_engine": "context_exec",
            "context_exec_mode": mode,
        },
    }


@pytest.mark.asyncio
async def test_context_exec_dispatch_selected(monkeypatch):
    from app import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "context_exec_enabled", True)
    monkeypatch.setattr(settings_mod.settings, "context_exec_legacy_fallback", False)

    fake_ctx = _FakeContextExecClient()
    fake_agent = _FakeAgentClient()
    sup = Supervisor(bus=object())  # type: ignore[arg-type]
    sup.context_exec_client = fake_ctx
    sup.agent_client = fake_agent

    step = await sup._agent_chain_escalation(
        source=ServiceRef(name="exec", version="0.2.0"),
        correlation_id="corr-x",
        ctx=_ctx_with_context_exec(),
        packs=[],
    )
    payload = step.result.get("ContextExecService", {})
    assert fake_ctx.run_calls == 1
    assert fake_agent.run_calls == 0
    assert payload.get("final_text") == "Belief provenance summary."
    assert step.step_name == "context_exec"
    assert payload.get("structured", {}).get("context_exec")
    assert payload.get("runtime_debug", {}).get("engine") == "context_exec"


@pytest.mark.asyncio
async def test_legacy_agent_chain_preserved_when_disabled(monkeypatch):
    from app import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "context_exec_enabled", False)

    fake_ctx = _FakeContextExecClient()
    fake_agent = _FallbackAgentClient()
    sup = Supervisor(bus=object())  # type: ignore[arg-type]
    sup.context_exec_client = fake_ctx
    sup.agent_client = fake_agent

    step = await sup._agent_chain_escalation(
        source=ServiceRef(name="exec", version="0.2.0"),
        correlation_id="corr-legacy",
        ctx=_ctx_with_context_exec(),
        packs=[],
    )
    assert fake_ctx.run_calls == 0
    assert fake_agent.run_calls == 1
    assert step.step_name == "agent_chain"
    assert "AgentChainService" in step.result


@pytest.mark.asyncio
async def test_context_exec_fallback_to_legacy_agent_chain(monkeypatch):
    from app import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "context_exec_enabled", True)
    monkeypatch.setattr(settings_mod.settings, "context_exec_legacy_fallback", True)

    fake_ctx = _FakeContextExecClient(raise_on_run=TimeoutError("context-exec timeout"))
    fake_agent = _FallbackAgentClient()
    sup = Supervisor(bus=object())  # type: ignore[arg-type]
    sup.context_exec_client = fake_ctx
    sup.agent_client = fake_agent

    step = await sup._agent_chain_escalation(
        source=ServiceRef(name="exec", version="0.2.0"),
        correlation_id="corr-fallback",
        ctx=_ctx_with_context_exec(),
        packs=[],
    )
    assert fake_ctx.run_calls == 1
    assert fake_agent.run_calls == 1
    assert step.step_name == "agent_chain"
    payload = step.result.get("AgentChainService", {})
    runtime_debug = payload.get("runtime_debug", {})
    assert runtime_debug.get("context_exec_attempted") is True
    assert runtime_debug.get("context_exec_fallback") == "legacy_agent_chain"


@pytest.mark.asyncio
async def test_no_fallback_when_disabled(monkeypatch):
    from app import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "context_exec_enabled", True)
    monkeypatch.setattr(settings_mod.settings, "context_exec_legacy_fallback", False)

    fake_ctx = _FakeContextExecClient(raise_on_run=RuntimeError("context-exec down"))
    fake_agent = _FallbackAgentClient()
    sup = Supervisor(bus=object())  # type: ignore[arg-type]
    sup.context_exec_client = fake_ctx
    sup.agent_client = fake_agent

    step = await sup._agent_chain_escalation(
        source=ServiceRef(name="exec", version="0.2.0"),
        correlation_id="corr-no-fallback",
        ctx=_ctx_with_context_exec(),
        packs=[],
    )
    assert fake_ctx.run_calls == 1
    assert fake_agent.run_calls == 0
    assert step.status == "fail"
    assert step.step_name == "context_exec"
    payload = step.result.get("ContextExecService", {})
    assert payload.get("runtime_debug", {}).get("context_exec_attempted") is True


def test_hub_probe_accepts_cortex_exec_success_payload():
    assert_hub_context_exec_routing(
        {
            "session_id": "probe-fixture",
            "raw": {
                "verb": "agent_runtime",
                "final_text": "Belief provenance for Denver",
                "steps": [
                    {
                        "step_name": "context_exec",
                        "result": {
                            "ContextExecService": {
                                "final_text": "Belief provenance for Denver",
                                "text": "Belief provenance for Denver",
                                "structured": {"context_exec": {"mode": "belief_provenance", "run_id": "r1"}},
                                "runtime_debug": {
                                    "engine": "context_exec",
                                    "context_exec_attempted": True,
                                    "context_exec_status": "ok",
                                },
                            }
                        },
                    }
                ],
            },
            "routing_debug": {
                "options": {
                    "agent_runtime_engine": "context_exec",
                    "context_exec_mode": "belief_provenance",
                }
            },
        },
        probe_name="cortex_exec_fixture",
        expected_mode="belief_provenance",
    )
