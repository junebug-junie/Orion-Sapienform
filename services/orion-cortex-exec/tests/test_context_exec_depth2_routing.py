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


class _FakeContextExecClient:
    async def run(self, **kwargs):
        return ContextExecRunV1(
            run_id="ctxrun_route",
            status="ok",
            mode="belief_provenance",
            text="Where did claim come from?",
            final_text="Belief provenance summary.",
            runtime_debug={"engine": "context_exec", "schema_valid": True},
        )


class _FakeAgentClient:
    async def run_chain(self, **kwargs):
        raise AssertionError("agent chain should not be called when context_exec enabled")


@pytest.mark.asyncio
async def test_supervisor_routes_context_exec_when_flagged(monkeypatch):
    from app import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "context_exec_enabled", True)
    monkeypatch.setattr(settings_mod.settings, "context_exec_legacy_fallback", False)

    sup = Supervisor(bus=object())  # type: ignore[arg-type]
    sup.context_exec_client = _FakeContextExecClient()
    sup.agent_client = _FakeAgentClient()

    step = await sup._agent_chain_escalation(
        source=ServiceRef(name="exec", version="0.2.0"),
        correlation_id="corr-x",
        ctx={
            "mode": "agent",
            "messages": [{"role": "user", "content": "Where did the Denver claim come from?"}],
            "options": {
                "agent_runtime_engine": "context_exec",
                "context_exec_mode": "belief_provenance",
            },
        },
        packs=[],
    )
    payload = step.result.get("ContextExecService", {})
    assert payload.get("final_text") == "Belief provenance summary."
    assert step.step_name == "context_exec"
