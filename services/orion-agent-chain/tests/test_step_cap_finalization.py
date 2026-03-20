"""Tests for step-cap finalization (answer depth overhaul)."""

from __future__ import annotations

import asyncio
from uuid import uuid4

from app import api as agent_api
from orion.schemas.agents.schemas import AgentChainRequest


class _FakeToolExecutor:
    def __init__(self):
        self.calls = []

    async def execute_llm_verb(self, tool_id, tool_input, *, parent_correlation_id=None):
        self.calls.append((tool_id, tool_input))
        if tool_id == "finalize_response":
            return {"llm_output": "Best-effort synthesis from trace."}
        return {"llm_output": f"obs for {tool_id}"}


def test_step_cap_yields_best_effort(monkeypatch):
    """When max steps reached, finalize_response invoked; user gets prose not error."""
    monkeypatch.setattr(agent_api.settings, "default_max_steps", 2)

    async def fake_planner(payload, *, parent_correlation_id=None, rpc_bus=None):
        trace = payload.get("trace") or []
        return {
            "status": "ok",
            "trace": trace + [{"step_index": len(trace), "thought": "need more", "action": {"tool_id": "plan_action", "input": {"goal": "x"}}, "observation": None}],
        }

    fake_exec = _FakeToolExecutor()
    monkeypatch.setattr(agent_api, "call_planner_react", fake_planner)
    monkeypatch.setattr(agent_api, "ToolExecutor", lambda *a, **k: fake_exec)
    monkeypatch.setattr(agent_api, "_resolve_tools", lambda _: [])

    req = AgentChainRequest(text="complex request", mode="agent", messages=[{"role": "user", "content": "x"}])
    out = asyncio.run(agent_api.execute_agent_chain(req, correlation_id=str(uuid4()), rpc_bus=object()))

    assert out.text
    assert "Best-effort" in out.text or "synthesis" in out.text.lower()
    assert "finalize_response" in [c[0] for c in fake_exec.calls]
