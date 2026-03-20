"""Tests for triage gating (answer depth overhaul)."""

from __future__ import annotations

import asyncio
from uuid import uuid4

from app import api as agent_api
from orion.schemas.agents.schemas import AgentChainRequest, ToolDef


class _FakeToolExecutor:
    def __init__(self):
        self.calls = []

    async def execute_llm_verb(self, tool_id, tool_input, *, parent_correlation_id=None):
        self.calls.append((tool_id, tool_input))
        if tool_id == "finalize_response":
            return {"llm_output": "Synthesized final answer from trace."}
        return {"llm_output": f"obs for {tool_id}"}


def test_triage_blocked_after_step_0(monkeypatch):
    """When planner returns triage at step_idx>0, override to finalize_response."""
    call_count = 0

    async def fake_planner(payload, *, parent_correlation_id=None, rpc_bus=None):
        nonlocal call_count
        call_count += 1
        trace = payload.get("trace") or []
        if call_count == 1:
            return {
                "status": "ok",
                "trace": [{"step_index": 0, "thought": "triage", "action": {"tool_id": "triage", "input": {"request": "deploy"}}, "observation": None}],
            }
        if call_count == 2:
            return {
                "status": "ok",
                "trace": trace + [{"step_index": 1, "thought": "triage again", "action": {"tool_id": "triage", "input": {}}, "observation": None}],
            }
        return {"status": "ok", "final_answer": {"content": "done"}}

    fake_exec = _FakeToolExecutor()
    monkeypatch.setattr(agent_api, "call_planner_react", fake_planner)
    monkeypatch.setattr(agent_api, "ToolExecutor", lambda *a, **k: fake_exec)
    tdef = ToolDef(tool_id="triage", description="t", input_schema={}, output_schema={})
    monkeypatch.setattr(
        agent_api,
        "_resolve_tools",
        lambda body, output_mode=None: ([tdef], ["executive_pack", "delivery_pack"]),
    )

    req = AgentChainRequest(text="how to deploy to Discord", mode="agent", messages=[{"role": "user", "content": "how to deploy"}])
    out = asyncio.run(agent_api.execute_agent_chain(req, correlation_id=str(uuid4()), rpc_bus=object()))

    tool_ids = [c[0] for c in fake_exec.calls]
    assert len(tool_ids) >= 1
    if len(tool_ids) >= 2:
        assert tool_ids[1] == "finalize_response"
    assert out.runtime_debug.get("triage_blocked_post_step0") is True
