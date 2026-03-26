from __future__ import annotations

import asyncio
from uuid import uuid4

from app import api as agent_api
from orion.schemas.agents.schemas import AgentChainRequest, ToolDef


def test_repeated_tool_breaker_finalizes(monkeypatch):
    calls: list[str] = []

    class _Exec:
        async def execute_llm_verb(self, tool_id, tool_input, *, parent_correlation_id=None):
            calls.append(tool_id)
            if tool_id == "finalize_response":
                return {"llm_output": "Best-effort final answer after repeated tool loop."}
            return {"llm_output": f"obs-{tool_id}"}

    async def fake_planner(payload, *, parent_correlation_id=None, rpc_bus=None):
        trace = list(payload.get("trace") or [])
        return {
            "status": "ok",
            "trace": trace
            + [
                {
                    "step_index": len(trace),
                    "thought": "repeat evaluate",
                    "action": {"tool_id": "evaluate", "input": {"output": "x"}},
                    "observation": None,
                }
            ],
        }

    monkeypatch.setattr(agent_api, "call_planner_react", fake_planner)
    monkeypatch.setattr(agent_api, "ToolExecutor", lambda *a, **k: _Exec())
    monkeypatch.setattr(agent_api.settings, "default_max_steps", 4)
    tdef = ToolDef(tool_id="evaluate", description="e", input_schema={}, output_schema={})
    fdef = ToolDef(tool_id="finalize_response", description="f", input_schema={}, output_schema={})
    monkeypatch.setattr(agent_api, "_resolve_tools", lambda body, output_mode=None: ([tdef, fdef], ["executive_pack"]))

    req = AgentChainRequest(text="Please evaluate this output.", mode="agent")
    out = asyncio.run(agent_api.execute_agent_chain(req, correlation_id=str(uuid4()), rpc_bus=object()))
    assert out.runtime_debug.get("repeated_tool_breaker") is True
    assert "finalize_response" in calls
    assert out.text
