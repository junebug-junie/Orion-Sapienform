"""Pass 2: second plan_action escalates to a delivery verb."""

from __future__ import annotations

import asyncio
from uuid import uuid4

from app import api as agent_api
from orion.schemas.agents.schemas import AgentChainRequest, ToolDef


def test_second_plan_action_becomes_write_guide(monkeypatch):
    calls = []

    class _Exec:
        async def execute_llm_verb(self, tool_id, tool_input, *, parent_correlation_id=None):
            calls.append(tool_id)
            return {"llm_output": f"ok-{tool_id}"}

    n = 0

    async def fake_planner(payload, *, parent_correlation_id=None, rpc_bus=None):
        nonlocal n
        n += 1
        tr = list(payload.get("trace") or [])
        return {
            "status": "ok",
            "trace": tr
            + [
                {
                    "step_index": len(tr),
                    "thought": "p",
                    "action": {"tool_id": "plan_action", "input": {"goal": "g"}},
                    "observation": None,
                }
            ],
        }

    monkeypatch.setattr(agent_api, "call_planner_react", fake_planner)
    monkeypatch.setattr(agent_api, "ToolExecutor", lambda *a, **k: _Exec())
    monkeypatch.setattr(agent_api.settings, "default_max_steps", 4)
    pa = ToolDef(tool_id="plan_action", description="p", input_schema={}, output_schema={})
    wg = ToolDef(tool_id="write_guide", description="w", input_schema={}, output_schema={})
    monkeypatch.setattr(
        agent_api,
        "_resolve_tools",
        lambda body, output_mode=None: ([pa, wg], ["executive_pack", "delivery_pack"]),
    )

    req = AgentChainRequest(text="instructions please", mode="agent", output_mode="implementation_guide")
    out = asyncio.run(agent_api.execute_agent_chain(req, correlation_id=str(uuid4()), rpc_bus=object()))
    assert "write_guide" in calls
    assert out.runtime_debug.get("repeated_plan_action_escalation") is True
