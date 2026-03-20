"""Pass 2: meta-plan final answer triggers finalize_response rewrite."""

from __future__ import annotations

import asyncio
from uuid import uuid4

from app import api as agent_api
from orion.schemas.agents.schemas import AgentChainRequest, ToolDef


def test_meta_plan_rewrite_invokes_finalize(monkeypatch):
    class _Exec:
        def __init__(self):
            self.calls = []

        async def execute_llm_verb(self, tool_id, tool_input, *, parent_correlation_id=None):
            self.calls.append(tool_id)
            if tool_id == "finalize_response":
                return {"llm_output": "Concrete step 1: do X. Step 2: do Y."}
            return {"llm_output": "x"}

    execu = _Exec()

    async def _planner(*args, **kwargs):
        return {
            "status": "ok",
            "final_answer": {
                "content": "Gather requirements, create a guide, then review and refine.",
                "structured": {},
            },
        }

    monkeypatch.setattr(agent_api, "call_planner_react", _planner)
    monkeypatch.setattr(agent_api, "ToolExecutor", lambda *a, **k: execu)
    tdef = ToolDef(tool_id="plan_action", description="p", input_schema={}, output_schema={})
    monkeypatch.setattr(
        agent_api,
        "_resolve_tools",
        lambda body, output_mode=None: ([tdef], ["executive_pack", "delivery_pack"]),
    )

    req = AgentChainRequest(
        text="how to install",
        mode="agent",
        output_mode="implementation_guide",
    )
    out = asyncio.run(agent_api.execute_agent_chain(req, correlation_id=str(uuid4()), rpc_bus=object()))
    assert "Concrete step" in out.text or "step 1" in out.text.lower()
    assert "finalize_response" in execu.calls
    assert out.runtime_debug.get("quality_evaluator_rewrite") is True
