from __future__ import annotations

import asyncio
from uuid import uuid4

from app import api as agent_api
from orion.schemas.agents.schemas import AgentChainRequest


class _FakeToolExecutor:
    def __init__(self, *_args, **_kwargs):
        self.calls = []

    async def execute_llm_verb(self, tool_id, tool_input, *, parent_correlation_id=None):
        self.calls.append((tool_id, tool_input, parent_correlation_id))
        return {"llm_output": f"obs for {tool_id}"}


def test_agent_chain_delegate_loop_executes_action_and_returns_final(monkeypatch):
    calls = {"planner": 0, "payloads": []}
    fake_exec = _FakeToolExecutor()

    async def _fake_planner(payload, *, parent_correlation_id=None, rpc_bus=None):
        calls["planner"] += 1
        calls["payloads"].append(payload)
        if calls["planner"] == 1:
            assert payload["preferences"]["delegate_tool_execution"] is True
            return {
                "status": "ok",
                "trace": [
                    {
                        "step_index": 0,
                        "thought": "need tool",
                        "action": {"tool_id": "analyze_text", "input": {"text": "hello"}},
                        "observation": None,
                    }
                ],
            }
        assert payload["trace"][-1]["observation"]["llm_output"] == "obs for analyze_text"
        return {
            "status": "ok",
            "final_answer": {"content": "done", "structured": {}},
            "trace": payload["trace"],
        }

    monkeypatch.setattr(agent_api, "call_planner_react", _fake_planner)
    monkeypatch.setattr(agent_api, "ToolExecutor", lambda *_a, **_k: fake_exec)
    monkeypatch.setattr(agent_api, "_resolve_tools", lambda _body: [])

    req = AgentChainRequest(
        text="hello",
        mode="agent",
        messages=[{"role": "user", "content": "hello"}],
    )
    out = asyncio.run(agent_api.execute_agent_chain(req, correlation_id=str(uuid4()), rpc_bus=object()))

    assert out.text == "done"
    assert calls["planner"] == 2
    assert len(fake_exec.calls) == 1
    assert fake_exec.calls[0][0] == "analyze_text"
