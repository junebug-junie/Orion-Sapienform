from __future__ import annotations

import asyncio
from uuid import uuid4

from app import api as agent_api
from orion.schemas.agents.schemas import AgentChainRequest, ToolDef


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
    ad = ToolDef(tool_id="analyze_text", description="a", input_schema={}, output_schema={})
    monkeypatch.setattr(
        agent_api,
        "_resolve_tools",
        lambda body, output_mode=None: ([ad], ["executive_pack", "memory_pack"]),
    )

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



def test_agent_chain_delegate_loop_allows_planner_no_action_terminal(monkeypatch):
    async def _fake_planner(payload, *, parent_correlation_id=None, rpc_bus=None):
        return {
            "status": "ok",
            "stop_reason": "continue",
            "continue_reason": "none",
            "trace": [
                {
                    "step_index": 0,
                    "thought": "Here is the final summary.",
                    "action": None,
                    "observation": None,
                }
            ],
        }

    monkeypatch.setattr(agent_api, "call_planner_react", _fake_planner)
    ad = ToolDef(tool_id="noop", description="n", input_schema={}, output_schema={})
    monkeypatch.setattr(
        agent_api,
        "_resolve_tools",
        lambda body, output_mode=None: ([ad], ["executive_pack"]),
    )

    req = AgentChainRequest(text="hello", mode="agent", messages=[{"role": "user", "content": "hello"}])
    out = asyncio.run(agent_api.execute_agent_chain(req, correlation_id=str(uuid4()), rpc_bus=object()))
    assert out.text == "Here is the final summary."


def test_agent_chain_stops_after_successful_finalize_response(monkeypatch):
    calls = {"planner": 0}

    class _FinalizeToolExecutor:
        def __init__(self, *_args, **_kwargs):
            self.calls = []

        async def execute_llm_verb(self, tool_id, tool_input, *, parent_correlation_id=None):
            self.calls.append((tool_id, tool_input, parent_correlation_id))
            return {"llm_output": "Finalized answer for the user."}

    fake_exec = _FinalizeToolExecutor()

    async def _fake_planner(payload, *, parent_correlation_id=None, rpc_bus=None):
        calls["planner"] += 1
        return {
            "status": "ok",
            "trace": [
                {
                    "step_index": 0,
                    "thought": "Finalize now.",
                    "action": {"tool_id": "finalize_response", "input": {"request": "hello"}},
                    "observation": None,
                }
            ],
        }

    monkeypatch.setattr(agent_api, "call_planner_react", _fake_planner)
    monkeypatch.setattr(agent_api, "ToolExecutor", lambda *_a, **_k: fake_exec)
    fr = ToolDef(tool_id="finalize_response", description="finalize", input_schema={}, output_schema={})
    monkeypatch.setattr(
        agent_api,
        "_resolve_tools",
        lambda body, output_mode=None: ([fr], ["executive_pack", "delivery_pack"]),
    )

    req = AgentChainRequest(text="hello", mode="agent", messages=[{"role": "user", "content": "hello"}])
    out = asyncio.run(agent_api.execute_agent_chain(req, correlation_id=str(uuid4()), rpc_bus=object()))

    assert out.text == "Finalized answer for the user."
    assert calls["planner"] == 1
    assert len(fake_exec.calls) == 1
    assert fake_exec.calls[0][0] == "finalize_response"
