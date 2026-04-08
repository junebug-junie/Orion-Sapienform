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


def test_third_consecutive_plan_action_is_suppressed_and_finalized(monkeypatch):
    calls = {"planner": 0, "toolsets": []}
    fake_exec = _FakeToolExecutor()

    async def _fake_planner(payload, *, parent_correlation_id=None, rpc_bus=None):
        calls["planner"] += 1
        calls["toolsets"].append([t.get("tool_id") for t in payload.get("toolset", [])])
        if calls["planner"] <= 3:
            return {
                "status": "ok",
                "trace": [
                    {
                        "step_index": calls["planner"] - 1,
                        "thought": f"plan step {calls['planner']}",
                        "action": {"tool_id": "plan_action", "input": {"goal": "ship"}},
                        "observation": None,
                    }
                ],
            }
        return {"status": "ok", "final_answer": {"content": "should not get here", "structured": {}}}

    monkeypatch.setattr(agent_api, "call_planner_react", _fake_planner)
    monkeypatch.setattr(agent_api, "ToolExecutor", lambda *_a, **_k: fake_exec)
    tools = [
        ToolDef(tool_id="plan_action", description="plan", input_schema={}, output_schema={}),
        ToolDef(tool_id="finalize_response", description="finalize", input_schema={}, output_schema={}),
    ]
    monkeypatch.setattr(
        agent_api,
        "_resolve_tools",
        lambda body, output_mode=None: (tools, ["executive_pack", "delivery_pack"]),
    )

    req = AgentChainRequest(text="hello", mode="agent", messages=[{"role": "user", "content": "hello"}])
    out = asyncio.run(agent_api.execute_agent_chain(req, correlation_id=str(uuid4()), rpc_bus=object()))

    executed_tools = [c[0] for c in fake_exec.calls]
    assert executed_tools == ["plan_action", "plan_action", "finalize_response"]
    # On the fourth planner turn (if reached), plan_action should no longer be visible.
    assert "plan_action" not in calls["toolsets"][-1]
    assert out.structured["finalization_reason"] == "repeated_plan_action"
    assert out.runtime_debug["finalization_reason"] == "repeated_plan_action"


def test_step_cap_sets_finalization_reason_and_returns_best_effort(monkeypatch):
    fake_exec = _FakeToolExecutor()

    async def _fake_planner(payload, *, parent_correlation_id=None, rpc_bus=None):
        return {
            "status": "ok",
            "trace": [
                {
                    "step_index": 0,
                    "thought": "keep planning",
                    "action": {"tool_id": "analyze_text", "input": {"text": "x"}},
                    "observation": None,
                }
            ],
        }

    monkeypatch.setattr(agent_api, "call_planner_react", _fake_planner)
    monkeypatch.setattr(agent_api, "ToolExecutor", lambda *_a, **_k: fake_exec)
    monkeypatch.setattr(agent_api.settings, "default_max_steps", 1)
    tools = [ToolDef(tool_id="analyze_text", description="a", input_schema={}, output_schema={})]
    monkeypatch.setattr(agent_api, "_resolve_tools", lambda body, output_mode=None: (tools, ["executive_pack"]))

    req = AgentChainRequest(text="hello", mode="agent", messages=[{"role": "user", "content": "hello"}])
    out = asyncio.run(agent_api.execute_agent_chain(req, correlation_id=str(uuid4()), rpc_bus=object()))
    assert out.structured["finalization_reason"] == "step_cap_best_effort"
    assert out.runtime_debug["finalization_reason"] == "step_cap_best_effort"
    assert "obs for finalize_response" in out.text or "Max steps reached" in out.text


def test_invalid_analyze_conversation_tool_is_remapped_to_available_tool(monkeypatch):
    calls = {"planner": 0}
    fake_exec = _FakeToolExecutor()

    async def _fake_planner(payload, *, parent_correlation_id=None, rpc_bus=None):
        calls["planner"] += 1
        if calls["planner"] == 1:
            return {
                "status": "ok",
                "trace": [
                    {
                        "step_index": 0,
                        "thought": "Need analysis first.",
                        "action": {"tool_id": "analyze_conversation", "input": {"text": "hello"}},
                        "observation": None,
                    }
                ],
            }
        return {"status": "ok", "final_answer": {"content": "done", "structured": {}}}

    monkeypatch.setattr(agent_api, "call_planner_react", _fake_planner)
    monkeypatch.setattr(agent_api, "ToolExecutor", lambda *_a, **_k: fake_exec)
    tools = [
        ToolDef(tool_id="analyze_text", description="analyze", input_schema={}, output_schema={}),
        ToolDef(tool_id="finalize_response", description="finalize", input_schema={}, output_schema={}),
    ]
    monkeypatch.setattr(agent_api, "_resolve_tools", lambda body, output_mode=None: (tools, ["executive_pack"]))

    req = AgentChainRequest(text="hello", mode="agent", messages=[{"role": "user", "content": "hello"}])
    out = asyncio.run(agent_api.execute_agent_chain(req, correlation_id=str(uuid4()), rpc_bus=object()))

    assert out.text == "done"
    assert fake_exec.calls[0][0] == "analyze_text"
    assert out.runtime_debug["invalid_tool_remap_count"] == 1
    assert out.runtime_debug["invalid_tool_last"]["requested"] == "analyze_conversation"
    assert out.runtime_debug["invalid_tool_last"]["resolved"] == "analyze_text"


def test_invalid_gather_info_tool_is_remapped_to_plan_action(monkeypatch):
    calls = {"planner": 0}
    fake_exec = _FakeToolExecutor()

    async def _fake_planner(payload, *, parent_correlation_id=None, rpc_bus=None):
        calls["planner"] += 1
        if calls["planner"] == 1:
            return {
                "status": "ok",
                "trace": [
                    {
                        "step_index": 0,
                        "thought": "Gather missing facts.",
                        "action": {"tool_id": "gather_info", "input": {"request": "hello"}},
                        "observation": None,
                    }
                ],
            }
        return {"status": "ok", "final_answer": {"content": "done", "structured": {}}}

    monkeypatch.setattr(agent_api, "call_planner_react", _fake_planner)
    monkeypatch.setattr(agent_api, "ToolExecutor", lambda *_a, **_k: fake_exec)
    tools = [
        ToolDef(tool_id="plan_action", description="plan", input_schema={}, output_schema={}),
        ToolDef(tool_id="finalize_response", description="finalize", input_schema={}, output_schema={}),
    ]
    monkeypatch.setattr(agent_api, "_resolve_tools", lambda body, output_mode=None: (tools, ["executive_pack"]))

    req = AgentChainRequest(text="hello", mode="agent", messages=[{"role": "user", "content": "hello"}])
    out = asyncio.run(agent_api.execute_agent_chain(req, correlation_id=str(uuid4()), rpc_bus=object()))

    assert out.text == "done"
    assert fake_exec.calls[0][0] == "plan_action"
    assert out.runtime_debug["invalid_tool_remap_count"] == 1
    assert out.runtime_debug["invalid_tool_last"]["requested"] == "gather_info"
    assert out.runtime_debug["invalid_tool_last"]["resolved"] == "plan_action"
