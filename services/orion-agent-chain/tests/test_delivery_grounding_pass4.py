from __future__ import annotations

import asyncio
from uuid import uuid4

from app import api as agent_api
from orion.schemas.agents.schemas import AgentChainRequest, ToolDef


def test_finalize_response_prefers_trace_grounded_output(monkeypatch):
    class _Exec:
        def __init__(self):
            self.calls = []

        async def execute_llm_verb(self, tool_id, tool_input, *, parent_correlation_id=None):
            self.calls.append((tool_id, tool_input))
            return {"llm_output": "Final grounded answer"}

    async def _planner(payload, *, parent_correlation_id=None, rpc_bus=None):
        trace = payload.get("trace") or []
        if not trace:
            return {
                "status": "ok",
                "trace": [
                    {
                        "step_index": 0,
                        "thought": "draft guide",
                        "action": {"tool_id": "write_guide", "input": {"request": payload["context"]["external_facts"]["text"]}},
                        "observation": None,
                    }
                ],
            }
        return {
            "status": "ok",
            "trace": trace + [
                {
                    "step_index": 1,
                    "thought": "finalize now",
                    "action": {"tool_id": "finalize_response", "input": {"request": payload["context"]["external_facts"]["text"]}},
                    "observation": None,
                }
            ],
        }

    fake_exec = _Exec()
    monkeypatch.setattr(agent_api, "call_planner_react", _planner)
    monkeypatch.setattr(agent_api, "ToolExecutor", lambda *a, **k: fake_exec)
    monkeypatch.setattr(
        agent_api,
        "_resolve_tools",
        lambda body, output_mode=None: (
            [
                ToolDef(tool_id="write_guide", description="guide", input_schema={}, output_schema={}),
                ToolDef(tool_id="finalize_response", description="finalize", input_schema={}, output_schema={}),
            ],
            ["executive_pack", "delivery_pack"],
        ),
    )

    req = AgentChainRequest(
        text="would you write up detailed developer instructions on how to build Orion (AI) into a Discord server?",
        mode="agent",
        output_mode="implementation_guide",
    )
    out = asyncio.run(agent_api.execute_agent_chain(req, correlation_id=str(uuid4()), rpc_bus=object()))

    finalize_call = [call for call in fake_exec.calls if call[0] == "finalize_response"][-1]
    fin_input = finalize_call[1]
    assert fin_input["trace_preferred_output"] == "Final grounded answer"
    assert fin_input["delivery_grounding_mode"] == "orion_repo_architecture"
    assert out.runtime_debug.get("finalization_source_trace_used") is True


def test_generic_drift_final_answer_triggers_finalize_rewrite(monkeypatch):
    class _Exec:
        def __init__(self):
            self.calls = []

        async def execute_llm_verb(self, tool_id, tool_input, *, parent_correlation_id=None):
            self.calls.append(tool_id)
            return {"llm_output": "Use the Orion Discord bridge via Orch and Exec."}

    async def _planner(*args, **kwargs):
        return {
            "status": "ok",
            "final_answer": {
                "content": "Deploy a Flask app on Ubuntu with Gunicorn and Nginx.",
                "structured": {},
            },
        }

    fake_exec = _Exec()
    monkeypatch.setattr(agent_api, "call_planner_react", _planner)
    monkeypatch.setattr(agent_api, "ToolExecutor", lambda *a, **k: fake_exec)
    monkeypatch.setattr(
        agent_api,
        "_resolve_tools",
        lambda body, output_mode=None: ([ToolDef(tool_id="finalize_response", description="finalize", input_schema={}, output_schema={})], ["executive_pack", "delivery_pack"]),
    )

    req = AgentChainRequest(
        text="would you write up detailed developer instructions on how to build Orion (AI) into a Discord server?",
        mode="agent",
        output_mode="implementation_guide",
    )
    out = asyncio.run(agent_api.execute_agent_chain(req, correlation_id=str(uuid4()), rpc_bus=object()))

    assert "Orion Discord bridge" in out.text
    assert "finalize_response" in fake_exec.calls
    assert out.runtime_debug.get("generic_drift_detected") is True
