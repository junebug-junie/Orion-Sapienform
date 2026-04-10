from __future__ import annotations

import asyncio
from uuid import uuid4

from app import api as agent_api
from orion.schemas.agents.bound_capability import (
    BoundCapabilityExecutionRequestV1,
    CapabilityRecoveryDecisionV1,
    CapabilityRecoveryReasonV1,
)
from orion.schemas.agents.schemas import AgentChainRequest, ToolDef


class _FakeCapabilityExecutor:
    def __init__(self, *_args, **_kwargs):
        self.calls: list[tuple[str, dict, str | None]] = []
        self._result = {
            "selected_skill": "skills.docker.prune_stopped_containers.v1",
            "selected_skill_family": "runtime_housekeeping",
            "execution_summary": "Dry-run cleanup completed",
            "capability_decision": {"selected_skill": "skills.docker.prune_stopped_containers.v1"},
        }

    async def execute_llm_verb(self, tool_id, tool_input, *, parent_correlation_id=None):
        self.calls.append((tool_id, tool_input, parent_correlation_id))
        return dict(self._result)


class _FakeCapabilityExecutorNoSkill(_FakeCapabilityExecutor):
    def __init__(self, *_args, **_kwargs):
        super().__init__()
        self._result = {
            "selected_skill": None,
            "selected_skill_family": "runtime_housekeeping",
            "execution_summary": "No skill",
            "capability_decision": {"selected_skill": None},
        }


async def _planner_returns_housekeep(_payload, *, parent_correlation_id=None, rpc_bus=None):
    return {
        "status": "ok",
        "trace": [
            {
                "step_index": 0,
                "thought": "execute runtime cleanup",
                "action": {"tool_id": "housekeep_runtime", "input": {"text": "dry-run cleanup"}},
                "observation": None,
            }
        ],
    }


def _tools():
    return [
        ToolDef(
            tool_id="housekeep_runtime",
            description="Runtime housekeeping",
            input_schema={},
            output_schema={},
            execution_mode="capability_backed",
            requires_capability_selector=True,
        ),
        ToolDef(tool_id="finalize_response", description="Finalize", input_schema={}, output_schema={}),
    ]


def test_planner_selected_capability_routes_to_bound_terminal_success(monkeypatch):
    fake_exec = _FakeCapabilityExecutor()
    monkeypatch.setattr(agent_api, "ToolExecutor", lambda *_a, **_k: fake_exec)
    monkeypatch.setattr(agent_api, "call_planner_react", _planner_returns_housekeep)
    monkeypatch.setattr(agent_api, "_resolve_tools", lambda body, output_mode=None: (_tools(), ["executive_pack"]))

    req = AgentChainRequest(text="dry-run cleanup of stopped containers", mode="agent", messages=[{"role": "user", "content": "cleanup"}])
    out = asyncio.run(agent_api.execute_agent_chain(req, correlation_id=str(uuid4()), rpc_bus=object()))

    assert fake_exec.calls and fake_exec.calls[0][0] == "housekeep_runtime"
    assert out.structured["finalization_reason"] == "bound_capability_execution"
    assert out.runtime_debug["bound_capability_terminal_path"] == "bound_direct_success"


def test_planner_selected_capability_fail_closed_terminal_reply(monkeypatch):
    fake_exec = _FakeCapabilityExecutorNoSkill()
    monkeypatch.setattr(agent_api, "ToolExecutor", lambda *_a, **_k: fake_exec)
    monkeypatch.setattr(agent_api, "call_planner_react", _planner_returns_housekeep)
    monkeypatch.setattr(agent_api, "_resolve_tools", lambda body, output_mode=None: (_tools(), ["executive_pack"]))

    req = AgentChainRequest(text="dry-run cleanup of stopped containers", mode="agent", messages=[{"role": "user", "content": "cleanup"}])
    out = asyncio.run(agent_api.execute_agent_chain(req, correlation_id=str(uuid4()), rpc_bus=object()))

    assert out.structured["finalization_reason"] == "bound_capability_fail_closed"
    assert out.runtime_debug["bound_capability_terminal_path"] == "bound_direct_no_compatible_capability"


def test_bound_and_planner_selected_paths_have_terminalization_parity(monkeypatch):
    fake_exec = _FakeCapabilityExecutor()
    monkeypatch.setattr(agent_api, "ToolExecutor", lambda *_a, **_k: fake_exec)
    monkeypatch.setattr(agent_api, "call_planner_react", _planner_returns_housekeep)
    monkeypatch.setattr(agent_api, "_resolve_tools", lambda body, output_mode=None: (_tools(), ["executive_pack"]))

    bound_req = AgentChainRequest(
        text="dry-run cleanup",
        mode="agent",
        tools=[t.model_dump() for t in _tools()],
        bound_capability_execution=BoundCapabilityExecutionRequestV1(
            selected_verb="housekeep_runtime",
            normalized_action_input={"text": "dry-run cleanup"},
            recovery=CapabilityRecoveryDecisionV1(
                reason=CapabilityRecoveryReasonV1.internal_contract_error,
                allow_replan=False,
                replanned=False,
            ),
        ),
    )
    planner_req = AgentChainRequest(text="dry-run cleanup", mode="agent", messages=[{"role": "user", "content": "cleanup"}])

    out_bound = asyncio.run(agent_api.execute_agent_chain(bound_req, correlation_id=str(uuid4()), rpc_bus=object()))
    out_planner = asyncio.run(agent_api.execute_agent_chain(planner_req, correlation_id=str(uuid4()), rpc_bus=object()))

    assert out_bound.structured["finalization_reason"] == "bound_capability_execution"
    assert out_planner.structured["finalization_reason"] == "bound_capability_execution"
    assert out_bound.runtime_debug["bound_capability_terminal_path"] == "bound_direct_success"
    assert out_planner.runtime_debug["bound_capability_terminal_path"] == "bound_direct_success"
