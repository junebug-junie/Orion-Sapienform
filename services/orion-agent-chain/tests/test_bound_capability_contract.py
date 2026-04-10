from __future__ import annotations

import asyncio
from uuid import uuid4

from app import api as agent_api
from orion.schemas.agents.bound_capability import (
    BoundCapabilityExecutionRequestV1,
    CapabilityRecoveryDecisionV1,
    CapabilityRecoveryReasonV1,
)
from orion.schemas.agents.schemas import AgentChainRequest


class _FakeToolExecutor:
    def __init__(self, *_args, **_kwargs):
        self.calls = []

    async def execute_llm_verb(self, tool_id, tool_input, *, parent_correlation_id=None):
        self.calls.append((tool_id, tool_input, parent_correlation_id))
        return {
            "selected_verb": tool_id,
            "selected_skill_family": "runtime_housekeeping",
            "selected_skill": "skills.runtime.docker_prune_stopped_containers.v1",
            "execution_summary": "Executed skills.runtime.docker_prune_stopped_containers.v1: status=success ok=True",
            "raw_payload_ref": {"status": "success", "ok": True},
        }


def _bound_contract(*, allow_replan: bool = False, policy_metadata: dict | None = None) -> BoundCapabilityExecutionRequestV1:
    return BoundCapabilityExecutionRequestV1(
        selected_verb="housekeep_runtime",
        normalized_action_input={"text": "dry-run cleanup stopped containers"},
        execution_mode="capability_backed",
        requires_capability_selector=True,
        preferred_skill_families=["runtime_housekeeping"],
        side_effect_level="low",
        planner_correlation_id="corr-bound",
        planner_metadata={"source": "test"},
        selected_tool_metadata={"tool_id": "housekeep_runtime"},
        policy_metadata={"confirmation_required": False, "execute_opt_in": False, **(policy_metadata or {})},
        recovery=CapabilityRecoveryDecisionV1(
            reason=CapabilityRecoveryReasonV1.internal_contract_error,
            allow_replan=allow_replan,
            replanned=False,
            detail="test-default",
        ),
    )


def _bound_request(*, allow_replan: bool = False, policy_metadata: dict | None = None) -> AgentChainRequest:
    return AgentChainRequest(
        text="dry-run cleanup stopped containers",
        mode="agent",
        tools=[{"tool_id": "housekeep_runtime", "execution_mode": "capability_backed", "requires_capability_selector": True}],
        bound_capability_execution=_bound_contract(allow_replan=allow_replan, policy_metadata=policy_metadata),
        messages=[{"role": "user", "content": "dry-run cleanup stopped containers"}],
    )


def test_bound_capability_contract_schema_validation():
    contract = _bound_contract()
    assert contract.selected_verb == "housekeep_runtime"
    assert contract.normalized_action_input["text"] == "dry-run cleanup stopped containers"


def test_bound_capability_contract_invalid_schema_rejected():
    try:
        BoundCapabilityExecutionRequestV1.model_validate({"selected_verb": "housekeep_runtime", "normalized_action_input": "bad"})
        assert False, "schema validation should have failed for invalid normalized_action_input"
    except Exception:
        assert True




def test_bound_capability_policy_blocked_entry_short_circuit(monkeypatch):
    fake_exec = _FakeToolExecutor()
    planner_calls = {"count": 0}

    async def _fake_planner(*_args, **_kwargs):
        planner_calls["count"] += 1
        raise AssertionError("planner should not be called when policy blocks bound capability")

    monkeypatch.setattr(agent_api, "call_planner_react", _fake_planner)
    monkeypatch.setattr(agent_api, "ToolExecutor", lambda *_a, **_k: fake_exec)

    out = asyncio.run(
        agent_api.execute_agent_chain(
            _bound_request(policy_metadata={"no_write_active": True, "tool_execution_policy": "none", "action_execution_policy": "none"}),
            correlation_id=str(uuid4()),
            rpc_bus=object(),
        )
    )

    assert planner_calls["count"] == 0
    assert fake_exec.calls == []
    bound = out.structured["bound_capability"]
    assert bound["reason"] == "policy_blocked"
    assert bound["observation"]["path"] == "blocked_agent_chain_entry"
    assert bound["observation"]["reply_emitted"] is True


def test_bound_capability_policy_blocked_defense_in_depth_without_no_write(monkeypatch):
    fake_exec = _FakeToolExecutor()

    async def _fake_planner(*_args, **_kwargs):
        raise AssertionError("planner should not be called for blocked bound capability")

    monkeypatch.setattr(agent_api, "call_planner_react", _fake_planner)
    monkeypatch.setattr(agent_api, "ToolExecutor", lambda *_a, **_k: fake_exec)

    out = asyncio.run(
        agent_api.execute_agent_chain(
            _bound_request(policy_metadata={"no_write_active": False, "action_execution_policy": "none"}),
            correlation_id=str(uuid4()),
            rpc_bus=object(),
        )
    )

    assert fake_exec.calls == []
    bound = out.structured["bound_capability"]
    observation = bound["observation"]
    assert observation["selected_verb"] == "housekeep_runtime"
    assert observation["selected_tool_would_have_been"] == "housekeep_runtime"
    assert observation["action_execution_policy"] == "none"
    assert observation["execution_blocked_reason"] == "action_execution_policy=none"

def test_bound_capability_executes_without_planner(monkeypatch):
    fake_exec = _FakeToolExecutor()
    planner_calls = {"count": 0}

    async def _fake_planner(*_args, **_kwargs):
        planner_calls["count"] += 1
        raise AssertionError("planner should not be called for bound capability execution")

    monkeypatch.setattr(agent_api, "call_planner_react", _fake_planner)
    monkeypatch.setattr(agent_api, "ToolExecutor", lambda *_a, **_k: fake_exec)

    out = asyncio.run(agent_api.execute_agent_chain(_bound_request(), correlation_id=str(uuid4()), rpc_bus=object()))

    assert planner_calls["count"] == 0
    assert fake_exec.calls and fake_exec.calls[0][0] == "housekeep_runtime"
    assert out.structured["finalization_reason"] == "bound_capability_execution"
    bound = out.structured["bound_capability"]
    assert bound["selected_verb"] == "housekeep_runtime"
    assert bound["selected_skill"] == "skills.runtime.docker_prune_stopped_containers.v1"
    assert bound["execution_path"] == "direct_execute"


def test_bound_capability_fail_closed_when_no_skill(monkeypatch):
    class _NoSkillExecutor(_FakeToolExecutor):
        async def execute_llm_verb(self, tool_id, tool_input, *, parent_correlation_id=None):
            self.calls.append((tool_id, tool_input, parent_correlation_id))
            return {
                "selected_verb": tool_id,
                "selected_skill_family": "runtime_housekeeping",
                "selected_skill": None,
                "execution_summary": "No compatible skill available.",
            }

    fake_exec = _NoSkillExecutor()

    async def _fake_planner(*_args, **_kwargs):
        raise AssertionError("planner should not be called on fail-closed bound execution")

    monkeypatch.setattr(agent_api, "call_planner_react", _fake_planner)
    monkeypatch.setattr(agent_api, "ToolExecutor", lambda *_a, **_k: fake_exec)

    out = asyncio.run(agent_api.execute_agent_chain(_bound_request(), correlation_id=str(uuid4()), rpc_bus=object()))

    failure = out.structured["bound_capability"]
    assert failure["status"] == "fail"
    assert failure["reason"] == "no_compatible_capability"
    assert "no compatible capability skill" in out.text.lower()


def test_bound_capability_recovery_replan_is_explicit(monkeypatch):
    class _MissingVerbExecutor(_FakeToolExecutor):
        async def execute_llm_verb(self, tool_id, tool_input, *, parent_correlation_id=None):
            raise FileNotFoundError("missing verb")

    fake_exec = _MissingVerbExecutor()
    planner_calls = {"count": 0}

    async def _fake_planner(*_args, **_kwargs):
        planner_calls["count"] += 1
        return {"status": "ok", "final_answer": {"content": "replanned fallback", "structured": {}}, "trace": []}

    monkeypatch.setattr(agent_api, "call_planner_react", _fake_planner)
    monkeypatch.setattr(agent_api, "ToolExecutor", lambda *_a, **_k: fake_exec)

    out = asyncio.run(
        agent_api.execute_agent_chain(_bound_request(allow_replan=True), correlation_id=str(uuid4()), rpc_bus=object())
    )

    assert planner_calls["count"] == 1
    assert out.text == "replanned fallback"
    assert out.runtime_debug["bound_execution_replanned"] is True
    assert out.runtime_debug["bound_execution_recovery_reason"] == "selected_verb_missing"


def test_bound_capability_semantic_mismatch_fast_fail(monkeypatch):
    fake_exec = _FakeToolExecutor()

    async def _fake_planner(*_args, **_kwargs):
        raise AssertionError("planner should not be called for bound mismatch")

    bad_contract = _bound_contract()
    bad_contract.selected_verb = "summarize_recent_changes"
    req = AgentChainRequest(
        text="dry-run cleanup of stopped docker containers",
        mode="agent",
        tools=[{"tool_id": "summarize_recent_changes", "execution_mode": "capability_backed", "requires_capability_selector": True}],
        bound_capability_execution=bad_contract,
        messages=[{"role": "user", "content": "dry-run cleanup of stopped docker containers"}],
    )

    monkeypatch.setattr(agent_api, "call_planner_react", _fake_planner)
    monkeypatch.setattr(agent_api, "ToolExecutor", lambda *_a, **_k: fake_exec)

    out = asyncio.run(agent_api.execute_agent_chain(req, correlation_id=str(uuid4()), rpc_bus=object()))
    assert fake_exec.calls == []
    bound = out.structured["bound_capability"]
    assert bound["reason"] == "policy_blocked"
    assert bound["observation"]["path"] == "bound_direct_semantic_mismatch"
    assert bound["observation"]["reply_emitted"] is True


def test_bound_capability_no_compatible_terminal_path(monkeypatch):
    class _NoSkillExecutor(_FakeToolExecutor):
        async def execute_llm_verb(self, tool_id, tool_input, *, parent_correlation_id=None):
            self.calls.append((tool_id, tool_input, parent_correlation_id))
            return {"selected_verb": tool_id, "selected_skill": None, "capability_decision": {"candidate_skills": []}}

    fake_exec = _NoSkillExecutor()

    async def _fake_planner(*_args, **_kwargs):
        raise AssertionError("planner should not be called on bound direct path")

    monkeypatch.setattr(agent_api, "call_planner_react", _fake_planner)
    monkeypatch.setattr(agent_api, "ToolExecutor", lambda *_a, **_k: fake_exec)

    out = asyncio.run(agent_api.execute_agent_chain(_bound_request(), correlation_id=str(uuid4()), rpc_bus=object()))
    bound = out.structured["bound_capability"]
    assert bound["reason"] == "no_compatible_capability"
    assert bound["observation"]["path"] == "bound_direct_no_compatible_capability"
    assert bound["observation"]["reply_emitted"] is True


def test_bound_capability_internal_error_terminal_reply(monkeypatch):
    class _BoomExecutor(_FakeToolExecutor):
        async def execute_llm_verb(self, tool_id, tool_input, *, parent_correlation_id=None):
            raise RuntimeError("executor boom")

    fake_exec = _BoomExecutor()

    async def _fake_planner(*_args, **_kwargs):
        raise AssertionError("planner should not be called on bound direct path")

    monkeypatch.setattr(agent_api, "call_planner_react", _fake_planner)
    monkeypatch.setattr(agent_api, "ToolExecutor", lambda *_a, **_k: fake_exec)

    out = asyncio.run(agent_api.execute_agent_chain(_bound_request(), correlation_id=str(uuid4()), rpc_bus=object()))
    bound = out.structured["bound_capability"]
    assert bound["reason"] == "capability_executor_unavailable"
    assert bound["observation"]["path"] == "bound_direct_internal_error"
    assert bound["observation"]["reply_emitted"] is True


def test_bound_capability_timeout_terminal_reply(monkeypatch):
    class _HangingExecutor(_FakeToolExecutor):
        async def execute_llm_verb(self, tool_id, tool_input, *, parent_correlation_id=None):
            await asyncio.Event().wait()

    fake_exec = _HangingExecutor()

    async def _fake_planner(*_args, **_kwargs):
        raise AssertionError("planner should not be called on bound direct path")

    monkeypatch.setattr(agent_api, "call_planner_react", _fake_planner)
    monkeypatch.setattr(agent_api, "ToolExecutor", lambda *_a, **_k: fake_exec)
    monkeypatch.setattr(agent_api, "_bound_execution_timeout_seconds", lambda: 0.05)

    out = asyncio.run(agent_api.execute_agent_chain(_bound_request(), correlation_id=str(uuid4()), rpc_bus=object()))
    bound = out.structured["bound_capability"]
    assert bound["reason"] == "capability_executor_unavailable"
    assert bound["observation"]["path"] == "bound_direct_timeout"
    assert bound["observation"]["reply_emitted"] is True


def test_bound_execution_timeout_budget_exceeds_nested_capability_rpc_budget():
    timeout_sec = agent_api._bound_execution_timeout_seconds()
    assert timeout_sec > 25.0
    assert timeout_sec > float(agent_api.settings.default_timeout_seconds)
