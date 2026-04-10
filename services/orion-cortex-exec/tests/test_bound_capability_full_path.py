from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.agents.schemas import AgentChainRequest, PlannerRequest, PlannerResponse
from orion.schemas.cortex.contracts import CortexClientResult
from orion.schemas.cortex.schemas import ExecutionPlan

from app.supervisor import Supervisor


@dataclass
class CapabilityHop:
    selected_verb: str
    selected_skill: str
    final_text: str


class _CapabilityBridgeBus:
    def __init__(self, hops: list[CapabilityHop]):
        self.hops = hops

    async def rpc_request(self, channel: str, env: BaseEnvelope, *, reply_channel: str, timeout_sec: float = 60.0):
        assert channel == "orion:cortex:request"
        payload = env.payload if isinstance(env.payload, dict) else {}
        metadata = payload.get("context", {}).get("metadata", {}) if isinstance(payload, dict) else {}
        selected_verb = str(metadata.get("requested_verb") or "")
        selected_skill = str(payload.get("verb") or "")
        assert selected_skill.startswith("skills.runtime.")

        concrete_final_text = "dry-run cleanup of stopped containers completed"
        self.hops.append(
            CapabilityHop(
                selected_verb=selected_verb,
                selected_skill=selected_skill,
                final_text=concrete_final_text,
            )
        )
        result = CortexClientResult(
            ok=True,
            mode="brain",
            verb=selected_skill,
            status="success",
            final_text=concrete_final_text,
            memory_used=False,
            recall_debug={},
            steps=[],
            metadata={"concrete_skill": selected_skill},
        )
        response = BaseEnvelope(
            kind="cortex.orch.result",
            source=ServiceRef(name="cortex-orch", version="test", node="test"),
            correlation_id=env.correlation_id,
            payload=result.model_dump(mode="json"),
        )
        return {"type": "message", "channel": reply_channel, "data": env.model_copy(update={"payload": response.payload})}


class _Codec:
    def decode(self, data: Any):
        return type("D", (), {"ok": True, "envelope": BaseEnvelope(kind="cortex.orch.result", source=ServiceRef(name="cortex-orch"), correlation_id=str(uuid4()), payload=data), "error": None})


class _HarnessBus:
    def __init__(self):
        self.codec = _Codec()
        self.handlers: dict[str, Any] = {}
        self.capability_hops: list[CapabilityHop] = []

    async def connect(self):
        return

    async def close(self):
        return

    def register(self, channel: str, handler):
        self.handlers[channel] = handler

    async def publish(self, channel: str, msg: BaseEnvelope | dict[str, Any]):
        return

    async def rpc_request(self, request_channel: str, envelope: BaseEnvelope, *, reply_channel: str, timeout_sec: float = 60.0):
        response_env = await self.handlers[request_channel](envelope)
        return {"type": "message", "channel": reply_channel, "data": response_env.payload}


async def _agent_handler(env: BaseEnvelope, *, capability_hops: list[CapabilityHop]) -> BaseEnvelope:
    req = AgentChainRequest.model_validate(env.payload)
    selected_verb = req.bound_capability_execution.selected_verb if req.bound_capability_execution else "housekeep_runtime"
    rpc_bus = _CapabilityBridgeBus(capability_hops)
    bridge_env = BaseEnvelope(
        kind="cortex.orch.request",
        source=ServiceRef(name="agent-chain", version="test", node="test"),
        correlation_id=env.correlation_id,
        payload={
            "verb": "skills.runtime.docker_prune_stopped_containers.v1",
            "context": {"metadata": {"requested_verb": selected_verb}},
        },
    )
    bridge_msg = await rpc_bus.rpc_request("orion:cortex:request", bridge_env, reply_channel=f"orion:cortex:result:{env.correlation_id}")
    bridge_payload = bridge_msg["data"].payload if isinstance(bridge_msg["data"], BaseEnvelope) else {}
    concrete_text = str((bridge_payload or {}).get("final_text") or "")
    result_payload = {
        "mode": req.mode,
        "text": concrete_text,
        "structured": {
            "finalization_reason": "bound_capability_execution",
            "bound_capability": {
                "selected_verb": selected_verb,
                "selected_skill": "skills.runtime.docker_prune_stopped_containers.v1",
                "execution_path": "direct_execute",
            },
        },
        "planner_raw": {"status": "ok"},
        "runtime_debug": {"bound_capability_terminal_path": "bound_direct_success"},
    }
    return BaseEnvelope(
        kind="agent.chain.result",
        source=ServiceRef(name="agent-chain", version="test", node="test"),
        correlation_id=env.correlation_id,
        payload=result_payload,
    )


def test_bound_capability_full_runtime_path_emits_non_empty_result_without_timeout():
    bus = _HarnessBus()

    async def planner_handler(env: BaseEnvelope) -> BaseEnvelope:
        req = PlannerRequest.model_validate(env.payload)
        planner_resp = PlannerResponse(
            request_id=req.request_id,
            status="ok",
            stop_reason="delegate",
            trace=[
                {
                    "step_index": 0,
                    "thought": "select semantic runtime tool",
                    "action": {"tool_id": "housekeep_runtime", "input": {"text": req.goal}},
                    "observation": {"selected_semantic_tool": "housekeep_runtime"},
                }
            ],
            usage={"steps": 1, "tokens_reason": 0, "tokens_answer": 0, "tools_called": ["housekeep_runtime"], "duration_ms": 1},
        )
        return BaseEnvelope(kind="agent.planner.result", source=ServiceRef(name="planner-react", version="test", node="test"), correlation_id=env.correlation_id, payload=planner_resp.model_dump(mode="json"))

    async def agent_handler(env: BaseEnvelope) -> BaseEnvelope:
        return await _agent_handler(env, capability_hops=bus.capability_hops)

    bus.register("orion:exec:request:PlannerReactService", planner_handler)
    bus.register("orion:exec:request:AgentChainService", agent_handler)

    supervisor = Supervisor(bus)

    corr = str(uuid4())
    req = ExecutionPlan(verb_name="chat_general", steps=[], metadata={"mode": "agent"})
    ctx = {
        "mode": "agent",
        "messages": [{"role": "user", "content": "Dry-run cleanup of stopped containers."}],
        "output_mode": "direct_answer",
        "response_profile": "direct_answer",
        "packs": ["executive_pack"],
        "options": {},
        "trace_id": corr,
        "session_id": "harness",
        "user_id": "harness",
    }

    result = asyncio.run(
        supervisor.execute(
            source=ServiceRef(name="cortex-orch", version="0", node="n"),
            req=req,
            correlation_id=corr,
            ctx=ctx,
            recall_cfg={"enabled": False, "required": False},
        )
    )

    agent_step = next(step for step in result.steps if step.step_name == "agent_chain")
    agent_payload = agent_step.result["AgentChainService"]
    bound = agent_payload["structured"]["bound_capability"]

    assert bound["selected_verb"] == "housekeep_runtime"
    assert bound["selected_skill"].startswith("skills.runtime.")
    assert bus.capability_hops
    assert bus.capability_hops[0].selected_verb == "housekeep_runtime"
    assert bus.capability_hops[0].selected_skill == bound["selected_skill"]
    assert bus.capability_hops[0].final_text
    assert (agent_payload.get("text") or "").strip()
    assert agent_payload["runtime_debug"]["bound_capability_terminal_path"] == "bound_direct_success"
    assert "timeout" not in (result.final_text or "").lower()
