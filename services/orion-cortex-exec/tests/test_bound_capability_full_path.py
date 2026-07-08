from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
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
        return {"type": "message", "channel": reply_channel, "data": response.model_dump(mode="json")}


class _Codec:
    def decode(self, data: Any):
        if isinstance(data, dict):
            env = BaseEnvelope.model_validate(data)
        else:
            env = data
        return type("D", (), {"ok": True, "envelope": env, "error": None})


class _HarnessBus:
    def __init__(self):
        self.codec = _Codec()
        self.capability_hops: list[CapabilityHop] = []

    async def connect(self):
        return

    async def close(self):
        return

    async def rpc_request(self, channel: str, env: BaseEnvelope, *, reply_channel: str, timeout_sec: float = 60.0):
        rpc_bus = _CapabilityBridgeBus(self.capability_hops)
        return await rpc_bus.rpc_request(channel, env, reply_channel=reply_channel, timeout_sec=timeout_sec)


def test_bound_capability_full_runtime_path_emits_non_empty_result_without_timeout():
    bus = _HarnessBus()
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

    cap_step = next(step for step in result.steps if step.step_name == "bound_capability_execution")
    agent_payload = cap_step.result["ContextExecService"]
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
