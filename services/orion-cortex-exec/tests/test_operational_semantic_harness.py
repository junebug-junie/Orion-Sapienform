import asyncio
import unittest
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.codec import OrionCodec
from orion.core.bus.enforce import enforcer
from orion.schemas.agents.schemas import AgentChainRequest, AgentChainResult, PlannerRequest, PlannerResponse
from orion.schemas.cortex.schemas import ExecutionPlan

from app.supervisor import Supervisor


@dataclass
class HarnessEvent:
    channel: str
    kind: str
    correlation_id: str
    payload: dict[str, Any] | Any


class HarnessBus:
    def __init__(self):
        self.codec = OrionCodec()
        self.handlers: dict[str, Any] = {}
        self.events: list[HarnessEvent] = []

    async def connect(self):
        return

    async def close(self):
        return

    def register(self, channel: str, handler):
        self.handlers[channel] = handler

    async def publish(self, channel: str, msg: BaseEnvelope | dict[str, Any]):
        enforcer.validate(channel)
        env = msg if isinstance(msg, BaseEnvelope) else BaseEnvelope.model_validate(msg)
        self.events.append(HarnessEvent(channel=channel, kind=env.kind, correlation_id=str(env.correlation_id), payload=env.payload))

    async def rpc_request(self, request_channel: str, envelope: BaseEnvelope, *, reply_channel: str, timeout_sec: float = 60.0):
        enforcer.validate(request_channel)
        response_env = await self.handlers[request_channel](envelope)
        return {"type": "message", "channel": reply_channel, "data": self.codec.encode(response_env)}


class OperationalSemanticHarnessTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        enforcer.enforce = True
        self.bus = HarnessBus()
        self.supervisor = Supervisor(self.bus)

        async def planner_handler(env: BaseEnvelope) -> BaseEnvelope:
            req = PlannerRequest.model_validate(env.payload)
            planner_resp = PlannerResponse(
                request_id=req.request_id,
                status="ok",
                stop_reason="delegate",
                trace=[{"step_index": 0, "thought": "generic-first", "action": {"tool_id": "generate_code_scaffold", "input": {"request": req.goal}}, "observation": {}}],
                usage={"steps": 1, "tokens_reason": 0, "tokens_answer": 0, "tools_called": ["generate_code_scaffold"], "duration_ms": 1},
            )
            return BaseEnvelope(kind="agent.planner.result", source=ServiceRef(name="planner-react", version="test", node="test"), correlation_id=env.correlation_id, payload=planner_resp.model_dump(mode="json"))

        async def agent_handler(env: BaseEnvelope) -> BaseEnvelope:
            req = AgentChainRequest.model_validate(env.payload)
            selected = req.bound_capability_execution.selected_verb if req.bound_capability_execution else "unknown"
            result = AgentChainResult(
                mode=req.mode,
                text=f"executed:{selected}",
                structured={"bound_capability": {"selected_verb": selected, "path": "bound_direct_success"}},
                planner_raw={"status": "ok"},
                runtime_debug={"bound_capability_terminal_path": "bound_direct_success"},
            )
            return BaseEnvelope(kind="agent.chain.result", source=ServiceRef(name="agent-chain", version="test", node="test"), correlation_id=env.correlation_id, payload=result.model_dump(mode="json"))

        self.bus.register("orion:exec:request:PlannerReactService", planner_handler)
        self.bus.register("orion:exec:request:AgentChainService", agent_handler)

    async def _run(self, ask: str):
        corr = str(uuid4())
        req = ExecutionPlan(verb_name="chat_general", steps=[], metadata={"mode": "agent"})
        ctx = {
            "mode": "agent",
            "messages": [{"role": "user", "content": ask}],
            "output_mode": "direct_answer",
            "response_profile": "direct_answer",
            "packs": ["executive_pack"],
            "options": {},
            "trace_id": corr,
            "session_id": "harness",
            "user_id": "harness",
        }
        recall_cfg = {"enabled": False, "required": False}
        return await self.supervisor.execute(
            source=ServiceRef(name="test", version="0", node="n"),
            req=req,
            correlation_id=corr,
            ctx=ctx,
            recall_cfg=recall_cfg,
        )

    async def test_operational_asks_commit_to_canonical_verbs(self):
        cases = {
            "Which nodes are up right now?": "assess_mesh_presence",
            "Check disk health across active nodes.": "assess_storage_health",
            "Summarize recent PR changes.": "summarize_recent_changes",
            "Dry-run cleanup of stopped containers.": "housekeep_runtime",
        }
        for ask, expected in cases.items():
            res = await self._run(ask)
            agent_step = next(step for step in res.steps if step.step_name == "agent_chain")
            selected = agent_step.result["AgentChainService"]["structured"]["bound_capability"]["selected_verb"]
            self.assertEqual(selected, expected)

    async def test_truthful_failure_is_preserved(self):
        async def failing_agent(env: BaseEnvelope) -> BaseEnvelope:
            req = AgentChainRequest.model_validate(env.payload)
            selected = req.bound_capability_execution.selected_verb if req.bound_capability_execution else "unknown"
            result = AgentChainResult(
                mode=req.mode,
                text="Bound capability execution failed: induced timeout",
                structured={"bound_capability": {"selected_verb": selected, "path": "bound_direct_timeout", "reason": "capability_executor_unavailable"}},
                planner_raw={"status": "error"},
                runtime_debug={"bound_capability_terminal_path": "bound_direct_timeout"},
            )
            return BaseEnvelope(kind="agent.chain.result", source=ServiceRef(name="agent-chain", version="test", node="test"), correlation_id=env.correlation_id, payload=result.model_dump(mode="json"))

        self.bus.register("orion:exec:request:AgentChainService", failing_agent)
        res = await self._run("Dry-run cleanup of stopped containers.")
        self.assertIn("Bound capability execution failed", res.final_text or "")


if __name__ == "__main__":
    unittest.main()
