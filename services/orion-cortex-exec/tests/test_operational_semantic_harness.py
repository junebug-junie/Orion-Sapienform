import asyncio
import unittest
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.codec import OrionCodec
from orion.core.bus.enforce import enforcer
from orion.schemas.cortex.contracts import CortexClientResult
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
        self.cortex_fail = False

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
        if request_channel not in self.handlers:
            raise RuntimeError(f"no handler for {request_channel}")
        response_env = await self.handlers[request_channel](envelope)
        return {"type": "message", "channel": reply_channel, "data": self.codec.encode(response_env)}


class OperationalSemanticHarnessTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        enforcer.enforce = True
        self.bus = HarnessBus()
        self.supervisor = Supervisor(self.bus)

        async def cortex_handler(env: BaseEnvelope) -> BaseEnvelope:
            payload = env.payload if isinstance(env.payload, dict) else {}
            metadata = payload.get("context", {}).get("metadata", {}) if isinstance(payload, dict) else {}
            selected_verb = str(metadata.get("requested_verb") or "")
            selected_skill = str(payload.get("verb") or "")
            if self.bus.cortex_fail:
                result = CortexClientResult(
                    ok=False,
                    mode="brain",
                    verb=selected_skill,
                    status="fail",
                    final_text="Bound capability execution failed: induced timeout",
                    memory_used=False,
                    recall_debug={},
                    steps=[],
                )
            else:
                result = CortexClientResult(
                    ok=True,
                    mode="brain",
                    verb=selected_skill,
                    status="success",
                    final_text=f"executed:{selected_verb}",
                    memory_used=False,
                    recall_debug={},
                    steps=[],
                )
            return BaseEnvelope(
                kind="cortex.orch.result",
                source=ServiceRef(name="cortex-orch", version="test", node="test"),
                correlation_id=env.correlation_id,
                payload=result.model_dump(mode="json"),
            )

        self.bus.register("orion:cortex:request", cortex_handler)

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
            cap_step = next(step for step in res.steps if step.step_name == "bound_capability_execution")
            selected = cap_step.result["ContextExecService"]["structured"]["bound_capability"]["selected_verb"]
            self.assertEqual(selected, expected)

    async def test_truthful_failure_is_preserved(self):
        self.bus.cortex_fail = True
        res = await self._run("Dry-run cleanup of stopped containers.")
        self.assertIn("Bound capability execution failed", res.final_text or "")


if __name__ == "__main__":
    unittest.main()
