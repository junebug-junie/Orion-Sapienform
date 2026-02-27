import asyncio
import unittest
import sys
from pathlib import Path
from unittest.mock import MagicMock

from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.agents.schemas import FinalAnswer, ToolDef
from orion.schemas.cortex.schemas import ExecutionPlan, StepExecutionResult

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.supervisor import Supervisor


class StubSupervisor(Supervisor):
    def __init__(self, planner_outputs, action_step_output=None):
        super().__init__(MagicMock())
        self._planner_outputs = list(planner_outputs)
        self.action_step_output = action_step_output or StepExecutionResult(
            status="success",
            verb_name="agent_chain",
            step_name="agent_chain",
            order=100,
            result={"AgentChainService": {"text": "agent-result"}},
            latency_ms=1,
            node="test",
            logs=["ok"],
            error=None,
        )
        self.executed_actions = 0

    def _toolset(self, packs=None, tags=None):
        return [ToolDef(tool_id="agent_chain", description="delegate", input_schema={}, output_schema={})]

    async def _planner_step(self, **kwargs):
        planner_final, action, stop_reason = self._planner_outputs.pop(0)
        planner_result = {
            "PlannerReactService": {
                "status": "ok",
                "trace": [{"thought": "test-thought", "action": action}],
                "stop_reason": stop_reason,
            }
        }
        step = StepExecutionResult(
            status="success",
            verb_name="planner",
            step_name="planner_react",
            order=0,
            result=planner_result,
            latency_ms=1,
            node="test",
            logs=["ok"],
            error=None,
        )
        return None, step, planner_final, action

    async def _execute_action(self, **kwargs):
        self.executed_actions += 1
        return self.action_step_output


class TestSupervisorPlannerDelegate(unittest.TestCase):
    def _run(self, planner_outputs, action_step_output=None):
        supervisor = StubSupervisor(planner_outputs, action_step_output=action_step_output)
        source = ServiceRef(name="test", node="test", version="1.0")
        req = ExecutionPlan(verb_name="chat", steps=[], metadata={"mode": "agent"})
        ctx = {"mode": "agent", "messages": [{"role": "user", "content": "help"}], "max_steps": 1}
        result = asyncio.run(
            supervisor.execute(
                source=source,
                req=req,
                correlation_id="corr-test",
                ctx=ctx,
                recall_cfg={"enabled": False},
            )
        )
        return supervisor, result

    def test_planner_final_only_stops_without_action_dispatch(self):
        supervisor, result = self._run(
            planner_outputs=[(FinalAnswer(content="hello"), None, "final_answer")]
        )
        self.assertEqual(supervisor.executed_actions, 0)
        self.assertEqual(result.final_text, "hello")
        step_names = [s.step_name for s in result.steps]
        self.assertEqual(step_names.count("planner_react"), 1)
        self.assertEqual(step_names.count("agent_chain"), 0)

    def test_planner_action_only_continues_and_runs_agent_chain(self):
        supervisor, result = self._run(
            planner_outputs=[(None, {"tool_id": "agent_chain", "input": {}}, "delegate")]
        )
        self.assertEqual(supervisor.executed_actions, 1)
        step_names = [s.step_name for s in result.steps]
        self.assertEqual(step_names.count("agent_chain"), 1)
        self.assertEqual(result.final_text, "agent-result")

    def test_planner_final_and_action_runs_agent_chain_and_preserves_final(self):
        action_step_output = StepExecutionResult(
            status="success",
            verb_name="agent_chain",
            step_name="agent_chain",
            order=100,
            result={"AgentChainService": {"text": ""}},
            latency_ms=1,
            node="test",
            logs=["ok"],
            error=None,
        )
        supervisor, result = self._run(
            planner_outputs=[(FinalAnswer(content="hello"), {"tool_id": "agent_chain", "input": {}}, "delegate")],
            action_step_output=action_step_output,
        )
        self.assertEqual(supervisor.executed_actions, 1)
        self.assertEqual(result.final_text, "hello")
        step_names = [s.step_name for s in result.steps]
        self.assertEqual(step_names.count("planner_react"), 1)
        self.assertEqual(step_names.count("agent_chain"), 1)


if __name__ == "__main__":
    unittest.main()
