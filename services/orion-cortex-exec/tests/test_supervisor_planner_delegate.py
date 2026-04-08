import asyncio
import unittest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.agents.schemas import FinalAnswer, ToolDef
from orion.schemas.cortex.schemas import ExecutionPlan, StepExecutionResult

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.supervisor import Supervisor, _ensure_agent_chain_tool, _extract_latest_planner_thought, _verb_to_step


class StubSupervisor(Supervisor):
    def __init__(self, planner_outputs, action_step_output=None, action_outputs=None):
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
        self.executed_tool_ids = []
        self.action_outputs = action_outputs or {}

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
        action = kwargs.get("action") or {}
        tool_id = action.get("tool_id")
        self.executed_tool_ids.append(tool_id)
        if tool_id in self.action_outputs:
            return self.action_outputs[tool_id]
        return self.action_step_output

    async def _agent_chain_escalation(self, **kwargs):
        self.executed_actions += 1
        self.executed_tool_ids.append("agent_chain")
        if "agent_chain" in self.action_outputs:
            return self.action_outputs["agent_chain"]
        return self.action_step_output


class CapabilityRoutingSupervisor(Supervisor):
    def __init__(self):
        super().__init__(MagicMock())
        self.chain_calls = 0

    def _toolset(self, packs=None, tags=None):
        return [ToolDef(tool_id="housekeep_runtime", description="cap", input_schema={}, output_schema={})]

    async def _planner_step(self, **kwargs):
        step = StepExecutionResult(
            status="success",
            verb_name="planner",
            step_name="planner_react",
            order=0,
            result={
                "PlannerReactService": {
                    "status": "ok",
                    "trace": [{"thought": "run cleanup", "action": {"tool_id": "housekeep_runtime", "input": {"text": "dry run"}}}],
                    "stop_reason": "delegate",
                }
            },
            latency_ms=1,
            node="test",
            logs=["ok"],
            error=None,
        )
        return None, step, None, {"tool_id": "housekeep_runtime", "input": {"text": "dry run"}}

    async def _agent_chain_escalation(self, **kwargs):
        self.chain_calls += 1
        return StepExecutionResult(
            status="success",
            verb_name="agent_chain",
            step_name="agent_chain",
            order=100,
            result={"AgentChainService": {"text": "dry-run cleanup complete", "runtime_debug": {"bound_execution_completed": True}}},
            latency_ms=1,
            node="test",
            logs=["ok"],
            error=None,
        )


class OperationalPreferenceSupervisor(Supervisor):
    def __init__(self):
        super().__init__(MagicMock())
        self.executed_tool_ids = []

    def _toolset(self, packs=None, tags=None):
        return [
            ToolDef(
                tool_id="housekeep_runtime",
                description="runtime cleanup semantic tool",
                input_schema={},
                output_schema={},
                execution_mode="capability_backed",
                side_effect_level="low",
            )
        ]

    async def _planner_step(self, **kwargs):
        step = StepExecutionResult(
            status="success",
            verb_name="planner",
            step_name="planner_react",
            order=0,
            result={"PlannerReactService": {"status": "ok", "trace": [{"thought": "answer directly", "action": None}], "stop_reason": "final_answer"}},
            latency_ms=1,
            node="test",
            logs=["ok"],
            error=None,
        )
        return None, step, FinalAnswer(content="Run: docker container prune --force"), None

    async def _execute_action(self, **kwargs):
        action = kwargs.get("action") or {}
        tool_id = action.get("tool_id")
        self.executed_tool_ids.append(tool_id)
        if tool_id == "housekeep_runtime":
            return StepExecutionResult(
                status="success",
                verb_name="housekeep_runtime",
                step_name="agent_chain",
                order=1,
                result={"AgentChainService": {"text": "dry-run cleanup complete"}},
                latency_ms=1,
                node="test",
                logs=["ok"],
                error=None,
            )
        return StepExecutionResult(
            status="success",
            verb_name="agent_chain",
            step_name="agent_chain",
            order=2,
            result={"AgentChainService": {"text": "chain-result"}},
            latency_ms=1,
            node="test",
            logs=["ok"],
            error=None,
        )

    async def _agent_chain_escalation(self, **kwargs):
        self.executed_tool_ids.append("agent_chain")
        return StepExecutionResult(
            status="success",
            verb_name="agent_chain",
            step_name="agent_chain",
            order=100,
            result={"AgentChainService": {"text": "agent-chain-fallback"}},
            latency_ms=1,
            node="test",
            logs=["ok"],
            error=None,
        )


class TestSupervisorPlannerDelegate(unittest.TestCase):
    def _run(self, planner_outputs, action_step_output=None, action_outputs=None):
        supervisor = StubSupervisor(planner_outputs, action_step_output=action_step_output, action_outputs=action_outputs)
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


    def test_ensure_agent_chain_tool_added_to_allowlist(self):
        tools = [ToolDef(tool_id="analyze_text", description="analyze", input_schema={}, output_schema={})]
        updated = _ensure_agent_chain_tool(tools)
        self.assertIn("agent_chain", [t.tool_id for t in updated])

    def test_planner_non_chain_action_then_runs_agent_chain(self):
        analyze_step = StepExecutionResult(
            status="success",
            verb_name="analyze_text",
            step_name="analyze_text",
            order=10,
            result={"AnalyzeService": {"text": "analysis"}},
            latency_ms=1,
            node="test",
            logs=["ok"],
            error=None,
        )
        chain_step = StepExecutionResult(
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
        supervisor, result = self._run(
            planner_outputs=[(None, {"tool_id": "analyze_text", "input": {}}, "delegate")],
            action_outputs={"analyze_text": analyze_step, "agent_chain": chain_step},
        )
        self.assertEqual(supervisor.executed_actions, 2)
        self.assertEqual(supervisor.executed_tool_ids, ["analyze_text", "agent_chain"])
        self.assertEqual(result.final_text, "agent-result")
        step_names = [s.step_name for s in result.steps]
        self.assertIn("analyze_text", step_names)
        self.assertIn("agent_chain", step_names)

    def test_capability_backed_action_routes_to_bound_agent_chain_without_llm_fallback(self):
        supervisor = CapabilityRoutingSupervisor()
        source = ServiceRef(name="test", node="test", version="1.0")
        req = ExecutionPlan(verb_name="chat", steps=[], metadata={"mode": "agent"})
        ctx = {"mode": "agent", "messages": [{"role": "user", "content": "dry-run cleanup"}], "max_steps": 1}

        async_mock = AsyncMock(side_effect=AssertionError("call_step_services should not execute for capability-backed verbs"))
        with patch("app.supervisor.is_active", return_value=True), patch("app.supervisor.call_step_services", async_mock):
            result = asyncio.run(
                supervisor.execute(
                    source=source,
                    req=req,
                    correlation_id="corr-cap",
                    ctx=ctx,
                    recall_cfg={"enabled": False},
                )
            )

        self.assertEqual(supervisor.chain_calls, 1)
        self.assertEqual(result.final_text, "dry-run cleanup complete")
        self.assertEqual(async_mock.await_count, 0)


    def test_no_write_blocks_bound_capability_pre_dispatch(self):
        supervisor = CapabilityRoutingSupervisor()
        source = ServiceRef(name="test", node="test", version="1.0")
        req = ExecutionPlan(verb_name="chat", steps=[], metadata={"mode": "agent"})
        ctx = {
            "mode": "agent",
            "messages": [{"role": "user", "content": "dry-run cleanup"}],
            "max_steps": 1,
            "options": {"no_write_active": True, "tool_execution_policy": "none", "action_execution_policy": "none"},
        }

        result = asyncio.run(
            supervisor.execute(
                source=source,
                req=req,
                correlation_id="corr-cap-no-write",
                ctx=ctx,
                recall_cfg={"enabled": False},
            )
        )

        self.assertEqual(supervisor.chain_calls, 0)
        self.assertIn("Execution is disabled in this session", result.final_text or "")
        blocked_step = next((s for s in result.steps if s.step_name == "bound_capability_pre_dispatch_blocked"), None)
        self.assertIsNotNone(blocked_step)
        payload = (blocked_step.result or {}).get("AgentChainService", {})
        bound = payload.get("bound_capability", {})
        self.assertEqual(bound.get("path"), "blocked_pre_dispatch")
        self.assertTrue(bound.get("dispatch_skipped"))

    def test_capability_backed_verb_cannot_fall_back_to_llm_step(self):
        cfg = CapabilityRoutingSupervisor().registry.get("housekeep_runtime")
        with self.assertRaises(RuntimeError):
            _verb_to_step(cfg)

    def test_operational_intent_prefers_semantic_tool_over_planner_final_answer(self):
        supervisor = OperationalPreferenceSupervisor()
        source = ServiceRef(name="test", node="test", version="1.0")
        req = ExecutionPlan(verb_name="chat", steps=[], metadata={"mode": "agent"})
        ctx = {
            "mode": "agent",
            "messages": [{"role": "user", "content": "would you please dry-run cleanup of stopped containers"}],
            "max_steps": 1,
            "options": {},
        }
        result = asyncio.run(
            supervisor.execute(
                source=source,
                req=req,
                correlation_id="corr-operational-prefer",
                ctx=ctx,
                recall_cfg={"enabled": False},
            )
        )
        self.assertIn("housekeep_runtime", supervisor.executed_tool_ids)
        self.assertIn("runtime_policy", result.metadata)
        self.assertTrue(result.metadata["runtime_policy"]["semantic_tool_preferred"])

    def test_no_write_explicit_downgrade_for_operational_request(self):
        supervisor = OperationalPreferenceSupervisor()
        source = ServiceRef(name="test", node="test", version="1.0")
        req = ExecutionPlan(verb_name="chat", steps=[], metadata={"mode": "agent"})
        ctx = {
            "mode": "agent",
            "messages": [{"role": "user", "content": "would you please dry-run cleanup of stopped containers"}],
            "max_steps": 1,
            "options": {"tool_execution_policy": "none", "no_write_active": True},
        }
        result = asyncio.run(
            supervisor.execute(
                source=source,
                req=req,
                correlation_id="corr-no-write",
                ctx=ctx,
                recall_cfg={"enabled": False},
            )
        )
        self.assertNotIn("housekeep_runtime", supervisor.executed_tool_ids)
        self.assertIn("Execution is disabled in this session", result.final_text or "")
        self.assertTrue(result.metadata["runtime_policy"]["no_write_active"])
        self.assertTrue(result.metadata["runtime_policy"]["downgraded_to_explanation"])

    def test_operational_guidance_blocked_without_validated_semantic_execution(self):
        supervisor = StubSupervisor(
            planner_outputs=[(FinalAnswer(content="```bash\ndocker rm -f $(docker ps -aq)\n```"), None, "final_answer")]
        )
        source = ServiceRef(name="test", node="test", version="1.0")
        req = ExecutionPlan(verb_name="chat", steps=[], metadata={"mode": "agent"})
        ctx = {
            "mode": "agent",
            "messages": [{"role": "user", "content": "cleanup stopped containers now"}],
            "max_steps": 1,
            "options": {},
        }
        result = asyncio.run(
            supervisor.execute(
                source=source,
                req=req,
                correlation_id="corr-guidance-block",
                ctx=ctx,
                recall_cfg={"enabled": False},
            )
        )
        self.assertIn("safe preview", (result.final_text or "").lower())

    def test_supervisor_prefers_effective_packs_from_context_metadata(self):
        supervisor = StubSupervisor(planner_outputs=[(FinalAnswer(content="hello"), None, "final_answer")])
        source = ServiceRef(name="test", node="test", version="1.0")
        req = ExecutionPlan(verb_name="chat", steps=[], metadata={"mode": "agent", "packs": "[\"executive_pack\",\"delivery_pack\"]"})
        ctx = {
            "mode": "agent",
            "messages": [{"role": "user", "content": "write a delivery guide"}],
            "packs": ["executive_pack"],
            "metadata": {"packs": ["executive_pack", "delivery_pack"]},
            "max_steps": 1,
        }

        asyncio.run(
            supervisor.execute(
                source=source,
                req=req,
                correlation_id="corr-packs",
                ctx=ctx,
                recall_cfg={"enabled": False},
            )
        )

        self.assertIn("delivery_pack", ctx["packs"])

    def test_grounding_guardrail_rejects_unrelated_scaffolding_answer(self):
        supervisor = StubSupervisor(
            planner_outputs=[(FinalAnswer(content="You are Orion.\nReturn your normal council output contract.\nUse Orion service setup workflow."), None, "final_answer")]
        )
        source = ServiceRef(name="test", node="test", version="1.0")
        req = ExecutionPlan(verb_name="chat", steps=[], metadata={"mode": "agent"})
        ctx = {
            "mode": "agent",
            "raw_user_text": "What runtime can I expect for V100s on an APC UPS battery backup?",
            "messages": [
                {"role": "user", "content": "What runtime can I expect for V100s on an APC UPS battery backup?"},
                {"role": "assistant", "content": "old unrelated Orion setup answer"},
            ],
            "prior_step_results": [
                {"step_name": "write_recommendation", "result": {"text": "Orion service setup playbook"}}
            ],
            "max_steps": 1,
        }

        result = asyncio.run(
            supervisor.execute(
                source=source,
                req=req,
                correlation_id="corr-grounding",
                ctx=ctx,
                recall_cfg={"enabled": True},
            )
        )
        assert result.final_text is not None
        self.assertIn("I may have drifted from your request", result.final_text)
        self.assertIn("V100s", result.final_text)
        self.assertNotIn("Return your normal council output contract", result.final_text)

    def test_grounding_guardrail_allows_direct_followup_answer_when_recent_thread_is_coherent(self):
        supervisor = StubSupervisor(
            planner_outputs=[(FinalAnswer(content="Given your Docker compose setup, start with 2 replicas and raise CPU limits."), None, "final_answer")]
        )
        source = ServiceRef(name="test", node="test", version="1.0")
        req = ExecutionPlan(verb_name="chat", steps=[], metadata={"mode": "agent"})
        ctx = {
            "mode": "agent",
            "raw_user_text": "yes",
            "messages": [
                {"role": "assistant", "content": "Do you want me to tune Docker compose replica counts or CPU limits first?"},
                {"role": "user", "content": "yes"},
            ],
            "prior_step_results": [
                {"step_name": "recall", "result": {"text": "hello Orion, good morning"}}
            ],
            "max_steps": 1,
        }

        result = asyncio.run(
            supervisor.execute(
                source=source,
                req=req,
                correlation_id="corr-followup",
                ctx=ctx,
                recall_cfg={"enabled": True},
            )
        )
        assert result.final_text is not None
        self.assertNotIn("I may have drifted from your request", result.final_text)

    def test_agent_mode_recall_defaults_to_chat_general_when_profile_is_inherited(self):
        supervisor = StubSupervisor(planner_outputs=[(FinalAnswer(content="ok"), None, "final_answer")])
        source = ServiceRef(name="test", node="test", version="1.0")
        req = ExecutionPlan(
            verb_name="chat",
            steps=[],
            metadata={"mode": "agent"},
        )
        captured = {}

        async def _fake_recall_step(*args, **kwargs):
            captured["profile"] = kwargs.get("recall_profile")
            step = StepExecutionResult(
                status="success",
                verb_name="chat",
                step_name="recall",
                order=-1,
                result={"RecallService": {"count": 1}},
                latency_ms=1,
                node="test",
                logs=[],
                error=None,
            )
            return step, {"count": 1}, ""

        with patch("app.supervisor.run_recall_step", new=AsyncMock(side_effect=_fake_recall_step)):
            asyncio.run(
                supervisor.execute(
                    source=source,
                    req=req,
                    correlation_id="corr-agent-recall-default",
                    ctx={
                        "mode": "agent",
                        "messages": [{"role": "user", "content": "yes"}],
                        "raw_user_text": "yes",
                        "max_steps": 1,
                    },
                    recall_cfg={"enabled": True, "required": True, "profile": "reflect.v1", "mode": "hybrid"},
                )
            )

        self.assertEqual(captured["profile"], "chat.general.v1")


if __name__ == "__main__":
    unittest.main()

class TestExtractLatestPlannerThought(unittest.TestCase):
    def test_returns_none_for_empty_trace(self):
        self.assertIsNone(_extract_latest_planner_thought({"PlannerReactService": {"trace": []}}))

    def test_returns_last_thought_when_available(self):
        thought = _extract_latest_planner_thought({"PlannerReactService": {"trace": [{"thought": "a"}, {"thought": "b"}]}})
        self.assertEqual(thought, "b")
