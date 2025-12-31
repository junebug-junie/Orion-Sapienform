from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import orion

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import LLMMessage, ServiceRef
from orion.cognition.packs_loader import PackManager
from orion.cognition.planner.loader import VerbRegistry
from orion.schemas.agents.schemas import (
    ContextBlock,
    DeliberationRequest,
    FinalAnswer,
    Goal,
    Limits,
    PlannerRequest,
    Preferences,
    ToolDef,
    TraceStep,
    AgentChainRequest,
)
from orion.schemas.cortex.schemas import ExecutionPlan, ExecutionStep, PlanExecutionResult, StepExecutionResult

from .clients import AgentChainClient, CouncilClient, LLMGatewayClient, PlannerReactClient
from .executor import _last_user_message, call_step_services, run_recall_step
from .settings import settings

logger = logging.getLogger("orion.cortex.exec.supervisor")

ORION_PKG_DIR = Path(orion.__file__).resolve().parent
PROMPTS_DIR = ORION_PKG_DIR / "cognition" / "prompts"
VERBS_DIR = ORION_PKG_DIR / "cognition" / "verbs"


def _load_prompt_content(template_ref: Optional[str]) -> Optional[str]:
    if not template_ref:
        return None
    ref = template_ref.strip()
    if not ref.endswith(".j2"):
        return ref
    path = PROMPTS_DIR / ref
    if path.exists():
        return path.read_text(encoding="utf-8")
    logger.warning("Prompt template not found: %s", path)
    return ref


def _verb_to_step(cfg) -> ExecutionStep:
    prompt_text = _load_prompt_content(cfg.prompt_template)
    services = cfg.services or ["LLMGatewayService"]
    return ExecutionStep(
        verb_name=cfg.name,
        step_name=cfg.name,
        description=cfg.description or "",
        order=0,
        services=services,
        prompt_template=prompt_text,
        requires_gpu=bool(cfg.requires_gpu),
        requires_memory=bool(cfg.requires_memory),
        timeout_ms=int(cfg.timeout_ms or settings.step_timeout_ms),
    )


def _extract_observation(step_res: StepExecutionResult) -> Dict[str, Any]:
    for payload in (step_res.result or {}).values():
        if not isinstance(payload, dict):
            continue
        if payload.get("llm_output"):
            return {"llm_output": payload.get("llm_output"), "raw": payload}
        if payload.get("content"):
            return {"llm_output": payload.get("content"), "raw": payload}
        if payload.get("text"):
            return {"llm_output": payload.get("text"), "raw": payload}
    return {"raw": step_res.result}


class Supervisor:
    """
    Hierarchical controller that can pick reasoning paths, run ReAct with verbs, checkpoint with Council,
    and escalate to Agent Chain.
    """

    def __init__(self, bus: OrionBusAsync):
        self.bus = bus
        self.registry = VerbRegistry(VERBS_DIR)
        self.pack_manager = PackManager(ORION_PKG_DIR / "cognition")
        self.llm_client = LLMGatewayClient(bus)
        self.planner_client = PlannerReactClient(bus)
        self.agent_client = AgentChainClient(bus)
        self.council_client = CouncilClient(bus)

    def _toolset(self, packs: List[str] | None = None, tags: List[str] | None = None) -> List[ToolDef]:
        try:
            names: List[str] = []
            if packs:
                self.pack_manager.load_packs()
                names = self.pack_manager.load_verb_set(packs)
            verbs = self.registry.filter(names=names or None, tags=tags)
        except Exception as exc:
            logger.warning("Verb registry load failed, falling back to all verbs: %s", exc)
            verbs = self.registry.list()

        tools: List[ToolDef] = []
        for v in verbs:
            tools.append(
                ToolDef(
                    tool_id=v.name,
                    description=v.description or f"Orion verb: {v.name}",
                    input_schema=v.input_schema or {},
                    output_schema=v.output_schema or {},
                )
            )
        return tools

    def _should_use_react(self, mode: str, tools: List[ToolDef]) -> bool:
        if mode == "agent":
            return True
        return bool(tools)

    async def _planner_step(
        self,
        *,
        source: ServiceRef,
        goal_text: str,
        toolset: List[ToolDef],
        trace: List[TraceStep],
        ctx: Dict[str, Any],
        correlation_id: str,
        diagnostic: bool = False,
    ) -> Tuple[PlannerRequest, StepExecutionResult, FinalAnswer | None, Dict[str, Any] | None]:
        planner_req = PlannerRequest(
            request_id=correlation_id,
            caller="cortex-exec",
            goal=Goal(description=goal_text, metadata={"verb": ctx.get("verb")}),
            context=ContextBlock(
                conversation_history=[LLMMessage(**m) if not isinstance(m, LLMMessage) else m for m in (ctx.get("messages") or [])],
                external_facts={"text": ctx.get("memory_digest", "")},
            ),
            toolset=toolset,
            trace=trace,
            limits=Limits(max_steps=1, timeout_seconds=int(settings.step_timeout_ms / 1000)),
            preferences=Preferences(plan_only=True, return_trace=True, delegate_tool_execution=True),
        )
        reply_channel = f"{settings.exec_result_prefix}:PlannerReactService:{correlation_id}"
        t0 = time.time()
        logs = [f"rpc -> PlannerReactService (plan_only) reply={reply_channel}"]
        planner_timeout = float(
            ctx.get("planner_timeout_sec")
            or (settings.step_timeout_ms / 1000.0)
        )
        if diagnostic:
            planner_timeout = min(planner_timeout, float(settings.diagnostic_agent_timeout_sec))
        planner_res = await self.planner_client.plan(
            source=source,
            req=planner_req,
            correlation_id=correlation_id,
            reply_to=reply_channel,
            timeout_sec=planner_timeout,
        )
        logs.append(f"ok <- PlannerReactService status={planner_res.status}")

        step_res = StepExecutionResult(
            status="success" if planner_res.status == "ok" else "fail",
            verb_name=ctx.get("verb") or "unknown",
            step_name="planner_react",
            order=-1,
            result={"PlannerReactService": planner_res.model_dump(mode="json")},
            latency_ms=int((time.time() - t0) * 1000),
            node=settings.node_name,
            logs=logs,
            error=None if planner_res.status == "ok" else (planner_res.error or {}).get("message"),
        )
        final = planner_res.final_answer
        action: Dict[str, Any] | None = None
        if planner_res.trace:
            last = planner_res.trace[-1]
            action = last.action
        return planner_req, step_res, final, action

    async def _execute_action(
        self,
        *,
        source: ServiceRef,
        action: Dict[str, Any],
        ctx: Dict[str, Any],
        correlation_id: str,
    ) -> StepExecutionResult:
        tool_id = action.get("tool_id")
        tool_input = action.get("input") or {}
        verb_cfg = self.registry.get(tool_id)
        step = _verb_to_step(verb_cfg)

        exec_ctx = {**ctx, **tool_input}
        exec_ctx.setdefault("verb", tool_id)

        step_res = await call_step_services(
            self.bus,
            source=source,
            step=step,
            ctx=exec_ctx,
            correlation_id=correlation_id,
            diagnostic=bool(ctx.get("diagnostic")),
        )
        return step_res

    async def _council_checkpoint(
        self,
        *,
        source: ServiceRef,
        correlation_id: str,
        prompt: str,
        history: List[Dict[str, Any]],
    ) -> StepExecutionResult:
        deliberation = DeliberationRequest(
            prompt=prompt,
            history=history,
            event="council_deliberation",
            trace_id=correlation_id,
        )
        reply_channel = f"{settings.channel_council_reply_prefix}:{correlation_id}"
        t0 = time.time()
        logs = [f"rpc -> CouncilService reply={reply_channel}"]
        try:
            council_res = await self.council_client.deliberate(
                source=source,
                req=deliberation,
                correlation_id=correlation_id,
                reply_to=reply_channel,
                timeout_sec=float(settings.diagnostic_agent_timeout_sec),
            )
            logs.append("ok <- CouncilService")
            status = "success"
            result_payload = council_res.model_dump(mode="json")
            error_msg = None
        except Exception as exc:
            logs.append(f"timeout/exception <- CouncilService: {exc}")
            status = "fail"
            result_payload = {"error": str(exc)}
            error_msg = str(exc)
        return StepExecutionResult(
            status=status,
            verb_name="council_checkpoint",
            step_name="council_checkpoint",
            order=99,
            result={"CouncilService": result_payload},
            latency_ms=int((time.time() - t0) * 1000),
            node=settings.node_name,
            logs=logs,
        )

    async def _agent_chain_escalation(
        self,
        *,
        source: ServiceRef,
        correlation_id: str,
        ctx: Dict[str, Any],
        packs: List[str],
    ) -> StepExecutionResult:
        agent_req = {
            "text": _last_user_message(ctx),
            "mode": ctx.get("mode") or "agent",
            "session_id": ctx.get("session_id"),
            "user_id": ctx.get("user_id"),
            "messages": ctx.get("messages") or [],
            "packs": packs,
        }
        reply_channel = f"{settings.exec_result_prefix}:AgentChainService:{correlation_id}"
        t0 = time.time()
        logs = [f"rpc -> AgentChainService reply={reply_channel}"]
        agent_res = await self.agent_client.run_chain(
            source=source,
            req=AgentChainRequest(**agent_req),
            correlation_id=correlation_id,
            reply_to=reply_channel,
            timeout_sec=float(settings.step_timeout_ms) / 1000.0,
        )
        logs.append("ok <- AgentChainService")
        return StepExecutionResult(
            status="success",
            verb_name="agent_chain",
            step_name="agent_chain",
            order=100,
            result={"AgentChainService": agent_res.model_dump(mode="json")},
            latency_ms=int((time.time() - t0) * 1000),
            node=settings.node_name,
            logs=logs,
        )

    async def execute(
        self,
        *,
        source: ServiceRef,
        req: ExecutionPlan,
        correlation_id: str,
        ctx: Dict[str, Any],
        recall_cfg: Dict[str, Any],
    ) -> PlanExecutionResult:
        step_results: List[StepExecutionResult] = []
        memory_used = False
        recall_debug: Dict[str, Any] = {}

        # Recall
        raw_enabled = recall_cfg.get("enabled", True)
        recall_enabled = str(raw_enabled).lower() not in {"false", "0", "no", "off"} if isinstance(raw_enabled, str) else bool(raw_enabled)
        if recall_enabled:
            recall_step, recall_debug, _ = await run_recall_step(
                self.bus,
                source=source,
                ctx=ctx,
                correlation_id=correlation_id,
                recall_cfg=recall_cfg,
                diagnostic=bool(ctx.get("diagnostic")),
            )
            step_results.append(recall_step)
            memory_used = recall_step.status == "success"

        packs = ctx.get("packs") or []
        tags = ctx.get("verb_tags") or []
        tools = self._toolset(packs=packs, tags=tags)
        mode = ctx.get("mode") or req.metadata.get("mode") or "agent"

        if not self._should_use_react(mode, tools):
            logger.info("Supervisor: using direct LLM path")
            direct_cfg = self.registry.get(req.verb_name)
            step = _verb_to_step(direct_cfg)
            direct_step = await call_step_services(
                self.bus,
                source=source,
                step=step,
                ctx=ctx,
                correlation_id=correlation_id,
                diagnostic=bool(ctx.get("diagnostic")),
            )
            step_results.append(direct_step)
            final_obs = _extract_observation(direct_step)
            return PlanExecutionResult(
                verb_name=req.verb_name,
                request_id=correlation_id,
                status="success" if direct_step.status == "success" else "fail",
                blocked=False,
                blocked_reason=None,
                steps=step_results,
                mode=mode,
                final_text=final_obs.get("llm_output"),
                memory_used=memory_used,
                recall_debug=recall_debug,
                error=direct_step.error,
            )

        trace: List[TraceStep] = []
        max_steps = int(ctx.get("max_steps") or 3)
        final_text: Optional[str] = None
        planner_thought: Optional[str] = None

        for _ in range(max_steps):
            goal_text = _last_user_message(ctx)
            planner_req, planner_step, planner_final, action = await self._planner_step(
                source=source,
                goal_text=goal_text,
                toolset=tools,
                trace=trace,
                ctx=ctx,
                correlation_id=correlation_id,
            )
            step_results.append(planner_step)

            planner_thought = planner_step.result.get("PlannerReactService", {}).get("trace", [{}])[-1].get("thought") if isinstance(planner_step.result, dict) else None

            if planner_final and planner_final.content:
                final_text = planner_final.content
                if not trace:
                    trace.append(
                        TraceStep(
                            step_index=0,
                            thought=planner_thought,
                            action=None,
                            observation={"llm_output": final_text},
                        )
                    )
                break
            if planner_step.status != "success" or not action:
                break

            action_step = await self._execute_action(
                source=source,
                action=action,
                ctx=ctx,
                correlation_id=correlation_id,
            )
            step_results.append(action_step)
            obs = _extract_observation(action_step)
            trace.append(
                TraceStep(
                    step_index=len(trace),
                    thought=None,
                    action=action,
                    observation=obs,
                )
            )
            if obs.get("llm_output"):
                final_text = obs.get("llm_output")

        # Council checkpoint (opt-in unless explicit mode=council)
        council_step: Optional[StepExecutionResult] = None

        require_council = bool(ctx.get("require_council")) or (ctx.get("mode") == "council")
        if require_council:
            prompt_lines = [f"Goal: { _last_user_message(ctx)}", "Trace:"]
            if trace:
                for step in trace:
                    prompt_lines.append(f"- action: {step.action} obs: {step.observation}")
            elif planner_thought:
                prompt_lines.append(f"- planner_thought: {planner_thought}")
            council_prompt = "\n".join(prompt_lines)
            council_step = await self._council_checkpoint(
                source=source,
                correlation_id=correlation_id,
                prompt=council_prompt,
                history=ctx.get("messages") or [],
            )
            step_results.append(council_step)
            try:
                council_payload = council_step.result.get("CouncilService", {})
                if isinstance(council_payload, dict):
                    final_text = council_payload.get("final_text") or final_text
            except Exception:
                pass

        if ctx.get("force_agent_chain"):
            final_text = None

        # Escalate to agent chain if still no final text
        if not final_text:
            agent_step = await self._agent_chain_escalation(
                source=source,
                correlation_id=correlation_id,
                ctx=ctx,
                packs=packs or [],
            )
            step_results.append(agent_step)
            try:
                agent_payload = agent_step.result.get("AgentChainService", {})
                if isinstance(agent_payload, dict):
                    final_text = agent_payload.get("text") or final_text
            except Exception:
                pass

        overall_status = "success" if step_results and all(s.status == "success" for s in step_results if s) else "partial"
        return PlanExecutionResult(
            verb_name=req.verb_name,
            request_id=correlation_id,
            status=overall_status,
            blocked=False,
            blocked_reason=None,
            steps=step_results,
            mode=mode,
            final_text=final_text,
            memory_used=memory_used,
            recall_debug=recall_debug,
            error=None,
        )
