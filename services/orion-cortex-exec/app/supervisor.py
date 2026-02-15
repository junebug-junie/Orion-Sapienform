from __future__ import annotations

import logging
import json
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
from orion.cognition.verb_activation import is_active

from .clients import AgentChainClient, CouncilClient, LLMGatewayClient, PlannerReactClient
from .executor import _last_user_message, call_step_services, run_recall_step
from .recall_utils import has_inline_recall, resolve_profile, should_run_recall
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


def _truncate_text(value: Any, limit: int = 240) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _compact_json(value: Any, limit: int = 220) -> str:
    try:
        rendered = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        rendered = str(value)
    return _truncate_text(rendered, limit=limit)


def _build_council_prompt(
    *,
    goal: str,
    mode: str,
    node_name: str,
    correlation_id: str,
    verb_name: str | None,
    allowed_verbs: List[str],
    toolset: List[ToolDef],
    trace: List[TraceStep],
    final_text: str | None,
) -> str:
    lines: List[str] = []
    lines.append("Council Supervision Context")
    lines.append(f"- goal: {_truncate_text(goal, 2000)}")
    lines.append(f"- mode: {mode}")
    lines.append(f"- node: {node_name}")
    lines.append(f"- correlation_id: {correlation_id}")
    if verb_name:
        lines.append(f"- entry_verb: {verb_name}")
    if allowed_verbs:
        lines.append(f"- allowed_verbs: {', '.join(allowed_verbs[:50])}")

    lines.append("\nToolset summary:")
    if not toolset:
        lines.append("- (none)")
    else:
        for tool in toolset[:25]:
            lines.append(f"- {tool.tool_id}: {_truncate_text(tool.description or '', 120)}")
        if len(toolset) > 25:
            lines.append(f"- ... ({len(toolset) - 25} more tools omitted)")

    lines.append("\nRecent trace:")
    if not trace:
        lines.append("- (no trace yet)")
    else:
        for step in trace[-8:]:
            action = step.action if isinstance(step.action, dict) else {}
            tool_id = action.get("tool_id") or "none"
            args = _compact_json(action.get("input") or {}, 220)
            obs = step.observation if isinstance(step.observation, dict) else {"raw": step.observation}
            obs_text = obs.get("llm_output") or obs.get("text") or obs.get("content") or obs.get("error") or _compact_json(obs, 280)
            lines.append(
                f"- step {step.step_index}: tool={tool_id} args={args} obs={_truncate_text(obs_text, 280)}"
            )

    if final_text:
        lines.append("\nProvisional final_text:")
        lines.append(_truncate_text(final_text, 1500))

    lines.append("\nReturn your normal council output contract.")
    return "\n".join(lines)


def _extract_council_debug(result_payload: Dict[str, Any]) -> Dict[str, Any]:
    opinions_raw = result_payload.get("opinions") if isinstance(result_payload, dict) else []
    opinions: List[Dict[str, Any]] = []
    if isinstance(opinions_raw, list):
        for item in opinions_raw[:12]:
            if not isinstance(item, dict):
                continue
            opinions.append(
                {
                    "agent_name": _truncate_text(item.get("agent_name") or item.get("name") or "unknown", 80),
                    "confidence": item.get("confidence"),
                    "text": _truncate_text(item.get("text") or "", 800),
                }
            )

    verdict_raw = result_payload.get("verdict") if isinstance(result_payload, dict) else {}
    verdict = {}
    if isinstance(verdict_raw, dict):
        verdict = {
            "action": verdict_raw.get("action"),
            "reason": _truncate_text(verdict_raw.get("reason") or "", 500),
            "constraints": verdict_raw.get("constraints") if isinstance(verdict_raw.get("constraints"), dict) else {},
        }

    blink_raw = result_payload.get("blink") if isinstance(result_payload, dict) else {}
    blink: Dict[str, Any] = {}
    if isinstance(blink_raw, dict):
        scores = blink_raw.get("scores") if isinstance(blink_raw.get("scores"), dict) else {}
        blink = {
            "proposed_answer": _truncate_text(blink_raw.get("proposed_answer") or "", 500),
            "scores": scores,
        }

    return {
        "opinions": opinions,
        "verdict": verdict,
        "blink": blink,
    }


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
            if not is_active(v.name, node_name=settings.node_name):
                continue
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
        if not tool_id or not is_active(str(tool_id), node_name=settings.node_name):
            logger.warning("Inactive verb selected by supervisor corr_id=%s verb=%s", correlation_id, tool_id)
            return StepExecutionResult(
                status="fail",
                verb_name=str(tool_id or "unknown"),
                step_name="inactive_verb_guard",
                order=0,
                result={"error": "inactive_verb", "verb": tool_id, "node": settings.node_name},
                latency_ms=0,
                node=settings.node_name,
                logs=[f"reject <- inactive verb {tool_id}"],
                error=f"inactive_verb:{tool_id}",
            )

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
        history: List[Dict[str, Any]],
        goal_text: str,
        mode: str,
        verb_name: str | None,
        allowed_verbs: List[str],
        toolset: List[ToolDef],
        trace: List[TraceStep],
        final_text: str | None,
        node_name: str,
    ) -> StepExecutionResult:
        prompt = _build_council_prompt(
            goal=goal_text,
            mode=mode,
            node_name=node_name,
            correlation_id=correlation_id,
            verb_name=verb_name,
            allowed_verbs=allowed_verbs,
            toolset=toolset,
            trace=trace,
            final_text=final_text,
        )
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
            result_payload["debug_compact"] = _extract_council_debug(result_payload)
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
        ctx.setdefault("debug", {})

        verb_recall_profile = None
        if isinstance(req.metadata, dict):
            verb_recall_profile = req.metadata.get("recall_profile") or None
        ctx.setdefault("plan_recall_profile", verb_recall_profile)
        recall_required = bool(recall_cfg.get("required", False))
        selected_profile, profile_source = resolve_profile(
            recall_cfg,
            verb_profile=verb_recall_profile,
        )
        inline_recall = has_inline_recall(req.steps)
        should_recall, recall_reason = should_run_recall(recall_cfg, req.steps)

        if should_recall and not inline_recall:
            logger.info(
                "Supervisor recall resolved profile=%s source=%s gating=%s",
                selected_profile,
                profile_source,
                recall_reason,
            )
            recall_step, recall_debug, _ = await run_recall_step(
                self.bus,
                source=source,
                ctx=ctx,
                correlation_id=correlation_id,
                recall_cfg=recall_cfg,
                recall_profile=selected_profile,
                diagnostic=bool(ctx.get("diagnostic")),
            )
            step_results.append(recall_step)
            memory_used = recall_step.status == "success"
            if recall_required and recall_step.status != "success":
                return PlanExecutionResult(
                    verb_name=req.verb_name,
                    request_id=correlation_id,
                    status="fail",
                    blocked=False,
                    blocked_reason=None,
                    steps=step_results,
                    mode=ctx.get("mode") or req.metadata.get("mode") or "agent",
                    final_text=None,
                    memory_used=memory_used,
                    recall_debug=recall_debug,
                    error=recall_step.error,
                )
        else:
            if inline_recall:
                recall_debug = {"skipped": "inline_recall_step_present"}
                logger.info(
                    "Supervisor recall skipped; inline RecallService step present",
                    extra={"correlation_id": correlation_id, "recall_cfg": recall_cfg},
                )
            else:
                recall_debug = {"skipped": recall_reason}
                logger.info(
                    "Supervisor recall skipped by gating (%s)",
                    recall_reason,
                    extra={"correlation_id": correlation_id, "recall_cfg": recall_cfg},
                )

        packs = ctx.get("packs") or []
        tags = ctx.get("verb_tags") or []
        tools = self._toolset(packs=packs, tags=tags)
        mode = ctx.get("mode") or req.metadata.get("mode") or "agent"
        options = ctx.get("options") if isinstance(ctx.get("options"), dict) else {}
        allowed_verbs = [str(v).strip() for v in (options.get("allowed_verbs") or []) if str(v).strip()]

        if not self._should_use_react(mode, tools):
            logger.info("Supervisor: using direct LLM path")
            if not is_active(req.verb_name, node_name=settings.node_name):
                return PlanExecutionResult(
                    verb_name=req.verb_name,
                    request_id=correlation_id,
                    status="fail",
                    blocked=False,
                    blocked_reason=None,
                    steps=step_results,
                    mode=mode,
                    final_text=f"Verb '{req.verb_name}' is inactive on node {settings.node_name}.",
                    memory_used=memory_used,
                    recall_debug=recall_debug,
                    error=f"inactive_verb:{req.verb_name}",
                )
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
            if action_step.status != "success" and str(action_step.error or "").startswith("inactive_verb:"):
                return PlanExecutionResult(
                    verb_name=req.verb_name,
                    request_id=correlation_id,
                    status="fail",
                    blocked=False,
                    blocked_reason=None,
                    steps=step_results,
                    mode=mode,
                    final_text=f"Verb '{action.get('tool_id')}' is inactive on node {settings.node_name}.",
                    memory_used=memory_used,
                    recall_debug=recall_debug,
                    error=action_step.error,
                )
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
            council_step = await self._council_checkpoint(
                source=source,
                correlation_id=correlation_id,
                history=ctx.get("messages") or [],
                goal_text=_last_user_message(ctx),
                mode=mode,
                verb_name=req.verb_name,
                allowed_verbs=allowed_verbs,
                toolset=tools,
                trace=trace,
                final_text=final_text,
                node_name=settings.node_name,
            )
            step_results.append(council_step)
            try:
                council_payload = council_step.result.get("CouncilService", {})
                if isinstance(council_payload, dict):
                    final_text = council_payload.get("final_text") or final_text
                    compact_debug = council_payload.get("debug_compact")
                    if isinstance(compact_debug, dict):
                        ctx.setdefault("debug", {})["council"] = compact_debug
                        recall_debug["council_debug"] = compact_debug
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
