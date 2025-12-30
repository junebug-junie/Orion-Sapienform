from __future__ import annotations

import logging
import time
from typing import Any, Dict, List
from uuid import uuid4

from jinja2 import Environment

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import ChatRequestPayload, RecallRequestPayload, ServiceRef
from orion.core.bus.contracts import KINDS
from orion.schemas.agents.schemas import AgentChainRequest, PlannerRequest
from orion.schemas.cortex.schemas import ExecutionStep, StepExecutionResult
from .settings import settings
from .clients import (
    AgentChainClient,
    CouncilClient,
    LLMGatewayClient,
    PlannerReactClient,
    RecallClient,
)

logger = logging.getLogger("orion.cortex.exec")


def _render_prompt(template_str: str, ctx: Dict[str, Any]) -> str:
    env = Environment(autoescape=False)
    tmpl = env.from_string(template_str or "")
    return tmpl.render(**ctx)


def _last_user_message(ctx: Dict[str, Any]) -> str:
    messages = ctx.get("messages") or []
    for m in reversed(messages):
        if m.get("role") == "user" and m.get("content"):
            return str(m["content"])
    return str(ctx.get("text") or ctx.get("query_text") or "")


async def call_step_services(
    bus: OrionBusAsync,
    *,
    source: ServiceRef,
    step: ExecutionStep,
    ctx: Dict[str, Any],
    correlation_id: str,
) -> StepExecutionResult:
    """
    Execute one plan step across its declared services, returning a unified result.
    Mutates `ctx` for downstream steps (e.g., recall fragments).
    """
    t0 = time.time()
    logs: List[str] = []
    merged_result: Dict[str, Any] = {}

    logger.info(
        "--- EXEC STEP '%s' (verb=%s) corr=%s services=%s ---",
        step.step_name,
        step.verb_name,
        correlation_id,
        step.services,
    )

    prompt = _render_prompt(step.prompt_template or "", ctx) if step.prompt_template else ""
    debug_prompt = (prompt[:200] + "...") if len(prompt) > 200 else prompt
    if prompt:
        logger.info("Rendered Prompt: %r", debug_prompt)

    step_timeout_sec = (step.timeout_ms or settings.step_timeout_ms) / 1000.0

    llm_client = LLMGatewayClient(bus)
    recall_client = RecallClient(bus)
    agent_chain_client = AgentChainClient(bus)
    planner_client = PlannerReactClient(bus)
    council_client = CouncilClient(bus)

    def _fail(status: str, error: str) -> StepExecutionResult:
        return StepExecutionResult(
            status=status,
            verb_name=step.verb_name,
            step_name=step.step_name,
            order=step.order,
            result=merged_result,
            latency_ms=int((time.time() - t0) * 1000),
            node=settings.node_name,
            logs=logs,
            error=error,
            soft_fail=(status == "soft_fail"),
        )

    for service in step.services:
        reply_channel = f"orion:rpc:{uuid4()}"
        try:
            if service == "LLMGatewayService":
                req_model = ctx.get("model") or ctx.get("llm_model") or None
                messages_payload = ctx.get("messages")
                if not messages_payload:
                    content = prompt or _last_user_message(ctx) or " "
                    messages_payload = [{"role": "user", "content": content}]

                request_object = ChatRequestPayload(
                    model=req_model,
                    messages=messages_payload,
                    options={
                        "temperature": float(ctx.get("temperature", 0.7)),
                        "max_tokens": int(ctx.get("max_tokens", 512)),
                        "stream": False,
                    },
                    user_id=ctx.get("user_id"),
                    session_id=ctx.get("session_id"),
                )

                logs.append(f"rpc -> LLMGatewayService ({KINDS.llm_chat_request})")
                result_object = await llm_client.chat(
                    source=source,
                    req=request_object,
                    correlation_id=correlation_id,
                    reply_to=reply_channel,
                    timeout_sec=step_timeout_sec,
                )
                merged_result[service] = result_object.model_dump(mode="json")
                logs.append("ok <- LLMGatewayService")

            elif service == "RecallService":
                query_text = _last_user_message(ctx)
                recall_request = RecallRequestPayload(
                    query_text=query_text,
                    max_items=int(ctx.get("recall_max_items", 8)),
                    time_window_days=int(ctx.get("recall_time_window_days", 90)),
                    mode=str(ctx.get("recall_mode", "hybrid")),
                    tags=list(ctx.get("tags") or []),
                    user_id=ctx.get("user_id"),
                    session_id=ctx.get("session_id"),
                    trace_id=str(correlation_id),
                )
                logs.append(f"rpc -> RecallService ({KINDS.recall_query_request})")
                recall_result = await recall_client.query(
                    source=source,
                    req=recall_request,
                    correlation_id=correlation_id,
                    reply_to=reply_channel,
                    timeout_sec=step_timeout_sec,
                )
                memory_used = bool(recall_result.fragments)
                ctx["memory_fragments"] = recall_result.fragments
                merged_result[service] = {
                    "memory_used": memory_used,
                    "recall_debug": recall_result.debug,
                    "error": recall_result.error,
                }
                logs.append(f"ok <- RecallService (items={len(recall_result.fragments)})")
                if not recall_result.ok:
                    return _fail("soft_fail", recall_result.error or "recall_error")

            elif service == "AgentChainService":
                text = _last_user_message(ctx)
                chain_request = AgentChainRequest(
                    text=text,
                    mode="chat",
                    session_id=ctx.get("session_id"),
                    user_id=ctx.get("user_id"),
                    goal_description=ctx.get("goal_description"),
                    messages=ctx.get("messages"),
                    tools=ctx.get("tools"),
                    packs=ctx.get("packs"),
                )
                logs.append(f"rpc -> AgentChainService ({KINDS.agent_chain_request})")
                chain_result = await agent_chain_client.run(
                    source=source,
                    req=chain_request,
                    correlation_id=correlation_id,
                    reply_to=reply_channel,
                )
                merged_result[service] = chain_result.model_dump(mode="json")
                ctx["agent_chain"] = chain_result.model_dump(mode="json")
                logs.append("ok <- AgentChainService")

            elif service == "PlannerReactService":
                text = _last_user_message(ctx)
                planner_request = PlannerRequest(
                    caller="cortex-exec",
                    goal={"type": "chat", "description": ctx.get("goal_description") or text},
                    context={"conversation_history": ctx.get("messages") or []},
                )
                logs.append(f"rpc -> PlannerReactService ({KINDS.planner_request})")
                planner_result = await planner_client.run(
                    source=source,
                    req=planner_request,
                    correlation_id=correlation_id,
                    reply_to=reply_channel,
                )
                merged_result[service] = planner_result.model_dump(mode="json")
                ctx["planner_result"] = planner_result.model_dump(mode="json")
                logs.append("ok <- PlannerReactService")

            elif service == "AgentCouncilService":
                logs.append(f"rpc -> AgentCouncilService ({KINDS.council_request})")
                council_result = await council_client.deliberate(
                    source=source,
                    req={"prompt": _last_user_message(ctx), "trace_id": str(correlation_id)},
                    correlation_id=correlation_id,
                    reply_to=reply_channel,
                )
                merged_result[service] = council_result
                ctx["council_result"] = council_result
                logs.append("ok <- AgentCouncilService")

            else:
                logs.append(f"skip <- {service} (unsupported service identifier)")
                merged_result[service] = {"skipped": True}

        except Exception as e:
            logs.append(f"exception <- {service}: {e}")
            logger.error("Service %s failed (corr=%s): %s", service, correlation_id, e)
            return _fail("fail", f"{service}: {e}")

    return StepExecutionResult(
        status="success",
        verb_name=step.verb_name,
        step_name=step.step_name,
        order=step.order,
        result=merged_result,
        latency_ms=int((time.time() - t0) * 1000),
        node=settings.node_name,
        logs=logs,
        soft_fail=False,
    )
