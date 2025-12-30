# services/orion-cortex-exec/app/executor.py
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Tuple
from uuid import uuid4

from jinja2 import Environment

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import ChatRequestPayload, LLMMessage, RecallRequestPayload, ServiceRef

from orion.schemas.agents.schemas import AgentChainRequest
from orion.schemas.cortex.schemas import ExecutionStep, StepExecutionResult
from .settings import settings
from .clients import AgentChainClient, LLMGatewayClient, RecallClient

logger = logging.getLogger("orion.cortex.exec")


def _render_prompt(template_str: str, ctx: Dict[str, Any]) -> str:
    env = Environment(autoescape=False)
    tmpl = env.from_string(template_str or "")
    return tmpl.render(**ctx)


def _last_user_message(ctx: Dict[str, Any]) -> str:
    msgs = ctx.get("messages") or []
    if isinstance(msgs, list):
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("role") == "user":
                return str(m.get("content") or "")
    return str(ctx.get("user_message") or "")


async def run_recall_step(
    bus: OrionBusAsync,
    *,
    source: ServiceRef,
    ctx: Dict[str, Any],
    correlation_id: str,
    recall_cfg: Dict[str, Any],
) -> Tuple[StepExecutionResult, Dict[str, Any], str]:
    t0 = time.time()
    recall_client = RecallClient(bus)
    reply_channel = f"{settings.exec_result_prefix}:RecallService:{uuid4()}"

    query_text = _last_user_message(ctx) or ""
    req = RecallRequestPayload(
        query_text=query_text,
        session_id=ctx.get("session_id"),
        user_id=ctx.get("user_id"),
        mode=str(recall_cfg.get("mode", "hybrid")),
        time_window_days=int(recall_cfg.get("time_window_days", 90)),
        max_items=int(recall_cfg.get("max_items", 8)),
        packs=list(ctx.get("packs") or []),
        trace_id=recall_cfg.get("trace_id") or correlation_id,
    )

    logs: List[str] = [f"rpc -> RecallService (mode={req.mode}, window={req.time_window_days})"]
    debug: Dict[str, Any] = {}
    try:
        res = await recall_client.query(
            source=source,
            req=req,
            correlation_id=correlation_id,
            reply_to=reply_channel,
            timeout_sec=float(settings.step_timeout_ms) / 1000.0,
        )
        fragments = res.fragments
        debug = {
            "count": len(fragments),
            "mode": req.mode,
            "time_window_days": req.time_window_days,
            "max_items": req.max_items,
            "error": None,
        }
        if getattr(res, "debug", None):
            try:
                debug.update(res.debug)  # type: ignore[arg-type]
            except Exception:
                debug["debug"] = res.debug

        digest_lines = [
            f"- {fr.get('text', '')}" for fr in fragments[:5] if isinstance(fr, dict)
        ]
        memory_digest = "\n".join(digest_lines)
        ctx["memory_digest"] = memory_digest
        ctx["memory_used"] = True
        ctx["recall_fragments"] = fragments
        logs.append(f"ok <- RecallService ({len(fragments)} fragments)")

        return (
            StepExecutionResult(
                status="success",
                verb_name=str(ctx.get("verb") or "unknown"),
                step_name="recall",
                order=-1,
                result={"RecallService": debug},
                latency_ms=int((time.time() - t0) * 1000),
                node=settings.node_name,
                logs=logs,
            ),
            debug,
            memory_digest,
        )
    except Exception as e:
        logs.append(f"exception <- RecallService: {e}")
        debug["error"] = str(e)
        return (
            StepExecutionResult(
                status="fail",
                verb_name=str(ctx.get("verb") or "unknown"),
                step_name="recall",
                order=-1,
                result={"RecallService": debug},
                latency_ms=int((time.time() - t0) * 1000),
                node=settings.node_name,
                logs=logs,
                error=str(e),
            ),
            debug,
            "",
        )


async def call_step_services(
    bus: OrionBusAsync,
    *,
    source: ServiceRef,
    step: ExecutionStep,
    ctx: Dict[str, Any],
    correlation_id: str,
) -> StepExecutionResult:
    t0 = time.time()
    logs: List[str] = []
    merged_result: Dict[str, Any] = {}

    # DEBUG: Log Context Keys to prove data is present
    logger.info(f"--- EXEC STEP '{step.step_name}' START ---")
    logger.info(f"Context Keys available: {list(ctx.keys())}")

    prompt = _render_prompt(step.prompt_template or "", ctx) if step.prompt_template else ""
    
    # DEBUG: Log Rendered Prompt (Truncated)
    debug_prompt = (prompt[:200] + "...") if len(prompt) > 200 else prompt
    logger.info(f"Rendered Prompt: {debug_prompt!r}")

    # Calculate Timeout from Step Definition (default to 60s if missing)
    # The YAML says 60000ms, so we convert to 60.0s
    step_timeout_sec = (step.timeout_ms or 60000) / 1000.0

    # Instantiate Clients
    llm_client = LLMGatewayClient(bus)
    agent_client = AgentChainClient(bus)

    for service in step.services:
        reply_channel = f"{settings.exec_result_prefix}:{service}:{uuid4()}"

        try:
            if service == "LLMGatewayService":
                # --- STRICT PATH ---
                # 1. Build Pydantic Model
                req_model = ctx.get("model") or ctx.get("llm_model") or None
                messages_payload = ctx.get("messages")
                
                if not messages_payload:
                    content = prompt or " "
                    messages_payload = [{"role": "user", "content": content}]

                request_object = ChatRequestPayload(
                    model=req_model,
                    messages=messages_payload,
                    options={
                        "temperature": float(ctx.get("temperature", 0.7)),
                        "max_tokens": int(ctx.get("max_tokens", 512)),
                        "stream": False,  # Keep this fix!
                    }
                )

                # 2. Delegate to Client WITH TIMEOUT [FIXED]
                logs.append(f"rpc -> LLMGateway via client (timeout={step_timeout_sec}s)")
                result_object = await llm_client.chat(
                    source=source,
                    req=request_object,
                    correlation_id=correlation_id,
                    reply_to=reply_channel,
                    timeout_sec=step_timeout_sec,  # <--- CRITICAL FIX
                )
                
                # Explicitly dump to dict for storage in result payload
                merged_result[service] = result_object.model_dump(mode="json")
                logs.append(f"ok <- {service}")

            elif service == "AgentChainService":
                agent_req = AgentChainRequest(
                    text=_last_user_message(ctx),
                    mode=ctx.get("mode") or "agent",
                    session_id=ctx.get("session_id"),
                    user_id=ctx.get("user_id"),
                    messages=[
                        LLMMessage(**m) if not isinstance(m, LLMMessage) else m
                        for m in (ctx.get("messages") or [])
                    ],
                    packs=ctx.get("packs") or [],
                )
                logs.append("rpc -> AgentChainService")
                agent_res = await agent_client.run_chain(
                    source=source,
                    req=agent_req,
                    correlation_id=correlation_id,
                    reply_to=reply_channel,
                    timeout_sec=step_timeout_sec,
                )
                merged_result[service] = agent_res.model_dump(mode="json")
                logs.append("ok <- AgentChainService")

            elif service == "CouncilService":
                logs.append("stub <- CouncilService (not implemented)")
                return StepExecutionResult(
                    status="fail",
                    verb_name=step.verb_name,
                    step_name=step.step_name,
                    order=step.order,
                    result={service: {"error": "CouncilService not implemented"}},
                    latency_ms=int((time.time() - t0) * 1000),
                    node=settings.node_name,
                    logs=logs,
                    error="CouncilService not implemented",
                )

            else:
                logs.append(f"skip <- {service} (generic path not implemented in example)")

        except Exception as e:
            logs.append(f"exception <- {service}: {e}")
            logger.error(f"Service {service} failed: {e}")
            return StepExecutionResult(
                status="fail",
                verb_name=step.verb_name,
                step_name=step.step_name,
                order=step.order,
                result=merged_result,
                latency_ms=int((time.time() - t0) * 1000),
                node=settings.node_name,
                logs=logs,
                error=f"{service}: {e}",
            )

    return StepExecutionResult(
        status="success",
        verb_name=step.verb_name,
        step_name=step.step_name,
        order=step.order,
        result=merged_result,
        latency_ms=int((time.time() - t0) * 1000),
        node=settings.node_name,
        logs=logs,
    )
