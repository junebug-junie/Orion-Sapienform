# services/orion-cortex-exec/app/executor.py
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

from jinja2 import Environment

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, ServiceRef

from .models import ExecutionStep, StepExecutionResult
from .settings import settings

logger = logging.getLogger("orion.cortex.exec")


def _render_prompt(template_str: str, ctx: Dict[str, Any]) -> str:
    env = Environment(autoescape=False)
    tmpl = env.from_string(template_str or "")
    return tmpl.render(**ctx)


def _kind_for(service_name: str) -> str:
    if service_name == "LLMGatewayService":
        return "llm.chat.request"
    return "exec.generic.request"


def _channel_for(service_name: str) -> str:
    if service_name == "LLMGatewayService":
        return "orion-llm:intake"
    return f"{settings.exec_request_prefix}:{service_name}"


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

    prompt = _render_prompt(step.prompt_template or "", ctx) if step.prompt_template else ""
    
    # [DEBUG] LOUD LOGGING OF PROMPT
    logger.warning(f"--- EXEC STEP '{step.step_name}' START ---")
    
    # [FIX] FORCE 120s TIMEOUT (Bypassing .env for now to ensure we wait long enough)
    timeout_sec = 120.0 

    for service in step.services:
        reply = f"orion:rpc:{uuid4()}"
        kind = _kind_for(service)
        channel = _channel_for(service)

        if service == "LLMGatewayService":
            # 1. Model Selection: Default to None (use active model)
            req_model = ctx.get("model") or ctx.get("llm_model") or None
            
            # 2. Payload Construction: [CRITICAL FIX]
            # Use the full conversation history from context!
            messages_payload = ctx.get("messages")
            
            if not messages_payload:
                logger.warning("EXEC: No messages in context! Falling back to prompt.")
                content = prompt or " "
                messages_payload = [{"role": "user", "content": content}]
            else:
                logger.warning(f"EXEC: Using {len(messages_payload)} messages from history.")

            payload = ChatRequestPayload(
                model=req_model,
                messages=messages_payload,
                options={
                    "temperature": float(ctx.get("temperature", 0.7)),
                    "max_tokens": int(ctx.get("max_tokens", 512)),
                    "stream": False,
                }
            ).model_dump(mode="json")
        else:
            payload = {"prompt": prompt, "context": ctx, "service": service}

        env = BaseEnvelope(
            kind=kind,
            source=source,
            correlation_id=correlation_id,
            reply_to=reply,
            payload=payload,
        )

        try:
            logs.append(f"rpc -> {channel} kind={kind} timeout={timeout_sec}s")
            msg = await bus.rpc_request(channel, env, reply_channel=reply, timeout_sec=timeout_sec)
            decoded = bus.codec.decode(msg.get("data"))
            if decoded.ok and isinstance(decoded.envelope.payload, dict):
                merged_result[service] = decoded.envelope.payload
                logs.append(f"ok <- {service}")
            else:
                err = decoded.error or "decode_failed"
                logs.append(f"fail <- {service}: {err}")
                return StepExecutionResult(
                    status="fail",
                    verb_name=step.verb_name,
                    step_name=step.step_name,
                    order=step.order,
                    result=merged_result,
                    latency_ms=int((time.time() - t0) * 1000),
                    node=settings.node_name,
                    logs=logs,
                    error=f"{service}: {err}",
                )
        except Exception as e:
            logs.append(f"exception <- {service}: {e}")
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
