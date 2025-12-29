# services/orion-cortex-exec/app/executor.py
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List
from uuid import uuid4

from jinja2 import Environment

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, ServiceRef

from orion.schemas.cortex.schemas import ExecutionStep, StepExecutionResult
from .settings import settings
from .clients import LLMGatewayClient

logger = logging.getLogger("orion.cortex.exec")


def _render_prompt(template_str: str, ctx: Dict[str, Any]) -> str:
    env = Environment(autoescape=False)
    tmpl = env.from_string(template_str or "")
    return tmpl.render(**ctx)


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

    for service in step.services:
        reply_channel = f"orion:rpc:{uuid4()}"

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
