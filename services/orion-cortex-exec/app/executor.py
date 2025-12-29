# services/orion-cortex-exec/app/executor.py
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List
from uuid import uuid4

from jinja2 import Environment

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, ServiceRef

from .models import ExecutionStep, StepExecutionResult
from .settings import settings
from .clients import LLMGatewayClient  # <--- IMPORT THE NEW CLIENT

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

    prompt = _render_prompt(step.prompt_template or "", ctx) if step.prompt_template else ""
    logger.info(f"--- EXEC STEP '{step.step_name}' START ---")
    
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
                        "stream": False,
                    }
                )

                # 2. Delegate to Client (No dicts, no strings)
                logs.append(f"rpc -> LLMGateway via client")
                result_data = await llm_client.chat(
                    source=source,
                    req=request_object,
                    correlation_id=correlation_id,
                    reply_to=reply_channel
                )
                
                merged_result[service] = result_data
                logs.append(f"ok <- {service}")

            else:
                # --- LEGACY/GENERIC PATH (Still loose for now) ---
                # ... (keep existing generic logic for non-LLM services if needed) ...
                logs.append(f"skip <- {service} (generic path not implemented in example)")

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
