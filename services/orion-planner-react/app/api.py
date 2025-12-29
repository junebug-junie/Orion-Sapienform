# services/orion-planner-react/app/api.py
from __future__ import annotations

import json
import logging
import os
import time
import uuid
import sys
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

from .settings import settings
from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ChatResultPayload

from orion.schemas.agents.schemas import (
    PlannerRequest,
    PlannerResponse,
    TraceStep,
    FinalAnswer,
    Usage,
    ToolDef,
    ContextBlock,
    Limits,
    Goal
)

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("planner-react.api")
logger.setLevel(logging.INFO)

router = APIRouter()

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _normalize_tool_id(requested: str, toolset: List[ToolDef]) -> str:
    if not requested: return ""
    requested = requested.strip()
    valid_ids = {t.tool_id for t in toolset}
    if requested in valid_ids: return requested
    no_space = requested.replace(" ", "")
    if no_space in valid_ids: return no_space
    snake = requested.replace(" ", "_")
    if snake in valid_ids: return snake
    kebab = requested.replace("_", "-")
    if kebab in valid_ids: return kebab
    return requested

def _repair_json(text: str) -> str:
    # 1. Strip Markdown Code Blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    # 2. Fix Python constants
    text = text.replace(" None", " null").replace(":None", ":null")
    text = text.replace(" True", " true").replace(":True", ":true")
    text = text.replace(" False", " false").replace(":False", ":false")
    text = re.sub(r",\s*([}\]])", r"\1", text)
    
    return text

# ─────────────────────────────────────────────
# Async RPC Helpers
# ─────────────────────────────────────────────

async def _call_planner_llm(
    bus: OrionBusAsync,
    *,
    goal: Goal,
    toolset: List[ToolDef],
    context: ContextBlock,
    prior_trace: List[TraceStep],
    limits: Limits,
) -> Dict[str, Any]:
    trace_id = str(uuid.uuid4())
    exec_channel = f"{settings.exec_request_prefix}:{settings.llm_gateway_service_name}"
    reply_channel = f"{settings.llm_reply_prefix}:{trace_id}"

    tools_description = "\n".join(f"- {t.tool_id}: {t.description or ''}" for t in toolset) or "(none)"
    
    prior_summary = ""
    if prior_trace:
        lines = []
        for s in prior_trace[-3:]:
            obs_str = str(s.observation)
            if len(obs_str) > 1000: obs_str = obs_str[:1000] + "... [TRUNCATED]"
            lines.append(f"Step {s.step_index}: thought={s.thought!r}, action={s.action}, observation={obs_str}")
        prior_summary = "\n".join(lines)

    last_user = ""
    if context.conversation_history:
        last_msg = context.conversation_history[-1]
        raw_user = last_msg.content if hasattr(last_msg, "content") else last_msg.get("content", "")
        last_user = raw_user[:14000] + "..." if len(raw_user) > 14000 else raw_user

    ext_text = (context.external_facts.get("text") or "")[:10000]

    system_msg = f"""
You are Orion's internal ReAct planner.
You must output PURE VALID JSON. No markdown, no text, no explanations outside the JSON.

TOOLS AVAILABLE:
{tools_description}

JSON FORMAT:
{{
  "thought": "Your reasoning here",
  "finish": true or false,
  "action": {{ "tool_id": "name", "input": {{ ... }} }} or null,
  "final_answer": {{ "content": "..." }} or null
}}

RULES:
1. If finish=false, "action" MUST NOT be null.
2. If finish=true, "final_answer" MUST NOT be null.
3. Do not generate "Observation:" lines. Only generate the next step JSON.
""".strip()

    # [FIX] Move JSON reminder to the END of the prompt (Recency Bias)
    user_prompt = f"""
GOAL: {goal.description}

HISTORY: 
{last_user}

CONTEXT/FACTS:
{ext_text}

PAST TRACE:
{prior_summary}

INSTRUCTION:
Based on the TRACE above, generate the NEXT step as a JSON object.
""".strip()

    planner_model = os.environ.get("PLANNER_MODEL") or "llama-3-8b-instruct-q4_k_m"
    
    payload = {
        "model": planner_model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt}
        ],
        "options": {"temperature": 0.1},
    }

    env = BaseEnvelope(
        kind="llm.chat.request",
        source={"name": "planner-react", "node": settings.node_name},
        correlation_id=trace_id,
        reply_to=reply_channel,
        payload=payload
    )

    msg = await bus.rpc_request(exec_channel, env, reply_channel=reply_channel, timeout_sec=float(limits.timeout_seconds))
    decoded = bus.codec.decode(msg.get("data"))
    
    if not decoded.ok:
        raise RuntimeError(f"LLM RPC Error: {decoded.error}")

    resp_payload = decoded.envelope.payload or {}
    
    if "error" in resp_payload:
        raise RuntimeError(f"LLM Gateway returned error: {resp_payload.get('error')} Details: {resp_payload.get('details')}")

    try:
        chat_res = ChatResultPayload(**resp_payload)
        text = chat_res.text
    except Exception:
        text = resp_payload.get("content") or resp_payload.get("text") or ""

    if not text:
        raise RuntimeError(f"LLM Gateway returned empty text. Payload: {resp_payload}")

    if "<think>" in text:
        text = text.split("</think>")[-1].strip()
    
    text = _repair_json(text)
    text = text.replace(r"\'", "'")

    try:
        return json.loads(text)
    except Exception as e:
        raise RuntimeError(f"Planner LLM returned non-JSON: {text!r}") from e


async def _call_cortex_verb(
    bus: OrionBusAsync,
    verb_name: str,
    context: Dict[str, Any],
    timeout_ms: int,
) -> Dict[str, Any]:
    trace_id = str(uuid.uuid4())
    reply_channel = f"{settings.cortex_result_prefix}:{trace_id}"

    payload = {
        "verb_name": verb_name,
        "origin_node": settings.service_name,
        "context": context or {},
        "steps": [],
        "timeout_ms": timeout_ms,
    }
    
    envelope = {
        "event": "orchestrate_verb",
        "trace_id": trace_id,
        "origin_node": settings.service_name,
        "reply_channel": reply_channel,
        "payload": payload,
        **payload 
    }

    msg = await bus.rpc_request(
        settings.cortex_request_channel, 
        envelope, 
        reply_channel=reply_channel, 
        timeout_sec=timeout_ms / 1000.0
    )
    
    decoded = bus.codec.decode(msg.get("data"))
    if not decoded.ok:
        raise RuntimeError(f"Cortex RPC decode failed: {decoded.error}")
    
    return decoded.envelope.payload if isinstance(decoded.envelope.payload, dict) else decoded.envelope.payload.model_dump(mode="json")


def _extract_llm_output_from_cortex(raw_cortex: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw_cortex, dict): 
        raise RuntimeError(f"Expected dict from Cortex, got {type(raw_cortex)}")
    
    if raw_cortex.get("ok") is not True: 
        raise RuntimeError(f"Cortex Error: {raw_cortex.get('error')}")
    
    candidate = raw_cortex
    steps = []
    
    for _ in range(4):
        if isinstance(candidate, dict) and (candidate.get("steps") or candidate.get("step_results")):
            steps = candidate.get("steps") or candidate.get("step_results")
            break
        if isinstance(candidate, dict) and isinstance(candidate.get("result"), dict):
            candidate = candidate["result"]
            continue
        if isinstance(candidate, dict) and isinstance(candidate.get("payload"), dict):
            candidate = candidate["payload"]
            continue
        break

    if not steps:
        if isinstance(candidate, list):
            steps = candidate
        else:
            return {"llm_output": "", "spark_meta": None, "raw_cortex": raw_cortex}
    
    first_step = steps[0]
    result_map = first_step.get("result")
    
    text = ""
    spark_meta = None

    if isinstance(result_map, dict):
        for val in result_map.values():
            if isinstance(val, dict):
                text = val.get("content") or val.get("text") or val.get("llm_output") or ""
                spark_meta = val.get("spark_meta")
                if text: break
    
    if not text:
        services = first_step.get("services") or []
        if isinstance(services, list) and services:
            srv_payload = services[0].get("payload") or {}
            res = srv_payload.get("result") or srv_payload
            if isinstance(res, dict):
                text = res.get("llm_output") or res.get("content") or res.get("text") or ""
                spark_meta = res.get("spark_meta")
            else:
                text = str(res)

    return {"llm_output": text.strip(), "spark_meta": spark_meta}


# ─────────────────────────────────────────────
# Core ReAct loop
# ─────────────────────────────────────────────

async def run_react_loop(payload: PlannerRequest) -> PlannerResponse:
    bus = OrionBusAsync(url=settings.orion_bus_url)
    await bus.connect()

    trace: List[TraceStep] = []
    tools_called: List[str] = []
    start = time.monotonic()
    final_answer = None
    steps_used = 0

    try:
        for step_index in range(payload.limits.max_steps):
            steps_used = step_index + 1
            
            planner_step = await _call_planner_llm(
                bus=bus,
                goal=payload.goal,
                toolset=payload.toolset,
                context=payload.context,
                prior_trace=trace,
                limits=payload.limits
            )
            
            thought = planner_step.get("thought", "")
            finish = planner_step.get("finish", False)
            action = planner_step.get("action")
            final = planner_step.get("final_answer")

            if finish or (final and not action):
                content = final.get("content", "") if final else thought
                structured = final.get("structured", {}) if final else {}
                
                final_answer = FinalAnswer(content=content, structured=structured)
                trace.append(TraceStep(step_index=step_index, thought=thought))
                break
            
            if not action:
                logger.warning(f"Step {step_index}: LLM returned no action and finish=False. Terminating.")
                final_answer = FinalAnswer(content=thought or "Planner halted: No action provided by LLM.")
                trace.append(TraceStep(step_index=step_index, thought=thought))
                break

            raw_tool_id = action.get("tool_id")
            tool_input = action.get("input", {})
            
            if not raw_tool_id:
                final_answer = FinalAnswer(content=thought)
                break

            tool_id = _normalize_tool_id(raw_tool_id, payload.toolset)

            try:
                raw_cortex = await _call_cortex_verb(
                    bus=bus,
                    verb_name=tool_id,
                    context=tool_input,
                    timeout_ms=payload.limits.timeout_seconds * 1000
                )
                obs_bundle = _extract_llm_output_from_cortex(raw_cortex)
                
                if not obs_bundle.get("llm_output"):
                    err = raw_cortex.get("error") or "Unknown error"
                    obs_bundle["llm_output"] = f"Tool executed but returned no text. (System Error: {err})"
                
                observation = obs_bundle
                tools_called.append(tool_id)
            except Exception as e:
                observation = {"error": str(e)}

            trace.append(TraceStep(
                step_index=step_index,
                thought=thought,
                action={"tool_id": tool_id, "input": tool_input},
                observation=observation
            ))

        if not final_answer:
            final_answer = FinalAnswer(content="Max steps reached without final answer.")

    except Exception as e:
        logger.exception("ReAct Loop Failed")
        return PlannerResponse(status="error", error={"message": str(e)}, request_id=payload.request_id)
    finally:
        await bus.close()

    return PlannerResponse(
        request_id=payload.request_id,
        status="ok",
        final_answer=final_answer,
        trace=trace,
        usage=Usage(
            steps=steps_used, 
            tokens_reason=0, 
            tokens_answer=0, 
            tools_called=tools_called, 
            duration_ms=int((time.monotonic() - start)*1000)
        )
    )

@router.post("/plan/react", response_model=PlannerResponse)
async def plan_react(payload: PlannerRequest) -> PlannerResponse:
    if not settings.orion_bus_enabled:
        raise HTTPException(status_code=500, detail="Bus disabled")
    return await run_react_loop(payload)
