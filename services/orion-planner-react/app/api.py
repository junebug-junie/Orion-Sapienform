from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

from .settings import settings
from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ChatResultPayload, LLMMessage
from orion.schemas.agents.schemas import (
    ContextBlock,
    FinalAnswer,
    Goal,
    Limits,
    PlannerRequest,
    PlannerResponse,
    ToolDef,
    TraceStep,
    Usage,
)
from orion.schemas.cortex.contracts import CortexClientContext, CortexClientRequest, RecallDirective

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("planner-react.api")
logger.setLevel(logging.INFO)

router = APIRouter()

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────


def _normalize_tool_id(requested: str, toolset: List[ToolDef]) -> str:
    if not requested:
        return ""
    requested = requested.strip()
    tool_map = {t.tool_id: t for t in toolset}
    if requested in tool_map:
        return requested

    variations = [
        requested.lower(),
        requested.replace(" ", ""),
        requested.replace(" ", "_").lower(),
        requested.replace("_", "-").lower(),
        requested.replace("-", "_").lower(),
    ]
    for v in variations:
        if v in tool_map:
            return v
        for valid_id in tool_map.keys():
            if valid_id.lower() == v:
                return valid_id

    req_lower = requested.lower()
    for t in toolset:
        tid = t.tool_id.lower()
        desc = (t.description or "").lower()
        if tid in req_lower:
            return t.tool_id
        if req_lower in tid and len(req_lower) > 4:
            return t.tool_id
        if req_lower in desc:
            return t.tool_id

    return requested


def _repair_json(text: str) -> str:
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    text = text.replace(" None", " null").replace(":None", ":null")
    text = text.replace(" True", " true").replace(":True", ":true")
    text = text.replace(" False", " false").replace(":False", ":false")
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def _format_schema(schema: Dict[str, Any]) -> str:
    if not schema or "properties" not in schema:
        return "{}"
    props = schema.get("properties", {})
    required = schema.get("required", [])
    fields = []
    for name, meta in props.items():
        type_str = meta.get("type", "any")
        req_str = " (required)" if name in required else ""
        fields.append(f'"{name}": "{type_str}{req_str}"')
    return "{ " + ", ".join(fields) + " }"


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

    tool_lines = []
    for t in toolset:
        schema_str = _format_schema(t.input_schema)
        tool_lines.append(
            f'- TOOL_ID: "{t.tool_id}"\n  DESCRIPTION: {t.description or "No description"}\n  INPUT_FORMAT: {schema_str}\n'
        )
    tools_description = "\n".join(tool_lines) or "(none)"

    prior_summary = ""
    if prior_trace:
        lines = []
        for s in prior_trace[-3:]:
            obs_str = str(s.observation)
            if len(obs_str) > 1000:
                obs_str = obs_str[:1000] + "... [TRUNCATED]"
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

AVAILABLE TOOLS:
{tools_description}

CORE INSTRUCTIONS:
1. **ANSWER THE USER'S INTENT EFFICIENTLY.** 2. **DEFINITION OF DONE:** As soon as you have the information to answer the user, STOP. Set "finish": true.
3. **DO NOT EXECUTE SUGGESTIONS:** If a tool result says "Mitigation: Consult a doctor", do NOT call a tool to do that. REPORT the mitigation to the user.
4. **FORMAT:** Output strict JSON. 
5. **ACTION FORMAT:** The "action" field MUST be a JSON object, NOT a string.
   - CORRECT: "action": {{ "tool_id": "assess_risk", "input": {{...}} }}
   - WRONG: "action": "assess_risk"

JSON FORMAT:
{{
  "thought": "I have the risk assessment results. I will summarize them for the user now.",
  "finish": true,
  "action": null,
  "final_answer": {{ "content": "Summary of findings..." }}
}}
""".strip()

    user_prompt = f"""
GOAL: {goal.description}

HISTORY: 
{last_user}

CONTEXT:
{ext_text}

TRACE:
{prior_summary}

NEXT STEP (JSON ONLY):
""".strip()

    planner_model = os.environ.get("PLANNER_MODEL") or "llama-3-8b-instruct-q4_k_m"

    payload = {
        "model": planner_model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
        ],
        "options": {"temperature": 0.1},
    }

    env = BaseEnvelope(
        kind="llm.chat.request",
        source={"name": "planner-react", "node": settings.node_name},
        correlation_id=trace_id,
        reply_to=reply_channel,
        payload=payload,
    )

    msg = await bus.rpc_request(
        exec_channel,
        env,
        reply_channel=reply_channel,
        timeout_sec=float(limits.timeout_seconds),
    )
    decoded = bus.codec.decode(msg.get("data"))
    if not decoded.ok:
        raise RuntimeError(f"LLM RPC Error: {decoded.error}")

    resp_payload = decoded.envelope.payload or {}
    if "error" in resp_payload:
        raise RuntimeError(f"LLM Gateway Error: {resp_payload.get('error')}")

    try:
        chat_res = ChatResultPayload(**resp_payload)
        text = chat_res.text
    except Exception:
        text = resp_payload.get("content") or resp_payload.get("text") or ""

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

    # Build a strict CortexClientRequest (agent mode, recall disabled)
    content = context.get("text") if isinstance(context.get("text"), str) else json.dumps(context or {}, ensure_ascii=False)
    ctx = CortexClientContext(
        messages=[LLMMessage(role="user", content=content)],
        session_id=context.get("session_id"),
        user_id=context.get("user_id"),
        trace_id=trace_id,
        metadata={},
    )
    cortex_req = CortexClientRequest(
        mode="agent",
        verb_name=verb_name,
        packs=[],
        options={},
        recall=RecallDirective(enabled=False),
        context=ctx,
    )

    envelope = BaseEnvelope(
        kind="cortex.orch.request",
        source={"name": settings.service_name, "node": settings.node_name},
        correlation_id=trace_id,
        reply_to=reply_channel,
        payload=cortex_req.model_dump(mode="json"),
    )

    msg = await bus.rpc_request(
        settings.cortex_request_channel,
        envelope,
        reply_channel=reply_channel,
        timeout_sec=timeout_ms / 1000.0,
    )

    decoded = bus.codec.decode(msg.get("data"))
    if not decoded.ok:
        raise RuntimeError(f"Cortex RPC decode failed: {decoded.error}")

    env_payload = decoded.envelope.payload
    return env_payload if isinstance(env_payload, dict) else decoded.envelope.payload.model_dump(mode="json")


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
                if text:
                    break

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


async def run_react_loop(payload: PlannerRequest) -> PlannerResponse:
    bus = OrionBusAsync(url=settings.orion_bus_url)
    await bus.connect()

    trace: List[TraceStep] = list(getattr(payload, "trace", []) or [])
    tools_called: List[str] = []
    start = time.monotonic()
    final_answer: Optional[FinalAnswer] = None
    steps_used = 0
    delegate_only = getattr(payload.preferences, "plan_only", False) or getattr(
        payload.preferences, "delegate_tool_execution", False
    )

    try:
        for step_index in range(payload.limits.max_steps):
            steps_used = step_index + 1

            planner_step = await _call_planner_llm(
                bus=bus,
                goal=payload.goal,
                toolset=payload.toolset,
                context=payload.context,
                prior_trace=trace,
                limits=payload.limits,
            )

            thought = planner_step.get("thought", "")
            finish = planner_step.get("finish", False)
            action = planner_step.get("action")
            final = planner_step.get("final_answer")

            if finish or (final and not action):
                raw_content = final.get("content") if final else thought
                if isinstance(raw_content, (dict, list)):
                    content = json.dumps(raw_content, ensure_ascii=False)
                elif raw_content is None:
                    content = ""
                else:
                    content = str(raw_content)

                structured = final.get("structured", {}) if final else {}
                final_answer = FinalAnswer(content=content, structured=structured)
                trace.append(TraceStep(step_index=step_index, thought=thought))
                break

            if not action:
                logger.warning(f"Step {step_index}: LLM returned no action and finish=False. Terminating.")
                final_answer = FinalAnswer(content=thought or "Planner halted: No action provided by LLM.")
                trace.append(TraceStep(step_index=step_index, thought=thought))
                break

            if isinstance(action, dict):
                raw_tool_id = action.get("tool_id")
                tool_input = action.get("input", {})
            elif isinstance(action, str):
                logger.warning(
                    f"Step {step_index}: LLM returned string action '{action}' instead of object. Attempting auto-fix."
                )
                raw_tool_id = action
                tool_input = {}
            else:
                raw_tool_id = None
                tool_input = {}

            if not raw_tool_id:
                final_answer = FinalAnswer(content=thought or "Invalid action format.")
                break

            tool_id = _normalize_tool_id(raw_tool_id, payload.toolset)

            if delegate_only:
                trace.append(
                    TraceStep(
                        step_index=step_index,
                        thought=thought,
                        action={"tool_id": tool_id, "input": tool_input},
                        observation=None,
                    )
                )
                final_answer = None
                break

            try:
                raw_cortex = await _call_cortex_verb(
                    bus=bus,
                    verb_name=tool_id,
                    context=tool_input,
                    timeout_ms=payload.limits.timeout_seconds * 1000,
                )
                obs_bundle = _extract_llm_output_from_cortex(raw_cortex)

                if not obs_bundle.get("llm_output"):
                    err = raw_cortex.get("error") or "Unknown error"
                    obs_bundle["llm_output"] = f"Tool executed but returned no text. (System Error: {err})"

                observation = obs_bundle
                tools_called.append(tool_id)
            except Exception as e:
                observation = {"error": str(e)}

            trace.append(
                TraceStep(
                    step_index=step_index,
                    thought=thought,
                    action={"tool_id": tool_id, "input": tool_input},
                    observation=observation,
                )
            )

        if not final_answer and not delegate_only:
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
            duration_ms=int((time.monotonic() - start) * 1000),
        ),
    )


@router.post("/plan/react", response_model=PlannerResponse)
async def plan_react(payload: PlannerRequest) -> PlannerResponse:
    if not settings.orion_bus_enabled:
        raise HTTPException(status_code=500, detail="Bus disabled")
    return await run_react_loop(payload)
