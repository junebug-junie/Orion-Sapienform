# services/orion-planner-react/app/api.py

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
import sys
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .settings import settings
from orion.core.bus.service import OrionBus

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger("planner-react.api")
logger.setLevel(logging.INFO)

router = APIRouter()


# ─────────────────────────────────────────────
# Pydantic models — ReAct contract
# ─────────────────────────────────────────────

Role = Literal["user", "assistant", "system"]


class Message(BaseModel):
    role: Role
    content: str


class Goal(BaseModel):
    type: str = "chat"
    description: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ContextBlock(BaseModel):
    conversation_history: List[Message] = Field(default_factory=list)
    orion_state_snapshot: Dict[str, Any] = Field(default_factory=dict)
    external_facts: Dict[str, Any] = Field(default_factory=dict)


class ToolDef(BaseModel):
    tool_id: str
    description: Optional[str] = None
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)


class Limits(BaseModel):
    max_steps: int = 4
    max_tokens_reason: int = 2048
    max_tokens_answer: int = 1024
    timeout_seconds: int = 60


class Preferences(BaseModel):
    style: str = "neutral"
    allow_internal_thought_logging: bool = True
    return_trace: bool = True


class PlannerRequest(BaseModel):
    request_id: Optional[str] = None
    caller: str = "hub"
    goal: Goal
    context: ContextBlock = Field(default_factory=ContextBlock)
    toolset: List[ToolDef] = Field(default_factory=list)
    limits: Limits = Field(default_factory=Limits)
    preferences: Preferences = Field(default_factory=Preferences)


class TraceStep(BaseModel):
    step_index: int
    thought: Optional[str] = None
    action: Optional[Dict[str, Any]] = None
    observation: Optional[Dict[str, Any]] = None


class Usage(BaseModel):
    steps: int
    tokens_reason: int
    tokens_answer: int
    tools_called: List[str]
    duration_ms: int


class FinalAnswer(BaseModel):
    content: str
    structured: Dict[str, Any] = Field(default_factory=dict)


class PlannerResponse(BaseModel):
    request_id: Optional[str] = None
    status: Literal["ok", "error", "timeout"] = "ok"
    error: Optional[Dict[str, Any]] = None
    final_answer: Optional[FinalAnswer] = None
    trace: List[TraceStep] = Field(default_factory=list)
    usage: Optional[Usage] = None


# ─────────────────────────────────────────────
# Hub-style RPC helper using OrionBus
# ─────────────────────────────────────────────

async def _request_and_wait(
    bus: OrionBus,
    channel_intake: str,
    channel_reply: str,
    payload: dict,
    trace_id: str,
    timeout_sec: float = 60.0,
) -> dict:
    """
    Planner-style clone of Hub's _request_and_wait:

    1) Subscribe FIRST on channel_reply.
    2) Spin a background thread to pump messages into an asyncio.Queue.
    3) Publish on channel_intake.
    4) Await first reply from queue or timeout.
    """
    if not bus or not getattr(bus, "enabled", False):
        raise RuntimeError("_request_and_wait used while OrionBus is disabled")

    sub = bus.raw_subscribe(channel_reply)
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def listener():
        try:
            for msg in sub:
                loop.call_soon_threadsafe(queue.put_nowait, msg)
                break
        finally:
            sub.close()

    asyncio.get_running_loop().run_in_executor(None, listener)

    # Publish after subscription is live
    bus.publish(channel_intake, payload)
    logger.info(
        "[%s] RPC Published -> %s (awaiting %s)",
        trace_id,
        channel_intake,
        channel_reply,
    )

    try:
        msg = await asyncio.wait_for(queue.get(), timeout=timeout_sec)
        reply = msg.get("data", {})
        logger.info("[%s] RPC reply received on %s.", trace_id, channel_reply)
        return reply
    except asyncio.TimeoutError:
        logger.error("[%s] RPC timed out waiting for %s", trace_id, channel_reply)
        return {"error": "timeout"}
    finally:
        # sub closed in listener
        pass


# ─────────────────────────────────────────────
# LLM planning step (LLM Gateway via Exec bus)
# ─────────────────────────────────────────────

async def _call_planner_llm(
    bus: OrionBus,
    *,
    goal: Goal,
    toolset: List[ToolDef],
    context: ContextBlock,
    prior_trace: List[TraceStep],
    limits: Limits,
) -> Dict[str, Any]:
    """
    Use LLM Gateway's 'chat' event to do a single ReAct planning step.
    Expects the LLM to return JSON in the body text.
    Handles DeepSeek R1 <think> blocks by extracting them and merging them into 'thought'.
    """
    trace_id = str(uuid.uuid4())

    # Same bus channels Hub uses (exec_request_prefix is fine as "llm-exec" style)
    exec_channel = f"{settings.exec_request_prefix}:{settings.llm_gateway_service_name}"
    reply_channel = f"{settings.llm_reply_prefix}:{trace_id}"

    # ─────────────────────────────────────────
    # Tools description for prompt
    # ─────────────────────────────────────────
    tools_description = "\n".join(
        f"- {t.tool_id}: {t.description or ''}" for t in toolset
    ) or "(none)"

    # Small summary of prior trace
    prior_summary = ""
    if prior_trace:
        lines: List[str] = []
        for s in prior_trace[-3:]:
            # ───────────────────────────────────────────────────────
            # FIX: Truncate historical observations to prevent bloat
            # ───────────────────────────────────────────────────────
            obs_str = str(s.observation)
            if len(obs_str) > 1000:  # Cap history items tightly
                obs_str = obs_str[:1000] + "... [TRUNCATED]"

            lines.append(
                f"Step {s.step_index}: thought={s.thought!r}, "
                f"action={s.action}, observation={obs_str}"
            )
        prior_summary = "\n".join(lines)
    # ─────────────────────────────────────────────────────────────
    # CONTEXT PREPARATION & SAFETY TRUNCATION
    # ─────────────────────────────────────────────────────────────

    # 1. Truncate User Message (History)
    last_user = ""
    if context.conversation_history:
        raw_user = context.conversation_history[-1].content or ""
        MAX_USER_CHARS = 14000
        if len(raw_user) > MAX_USER_CHARS:
            logger.warning(f"Truncating last_user from {len(raw_user)} to {MAX_USER_CHARS} chars")
            last_user = raw_user[:MAX_USER_CHARS] + "\n... [TRUNCATED]"
        else:
            last_user = raw_user

    # 2. Truncate External Facts
    ext_text = context.external_facts.get("text") or ""
    MAX_EXT_CHARS = 10000 
    if len(ext_text) > MAX_EXT_CHARS:
        logger.warning(
            "Truncating external_facts from %d to %d chars to prevent context overflow.", 
            len(ext_text), 
            MAX_EXT_CHARS
        )
        ext_text = ext_text[:MAX_EXT_CHARS] + "\n... [TRUNCATED DUE TO CONTEXT LIMITS]"

    # ─────────────────────────────────────────
    # System instructions
    # ─────────────────────────────────────────
    system_msg = f"""
You are Orion's internal ReAct planner.

Your job:
- Look at the goal, tools, and recent history.
- Decide either to call ONE tool or to finish with a final answer.
- Respond ONLY with a single JSON object, no backticks, no commentary.

JSON schema:
{{
  "thought": "string",
  "finish": true | false,
  "action": {{
    "tool_id": "string | null",
    "input": {{ "..." : "..." }}
  }} | null,
  "final_answer": {{
    "content": "string",
    "structured": {{ "..." : "..." }}
  }} | null
}}

Rules:
- If "finish" = true, then "final_answer" MUST be non-null and "action" MUST be null.
- If "finish" = false, then "action" MUST be non-null and "final_answer" MUST be null.
- tool_id MUST be one of: {[t.tool_id for t in toolset]}.
""".strip()

    # ─────────────────────────────────────────
    # User-side prompt with the concrete planning context
    # ─────────────────────────────────────────
    user_prompt = f"""
GOAL:
{goal.description}

TOOLS:
{tools_description}

RECENT HISTORY (last user message):
{last_user or "(none)"}

OPTIONAL TEXT (for tools like extract_facts):
{ext_text or "(none)"}

PRIOR TRACE SUMMARY:
{prior_summary or "(no previous steps)"}

Now decide the next step according to the JSON schema above.
""".strip()

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_prompt},
    ]

    body = {
        # Let env var specify model, or fall back to None (Gateway default)
        "model": os.environ.get("PLANNER_MODEL"),
        "messages": messages,
        "options": {
            "temperature": 0.1,
            # You can add "backend": "vllm" here explicitly if you want:
            # "backend": "vllm",
        },
        "stream": False,
        "return_json": False,
        "trace_id": trace_id,
        "user_id": None,
        "session_id": None,
        "source": "planner-react",
        "verb": "planner-react",
        "profile_name": None,
    }

    envelope = {
        "event": "chat",
        "service": settings.llm_gateway_service_name,
        "correlation_id": trace_id,
        "reply_channel": reply_channel,
        "payload": {
            "body": body,
        },
    }

    raw_reply = await _request_and_wait(
        bus=bus,
        channel_intake=exec_channel,
        channel_reply=reply_channel,
        payload=envelope,
        trace_id=trace_id,
        timeout_sec=float(limits.timeout_seconds),
    )

    if not isinstance(raw_reply, dict):
        raise RuntimeError(f"LLM Gateway reply not a dict: {raw_reply!r}")

    payload = raw_reply.get("payload") or {}
    text = (payload.get("text") or "").strip()
    if not text:
        raise RuntimeError("LLM Gateway returned empty text for planner-react")

    # ─────────────────────────────────────────────────────────────
    # FIX: Sanitize LLM Output (DeepSeek / Markdown / Escapes)
    # ─────────────────────────────────────────────────────────────
    
    internal_monologue = ""
    
    # 1. Extract <think> blocks (DeepSeek R1 style)
    # We strip it out for JSON parsing, but save it to append to "thought" later
    if "<think>" in text:
        try:
            # We want to capture the content between tags, and remove the whole block from 'text'
            # Robust split in case of multiple blocks (takes the first/main one)
            parts = text.split("</think>")
            if len(parts) > 1:
                # Part 0 contains <think>...
                pre_think = parts[0]
                # Check where it starts
                start_idx = pre_think.find("<think>")
                if start_idx != -1:
                    internal_monologue = pre_think[start_idx + 7:].strip()
                    # The JSON should be in the remainder (parts[1])
                    text = parts[1].strip()
        except Exception as e:
            logger.warning(f"Error extracting <think> block: {e}")
            # fall back to original text, JSON parse might fail but we try
            pass

    # 2. Extract strictly the JSON part
    # Finds the first '{' and the last '}'. This handles ```json wrappers, 
    # conversational filler, or trailing newlines effectively.
    idx_start = text.find("{")
    idx_end = text.rfind("}")

    if idx_start != -1 and idx_end != -1:
        text = text[idx_start : idx_end + 1]

    # 3. Fix Invalid Escapes (The "Alice\'s" bug)
    # JSON standard does NOT allow escaping single quotes, but LLMs often do it.
    text = text.replace(r"\'", "'")

    # ─────────────────────────────────────────────────────────────

    try:
        step = json.loads(text)
    except Exception as exc:
        # Surface the bad text so we can see what the model is doing
        raise RuntimeError(f"Planner LLM returned non-JSON text: {text!r}") from exc

    # 4. Merge internal monologue into the JSON thought field
    # This ensures the deep reasoning is preserved in the trace
    if internal_monologue:
        original_thought = step.get("thought", "")
        # Format it clearly so we know what is what
        combined_thought = f"[Deep Thought]\n{internal_monologue}\n\n[Planner]\n{original_thought}"
        step["thought"] = combined_thought.strip()

    return step


def _extract_llm_output_from_cortex(raw_cortex: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a Cortex-Orch reply into a compact observation bundle.
    """
    if not isinstance(raw_cortex, dict):
        raise RuntimeError(f"Cortex-Orch reply is not a dict: {type(raw_cortex)}")

    ok = raw_cortex.get("ok", None)
    if ok is not True:
        raise RuntimeError(
            f"Cortex-Orch reported error: {raw_cortex.get('error')!r}"
        )

    step_results = raw_cortex.get("step_results") or []
    if not step_results:
        # Some verbs return no steps but are OK (empty).
        return {"llm_output": "", "raw_cortex": raw_cortex}

    first_step = step_results[0] or {}
    services = first_step.get("services") or []
    if not services:
        raise RuntimeError(f"Cortex-Orch step has no services: {first_step!r}")

    first_service = services[0] or {}
    srv_payload = first_service.get("payload") or {}

    # LLM-Gateway exec_step usually nests under "result"
    result = srv_payload.get("result") or srv_payload

    # ─────────────────────────────────────────────────────────────
    # FIX: Handle case where 'result' is a plain string
    # ─────────────────────────────────────────────────────────────
    spark_meta = None
    
    if isinstance(result, dict):
        text = (
            result.get("llm_output")
            or result.get("text")
            or result.get("response")
            or ""
        )
        spark_meta = result.get("spark_meta")
    else:
        # Fallback for raw strings (backward compatible safety net)
        text = str(result)

    text = text.strip()

    return {
        "llm_output": text,
        "spark_meta": spark_meta,
        "raw_cortex": raw_cortex,
        "step_meta": {
            "verb_name": first_step.get("verb_name"),
            "step_name": first_step.get("step_name"),
            "order": first_step.get("order"),
            "service": first_service.get("service"),
        },
    }

# ─────────────────────────────────────────────
# Cortex-Orch verb call (tools)
# ─────────────────────────────────────────────


async def _call_cortex_verb(
    *,
    bus: OrionBus,
    verb_name: str,
    context: Dict[str, Any],
    timeout_ms: int,
) -> Dict[str, Any]:
    """
    Bus RPC helper: planner-react → cortex-orch.
    """
    if not bus or not getattr(bus, "enabled", False):
        raise RuntimeError("OrionBus is disabled; cannot call cortex-orch.")

    trace_id = str(uuid.uuid4())

    request_channel = settings.cortex_request_channel       # e.g. "orion-cortex:request"
    result_prefix = settings.cortex_result_prefix           # e.g. "orion-cortex:result"
    reply_channel = f"{result_prefix}:{trace_id}"

    core_payload: Dict[str, Any] = {
        "verb_name": verb_name,
        "origin_node": settings.service_name,
        "context": context or {},
        "steps": [],          # let cortex-orch load cognition/verbs/<verb>.yaml
        "timeout_ms": timeout_ms,
    }

    envelope: Dict[str, Any] = {
        # Bus/meta fields
        "event": "orchestrate_verb",
        "trace_id": trace_id,
        "origin_node": settings.service_name,
        "reply_channel": reply_channel,

        # OrchestrateVerbRequest fields at TOP LEVEL
        **core_payload,

        # And again under `payload` for older/newer handlers
        "payload": core_payload,
    }

    # Subscribe first, then publish — avoid race
    sub = bus.raw_subscribe(reply_channel)
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def listener():
        try:
            for msg in sub:
                loop.call_soon_threadsafe(queue.put_nowait, msg)
                break
        finally:
            # raw_subscribe closes pubsub in its own finally
            pass

    asyncio.get_running_loop().run_in_executor(None, listener)

    # Now publish the request
    bus.publish(request_channel, envelope)
    logger.info(
        "[%s] planner-react -> cortex-orch published (%s → %s)",
        trace_id,
        request_channel,
        reply_channel,
    )

    try:
        msg = await asyncio.wait_for(queue.get(), timeout=timeout_ms / 1000.0)
        data = msg.get("data") or {}
        return data
    except asyncio.TimeoutError:
        logger.error(
            "[%s] planner-react timed out waiting for cortex-orch on %s",
            trace_id,
            reply_channel,
        )
        return {"error": "timeout"}


# ─────────────────────────────────────────────
# Core ReAct loop
# ─────────────────────────────────────────────
async def run_react_loop(payload: PlannerRequest) -> PlannerResponse:
    bus = OrionBus(
        url=settings.orion_bus_url,
        enabled=settings.orion_bus_enabled,
    )
    if not bus.enabled:
        raise RuntimeError("OrionBus is disabled; planner-react cannot run.")

    trace: List[TraceStep] = []
    tools_called: List[str] = []
    steps_used = 0
    start = time.monotonic()

    final_answer: Optional[FinalAnswer] = None
    last_planner_step: Optional[Dict[str, Any]] = None

    try:
        for step_index in range(payload.limits.max_steps):
            steps_used = step_index + 1

            # 1) Reason: LLM chooses either tool or final answer
            planner_step = await _call_planner_llm(
                bus=bus,
                goal=payload.goal,
                toolset=payload.toolset,
                context=payload.context,
                prior_trace=trace,
                limits=payload.limits,
            )
            last_planner_step = planner_step

            thought = planner_step.get("thought", "")
            finish = bool(planner_step.get("finish", False))
            action = planner_step.get("action")
            final = planner_step.get("final_answer")

            # ─────────────────────────────────────────
            # Case 1: explicit finish=true
            # ─────────────────────────────────────────
            if finish:
                # If planner forgot to attach final_answer but gave us a thought,
                # treat the thought as the answer instead of crashing.
                if not final:
                    final_answer = FinalAnswer(
                        content=thought or "",
                        structured={},
                    )
                else:
                    final_answer = FinalAnswer(
                        content=final.get("content", ""),
                        structured=final.get("structured") or {},
                    )

                trace.append(
                    TraceStep(
                        step_index=step_index,
                        thought=thought,
                        action=None,
                        observation=None,
                    )
                )
                break

            # ─────────────────────────────────────────
            # Case 2: finish=false — try to get a tool_id
            # ─────────────────────────────────────────
            tool_id: Optional[str] = None
            tool_input: Dict[str, Any] = {}

            if isinstance(action, dict):
                tool_id = action.get("tool_id")
                tool_input = action.get("input") or {}

            # If we don't have a tool_id, treat this as an implicit finish:
            # use final_answer if present, otherwise fall back to thought.
            if not tool_id:
                if isinstance(final, dict) and final.get("content"):
                    final_answer = FinalAnswer(
                        content=final.get("content", ""),
                        structured=final.get("structured") or {},
                    )
                else:
                    final_answer = FinalAnswer(
                        content=thought or "",
                        structured={},
                    )

                trace.append(
                    TraceStep(
                        step_index=step_index,
                        thought=thought,
                        action=None,
                        observation=None,
                    )
                )
                break

            # ─────────────────────────────────────────
            # Case 3: normal tool call path
            # ─────────────────────────────────────────
            tool_id = str(tool_id)

            try:
                raw_cortex = await _call_cortex_verb(
                    bus=bus,
                    verb_name=tool_id,
                    context=tool_input,
                    timeout_ms=payload.limits.timeout_seconds * 1000,
                )

                try:
                    obs_bundle = _extract_llm_output_from_cortex(raw_cortex)
                    observation: Dict[str, Any] = obs_bundle
                    tools_called.append(tool_id)
                except Exception as exc:
                    observation = {
                        "error": str(exc),
                        "raw_cortex": raw_cortex,
                    }

            except Exception as exc:
                observation = {"error": str(exc)}

            trace.append(
                TraceStep(
                    step_index=step_index,
                    thought=thought,
                    action={"tool_id": tool_id, "input": tool_input},
                    observation=observation,
                )
            )

        # ─────────────────────────────────────────
        # If we exit the loop with no final_answer, salvage from last step
        # ─────────────────────────────────────────
        if final_answer is None:
            if last_planner_step is not None:
                thought = last_planner_step.get("thought", "")
                final = last_planner_step.get("final_answer")

                if isinstance(final, dict) and final.get("content"):
                    coerced_content = final.get("content", "")
                    structured = final.get("structured") or {}
                else:
                    if isinstance(final, dict):
                        maybe_content = final.get("content")
                    else:
                        maybe_content = None

                    coerced_content = (
                        maybe_content
                        or last_planner_step.get("answer")
                        or last_planner_step.get("text")
                        or thought
                        or ""
                    )
                    structured = {"planner_raw": last_planner_step}

                final_answer = FinalAnswer(
                    content=coerced_content,
                    structured=structured,
                )

                # Ensure we have a closing trace step
                if not trace or trace[-1].step_index != steps_used - 1:
                    trace.append(
                        TraceStep(
                            step_index=steps_used - 1,
                            thought=thought,
                            action=None,
                            observation=None,
                        )
                    )
            else:
                # Truly pathological: we never got even one planner_step
                raise RuntimeError("ReAct planner did not produce a final answer.")

    finally:
        # OrionBus doesn't need explicit close, but leaving here for future parity.
        pass

    duration_ms = int((time.monotonic() - start) * 1000)

    return PlannerResponse(
        request_id=payload.request_id,
        status="ok",
        error=None,
        final_answer=final_answer,
        trace=trace if payload.preferences.return_trace else [],
        usage=Usage(
            steps=steps_used,
            tokens_reason=0,  # could be filled from llm-gateway later
            tokens_answer=len(final_answer.content.split()),
            tools_called=tools_called,
            duration_ms=duration_ms,
        ),
    )


# ─────────────────────────────────────────────
# FastAPI route
# ─────────────────────────────────────────────

@router.post("/plan/react", response_model=PlannerResponse)
async def plan_react(payload: PlannerRequest) -> PlannerResponse:
    if not settings.orion_bus_enabled:
        raise HTTPException(
            status_code=500,
            detail="Orion bus is disabled; planner-react requires bus to talk to LLM and Cortex.",
        )

    try:
        return await run_react_loop(payload)
    except TimeoutError as exc:
        return PlannerResponse(
            request_id=payload.request_id,
            status="timeout",
            error={"message": str(exc)},
            final_answer=None,
            trace=[],
            usage=None,
        )
    except Exception as exc:
        logger.exception("planner-react error")
        return PlannerResponse(
            request_id=payload.request_id,
            status="error",
            error={"message": str(exc)},
            final_answer=None,
            trace=[],
            usage=None,
        )
