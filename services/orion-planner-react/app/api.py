from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException

from orion.core.llm_json import parse_json_object, repair_json
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


class PlannerTransportError(RuntimeError):
    """LLM transport, RPC, or gateway failure."""


class PlannerParseError(RuntimeError):
    """Planner response could not be parsed as JSON."""

    def __init__(self, message: str, *, raw_text: str = "", salvage_succeeded: bool = False) -> None:
        super().__init__(message)
        self.raw_text = raw_text
        self.salvage_succeeded = salvage_succeeded


class PlannerSchemaError(RuntimeError):
    """Planner response JSON was present but semantically invalid."""


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────


def _truncate_for_log(value: Any, limit: int = 300) -> str:
    if isinstance(value, str):
        raw = value
    else:
        try:
            raw = json.dumps(value, ensure_ascii=False)
        except Exception:
            raw = repr(value)
    if len(raw) > limit:
        return raw[:limit] + "... [TRUNCATED]"
    return raw




def _strip_code_fence_wrapper(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if not lines:
        return stripped

    opening = lines[0].strip()
    if not opening.startswith("```"):
        return stripped

    body = lines[1:]
    if body and body[-1].strip().startswith("```"):
        body = body[:-1]
    return "\n".join(body).strip()


def _strip_leading_language_marker(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return stripped

    lines = stripped.splitlines()
    if not lines:
        return stripped

    marker = lines[0].strip().lower()
    if marker in {"bash", "sh", "plaintext", "json"} and len(lines) > 1:
        return "\n".join(lines[1:]).strip()
    return stripped


def _extract_first_balanced_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return None


def _extract_text_field(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, dict):
        for key in ("content", "text", "answer"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate
    return None


def _final_answer_type_name(payload: Any) -> str:
    if isinstance(payload, dict) and "final_answer" in payload:
        return type(payload.get("final_answer")).__name__
    return "missing"


def _parse_planner_response_text(raw_text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    raw = raw_text if isinstance(raw_text, str) else str(raw_text or "")
    stripped_raw = raw.strip()
    first_line = stripped_raw.splitlines()[0].strip().lower() if stripped_raw.splitlines() else ""
    needs_salvage_hint = (
        stripped_raw.startswith("```")
        or first_line in {"bash", "sh", "plaintext", "json"}
        or not stripped_raw.startswith("{")
        or not stripped_raw.endswith("}")
    )

    parse_attempts: List[Tuple[str, bool]] = []

    def add_candidate(candidate: Optional[str], *, salvaged: bool) -> None:
        if not candidate:
            return
        normalized = candidate.strip()
        if not normalized:
            return
        if any(existing == normalized for existing, _ in parse_attempts):
            return
        parse_attempts.append((normalized, salvaged))

    outer_balanced = _extract_first_balanced_json_object(raw)
    add_candidate(outer_balanced, salvaged=bool(outer_balanced and outer_balanced.strip() != raw.strip()))
    add_candidate(raw if stripped_raw else None, salvaged=False)

    fence_stripped = _strip_code_fence_wrapper(raw)
    marker_stripped = _strip_leading_language_marker(raw)
    fence_then_marker = _strip_leading_language_marker(fence_stripped)
    marker_then_fence = _strip_code_fence_wrapper(marker_stripped)

    add_candidate(fence_stripped, salvaged=fence_stripped.strip() != raw.strip())
    add_candidate(marker_stripped, salvaged=marker_stripped.strip() != raw.strip())
    add_candidate(fence_then_marker, salvaged=fence_then_marker.strip() != raw.strip())
    add_candidate(marker_then_fence, salvaged=marker_then_fence.strip() != raw.strip())

    for candidate, _ in list(parse_attempts):
        balanced = _extract_first_balanced_json_object(candidate)
        add_candidate(balanced, salvaged=bool(balanced and balanced.strip() != raw.strip()))

    for candidate, salvaged in parse_attempts:
        try:
            parsed = parse_json_object(candidate)
            return parsed, {
                "salvage_succeeded": salvaged,
                "raw_snippet": _truncate_for_log(raw),
                "salvaged_snippet": _truncate_for_log(candidate),
            }
        except Exception:
            continue

    if not needs_salvage_hint:
        try:
            parsed = parse_json_object(raw)
            return parsed, {"salvage_succeeded": False, "raw_snippet": _truncate_for_log(raw)}
        except Exception:
            pass

    repaired_preview = repair_json(raw)
    if len(repaired_preview) > 500:
        repaired_preview = repaired_preview[:500] + "... [TRUNCATED]"
    raise PlannerParseError(
        f"Planner LLM returned non-JSON (len={len(raw)}): {repaired_preview!r}",
        raw_text=raw,
        salvage_succeeded=False,
    )

def _stringify_final_answer_node(value: Any, *, depth: int = 0) -> str:
    if depth > 4:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, dict):
        lines: List[str] = []
        for key, item in value.items():
            heading = str(key).replace("_", " ").strip().title() or "Section"
            if isinstance(item, str):
                text = item.strip()
                if text:
                    lines.append(f"### {heading}\n{text}")
                continue
            if isinstance(item, (dict, list)):
                rendered = _stringify_final_answer_node(item, depth=depth + 1)
                if rendered.strip():
                    lines.append(f"### {heading}\n{rendered}")
                continue
            if item is not None:
                lines.append(f"- **{key}**: {item}")
        return "\n\n".join(line for line in lines if line.strip())

    if isinstance(value, list):
        if not value:
            return ""
        rendered_items: List[str] = []
        for item in value:
            if isinstance(item, (dict, list)):
                rendered = _stringify_final_answer_node(item, depth=depth + 1)
                if rendered.strip():
                    rendered_items.append(f"- {rendered.replace(chr(10), chr(10) + '  ')}")
            elif item is not None:
                rendered_items.append(f"- {item}")
        return "\n".join(rendered_items)

    return ""


def _normalize_finished_final_answer(final_answer: Any, thought: str) -> Optional[Dict[str, Any]]:
    if isinstance(final_answer, str):
        return {
            "content": final_answer,
            "structured": {},
            "normalized": False,
            "type": "str",
        }

    if isinstance(final_answer, dict):
        preferred_content = _extract_text_field(final_answer)
        if preferred_content is not None:
            return {
                "content": preferred_content,
                "structured": final_answer.get("structured", final_answer) if isinstance(final_answer.get("structured"), dict) else final_answer,
                "normalized": preferred_content != final_answer.get("content"),
                "type": "dict",
            }

        rendered = _stringify_final_answer_node(final_answer)
        if not rendered.strip():
            rendered = json.dumps(final_answer, ensure_ascii=False, separators=(",", ":"))
        if rendered.strip():
            return {
                "content": rendered,
                "structured": final_answer,
                "normalized": True,
                "type": "dict",
            }
        return None

    if isinstance(final_answer, list):
        rendered = _stringify_final_answer_node(final_answer)
        if not rendered.strip():
            rendered = json.dumps(final_answer, ensure_ascii=False, separators=(",", ":"))
        if rendered.strip():
            return {
                "content": rendered,
                "structured": {"items": final_answer},
                "normalized": True,
                "type": "list",
            }
        return None

    if final_answer is None and isinstance(thought, str) and thought.strip():
        return None

    return None


def _validate_or_normalize_planner_step(planner_step: Any) -> Dict[str, Any]:
    if not isinstance(planner_step, dict):
        raise PlannerSchemaError(f"planner response must be an object, got {type(planner_step).__name__}")

    normalized = dict(planner_step)
    thought = normalized.get("thought", "")
    normalized["thought"] = thought if isinstance(thought, str) else str(thought or "")

    finish = normalized.get("finish", False)
    if not isinstance(finish, bool):
        raise PlannerSchemaError(f"finish must be boolean, got {type(finish).__name__}")
    normalized["finish"] = finish

    action = normalized.get("action")
    if action is not None and not isinstance(action, (dict, str)):
        raise PlannerSchemaError(f"action must be object|string|null, got {type(action).__name__}")

    final_info = _normalize_finished_final_answer(normalized.get("final_answer"), normalized["thought"]) if finish else None
    if finish:
        if final_info is None or not final_info["content"].strip():
            raise PlannerSchemaError("finish=true requires a usable final_answer")
        normalized["final_answer"] = final_info["content"]
        normalized["_final_answer_structured"] = final_info["structured"]
        normalized["_final_answer_normalized"] = final_info["normalized"]
        normalized["_final_answer_type"] = final_info["type"]
        normalized["action"] = action if isinstance(action, (dict, str)) else None
        return normalized

    if action is None:
        raise PlannerSchemaError("finish=false requires an action")

    normalized["_final_answer_structured"] = {}
    normalized["_final_answer_normalized"] = False
    normalized["_final_answer_type"] = type(normalized.get("final_answer")).__name__
    return normalized


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
    system_override: Optional[str] = None,
    options_override: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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

    ext_facts = context.external_facts if isinstance(context.external_facts, dict) else {}
    ext_text = (ext_facts.get("text") or "")[:10000]
    om = ext_facts.get("output_mode") or ""
    rp = ext_facts.get("response_profile") or ""

    system_msg = (system_override or f"""
You are Orion's internal ReAct planner.

RUNTIME ROUTING (must respect):
- output_mode: {om or "(not set)"}
- response_profile: {rp or "(not set)"}
When output_mode is implementation_guide, tutorial, code_delivery, or direct_answer for instructional asks:
  Prefer delivery tools (write_guide, write_tutorial, answer_direct, finalize_response) over plan_action.
When output_mode is code_delivery: prefer generate_code_scaffold.
When output_mode is comparative_analysis: prefer compare_options.
When output_mode is decision_support: prefer write_recommendation.

AVAILABLE TOOLS:
{tools_description}

CORE INSTRUCTIONS:
1. **ANSWER THE USER'S INTENT EFFICIENTLY.** 2. **DEFINITION OF DONE:** As soon as you have the information to answer the user, STOP. Set "finish": true.
3. **DO NOT EXECUTE SUGGESTIONS:** If a tool result says "Mitigation: Consult a doctor", do NOT call a tool to do that. REPORT the mitigation to the user.
4. **FORMAT:** Output strict JSON. 
5. **TRIAGE RULE:** triage is ONLY allowed when TRACE is empty (step 0). After step 0, do NOT call triage. If TRACE is empty and the toolset includes "triage", you may choose "triage" as the first action. Triage output is internal—use it to select the next tool, then produce the actual answer.
6. **INPUT FIELD MAPPING:** Populate tool-specific intent fields when calling tools:
   - triage.request
   - plan_action.goal
   - goal_formulate.intention
   - summarize_context.context_raw
   - tag_enrich.fragment
   - pattern_detect.fragments
   - evaluate.output
   - assess_risk.scenario
   - answer_direct.request, finalize_response.original_request, write_guide.request
   - write_tutorial.request, write_runbook.request, write_recommendation.request
   - compare_options.request, synthesize_patterns.request, generate_code_scaffold.request
   Also include "text" with the raw user request whenever possible.
7. **ACTION FORMAT:** The "action" field MUST be a JSON object, NOT a string.
   - CORRECT: "action": {{ "tool_id": "assess_risk", "input": {{...}} }}
   - WRONG: "action": "assess_risk"
8. **NO DOUBLE ENCODING:** Output must be a JSON OBJECT, not a JSON string.
   - Do not wrap output in single or double quotes.
   - Do not escape quotes with backslashes.
   - WRONG: "{{\"thought\": \"...\"}}"
   - RIGHT: {{"thought": "..."}}

9. **NO REPEATED TOOLS:** Do not call the same tool twice. Use the prior result; if you can answer, set finish: true.

JSON FORMAT:
{{
  "thought": "I have the risk assessment results. I will summarize them for the user now.",
  "finish": true,
  "action": null,
  "final_answer": {{ "content": "Summary of findings..." }}
}}
WRONG:
{{"thought":"Need more info","finish":false,"action":null,"final_answer":null}}

RIGHT:
{{"thought":"Need tool output first.","finish":false,"action":{{"tool_id":"triage","input":{{"text":"..."}}}},"final_answer":null}}
""".strip())

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
        "options": options_override or {"temperature": 0.1, "max_tokens": 256, "return_json": True},
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
        raise PlannerTransportError(f"LLM RPC Error: {decoded.error}")

    resp_payload = decoded.envelope.payload or {}
    if "error" in resp_payload:
        raise PlannerTransportError(f"LLM Gateway Error: {resp_payload.get('error')}")

    try:
        chat_res = ChatResultPayload(**resp_payload)
        text = chat_res.text
    except Exception:
        text = resp_payload.get("content") or resp_payload.get("text") or ""

    if "<think>" in text:
        text = text.split("</think>")[-1].strip()

    if text.strip().startswith("[Error:"):
        raise PlannerTransportError(f"LLM Gateway error: {text.strip()}")

    return _parse_planner_response_text(text)


def _planner_error_step(message: str) -> Dict[str, Any]:
    return {
        "thought": message,
        "finish": False,
        "action": None,
        "final_answer": None,
    }


def _fallback_tool_id(step_index: int, toolset: List[ToolDef]) -> Optional[str]:
    tool_ids = {t.tool_id for t in toolset}
    if step_index == 0 and "triage" in tool_ids:
        return "triage"
    if "plan_action" in tool_ids:
        return "plan_action"
    if "analyze_text" in tool_ids:
        return "analyze_text"
    if toolset:
        return toolset[0].tool_id
    return None


async def _repair_or_fallback_step(
    *,
    bus: OrionBusAsync,
    corr_id: str,
    step_index: int,
    planner_step: Dict[str, Any],
    goal: Goal,
    toolset: List[ToolDef],
    context: ContextBlock,
    prior_trace: List[TraceStep],
    limits: Limits,
    delegate_only: bool,
) -> Dict[str, Any]:
    repair_system = (
        "You violated planner contract. Output JSON ONLY with either "
        "(finish=true and final_answer) OR (finish=false and action object). "
        "Keep action.input compact; never paste full user text. Use text <= 300 chars."
    )
    try:
        repaired, repair_meta = await _call_planner_llm(
            bus=bus,
            goal=goal,
            toolset=toolset,
            context=context,
            prior_trace=prior_trace,
            limits=limits,
            system_override=repair_system,
            options_override={"temperature": 0, "max_tokens": 128, "return_json": True},
        )
    except PlannerTransportError as e:
        logger.warning(
            "corr_id=%s step=%s failure_category=repair_transport_failure normalization_succeeded=false detail=%s",
            corr_id,
            step_index,
            e,
        )
        repaired = _planner_error_step(f"Planner repair failed: {e}")
        repair_meta = {"salvage_succeeded": False, "raw_snippet": _truncate_for_log(getattr(e, "raw_text", str(e)))}
    except PlannerParseError as e:
        logger.warning(
            "corr_id=%s step=%s failure_category=repair_parse_failure final_answer_type=missing normalization_succeeded=false salvage_succeeded=%s raw_snippet=%s detail=%s",
            corr_id,
            step_index,
            str(bool(getattr(e, "salvage_succeeded", False))).lower(),
            _truncate_for_log(getattr(e, "raw_text", str(e))),
            e,
        )
        repaired = _planner_error_step(f"Planner repair failed: {e}")
        repair_meta = {"salvage_succeeded": False, "raw_snippet": _truncate_for_log(getattr(e, "raw_text", str(e)))}
    except Exception as e:
        logger.warning(
            "corr_id=%s step=%s failure_category=repair_unknown_failure normalization_succeeded=false detail=%s",
            corr_id,
            step_index,
            e,
        )
        repaired = _planner_error_step(f"Planner repair failed: {e}")
        repair_meta = {"salvage_succeeded": False, "raw_snippet": _truncate_for_log(getattr(e, "raw_text", str(e)))}

    try:
        normalized_repaired = _validate_or_normalize_planner_step(repaired)
        if repair_meta.get("salvage_succeeded"):
            logger.info(
                "corr_id=%s step=%s failure_category=repair_salvaged final_answer_type=%s normalization_succeeded=%s salvage_succeeded=true raw_snippet=%s",
                corr_id,
                step_index,
                normalized_repaired.get("_final_answer_type", _final_answer_type_name(repaired)),
                str(bool(normalized_repaired.get("_final_answer_normalized"))).lower(),
                repair_meta.get("raw_snippet", _truncate_for_log(repaired)),
            )
        if normalized_repaired.get("_final_answer_normalized"):
            logger.info(
                "corr_id=%s step=%s failure_category=normalization_applied final_answer_type=%s normalization_succeeded=true salvage_succeeded=%s raw_snippet=%s",
                corr_id,
                step_index,
                normalized_repaired.get("_final_answer_type"),
                str(bool(repair_meta.get("salvage_succeeded"))).lower(),
                repair_meta.get("raw_snippet", _truncate_for_log(repaired)),
            )
        return normalized_repaired
    except PlannerSchemaError as e:
        logger.warning(
            "corr_id=%s step=%s failure_category=repair_schema_validation final_answer_type=%s normalization_succeeded=false salvage_succeeded=%s raw_snippet=%s detail=%s",
            corr_id,
            step_index,
            _final_answer_type_name(repaired),
            str(bool(repair_meta.get("salvage_succeeded"))).lower(),
            repair_meta.get("raw_snippet", _truncate_for_log(repaired)),
            e,
        )

    if delegate_only:
        logger.warning(
            "corr_id=%s step=%s failure_category=repair_fallback_invoked normalization_succeeded=false salvage_succeeded=%s raw_snippet=%s",
            corr_id,
            step_index,
            str(bool(repair_meta.get("salvage_succeeded"))).lower(),
            repair_meta.get("raw_snippet", _truncate_for_log(planner_step)),
        )
        tool_id = _fallback_tool_id(step_index, toolset)
        if not tool_id:
            return {
                "thought": f"{planner_step.get('thought', '')} Planner contract failed and no tool is available.",
                "finish": True,
                "action": None,
                "final_answer": {"content": "Planner failed: no available delegate tool."},
            }

        last_user = ""
        if context.conversation_history:
            last_msg = context.conversation_history[-1]
            last_user = last_msg.content if hasattr(last_msg, "content") else last_msg.get("content", "")
        text_value = last_user or goal.description

        return {
            "thought": f"{planner_step.get('thought', '')} [Fallback action injected after contract violation.]".strip(),
            "finish": False,
            "action": {"tool_id": tool_id, "input": {"text": text_value, "request": goal.description}},
            "final_answer": None,
        }

    logger.warning(
        "corr_id=%s step=%s failure_category=repair_fallback_invoked normalization_succeeded=false salvage_succeeded=%s raw_snippet=%s",
        corr_id,
        step_index,
        str(bool(repair_meta.get("salvage_succeeded"))).lower(),
        repair_meta.get("raw_snippet", _truncate_for_log(planner_step)),
    )
    return {
        "thought": planner_step.get("thought", ""),
        "finish": True,
        "action": None,
        "final_answer": {"content": "Planner failed contract: finish=false with no action after repair attempt."},
    }


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

    corr_id = payload.request_id or f"planner-react-{uuid.uuid4()}"
    trace: List[TraceStep] = list(payload.trace or [])
    tools_called: List[str] = []
    start = time.monotonic()
    final_answer: Optional[FinalAnswer] = None
    steps_used = 0
    delegate_only = getattr(payload.preferences, "plan_only", False) or getattr(payload.preferences, "delegate_tool_execution", False)

    try:
        for step_index in range(payload.limits.max_steps):
            steps_used = step_index + 1

            try:
                raw_planner_step, planner_meta = await _call_planner_llm(
                    bus=bus,
                    goal=payload.goal,
                    toolset=payload.toolset,
                    context=payload.context,
                    prior_trace=trace,
                    limits=payload.limits,
                )
            except PlannerTransportError as e:
                logger.warning(
                    "corr_id=%s step=%s failure_category=llm_transport_failure normalization_succeeded=false detail=%s",
                    corr_id,
                    step_index,
                    e,
                )
                planner_step = _planner_error_step(f"Planner call failed: {e}")
                planner_meta = {"salvage_succeeded": False, "raw_snippet": _truncate_for_log(getattr(e, "raw_text", str(e)))}
            except PlannerParseError as e:
                logger.warning(
                    "corr_id=%s step=%s failure_category=planner_response_parse_failure final_answer_type=missing normalization_succeeded=false salvage_succeeded=%s raw_snippet=%s detail=%s",
                    corr_id,
                    step_index,
                    str(bool(getattr(e, "salvage_succeeded", False))).lower(),
                    _truncate_for_log(getattr(e, "raw_text", str(e))),
                    e,
                )
                planner_step = _planner_error_step(f"Planner call failed: {e}")
                planner_meta = {"salvage_succeeded": False, "raw_snippet": _truncate_for_log(getattr(e, "raw_text", str(e)))}
            except Exception as e:
                logger.warning(
                    "corr_id=%s step=%s failure_category=llm_unknown_failure normalization_succeeded=false detail=%s",
                    corr_id,
                    step_index,
                    e,
                )
                planner_step = _planner_error_step(f"Planner call failed: {e}")
                planner_meta = {"salvage_succeeded": False, "raw_snippet": _truncate_for_log(getattr(e, "raw_text", str(e)))}
            else:
                try:
                    planner_step = _validate_or_normalize_planner_step(raw_planner_step)
                    if planner_meta.get("salvage_succeeded"):
                        logger.info(
                            "corr_id=%s step=%s failure_category=planner_response_salvaged final_answer_type=%s normalization_succeeded=%s salvage_succeeded=true raw_snippet=%s",
                            corr_id,
                            step_index,
                            planner_step.get("_final_answer_type", _final_answer_type_name(raw_planner_step)),
                            str(bool(planner_step.get("_final_answer_normalized"))).lower(),
                            planner_meta.get("raw_snippet", _truncate_for_log(raw_planner_step)),
                        )
                    if planner_step.get("_final_answer_normalized"):
                        logger.info(
                            "corr_id=%s step=%s failure_category=normalization_applied final_answer_type=%s normalization_succeeded=true salvage_succeeded=%s raw_snippet=%s",
                            corr_id,
                            step_index,
                            planner_step.get("_final_answer_type"),
                            str(bool(planner_meta.get("salvage_succeeded"))).lower(),
                            planner_meta.get("raw_snippet", _truncate_for_log(raw_planner_step)),
                        )
                except PlannerSchemaError as e:
                    logger.warning(
                        "corr_id=%s step=%s failure_category=planner_response_schema_validation final_answer_type=%s normalization_succeeded=false salvage_succeeded=%s raw_snippet=%s detail=%s",
                        corr_id,
                        step_index,
                        _final_answer_type_name(raw_planner_step),
                        str(bool(planner_meta.get("salvage_succeeded"))).lower(),
                        planner_meta.get("raw_snippet", _truncate_for_log(raw_planner_step)),
                        e,
                    )
                    planner_step = _planner_error_step(f"Planner response invalid: {e}")

            thought = planner_step.get("thought", "")
            finish = planner_step.get("finish", False)
            action = planner_step.get("action")
            final = planner_step.get("final_answer")

            if not finish and action is None:
                planner_step = await _repair_or_fallback_step(
                    bus=bus,
                    corr_id=corr_id,
                    step_index=step_index,
                    planner_step=planner_step,
                    goal=payload.goal,
                    toolset=payload.toolset,
                    context=payload.context,
                    prior_trace=trace,
                    limits=payload.limits,
                    delegate_only=delegate_only,
                )
                thought = planner_step.get("thought", "")
                finish = planner_step.get("finish", False)
                action = planner_step.get("action")
                final = planner_step.get("final_answer")

            if finish or (final and not action):
                content = final if isinstance(final, str) else (thought or "")
                structured = planner_step.get("_final_answer_structured", {})
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
        return PlannerResponse(status="error", stop_reason="error", error={"message": str(e)}, request_id=payload.request_id)
    finally:
        await bus.close()

    stop_reason = "continue"
    continue_reason = None
    if final_answer and final_answer.content:
        stop_reason = "final_answer"
    if trace:
        last_action = trace[-1].action if isinstance(trace[-1].action, dict) else None
        if isinstance(last_action, dict):
            stop_reason = "delegate"
            continue_reason = "action_present"
    if final_answer is None and not trace:
        stop_reason = "continue"

    return PlannerResponse(
        request_id=payload.request_id,
        status="ok",
        stop_reason=stop_reason,
        continue_reason=continue_reason,
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
