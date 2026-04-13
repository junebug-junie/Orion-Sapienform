from __future__ import annotations

import json
import logging
import os
import re
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

    def __init__(
        self,
        message: str,
        *,
        raw_text: str = "",
        salvage_succeeded: bool = False,
        parse_mode: str = "unrecoverable_parse_failure",
        salvage_source: str = "none",
    ) -> None:
        super().__init__(message)
        self.raw_text = raw_text
        self.salvage_succeeded = salvage_succeeded
        self.parse_mode = parse_mode
        self.salvage_source = salvage_source


class PlannerSchemaError(RuntimeError):
    """Planner response JSON was present but semantically invalid."""


class PlannerContractViolation(PlannerSchemaError):
    """Planner output violated the offered tool contract."""

    def __init__(
        self,
        message: str,
        *,
        failure_category: str,
        raw_action_name: Optional[str] = None,
        raw_action_id: Optional[str] = None,
        normalized_action_name: Optional[str] = None,
        normalized_action_id: Optional[str] = None,
        from_salvage: bool = False,
    ) -> None:
        super().__init__(message)
        self.failure_category = failure_category
        self.raw_action_name = raw_action_name
        self.raw_action_id = raw_action_id
        self.normalized_action_name = normalized_action_name
        self.normalized_action_id = normalized_action_id
        self.from_salvage = from_salvage


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


def _thought_reads_as_internal_planner_routing(thought: str) -> bool:
    """
    Detect planner-internal monologue (output mode / tool choice) that must not be shown as final_answer.
    """
    t = (thought or "").lower()
    signals = 0
    if "output mode" in t:
        signals += 1
    if "write_guide" in t or "write guide" in t:
        signals += 1
    if "implementation_guide" in t or "implementation guide" in t:
        signals += 1
    if "i should use" in t or "i should call" in t:
        signals += 1
    if "response profile" in t or "response_profile" in t:
        signals += 1
    if "the user is asking" in t and ("i should" in t or "write_guide" in t or "write guide" in t):
        signals += 1
    return signals >= 2


def _extract_text_field(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, dict):
        for key in ("content", "text", "answer", "message", "body", "summary", "reply"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate
    return None


def _looks_like_planner_contract_text(text: str) -> bool:
    lowered = (text or "").lower()
    return '"thought"' in lowered or '"finish"' in lowered or '"action"' in lowered or '"final_answer"' in lowered


def _extract_markdown_code_block(text: str) -> Optional[Tuple[str, str]]:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return None
    lines = stripped.splitlines()
    if len(lines) < 2:
        return None
    opening = lines[0].strip()
    language = opening[3:].strip().lower()
    body = lines[1:]
    if body and body[-1].strip().startswith("```"):
        body = body[:-1]
    body_text = "\n".join(body).strip()
    if not body_text:
        return None
    return language, body_text


def _strip_think_blocks(text: str) -> str:
    cleaned = str(text or "")
    while "<think>" in cleaned and "</think>" in cleaned:
        start = cleaned.find("<think>")
        end = cleaned.find("</think>", start)
        if end == -1:
            break
        cleaned = (cleaned[:start] + cleaned[end + len("</think>") :]).strip()
    return cleaned.strip()


def _salvage_final_answer_text(raw_text: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    stripped = raw_text.strip()
    if not stripped:
        return None, {}

    code_block = _extract_markdown_code_block(stripped)
    if code_block is not None:
        language, body = code_block
        if not _looks_like_planner_contract_text(body):
            return (
                {
                    "thought": "Planner emitted final answer content directly.",
                    "finish": True,
                    "action": None,
                    "final_answer": {"content": body},
                },
                {
                    "salvage_succeeded": True,
                    "parse_mode": "salvaged_from_code_block",
                    "salvage_source": language or "code_block",
                    "final_answer_salvaged": True,
                    "action_salvaged": False,
                    "raw_snippet": _truncate_for_log(raw_text),
                    "salvaged_snippet": _truncate_for_log(body),
                },
            )

    if not _looks_like_planner_contract_text(stripped):
        has_answer_shape = any(marker in stripped for marker in ("##", "###", "1.", "2.", "- ", "* ", "Step "))
        if has_answer_shape or len(stripped) >= 80:
            return (
                {
                    "thought": "Planner emitted mixed text final answer content.",
                    "finish": True,
                    "action": None,
                    "final_answer": {"content": stripped},
                },
                {
                    "salvage_succeeded": True,
                    "parse_mode": "salvaged_final_answer_from_mixed_text",
                    "salvage_source": "mixed_text",
                    "final_answer_salvaged": True,
                    "action_salvaged": False,
                    "raw_snippet": _truncate_for_log(raw_text),
                    "salvaged_snippet": _truncate_for_log(stripped),
                },
            )

    return None, {}


def _final_answer_type_name(payload: Any) -> str:
    if isinstance(payload, dict) and "final_answer" in payload:
        return type(payload.get("final_answer")).__name__
    return "missing"


_DESTRUCTIVE_SHELL_PATTERNS = (
    "docker rm",
    "rm -rf",
    "docker system prune",
    "docker volume rm",
    "docker network rm",
)


def _contains_destructive_shell_pattern(value: Any) -> bool:
    if value is None:
        return False
    blob = ""
    if isinstance(value, str):
        blob = value
    elif isinstance(value, dict):
        blob = json.dumps(value, ensure_ascii=False)
    else:
        blob = str(value)
    lowered = re.sub(r"\s+", " ", blob.lower())
    return any(pattern in lowered for pattern in _DESTRUCTIVE_SHELL_PATTERNS)


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

    parse_attempts: List[Tuple[str, bool, str]] = []

    def add_candidate(candidate: Optional[str], *, salvaged: bool, source: str) -> None:
        if not candidate:
            return
        normalized = candidate.strip()
        if not normalized:
            return
        if any(existing == normalized for existing, _, _ in parse_attempts):
            return
        parse_attempts.append((normalized, salvaged, source))

    outer_balanced = _extract_first_balanced_json_object(raw)
    add_candidate(
        outer_balanced,
        salvaged=bool(outer_balanced and outer_balanced.strip() != raw.strip()),
        source="balanced_json",
    )
    add_candidate(raw if stripped_raw else None, salvaged=False, source="raw")

    fence_stripped = _strip_code_fence_wrapper(raw)
    marker_stripped = _strip_leading_language_marker(raw)
    fence_then_marker = _strip_leading_language_marker(fence_stripped)
    marker_then_fence = _strip_code_fence_wrapper(marker_stripped)

    add_candidate(fence_stripped, salvaged=fence_stripped.strip() != raw.strip(), source="code_fence")
    add_candidate(marker_stripped, salvaged=marker_stripped.strip() != raw.strip(), source="language_marker")
    add_candidate(fence_then_marker, salvaged=fence_then_marker.strip() != raw.strip(), source="fence_then_marker")
    add_candidate(marker_then_fence, salvaged=marker_then_fence.strip() != raw.strip(), source="marker_then_fence")

    for candidate, _, source in list(parse_attempts):
        balanced = _extract_first_balanced_json_object(candidate)
        add_candidate(
            balanced,
            salvaged=bool(balanced and balanced.strip() != raw.strip()),
            source=f"{source}_balanced",
        )

    for candidate, salvaged, source in parse_attempts:
        try:
            parsed = parse_json_object(candidate)
            return parsed, {
                "salvage_succeeded": salvaged,
                "parse_mode": "raw_parse_success" if not salvaged and source == "raw" else "normalized_from_jsonish",
                "salvage_source": source if salvaged else "none",
                "final_answer_salvaged": False,
                "action_salvaged": False,
                "raw_snippet": _truncate_for_log(raw),
                "salvaged_snippet": _truncate_for_log(candidate),
            }
        except Exception:
            continue

    # One top-level object then trailing prose; apply same repairs as parse_json_object (e.g. invalid \').
    repaired_for_decode = repair_json(raw)
    start_obj = repaired_for_decode.find("{")
    if start_obj != -1:
        try:
            obj, end = json.JSONDecoder().raw_decode(repaired_for_decode, start_obj)
            if isinstance(obj, dict) and any(k in obj for k in ("thought", "finish", "action", "final_answer")):
                return obj, {
                    "salvage_succeeded": True,
                    "parse_mode": "raw_decode_leading_object",
                    "salvage_source": "json_decoder_raw_decode",
                    "final_answer_salvaged": False,
                    "action_salvaged": False,
                    "raw_snippet": _truncate_for_log(raw),
                    "salvaged_snippet": _truncate_for_log(repaired_for_decode[start_obj:end]),
                }
        except Exception:
            pass

    if not needs_salvage_hint:
        try:
            parsed = parse_json_object(raw)
            return parsed, {
                "salvage_succeeded": False,
                "parse_mode": "raw_parse_success",
                "salvage_source": "none",
                "final_answer_salvaged": False,
                "action_salvaged": False,
                "raw_snippet": _truncate_for_log(raw),
            }
        except Exception:
            pass

    salvaged_answer, salvage_meta = _salvage_final_answer_text(raw)
    if salvaged_answer is not None:
        return salvaged_answer, salvage_meta

    repaired_preview = repair_json(raw)
    if len(repaired_preview) > 500:
        repaired_preview = repaired_preview[:500] + "... [TRUNCATED]"
    raise PlannerParseError(
        f"Planner LLM returned non-JSON (len={len(raw)}): {repaired_preview!r}",
        raw_text=raw,
        salvage_succeeded=False,
        parse_mode="unrecoverable_parse_failure",
        salvage_source="none",
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


def _normalize_finished_final_answer(
    final_answer: Any,
    thought: str,
    *,
    thought_fallback_for_finish: bool = False,
) -> Optional[Dict[str, Any]]:
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
        if thought_fallback_for_finish:
            if _thought_reads_as_internal_planner_routing(thought):
                return None
            # finish=true path: models sometimes put the user-visible reply only in "thought".
            return {
                "content": thought.strip(),
                "structured": {},
                "normalized": True,
                "type": "thought_fallback",
            }
        return None

    return None


def _output_mode_coercion_tool(
    output_mode: Optional[str],
    goal_txt: str,
    offered_ids: set[str],
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    When the model sets finish=true with only internal-routing thought, pick a delivery
    tool from output_mode instead of always defaulting to answer_direct.
    """
    om = (output_mode or "").strip().lower()
    if not goal_txt:
        return None
    if om == "implementation_guide" and "write_guide" in offered_ids:
        return ("write_guide", {"request": goal_txt, "text": goal_txt})
    if om == "code_delivery" and "generate_code_scaffold" in offered_ids:
        return ("generate_code_scaffold", {"request": goal_txt, "text": goal_txt})
    if om == "comparative_analysis" and "compare_options" in offered_ids:
        return ("compare_options", {"request": goal_txt, "text": goal_txt})
    if om == "decision_support" and "write_recommendation" in offered_ids:
        return ("write_recommendation", {"request": goal_txt, "text": goal_txt})
    if "answer_direct" in offered_ids:
        return ("answer_direct", {"text": goal_txt})
    return None


def _external_output_mode_from_context(context: Optional[ContextBlock]) -> Optional[str]:
    if context is None:
        return None
    ext = context.external_facts if isinstance(context.external_facts, dict) else {}
    raw = ext.get("output_mode")
    if raw is None or raw == "":
        return None
    return str(raw).strip()


def _validate_or_normalize_planner_step(
    planner_step: Any,
    *,
    toolset: List[ToolDef],
    from_salvage: bool = False,
    planning_goal_text: Optional[str] = None,
    output_mode: Optional[str] = None,
) -> Dict[str, Any]:
    if not isinstance(planner_step, dict):
        raise PlannerSchemaError(f"planner response must be an object, got {type(planner_step).__name__}")

    normalized = dict(planner_step)
    thought = normalized.get("thought", "")
    normalized["thought"] = thought if isinstance(thought, str) else str(thought or "")

    finish = normalized.get("finish")
    action_salvaged = False
    final_answer_salvaged = False
    normalization_mode = "raw_parse_success"
    if finish is None:
        inferred_final = _normalize_finished_final_answer(
            normalized.get("final_answer"),
            normalized["thought"],
            thought_fallback_for_finish=False,
        )
        raw_action = normalized.get("action")
        if inferred_final is not None:
            finish = True
            final_answer_salvaged = True
            normalization_mode = "normalized_from_jsonish"
        elif raw_action is not None:
            finish = False
            action_salvaged = True
            normalization_mode = "normalized_from_jsonish"
        else:
            finish = False
    if not isinstance(finish, bool):
        raise PlannerSchemaError(f"finish must be boolean, got {type(finish).__name__}")
    normalized["finish"] = finish

    action = normalized.get("action")
    if isinstance(action, str) and action.strip():
        action = {"tool_id": action.strip(), "input": {}}
        action_salvaged = True
        normalization_mode = "normalized_from_jsonish"
    elif isinstance(action, dict):
        action = dict(action)
        tool_id = action.get("tool_id") or action.get("tool") or action.get("verb_name")
        if "name" in action and "tool_id" not in action and isinstance(action.get("name"), str):
            action["name"] = action.get("name")
        if "action_id" in action and isinstance(action.get("action_id"), str):
            action["action_id"] = action.get("action_id")
        tool_input = action.get("input")
        if tool_input is None:
            tool_input = action.get("args") or action.get("arguments") or action.get("payload") or {}
            if tool_input:
                action_salvaged = True
                normalization_mode = "normalized_from_jsonish"
        if tool_id and ("tool_id" not in action or "input" not in action):
            action["tool_id"] = str(tool_id)
            action["input"] = tool_input if isinstance(tool_input, dict) else {"value": tool_input}
            action_salvaged = True
            normalization_mode = "normalized_from_jsonish"
    if action is not None and not isinstance(action, dict):
        raise PlannerSchemaError(f"action must be object|string|null, got {type(action).__name__}")
    if isinstance(action, dict):
        raw_action_name = action.get("name")
        raw_action_id = action.get("action_id")
        raw_tool_id = action.get("tool_id")
        offered_tool_ids = {t.tool_id for t in toolset}
        normalized_action_name = str(raw_action_name).strip() if isinstance(raw_action_name, str) and raw_action_name.strip() else None
        normalized_action_id = str(raw_action_id).strip() if isinstance(raw_action_id, str) and raw_action_id.strip() else None
        normalized_tool_id = str(raw_tool_id).strip() if isinstance(raw_tool_id, str) and raw_tool_id.strip() else None

        if normalized_tool_id is not None:
            normalized_action_id = normalized_tool_id
        elif normalized_action_id is not None:
            normalized_tool_id = normalized_action_id
        elif normalized_action_name is not None:
            normalized_tool_id = normalized_action_name

        if normalized_action_name and normalized_action_id and normalized_action_name != normalized_action_id:
            raise PlannerContractViolation(
                "action.name and action_id disagree",
                failure_category="action_name_id_mismatch",
                raw_action_name=str(raw_action_name) if raw_action_name is not None else None,
                raw_action_id=str(raw_action_id) if raw_action_id is not None else None,
                normalized_action_name=normalized_action_name,
                normalized_action_id=normalized_action_id,
                from_salvage=from_salvage,
            )

        if normalized_action_name and normalized_action_name not in offered_tool_ids:
            failure_category = "invalid_action_not_in_toolset"
            if from_salvage and _contains_destructive_shell_pattern(action.get("input")):
                failure_category = "salvage_out_of_contract"
            raise PlannerContractViolation(
                "action.name is not in planner-visible toolset",
                failure_category=failure_category,
                raw_action_name=str(raw_action_name) if raw_action_name is not None else None,
                raw_action_id=str(raw_action_id) if raw_action_id is not None else None,
                normalized_action_name=normalized_action_name,
                normalized_action_id=normalized_action_id,
                from_salvage=from_salvage,
            )

        if normalized_action_id and normalized_action_id not in offered_tool_ids:
            failure_category = "invalid_action_id_not_in_toolset"
            if from_salvage and _contains_destructive_shell_pattern(action.get("input")):
                failure_category = "salvage_out_of_contract"
            raise PlannerContractViolation(
                "action_id is not in planner-visible toolset",
                failure_category=failure_category,
                raw_action_name=str(raw_action_name) if raw_action_name is not None else None,
                raw_action_id=str(raw_action_id) if raw_action_id is not None else None,
                normalized_action_name=normalized_action_name,
                normalized_action_id=normalized_action_id,
                from_salvage=from_salvage,
            )

        if not normalized_tool_id or normalized_tool_id not in offered_tool_ids:
            failure_category = "planner_contract_violation"
            if from_salvage and _contains_destructive_shell_pattern(action.get("input")):
                failure_category = "salvage_out_of_contract"
            raise PlannerContractViolation(
                "action cannot be validated against offered planner toolset",
                failure_category=failure_category,
                raw_action_name=str(raw_action_name) if raw_action_name is not None else None,
                raw_action_id=str(raw_action_id) if raw_action_id is not None else None,
                normalized_action_name=normalized_action_name,
                normalized_action_id=normalized_action_id,
                from_salvage=from_salvage,
            )

        action = dict(action)
        action["tool_id"] = normalized_tool_id
        normalized["action"] = action

    normalized["action"] = action

    final_info = (
        _normalize_finished_final_answer(
            normalized.get("final_answer"),
            normalized["thought"],
            thought_fallback_for_finish=True,
        )
        if finish
        else None
    )
    if finish:
        if final_info is None or not final_info["content"].strip():
            thought_only = _normalize_finished_final_answer(
                None,
                normalized["thought"],
                thought_fallback_for_finish=True,
            )
            if thought_only is not None and thought_only["content"].strip():
                final_info = thought_only
        if final_info is None or not final_info["content"].strip():
            goal_txt = (planning_goal_text or "").strip()
            offered_ids = {t.tool_id for t in toolset}
            if goal_txt and _thought_reads_as_internal_planner_routing(str(normalized.get("thought") or "")):
                coerced_tool = _output_mode_coercion_tool(output_mode, goal_txt, offered_ids)
                if coerced_tool is not None:
                    tool_id, tool_input = coerced_tool
                    coerced = {
                        **normalized,
                        "finish": False,
                        "final_answer": None,
                        "action": {"tool_id": tool_id, "input": tool_input},
                    }
                    return _validate_or_normalize_planner_step(
                        coerced,
                        toolset=toolset,
                        from_salvage=from_salvage,
                        planning_goal_text=planning_goal_text,
                        output_mode=output_mode,
                    )
            raise PlannerSchemaError("finish=true requires a usable final_answer")
        normalized["final_answer"] = final_info["content"]
        normalized["_final_answer_structured"] = final_info["structured"]
        normalized["_final_answer_normalized"] = final_info["normalized"]
        normalized["_final_answer_type"] = final_info["type"]
        normalized["_planner_normalization_mode"] = normalization_mode
        normalized["_action_salvaged"] = action_salvaged
        normalized["_final_answer_salvaged"] = final_answer_salvaged or final_info["normalized"]
        return normalized

    if action is None:
        goal_txt = (planning_goal_text or "").strip()
        offered_ids = {t.tool_id for t in toolset}
        salvaged_tool = _output_mode_coercion_tool(output_mode, goal_txt, offered_ids)
        if salvaged_tool is not None:
            tool_id, tool_input = salvaged_tool
            coerced = {
                **normalized,
                "finish": False,
                "final_answer": None,
                "action": {"tool_id": tool_id, "input": tool_input},
            }
            return _validate_or_normalize_planner_step(
                coerced,
                toolset=toolset,
                from_salvage=True,
                planning_goal_text=planning_goal_text,
                output_mode=output_mode,
            )
        raise PlannerSchemaError("finish=false requires an action")

    normalized["_final_answer_structured"] = {}
    normalized["_final_answer_normalized"] = False
    normalized["_final_answer_type"] = type(normalized.get("final_answer")).__name__
    normalized["_planner_normalization_mode"] = normalization_mode
    normalized["_action_salvaged"] = action_salvaged
    normalized["_final_answer_salvaged"] = final_answer_salvaged
    return normalized


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
    grounding_mode = ext_facts.get("delivery_grounding_mode") or "default_delivery"
    grounding_context = (ext_facts.get("grounding_context") or "")[:2000]
    anti_generic_drift = ext_facts.get("anti_generic_drift") or ""
    no_write_active = bool(ext_facts.get("no_write_active"))
    execution_blocked_reason = ext_facts.get("execution_blocked_reason") or ""
    operational_intent_detected = bool(ext_facts.get("operational_intent_detected"))
    available_operational_tools = ext_facts.get("available_operational_tools") or []

    system_msg = (system_override or f"""
You are Orion's internal ReAct planner.

RUNTIME ROUTING (must respect):
- output_mode: {om or "(not set)"}
- response_profile: {rp or "(not set)"}
- delivery_grounding_mode: {grounding_mode}
When output_mode is implementation_guide, tutorial, code_delivery, or direct_answer for instructional asks:
  Prefer delivery tools (write_guide, write_tutorial, answer_direct, finalize_response) over plan_action.
When output_mode is code_delivery: prefer generate_code_scaffold.
When output_mode is comparative_analysis: prefer compare_options.
When output_mode is decision_support: prefer write_recommendation.
When delivery_grounding_mode is orion_repo_architecture:
  Keep answers grounded in Orion's actual architecture (Hub/Client -> Cortex-Orch -> Cortex-Exec -> PlannerReact/AgentChain -> LLM Gateway over the bus).
  Do not silently substitute a generic Flask/Ubuntu deployment stack unless the user explicitly asks for that stack.
Grounding context: {grounding_context or "(none)"}
Anti-generic drift: {anti_generic_drift or "(none)"}
Operational intent detected: {operational_intent_detected}
No-write active: {no_write_active}
Execution blocked reason: {execution_blocked_reason or "(none)"}
Available operational semantic tools: {available_operational_tools}

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
10. **OPERATIONAL TOOL PREFERENCE:** If the user asks for concrete runtime/operational action and an operational semantic tool is available, prefer delegating to that tool over writing CLI prose.
11. **NO-WRITE HONESTY:** If no-write is active, never imply execution occurred. Prefer explicit blocked-execution framing instead of command instructions.

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

    planner_opts: Dict[str, Any] = {
        "temperature": 0.1,
        "max_tokens": int(settings.planner_max_completion_tokens),
        "return_json": True,
    }
    if options_override:
        planner_opts = {**planner_opts, **options_override}
    payload = {
        "model": planner_model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
        ],
        "route": "agent",
        "options": planner_opts,
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
        if chat_res.reasoning_content:
            text = _strip_think_blocks(text)
    except Exception:
        text = resp_payload.get("content") or resp_payload.get("text") or ""
    text = _strip_think_blocks(text)

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
            "corr_id=%s step=%s failure_category=repair_parse_failure planner_normalization_mode=%s salvage_source=%s final_answer_type=missing normalization_succeeded=false salvage_succeeded=%s raw_snippet=%s detail=%s",
            corr_id,
            step_index,
            getattr(e, "parse_mode", "unrecoverable_parse_failure"),
            getattr(e, "salvage_source", "none"),
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

    logger.info(
        "event=planner_salvage_attempt corr_id=%s step=%s offered_tools=%s salvage_source=%s",
        corr_id,
        step_index,
        [t.tool_id for t in toolset],
        repair_meta.get("salvage_source", "none"),
    )

    try:
        normalized_repaired = _validate_or_normalize_planner_step(
            repaired,
            toolset=toolset,
            from_salvage=True,
            planning_goal_text=goal.description,
            output_mode=_external_output_mode_from_context(context),
        )
        if repair_meta.get("salvage_succeeded") or normalized_repaired.get("_action_salvaged"):
            logger.info(
                "corr_id=%s step=%s failure_category=repair_salvaged planner_normalization_mode=%s salvage_source=%s final_answer_type=%s final_answer_salvaged=%s action_salvaged=%s normalization_succeeded=%s salvage_succeeded=%s raw_snippet=%s",
                corr_id,
                step_index,
                normalized_repaired.get("_planner_normalization_mode", repair_meta.get("parse_mode", "normalized_from_jsonish")),
                repair_meta.get("salvage_source", "none"),
                normalized_repaired.get("_final_answer_type", _final_answer_type_name(repaired)),
                str(bool(normalized_repaired.get("_final_answer_salvaged"))).lower(),
                str(bool(normalized_repaired.get("_action_salvaged"))).lower(),
                str(bool(normalized_repaired.get("_final_answer_normalized"))).lower(),
                str(bool(repair_meta.get("salvage_succeeded"))).lower(),
                repair_meta.get("raw_snippet", _truncate_for_log(repaired)),
            )
        if normalized_repaired.get("_final_answer_normalized"):
            logger.info(
                "corr_id=%s step=%s failure_category=normalization_applied planner_normalization_mode=%s salvage_source=%s final_answer_type=%s final_answer_salvaged=%s action_salvaged=%s normalization_succeeded=true salvage_succeeded=%s raw_snippet=%s",
                corr_id,
                step_index,
                normalized_repaired.get("_planner_normalization_mode", repair_meta.get("parse_mode", "normalized_from_jsonish")),
                repair_meta.get("salvage_source", "none"),
                normalized_repaired.get("_final_answer_type"),
                str(bool(normalized_repaired.get("_final_answer_salvaged"))).lower(),
                str(bool(normalized_repaired.get("_action_salvaged"))).lower(),
                str(bool(repair_meta.get("salvage_succeeded"))).lower(),
                repair_meta.get("raw_snippet", _truncate_for_log(repaired)),
            )
        logger.info(
            "event=planner_action_validated corr_id=%s step=%s offered_tools=%s raw_action_name=%s raw_action_id=%s normalized_action_name=%s normalized_action_id=%s parse_path=salvage",
            corr_id,
            step_index,
            [t.tool_id for t in toolset],
            ((repaired.get("action") or {}).get("name") if isinstance(repaired, dict) and isinstance(repaired.get("action"), dict) else None),
            ((repaired.get("action") or {}).get("action_id") if isinstance(repaired, dict) and isinstance(repaired.get("action"), dict) else None),
            ((normalized_repaired.get("action") or {}).get("name") if isinstance(normalized_repaired.get("action"), dict) else None),
            ((normalized_repaired.get("action") or {}).get("tool_id") if isinstance(normalized_repaired.get("action"), dict) else None),
        )
        return normalized_repaired
    except PlannerContractViolation as e:
        logger.warning(
            "event=planner_salvage_rejected_out_of_toolset corr_id=%s step=%s offered_tools=%s raw_action_name=%s raw_action_id=%s normalized_action_name=%s normalized_action_id=%s failure_category=%s parse_path=salvage detail=%s",
            corr_id,
            step_index,
            [t.tool_id for t in toolset],
            e.raw_action_name,
            e.raw_action_id,
            e.normalized_action_name,
            e.normalized_action_id,
            e.failure_category,
            e,
        )
    except PlannerSchemaError as e:
        logger.warning(
            "corr_id=%s step=%s failure_category=repair_schema_validation planner_normalization_mode=%s salvage_source=%s final_answer_type=%s normalization_succeeded=false salvage_succeeded=%s raw_snippet=%s detail=%s",
            corr_id,
            step_index,
            repair_meta.get("parse_mode", "unrecoverable_parse_failure"),
            repair_meta.get("salvage_source", "none"),
            _final_answer_type_name(repaired),
            str(bool(repair_meta.get("salvage_succeeded"))).lower(),
            repair_meta.get("raw_snippet", _truncate_for_log(repaired)),
            e,
        )

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
        offered_tools = [t.tool_id for t in payload.toolset]
        logger.info("event=planner_contract_toolset corr_id=%s offered_tools=%s", corr_id, offered_tools)
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
                    "corr_id=%s step=%s failure_category=planner_response_parse_failure planner_normalization_mode=%s salvage_source=%s final_answer_type=missing normalization_succeeded=false salvage_succeeded=%s raw_snippet=%s detail=%s",
                    corr_id,
                    step_index,
                    getattr(e, "parse_mode", "unrecoverable_parse_failure"),
                    getattr(e, "salvage_source", "none"),
                    str(bool(getattr(e, "salvage_succeeded", False))).lower(),
                    _truncate_for_log(getattr(e, "raw_text", str(e))),
                    e,
                )
                logger.warning(
                    "event=planner_parse_failure corr_id=%s step=%s offered_tools=%s parse_mode=%s salvage_source=%s",
                    corr_id,
                    step_index,
                    offered_tools,
                    getattr(e, "parse_mode", "unrecoverable_parse_failure"),
                    getattr(e, "salvage_source", "none"),
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
                    planner_step = _validate_or_normalize_planner_step(
                        raw_planner_step,
                        toolset=payload.toolset,
                        from_salvage=bool(planner_meta.get("salvage_succeeded")),
                        planning_goal_text=payload.goal.description if payload.goal else None,
                        output_mode=_external_output_mode_from_context(payload.context),
                    )
                    if planner_meta.get("salvage_succeeded"):
                        logger.info(
                            "corr_id=%s step=%s failure_category=planner_response_salvaged planner_normalization_mode=%s salvage_source=%s final_answer_type=%s final_answer_salvaged=%s action_salvaged=%s normalization_succeeded=%s salvage_succeeded=%s raw_snippet=%s",
                            corr_id,
                            step_index,
                            planner_step.get("_planner_normalization_mode", planner_meta.get("parse_mode", "normalized_from_jsonish")),
                            planner_meta.get("salvage_source", "none"),
                            planner_step.get("_final_answer_type", _final_answer_type_name(raw_planner_step)),
                            str(bool(planner_step.get("_final_answer_salvaged"))).lower(),
                            str(bool(planner_step.get("_action_salvaged"))).lower(),
                            str(bool(planner_step.get("_final_answer_normalized"))).lower(),
                            str(bool(planner_meta.get("salvage_succeeded"))).lower(),
                            planner_meta.get("raw_snippet", _truncate_for_log(raw_planner_step)),
                        )
                    if planner_step.get("_final_answer_normalized") or planner_step.get("_action_salvaged"):
                        logger.info(
                            "corr_id=%s step=%s failure_category=normalization_applied planner_normalization_mode=%s salvage_source=%s final_answer_type=%s final_answer_salvaged=%s action_salvaged=%s normalization_succeeded=true salvage_succeeded=%s raw_snippet=%s",
                            corr_id,
                            step_index,
                            planner_step.get("_planner_normalization_mode", planner_meta.get("parse_mode", "normalized_from_jsonish")),
                            planner_meta.get("salvage_source", "none"),
                            planner_step.get("_final_answer_type"),
                            str(bool(planner_step.get("_final_answer_salvaged"))).lower(),
                            str(bool(planner_step.get("_action_salvaged"))).lower(),
                            str(bool(planner_meta.get("salvage_succeeded"))).lower(),
                            planner_meta.get("raw_snippet", _truncate_for_log(raw_planner_step)),
                        )
                    logger.info(
                        "event=planner_action_validated corr_id=%s step=%s offered_tools=%s raw_action_name=%s raw_action_id=%s normalized_action_name=%s normalized_action_id=%s parse_path=%s",
                        corr_id,
                        step_index,
                        offered_tools,
                        ((raw_planner_step.get("action") or {}).get("name") if isinstance(raw_planner_step, dict) and isinstance(raw_planner_step.get("action"), dict) else None),
                        ((raw_planner_step.get("action") or {}).get("action_id") if isinstance(raw_planner_step, dict) and isinstance(raw_planner_step.get("action"), dict) else None),
                        ((planner_step.get("action") or {}).get("name") if isinstance(planner_step.get("action"), dict) else None),
                        ((planner_step.get("action") or {}).get("tool_id") if isinstance(planner_step.get("action"), dict) else None),
                        ("salvage" if planner_meta.get("salvage_succeeded") else "clean_parse"),
                    )
                except PlannerContractViolation as e:
                    logger.warning(
                        "event=planner_contract_violation_blocked corr_id=%s step=%s offered_tools=%s raw_action_name=%s raw_action_id=%s normalized_action_name=%s normalized_action_id=%s failure_category=%s parse_path=%s detail=%s",
                        corr_id,
                        step_index,
                        offered_tools,
                        e.raw_action_name,
                        e.raw_action_id,
                        e.normalized_action_name,
                        e.normalized_action_id,
                        e.failure_category,
                        ("salvage" if planner_meta.get("salvage_succeeded") else "clean_parse"),
                        e,
                    )
                    planner_step = _planner_error_step(f"Planner contract violation: {e.failure_category}")
                except PlannerSchemaError as e:
                    logger.warning(
                        "corr_id=%s step=%s failure_category=planner_response_schema_validation planner_normalization_mode=%s salvage_source=%s final_answer_type=%s normalization_succeeded=false salvage_succeeded=%s raw_snippet=%s detail=%s",
                        corr_id,
                        step_index,
                        planner_meta.get("parse_mode", "unrecoverable_parse_failure"),
                        planner_meta.get("salvage_source", "none"),
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

            tool_id = str(raw_tool_id).strip()

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
