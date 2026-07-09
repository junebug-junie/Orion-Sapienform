# services/orion-cortex-exec/app/router.py
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import ServiceRef

from .executor import (
    _forward_llm_uncertainty_metadata,
    call_step_services,
    prepare_brain_reply_context,
    prepare_chat_quick_reply_context,
    run_recall_step,
)
from .grounding_capsule import assemble_stance_grounding
from .pcr_chat_memory import CONTINUITY_PROFILE, run_pcr_phase0_and_1, run_pcr_phase3
from .situation import mark_orion_turn
from .recall_utils import (
    delivery_safe_recall_decision,
    has_inline_recall,
    hub_chat_lane_from_ctx,
    plan_ctx_latest_user_text,
    recall_enabled_value,
    resolve_profile,
    should_run_recall,
)
from orion.schemas.cortex.schemas import ExecutionPlan, PlanExecutionRequest, PlanExecutionResult, StepExecutionResult
from orion.schemas.metacognitive_trace import MetacognitiveTraceV1
from .supervisor import Supervisor
from .settings import settings
from .grammar_emit import (
    begin_plan_grammar,
    get_or_create_collector,
    record_assembled_grammar,
    short_error_kind,
)
from .metacog_enrichment import extract_reasoning_features
from .reasoning_emit import build_reasoning_call, publish_reasoning_call
from .chat_stance import strip_identity_recital_leadin, strip_transactional_closers
from orion.cognition.verb_activation import is_active

logger = logging.getLogger("orion.cortex.router")


def _thought_debug_enabled() -> bool:
    return str(os.getenv("DEBUG_THOUGHT_PROCESS", "false")).strip().lower() in {"1", "true", "yes", "on"}


def _debug_len(value: Any) -> int:
    return len(str(value or ""))


def _debug_snippet(value: Any, max_len: int = 200) -> str:
    text = str(value or "").strip()
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}…"


def _preview_text(value: str | None, limit: int = 220) -> str:
    if not value:
        return ""
    return repr(value[:limit])


_THINK_BLOCK_RE = re.compile(r"<think>\s*.*?\s*</think>", flags=re.IGNORECASE | re.DOTALL)
_THINK_CLOSE_ONLY_RE = re.compile(r"</think\s*>", flags=re.IGNORECASE)
_PLANNING_LINE_RE = re.compile(
    r"^\s*(okay,?\s+i\s+need\s+to|okay,?\s+so\s+the\s+user\s+wants|the\s+user\s+wants|i\s+should|need\s+to|let\s+me(?:\s+think)?|check\s+response\s+hazards|looking\s+at\s+the\s+memory\s+digest|plan:|steps?:)\b",
    flags=re.IGNORECASE,
)
_PLANNING_TEXT_RE = re.compile(
    r"^\s*(okay,?\s+i\s+need\s+to|okay,?\s+so\s+the\s+user\s+wants|the\s+user\s+wants|i\s+should|need\s+to|let\s+me(?:\s+think)?|check\s+response\s+hazards|looking\s+at\s+the\s+memory\s+digest)\b",
    flags=re.IGNORECASE,
)


def _structured_output_expected(verb_name: str | None) -> bool:
    return str(verb_name or "").strip().lower() in {
        "journal.compose",
        "concept_induction_journal_synthesize",
        "github_compactor_digest_v1",
        "memory_graph_suggest",
        "stance_react",
        "harness_finalize_reflect",
    }


def _strip_think_content(text: str) -> tuple[str, dict[str, Any]]:
    raw = str(text or "")
    raw_trimmed = raw.strip()
    has_open = "<think>" in raw.lower()
    has_close = bool(_THINK_CLOSE_ONLY_RE.search(raw))
    if has_open and has_close:
        stripped = _THINK_BLOCK_RE.sub(" ", raw).strip()
        if "<think>" in stripped.lower():
            stripped = stripped[: stripped.lower().find("<think>")].strip()
        return stripped, {
            "has_think_tags": True,
            "think_stripped": stripped != raw_trimmed,
            "full_block_detected": True,
            "close_tag_only_detected": False,
        }
    if (not has_open) and has_close:
        first_close = _THINK_CLOSE_ONLY_RE.search(raw)
        cleaned = raw[first_close.end() :] if first_close else raw
        stripped = cleaned.strip()
        return stripped, {
            "has_think_tags": True,
            "think_stripped": stripped != raw_trimmed,
            "full_block_detected": False,
            "close_tag_only_detected": True,
        }
    if has_open and not has_close:
        stripped = raw[: raw.lower().find("<think>")].strip()
        return stripped, {
            "has_think_tags": True,
            "think_stripped": stripped != raw_trimmed,
            "full_block_detected": False,
            "close_tag_only_detected": False,
        }
    return raw_trimmed, {
        "has_think_tags": False,
        "think_stripped": False,
        "full_block_detected": False,
        "close_tag_only_detected": False,
    }


def _strip_planning_preamble(text: str) -> tuple[str, bool, int]:
    lines = [line for line in str(text or "").splitlines()]
    dropped = 0
    while lines and _PLANNING_LINE_RE.search(lines[0] or ""):
        lines.pop(0)
        dropped += 1
    cleaned = "\n".join(lines).strip()
    return cleaned, dropped > 0, dropped


def _looks_like_internal_planning(text: str) -> bool:
    candidate = str(text or "").strip()
    if not candidate:
        return False
    head = "\n".join(candidate.splitlines()[:3]).strip()
    return bool(_PLANNING_TEXT_RE.search(head))


_ORION_NAME = r"or[ií\u00ed]on"
_IDENTITY_BOUNDARY_REPAIRS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(rf"\byou['\u2019]re\s+{_ORION_NAME}\b", re.IGNORECASE), "You're Oríon"),
    (re.compile(rf"\byou\s+are\s+{_ORION_NAME}\b", re.IGNORECASE), "You are Oríon"),
    (re.compile(rf"\byoure\s+{_ORION_NAME}\b", re.IGNORECASE), "youre orion"),
    (re.compile(rf"\byour\s+name\s+is\s+{_ORION_NAME}\b", re.IGNORECASE), "your name is Orion"),
)
_IDENTITY_BOUNDARY_ASSISTANT_CLAIM = "I'm Oríon"


def _apply_chat_general_identity_boundary_guard(
    text: str,
    *,
    verb_name: str | None,
) -> tuple[str, dict[str, Any]]:
    diagnostics: dict[str, Any] = {
        "identity_boundary_applied": False,
        "identity_boundary_violations": [],
    }
    if str(verb_name or "").strip().lower() != "chat_general":
        return text, diagnostics
    candidate = str(text or "")
    if not candidate.strip():
        return text, diagnostics
    violations: list[str] = []
    repaired = candidate
    for pattern, label in _IDENTITY_BOUNDARY_REPAIRS:
        if pattern.search(repaired):
            violations.append(label)
            repaired = pattern.sub(_IDENTITY_BOUNDARY_ASSISTANT_CLAIM, repaired)
    if violations:
        diagnostics["identity_boundary_applied"] = True
        diagnostics["identity_boundary_violations"] = violations
    return repaired, diagnostics


def _provider_completion_meta(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("raw") if isinstance(payload.get("raw"), dict) else {}
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    raw_usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
    choices = raw.get("choices") if isinstance(raw.get("choices"), list) else []
    first = choices[0] if choices and isinstance(choices[0], dict) else {}
    finish_reason = first.get("finish_reason")
    completion_tokens = usage.get("completion_tokens") or raw_usage.get("completion_tokens")
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    return {
        "finish_reason": finish_reason,
        "completion_tokens": completion_tokens,
        "has_reasoning_content": bool(str(payload.get("reasoning_content") or "").strip()),
        "has_reasoning_trace": isinstance(payload.get("reasoning_trace"), dict),
        "provider_reasoning_available": meta.get("provider_reasoning_available"),
        "inline_think_extracted": meta.get("inline_think_extracted"),
    }


def _extract_first_json_object_text(text: str) -> str | None:
    candidate = (text or "").strip()
    if not candidate:
        return None
    start = candidate.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(candidate)):
        ch = candidate[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return candidate[start : idx + 1]
    return None


def _extract_final_text(steps: List[StepExecutionResult], *, verb_name: str | None) -> tuple[str, Dict[str, Any]]:
    diagnostics: Dict[str, Any] = {
        "source_field": None,
        "source_service": None,
        "think_tags_detected": False,
        "think_stripping_applied": False,
        "think_full_block_detected": False,
        "think_close_tag_only_detected": False,
        "structured_output_sanitized": False,
        "structured_output_rejected": False,
        "structured_json_extraction_attempted": False,
        "candidate_count": 0,
        "rejected_candidate_count": 0,
        "candidate_fields_considered": [],
        "provider_finish_reason": None,
        "provider_completion_tokens": None,
        "provider_has_reasoning_content": False,
        "provider_has_reasoning_trace": False,
        "provider_reasoning_available": None,
        "inline_think_extracted": None,
        "planning_stripping_applied": False,
        "planning_lines_dropped": 0,
        "planning_candidate_rejected": False,
        "planning_candidate_rejection_reason": None,
        "raw_len": 0,
        "clean_len": 0,
        "final_len": 0,
        "truncation_detected": False,
        "result_len": 0,
    }
    structured_expected = _structured_output_expected(verb_name)
    # Skill/service adapters frequently emit terminal text in `final_text`.
    # Consider it first for all verbs so capability-backed skills do not lose output.
    candidate_fields = ("final_text", "content", "text")
    for step in reversed(steps):
        for service_name, payload in (step.result or {}).items():
            if not isinstance(payload, dict):
                continue
            for field in candidate_fields:
                candidate = payload.get(field)
                if not isinstance(candidate, str) or not candidate.strip():
                    continue
                diagnostics["candidate_count"] += 1
                cleaned, think_diag = _strip_think_content(candidate)
                pre_len = len(candidate.strip())
                post_len = len(cleaned)
                candidate_diag = {
                    "service": service_name,
                    "field": field,
                    "raw_len": pre_len,
                    "clean_len": post_len,
                    "think_tags": bool(think_diag["has_think_tags"]),
                    "think_stripped": bool(think_diag["think_stripped"]),
                    "think_full_block_detected": bool(think_diag["full_block_detected"]),
                    "think_close_tag_only_detected": bool(think_diag["close_tag_only_detected"]),
                }
                provider_meta = _provider_completion_meta(payload)
                diagnostics["provider_finish_reason"] = provider_meta.get("finish_reason")
                diagnostics["provider_completion_tokens"] = provider_meta.get("completion_tokens")
                diagnostics["provider_has_reasoning_content"] = provider_meta.get("has_reasoning_content")
                diagnostics["provider_has_reasoning_trace"] = provider_meta.get("has_reasoning_trace")
                diagnostics["provider_reasoning_available"] = provider_meta.get("provider_reasoning_available")
                diagnostics["inline_think_extracted"] = provider_meta.get("inline_think_extracted")
                no_plan_text = cleaned
                planning_applied = False
                dropped_lines = 0
                if str(verb_name or "").strip().lower() == "chat_general":
                    no_plan_text, planning_applied, dropped_lines = _strip_planning_preamble(cleaned)
                if len(diagnostics["candidate_fields_considered"]) < 8:
                    diagnostics["candidate_fields_considered"].append(candidate_diag)
                if structured_expected:
                    diagnostics["structured_json_extraction_attempted"] = True
                    extra_fields = (
                        ("reasoning_content", payload.get("reasoning_content")),
                        ("text", payload.get("text")),
                        ("final_text", payload.get("final_text")),
                    )
                    json_text = _extract_first_json_object_text(cleaned)
                    if not json_text and provider_meta.get("finish_reason") == "length" and cleaned.lstrip().startswith("{"):
                        from orion.memory_graph.json_extract import salvage_truncated_json_object_text

                        json_text = salvage_truncated_json_object_text(cleaned)
                        if json_text:
                            diagnostics["structured_output_salvaged"] = True
                    if not json_text:
                        for extra_field, extra_val in extra_fields:
                            if extra_field == field:
                                continue
                            if not isinstance(extra_val, str) or not extra_val.strip():
                                continue
                            json_text = _extract_first_json_object_text(extra_val)
                            if json_text:
                                candidate_diag = dict(candidate_diag)
                                candidate_diag["field"] = extra_field
                                candidate_diag["raw_len"] = len(extra_val.strip())
                                break
                    if json_text:
                        diagnostics["source_field"] = candidate_diag.get("field", field)
                        diagnostics["source_service"] = service_name
                        diagnostics["think_tags_detected"] = bool(think_diag["has_think_tags"])
                        diagnostics["think_stripping_applied"] = bool(think_diag["think_stripped"])
                        diagnostics["think_full_block_detected"] = bool(think_diag["full_block_detected"])
                        diagnostics["think_close_tag_only_detected"] = bool(think_diag["close_tag_only_detected"])
                        diagnostics["structured_output_sanitized"] = bool(json_text != cleaned)
                        diagnostics["result_len"] = len(json_text)
                        diagnostics["raw_len"] = pre_len
                        diagnostics["clean_len"] = len(json_text)
                        diagnostics["final_len"] = len(json_text)
                        return json_text, diagnostics
                    diagnostics["rejected_candidate_count"] += 1
                    if not diagnostics.get("structured_rejection_preview"):
                        diagnostics["structured_rejection_preview"] = cleaned[:500]
                    continue
                diagnostics["source_field"] = field
                diagnostics["source_service"] = service_name
                diagnostics["think_tags_detected"] = bool(think_diag["has_think_tags"])
                diagnostics["think_stripping_applied"] = bool(think_diag["think_stripped"])
                diagnostics["think_full_block_detected"] = bool(think_diag["full_block_detected"])
                diagnostics["think_close_tag_only_detected"] = bool(think_diag["close_tag_only_detected"])
                diagnostics["planning_stripping_applied"] = planning_applied
                diagnostics["planning_lines_dropped"] = dropped_lines
                final_text = no_plan_text or cleaned
                if str(verb_name or "").strip().lower() == "chat_general" and _looks_like_internal_planning(final_text):
                    diagnostics["planning_candidate_rejected"] = True
                    diagnostics["planning_candidate_rejection_reason"] = "looks_like_internal_planning_after_cleanup"
                    diagnostics["rejected_candidate_count"] += 1
                    continue
                if provider_meta.get("finish_reason") == "length":
                    diagnostics["truncation_detected"] = True
                    if final_text and final_text[-1] not in ".!?…":
                        final_text = f"{final_text}…"
                diagnostics["result_len"] = len(final_text)
                diagnostics["raw_len"] = pre_len
                diagnostics["clean_len"] = post_len
                diagnostics["final_len"] = len(final_text)
                return final_text, diagnostics
    if structured_expected and diagnostics["candidate_count"] > 0:
        diagnostics["structured_output_rejected"] = True
        previews = [
            str(c.get("raw_len") or "")
            for c in (diagnostics.get("candidate_fields_considered") or [])
            if isinstance(c, dict) and int(c.get("raw_len") or 0) > 0
        ]
        if previews:
            diagnostics["structured_rejection_raw_lens"] = previews
    if _is_runtime_skill_verb(verb_name):
        fallback_message = None
        fallback_status = None
        for step in reversed(steps):
            for _service_name, payload in (step.result or {}).items():
                if not isinstance(payload, dict):
                    continue
                if isinstance(payload.get("status"), str):
                    ps = str(payload.get("status")).strip()
                    if ps:
                        fallback_status = ps
                err = payload.get("error")
                if isinstance(err, dict) and isinstance(err.get("message"), str):
                    candidate = str(err.get("message")).strip()
                    if candidate:
                        fallback_message = candidate
                elif isinstance(err, str) and err.strip():
                    fallback_message = err.strip()
            if not fallback_status and isinstance(step.status, str) and step.status.strip():
                fallback_status = step.status.strip()
            if not fallback_message and isinstance(step.error, str) and step.error.strip():
                fallback_message = step.error.strip()
            if fallback_message or fallback_status:
                break
        if fallback_message or fallback_status:
            runtime_text = (
                f"Runtime skill result: status={fallback_status or 'fail'} "
                f"message={fallback_message or 'empty_terminal_output'}"
            )
            diagnostics["source_field"] = diagnostics["source_field"] or "runtime_fallback"
            diagnostics["source_service"] = diagnostics["source_service"] or "runtime_fallback"
            diagnostics["result_len"] = len(runtime_text)
            diagnostics["final_len"] = len(runtime_text)
            diagnostics["clean_len"] = len(runtime_text)
            diagnostics["raw_len"] = len(runtime_text)
            return runtime_text, diagnostics
    return "", diagnostics


def _is_runtime_skill_verb(verb_name: str | None) -> bool:
    return str(verb_name or "").strip().lower().startswith("skills.runtime.")


def _should_fail_empty_runtime_skill_output(*, overall_status: str, verb_name: str | None, final_text: str | None) -> bool:
    return overall_status == "success" and _is_runtime_skill_verb(verb_name) and not str(final_text or "").strip()


def _should_fail_empty_structured_verb_output(
    *,
    overall_status: str,
    verb_name: str | None,
    final_text: str | None,
    final_text_diag: Dict[str, Any],
) -> bool:
    if overall_status != "success" or not _structured_output_expected(verb_name):
        return False
    if str(final_text or "").strip():
        return False
    if bool(final_text_diag.get("structured_output_rejected")):
        return True
    return int(final_text_diag.get("candidate_count") or 0) > 0 and int(final_text_diag.get("result_len") or 0) == 0


def _autonomy_goal_lineage_from_state(state: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not isinstance(state, dict):
        return None
    headlines = state.get("goal_headlines")
    if not isinstance(headlines, list) or not headlines:
        return None
    first = headlines[0]
    if not isinstance(first, dict):
        return None
    aid = first.get("artifact_id")
    sig = first.get("proposal_signature")
    if not aid and not sig:
        return None
    out: Dict[str, Any] = {}
    if aid:
        out["goal_artifact_id"] = aid
    if sig:
        out["proposal_signature"] = sig
    return out or None


def _autonomy_state_preview(ctx: Dict[str, Any]) -> Dict[str, Any] | None:
    summary = ctx.get("chat_autonomy_summary") if isinstance(ctx.get("chat_autonomy_summary"), dict) else {}
    state = ctx.get("chat_autonomy_state") if isinstance(ctx.get("chat_autonomy_state"), dict) else {}
    if not summary and not state:
        return None
    dominant_drive = state.get("dominant_drive")
    if not dominant_drive:
        top_drives = list(summary.get("top_drives") or [])
        if top_drives:
            dominant_drive = top_drives[0]
        else:
            active_drives = list(state.get("active_drives") or [])
            dominant_drive = active_drives[0] if active_drives else None
    preview = {
        "dominant_drive": dominant_drive,
        "top_drives": list(summary.get("top_drives") or state.get("active_drives") or [])[:3],
        "active_tensions": list(summary.get("active_tensions") or state.get("tension_kinds") or [])[:3],
        "proposal_headlines": list(summary.get("proposal_headlines") or [])[:3],
    }
    dc = summary.get("drive_competition") if isinstance(summary.get("drive_competition"), dict) else None
    if dc:
        preview["drive_competition"] = dc
    lineage = _autonomy_goal_lineage_from_state(state)
    if lineage:
        preview["goal_lineage"] = lineage
    if not any(preview.values()):
        return None
    return preview


def _situation_grounding_metadata_from_ctx(ctx: Dict[str, Any]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    if isinstance(ctx.get("situation_brief"), dict):
        metadata["situation_brief"] = ctx.get("situation_brief")
    if isinstance(ctx.get("situation_prompt_fragment"), dict) and ctx.get("situation_prompt_fragment"):
        metadata["situation_prompt_fragment"] = ctx.get("situation_prompt_fragment")
    if isinstance(ctx.get("presence_context"), dict):
        metadata["presence_context"] = ctx.get("presence_context")
    presence_ctx = ctx.get("presence_context") if isinstance(ctx.get("presence_context"), dict) else {}
    metadata["presence_context_present"] = bool(presence_ctx)
    metadata["audience_mode"] = str(presence_ctx.get("audience_mode") or "solo")
    fragment = ctx.get("situation_prompt_fragment")
    metadata["situation_fragment_present"] = isinstance(fragment, dict) and bool(fragment)
    if isinstance(ctx.get("situation_affordances"), list):
        metadata["situation_affordances"] = ctx.get("situation_affordances")
    if ctx.get("temporal_phase") is not None:
        metadata["temporal_phase"] = ctx.get("temporal_phase")
    return metadata


def _active_goals_from_ctx(ctx: Dict[str, Any]) -> list[Dict[str, Any]]:
    summary = ctx.get("chat_autonomy_summary") if isinstance(ctx.get("chat_autonomy_summary"), dict) else {}
    return [g for g in (summary.get("active_goals") or []) if isinstance(g, dict)]


def _resolve_autonomy_execution_mode(ctx: Dict[str, Any]) -> str:
    active_goals = _active_goals_from_ctx(ctx)
    lifecycle_goals = [
        g
        for g in active_goals
        if str(g.get("proposal_status") or "proposed").strip().lower()
        not in {"completed", "archived", "superseded"}
    ]
    statuses = {str(g.get("proposal_status") or "proposed").strip().lower() for g in lifecycle_goals}
    ctx_mode = str(ctx.get("chat_autonomy_execution_mode") or "").strip().lower()
    if "executing" in statuses or (ctx_mode == "executing" and lifecycle_goals):
        return "executing"
    if "planned" in statuses or (ctx_mode == "planned" and lifecycle_goals):
        return "planned"
    if ctx_mode == "hint_only":
        return "hint_only"
    summary = ctx.get("chat_autonomy_summary") if isinstance(ctx.get("chat_autonomy_summary"), dict) else {}
    goals_present = bool(
        summary.get("goals_present")
        or summary.get("proposal_headlines")
        or active_goals
    )
    stance_mode = str(summary.get("stance_mode") or "")
    if stance_mode == "proposal_only" or (
        stance_mode not in {"normal", "fallback_contextual"} and goals_present
    ):
        return "none"
    return "none"


def _autonomy_payload_from_ctx(ctx: Dict[str, Any]) -> Dict[str, Any]:
    summary = ctx.get("chat_autonomy_summary") if isinstance(ctx.get("chat_autonomy_summary"), dict) else None
    debug = ctx.get("chat_autonomy_debug") if isinstance(ctx.get("chat_autonomy_debug"), dict) else None
    state = ctx.get("chat_autonomy_state") if isinstance(ctx.get("chat_autonomy_state"), dict) else None
    backend = ctx.get("chat_autonomy_backend")
    selected_subject = ctx.get("chat_autonomy_selected_subject")
    repository_status = ctx.get("chat_autonomy_repository_status") if isinstance(ctx.get("chat_autonomy_repository_status"), dict) else None
    has_autonomy_context = any(item is not None for item in (summary, debug, state, backend, selected_subject, repository_status))
    payload: Dict[str, Any] = {}
    if summary:
        payload["autonomy_summary"] = summary
    if debug:
        payload["autonomy_debug"] = debug
    preview = _autonomy_state_preview(ctx)
    if preview:
        payload["autonomy_state_preview"] = preview
    lineage = _autonomy_goal_lineage_from_state(state if isinstance(state, dict) else None)
    if lineage:
        payload["autonomy_goal_lineage"] = lineage
    summary_dict = summary if isinstance(summary, dict) else {}
    active_goals = _active_goals_from_ctx(ctx)
    goals_present = bool(
        summary_dict.get("goals_present")
        or summary_dict.get("proposal_headlines")
        or active_goals
    )
    if goals_present:
        payload["autonomy_goals_present"] = True
    exec_mode = _resolve_autonomy_execution_mode(ctx)
    if exec_mode != "none" or has_autonomy_context or goals_present or lineage:
        payload["autonomy_execution_mode"] = exec_mode
    if has_autonomy_context:
        if backend:
            payload["autonomy_backend"] = backend
        elif state:
            payload["autonomy_backend"] = state.get("source")
        else:
            payload["autonomy_backend"] = None
        if selected_subject is not None:
            payload["autonomy_selected_subject"] = selected_subject
        elif state:
            payload["autonomy_selected_subject"] = state.get("subject")
        else:
            payload["autonomy_selected_subject"] = None
    if repository_status:
        payload["autonomy_repository_status"] = repository_status
    mutation_cognition = ctx.get("chat_mutation_cognition_context")
    if isinstance(mutation_cognition, dict) and mutation_cognition:
        payload["mutation_cognition_context"] = mutation_cognition
    runtime_response_diagnostics = ctx.get("runtime_response_diagnostics")
    if isinstance(runtime_response_diagnostics, dict) and runtime_response_diagnostics:
        payload["runtime_response_diagnostics"] = runtime_response_diagnostics
    inline_think_content = ctx.get("inline_think_content")
    thinking_source = ctx.get("thinking_source")
    final_text_clean = ctx.get("chat_general_final_text_clean")
    chat_stance_debug = ctx.get("chat_stance_debug") if isinstance(ctx.get("chat_stance_debug"), dict) else None
    if isinstance(inline_think_content, str):
        payload["inline_think_content"] = inline_think_content
    if isinstance(thinking_source, str):
        payload["thinking_source"] = thinking_source
    if isinstance(final_text_clean, str):
        payload["final_text_clean"] = final_text_clean
    if chat_stance_debug:
        attention_frame = chat_stance_debug.get("source_inputs", {}).get("attention_frame") if isinstance(chat_stance_debug.get("source_inputs"), dict) else {}
        selected = attention_frame.get("selected_action") if isinstance(attention_frame, dict) else None
        if isinstance(selected, dict) and selected.get("action_type") == "ask" and isinstance(final_text_clean, str):
            chat_stance_debug["speech_followed_recommendation"] = {"selected_action": "ask", "final_text_has_question": "?" in final_text_clean}
        elif isinstance(selected, dict):
            chat_stance_debug["speech_followed_recommendation"] = {"selected_action": selected.get("action_type"), "final_text_has_question": ("?" in final_text_clean) if isinstance(final_text_clean, str) else None}
        payload["chat_stance_debug"] = chat_stance_debug
    v2_state = ctx.get("chat_autonomy_state_v2")
    if isinstance(v2_state, dict):
        payload["autonomy_state_v2_preview"] = {
            "schema_version": v2_state.get("schema_version"),
            "dominant_drive": v2_state.get("dominant_drive"),
            "active_drives": (v2_state.get("active_drives") or [])[:3],
            "confidence": v2_state.get("confidence"),
            "unknowns": (v2_state.get("unknowns") or [])[:5],
            "top_attention_summaries": [
                item["summary"]
                for item in (v2_state.get("attention_items") or [])[:3]
                if isinstance(item, dict)
            ],
            "top_inhibition_reasons": [
                item["inhibition_reason"]
                for item in (v2_state.get("inhibited_impulses") or [])[:3]
                if isinstance(item, dict)
            ],
        }
    delta = ctx.get("chat_autonomy_state_delta")
    if isinstance(delta, dict):
        payload["autonomy_state_delta"] = delta
    # Hub merges cortex_result.metadata into chat turn spark_meta; concept-induction reads
    # payload.spark_meta.turn_effect (see orion.spark.concept_induction.tensions._extract_turn_effect).
    te = ctx.get("turn_effect")
    if isinstance(te, dict) and te:
        payload["turn_effect"] = te
        payload["turn_effect_status"] = "present"
    te_ev = ctx.get("turn_effect_evidence")
    if isinstance(te_ev, dict) and te_ev:
        payload["turn_effect_evidence"] = te_ev
    elif payload.get("turn_effect_status") != "present":
        payload["turn_effect_status"] = "missing"
        reason = "no_turn_effect_in_ctx"
        biometrics = ctx.get("biometrics")
        if isinstance(biometrics, dict):
            reason = str(biometrics.get("reason") or reason)
        payload["turn_effect_missing_reason"] = reason
    ctx_meta = ctx.get("metadata")
    if isinstance(ctx_meta, dict):
        for key in (
            "mind_requested",
            "mind_skip_reason",
            "mind_invocation_failed",
            "mind_artifact_persist_failed",
            "llm_uncertainty",
        ):
            if key in ctx_meta:
                payload[key] = ctx_meta[key]
    return payload


def _harvest_llm_uncertainty_from_steps(ctx: Dict[str, Any], step_results: List[StepExecutionResult]) -> None:
    for step in reversed(step_results):
        if not isinstance(step.result, dict):
            continue
        gw_payload = step.result.get("LLMGatewayService")
        if isinstance(gw_payload, dict):
            _forward_llm_uncertainty_metadata(gw_payload, ctx)
            return




def _normalize_execution_depth(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _collect_metacog_traces(step_results: List[StepExecutionResult], *, correlation_id: str, session_id: str | None) -> List[MetacognitiveTraceV1]:
    traces: List[MetacognitiveTraceV1] = []
    for step in step_results:
        if not isinstance(step.result, dict):
            continue
        payload = step.result.get("LLMGatewayService")
        if not isinstance(payload, dict):
            continue
        reasoning_content = None
        reasoning_trace = None
        payload_keys = []
        if isinstance(payload, dict):
            payload_keys = sorted(payload.keys())
            reasoning_content = payload.get("reasoning_content")
            reasoning_trace = payload.get("reasoning_trace")
        else:
            try:
                dumped = payload.model_dump()
                payload_keys = sorted(dumped.keys())
                reasoning_content = dumped.get("reasoning_content")
                reasoning_trace = dumped.get("reasoning_trace")
            except Exception:
                payload_keys = [type(payload).__name__]
        trace_content = reasoning_trace.get("content") if isinstance(reasoning_trace, dict) else None
        print(
            "===THINK_HOP=== hop=exec_in "
            f"corr={correlation_id} "
            f"keys={payload_keys} "
            f"reasoning_len={len(reasoning_content) if isinstance(reasoning_content, str) else 0} "
            f"trace_len={len(trace_content) if isinstance(trace_content, str) else 0} "
            f"provider_reasoning_available={(payload.get('meta') or {}).get('provider_reasoning_available') if isinstance(payload, dict) else None} "
            f"inline_think_extracted={(payload.get('meta') or {}).get('inline_think_extracted') if isinstance(payload, dict) else None} "
            f"preview={_preview_text((reasoning_content if isinstance(reasoning_content, str) else None) or trace_content)}",
            flush=True,
        )
        reasoning = reasoning_content
        if not isinstance(reasoning, str) or not reasoning.strip():
            continue
        model_name = payload.get("model_used") or payload.get("model") or "unknown"
        token_count = None
        usage = payload.get("usage")
        if isinstance(usage, dict):
            token_count = usage.get("completion_tokens") or usage.get("total_tokens")
        traces.append(
            MetacognitiveTraceV1(
                correlation_id=str(correlation_id),
                session_id=session_id,
                message_id=str(correlation_id),
                trace_role="reasoning" if step.step_name != "synthesize_chat_stance_brief" else "stance",
                trace_stage="post_answer",
                content=reasoning.strip(),
                model=str(model_name),
                token_count=int(token_count) if isinstance(token_count, (int, float)) else None,
                metadata={
                    "step_name": step.step_name,
                    "verb_name": step.verb_name,
                    **extract_reasoning_features(reasoning),
                },
            )
        )
        if _thought_debug_enabled():
            logger.info(
                "THOUGHT_DEBUG_EXEC stage=trace_collected corr=%s verb=%s step=%s trace_role=%s trace_stage=%s content_len=%s content_snippet=%r",
                correlation_id,
                step.verb_name,
                step.step_name,
                "reasoning" if step.step_name != "synthesize_chat_stance_brief" else "stance",
                "post_answer",
                _debug_len(reasoning),
                _debug_snippet(reasoning),
            )
    return traces


def _extract_reasoning_payload(
    step_results: List[StepExecutionResult],
    *,
    think_close_tag_only_detected: bool = False,
    prior_step_results: List[Dict[str, Any]] | None = None,
    correlation_id: str | None = None,
    canonical_verb_name: str | None = None,
    canonical_step_name: str | None = None,
) -> tuple[str | None, str | None, str, Dict[str, Any] | None]:
    def _iter_payloads() -> List[Dict[str, Any]]:
        payloads: List[Dict[str, Any]] = []
        if canonical_verb_name and canonical_step_name:
            for step in step_results:
                if (
                    str(step.verb_name or "").strip().lower() == str(canonical_verb_name).strip().lower()
                    and str(step.step_name or "").strip().lower() == str(canonical_step_name).strip().lower()
                    and isinstance(step.result, dict)
                ):
                    payload = step.result.get("LLMGatewayService")
                    if isinstance(payload, dict):
                        payloads.append(payload)
            return payloads
        for step in step_results:
            if not isinstance(step.result, dict):
                continue
            payload = step.result.get("LLMGatewayService")
            if isinstance(payload, dict):
                payloads.append(payload)
        for prior in prior_step_results or []:
            if not isinstance(prior, dict):
                continue
            svc_payload = prior.get("result")
            if not isinstance(svc_payload, dict):
                continue
            payload = svc_payload.get("LLMGatewayService")
            if isinstance(payload, dict):
                payloads.append(payload)
        return payloads

    for payload in _iter_payloads():
        reasoning_content = payload.get("reasoning_content")
        inline_think_content = payload.get("inline_think_content")
        raw_thinking_source = payload.get("thinking_source")
        reasoning_trace = payload.get("reasoning_trace")
        if correlation_id:
            logger.info(
                "exec_step_reasoning_capture corr_id=%s reasoning_content_len=%s inline_think_content_len=%s thinking_source=%s payload_keys=%s",
                correlation_id,
                len(reasoning_content) if isinstance(reasoning_content, str) else 0,
                len(inline_think_content) if isinstance(inline_think_content, str) else 0,
                raw_thinking_source if isinstance(raw_thinking_source, str) else "none",
                sorted(list(payload.keys())),
            )
        if isinstance(reasoning_content, str) and reasoning_content.strip():
            trace_dict = reasoning_trace if isinstance(reasoning_trace, dict) else None
            return reasoning_content.strip(), (
                inline_think_content.strip() if isinstance(inline_think_content, str) and inline_think_content.strip() else None
            ), "provider_reasoning", trace_dict
        if isinstance(inline_think_content, str) and inline_think_content.strip():
            thinking_source = "inline_think_close_tag_only" if think_close_tag_only_detected else "inline_think_full_block"
            if isinstance(raw_thinking_source, str) and raw_thinking_source.strip() == "inline_think":
                return None, inline_think_content.strip(), thinking_source, None
        if isinstance(reasoning_trace, dict):
            trace_content = reasoning_trace.get("content")
            if isinstance(trace_content, str) and trace_content.strip():
                return trace_content.strip(), None, "provider_reasoning", reasoning_trace
    return None, None, "none", None


class PlanRunner:
    async def run_plan(
        self,
        bus: OrionBusAsync,
        *,
        source: ServiceRef,
        req: PlanExecutionRequest,
        correlation_id: str,
        ctx: Dict[str, Any],
    ) -> PlanExecutionResult:
        plan: ExecutionPlan = req.plan
        depth = None
        if isinstance(plan.metadata, dict):
            depth = _normalize_execution_depth(plan.metadata.get("execution_depth"))
        start_mode = (req.args.extra or {}).get("mode") or ctx.get("mode") or "brain"
        logger.info(
            "plan_start corr_id=%s depth=%s mode=%s verb=%s steps=%s",
            correlation_id,
            depth,
            start_mode,
            plan.verb_name,
            [s.step_name for s in plan.steps],
        )
        if _is_runtime_skill_verb(plan.verb_name):
            logger.info(
                "runtime_skill_invocation corr_id=%s verb=%s mode=%s",
                correlation_id,
                plan.verb_name,
                start_mode,
            )
        step_results: List[StepExecutionResult] = []
        overall_status = "success"
        recall_debug: Dict[str, Any] = {}
        memory_used = False
        soft_failure = False

        extra = req.args.extra or {}
        options = extra.get("options") if isinstance(extra, dict) else {}
        diagnostic = bool(
            settings.diagnostic_mode
            or extra.get("diagnostic")
            or (isinstance(options, dict) and (options.get("diagnostic") or options.get("diagnostic_mode")))
        )
        mode = extra.get("mode") or ctx.get("mode") or "brain"
        recall_cfg = extra.get("recall") or ctx.get("recall") or {}
        if (
            isinstance(plan.metadata, dict)
            and str(plan.metadata.get("recall_enabled_default") or "").lower() == "false"
            and "enabled" not in recall_cfg
        ):
            recall_cfg = dict(recall_cfg)
            recall_cfg["enabled"] = False
        raw_enabled = recall_cfg.get("enabled", True)
        ctx.setdefault("recall", recall_cfg)
        recall_enabled = recall_enabled_value(recall_cfg)
        recall_required = bool(recall_cfg.get("required", False))
        verb_recall_profile = None
        if isinstance(plan.metadata, dict):
            verb_recall_profile = plan.metadata.get("recall_profile") or None
        ctx.setdefault("plan_recall_profile", verb_recall_profile)
        recall_policy = delivery_safe_recall_decision(
            recall_cfg,
            plan.steps,
            output_mode=ctx.get("output_mode"),
            verb_profile=verb_recall_profile,
            user_text=plan_ctx_latest_user_text(ctx),
            runtime_mode=mode,
            plan_verb_name=plan.verb_name,
            hub_chat_lane=hub_chat_lane_from_ctx(ctx),
            chat_quick_recall_profile=settings.chat_quick_recall_profile,
            chat_kids_story_recall_profile=settings.chat_kids_story_recall_profile,
        )
        selected_profile = recall_policy["profile"]
        profile_source = recall_policy["profile_source"]
        ctx.setdefault("debug", {})["recall_gating_reason"] = recall_policy["recall_gating_reason"]
        ctx.setdefault("debug", {})["recall_profile"] = selected_profile
        ctx.setdefault("debug", {})["recall_profile_source"] = profile_source
        ctx.setdefault("debug", {})["recall_profile_override_source"] = recall_policy.get("profile_override_source")

        grammar_collector = begin_plan_grammar(
            ctx,
            correlation_id=correlation_id,
            req=req,
            mode=str(mode),
            depth=depth,
            run_recall=bool(recall_policy["run_recall"]),
            recall_profile=str(selected_profile) if selected_profile else None,
            recall_reason=str(recall_policy.get("reason") or recall_policy.get("recall_gating_reason") or "unknown"),
            node_name=settings.node_name,
            code_version=settings.service_version,
        )

        if diagnostic:
            logger.info("Diagnostic PlanExecutionRequest json=%s", req.model_dump_json())
            logger.info(
                "Recall directive (raw) corr=%s enabled=%s required=%s cfg=%s",
                correlation_id,
                recall_enabled,
                recall_required,
                recall_cfg,
            )
            logger.info(
                "Recall selected profile=%s source=%s override_source=%s gating_reason=%s",
                selected_profile,
                profile_source,
                recall_policy.get("profile_override_source"),
                recall_policy["recall_gating_reason"],
            )

        logger.info(
            "Exec plan start: corr=%s mode=%s verb=%s recall_enabled=%s recall_required=%s steps=%s recall_cfg=%s",
            correlation_id,
            mode,
            plan.verb_name,
            recall_enabled,
            recall_required,
            [s.step_name for s in plan.steps],
            recall_cfg,
        )

        options = extra.get("options") or ctx.get("options") or {}
        if isinstance(options, dict):
            ctx.setdefault("options", options)
            for key, val in options.items():
                ctx.setdefault(key, val)

        if diagnostic:
            ctx["diagnostic"] = True

        ctx["verb"] = plan.verb_name
        if mode == "brain" and not _is_runtime_skill_verb(plan.verb_name):
            verb_lc = str(plan.verb_name or "").strip().lower()
            if verb_lc == "chat_quick":
                opts_now = ctx.get("options") if isinstance(ctx.get("options"), dict) else {}
                hub_full = bool(opts_now.get("chat_quick_full_stance"))
                if hub_full:
                    prepare_brain_reply_context(ctx)
                else:
                    prepare_chat_quick_reply_context(ctx)
            elif verb_lc == "chat_kids_story":
                prepare_chat_quick_reply_context(ctx)
            elif (
                ctx.get("skip_brain_reply_context")
                or str(plan.verb_name or "").strip().lower()
                in {"introspect_spark", "memory_graph_suggest"}
            ):
                logger.info(
                    "router_skip_prepare_brain_reply_context corr=%s verb=%s reason=spark_or_skip_flag",
                    correlation_id,
                    plan.verb_name,
                )
            else:
                prepare_brain_reply_context(ctx)
        existing_scope = str(ctx.get("_run_scope_corr_id") or "")
        if existing_scope and existing_scope != correlation_id:
            logger.warning(
                "router_scope_reset corr_id=%s previous_scope=%s",
                correlation_id,
                existing_scope,
            )
        ctx["_run_scope_corr_id"] = correlation_id
        scoped_results = ctx.setdefault("prior_step_results_by_corr", {})
        existing_scoped_results = scoped_results.get(correlation_id)
        if not isinstance(existing_scoped_results, list):
            existing_scoped_results = []
            scoped_results[correlation_id] = existing_scoped_results
        ctx["prior_step_results"] = list(existing_scoped_results)
        if mode == "brain" and plan.verb_name and not is_active(plan.verb_name, node_name=settings.node_name):
            logger.warning("Inactive verb blocked in router corr_id=%s verb=%s", correlation_id, plan.verb_name)
            record_assembled_grammar(
                ctx,
                status="fail",
                final_text_present=True,
                reasoning_present=False,
                thinking_source="none",
            )
            return PlanExecutionResult(
                verb_name=plan.verb_name,
                request_id=req.args.request_id,
                status="fail",
                blocked=False,
                blocked_reason=None,
                steps=step_results,
                mode=mode,
                final_text=f"Verb '{plan.verb_name}' is inactive on node {settings.node_name}.",
                memory_used=memory_used,
                recall_debug=recall_debug,
                error=f"inactive_verb:{plan.verb_name}",
            )
        # Supervised path only for depth=2 agent runtime flows
        if (depth == 2) or (mode == "agent" and plan.verb_name == "agent_runtime") or extra.get("supervised"):
            supervisor = Supervisor(bus)
            return await supervisor.execute(
                source=source,
                req=plan,
                correlation_id=correlation_id,
                ctx=ctx,
                recall_cfg=recall_cfg,
            )

        inline_recall = has_inline_recall(plan.steps)
        should_recall = bool(recall_policy["run_recall"])
        recall_reason = str(recall_policy["reason"])
        use_pcr_pre_recall = (
            settings.chat_pcr_enabled
            and str(plan.verb_name or "").strip().lower()
            in ("chat_general", "chat_quick", "stance_react")
            and should_recall
            and not inline_recall
        )

        if use_pcr_pre_recall:
            logger.info(
                "PCR phase0+1 start corr=%s profile=%s gating=%s",
                correlation_id,
                selected_profile,
                recall_reason,
            )
            _pcr_memory, recall_step, recall_debug = await run_pcr_phase0_and_1(
                bus,
                source=source,
                ctx=ctx,
                correlation_id=correlation_id,
                recall_cfg=recall_cfg,
            )
            if recall_step is not None:
                step_results.append(recall_step)
                grammar_collector.record_step_started(
                    order=0,
                    step_name=recall_step.step_name or "pcr_continuity_recall",
                    verb_name=recall_step.verb_name or "recall",
                    services=["RecallService"],
                )
                if recall_step.status == "success":
                    keys = sorted(recall_step.result.keys()) if isinstance(recall_step.result, dict) else []
                    grammar_collector.record_step_completed(
                        order=0,
                        step_name=recall_step.step_name or "pcr_continuity_recall",
                        latency_ms=recall_step.latency_ms,
                        result_service_keys=keys,
                    )
                else:
                    grammar_collector.record_step_failed(
                        order=0,
                        step_name=recall_step.step_name or "pcr_continuity_recall",
                        error_kind=short_error_kind(recall_step.error),
                    )
            memory_used = recall_step is not None and recall_step.status == "success"
            recall_count = 0
            if recall_step is not None and isinstance(recall_step.result, dict):
                recall_payload = recall_step.result.get("RecallService")
                if isinstance(recall_payload, dict):
                    recall_count = int(recall_payload.get("count") or 0)
            if isinstance(recall_debug, dict):
                recall_debug.setdefault("profile", CONTINUITY_PROFILE if recall_step else None)
                recall_debug.setdefault("profile_source", "pcr")
                recall_debug.setdefault("profile_override_source", recall_policy.get("profile_override_source"))
                recall_debug.setdefault("recall_gating_reason", recall_policy.get("recall_gating_reason"))
        elif should_recall and not inline_recall:
            logger.info(
                "Recall resolved profile=%s source=%s gating=%s recall_gating_reason=%s",
                selected_profile,
                profile_source,
                recall_reason,
                recall_policy["recall_gating_reason"],
            )
            recall_step, recall_debug, _ = await run_recall_step(
                bus,
                source=source,
                ctx=ctx,
                correlation_id=correlation_id,
                recall_cfg=recall_cfg,
                recall_profile=selected_profile,
                diagnostic=diagnostic,
                rpc_timeout_sec=None,
            )
            step_results.append(recall_step)
            grammar_collector.record_step_started(
                order=0,
                step_name=recall_step.step_name or "pre_recall",
                verb_name=recall_step.verb_name or "recall",
                services=["RecallService"],
            )
            if recall_step.status == "success":
                keys = sorted(recall_step.result.keys()) if isinstance(recall_step.result, dict) else []
                grammar_collector.record_step_completed(
                    order=0,
                    step_name=recall_step.step_name or "pre_recall",
                    latency_ms=recall_step.latency_ms,
                    result_service_keys=keys,
                )
            else:
                grammar_collector.record_step_failed(
                    order=0,
                    step_name=recall_step.step_name or "pre_recall",
                    error_kind=short_error_kind(recall_step.error),
                )
            memory_used = recall_step.status == "success"
            recall_count = 0
            if isinstance(recall_step.result, dict):
                recall_payload = recall_step.result.get("RecallService")
                if isinstance(recall_payload, dict):
                    recall_count = int(recall_payload.get("count") or 0)
                    recall_debug = recall_payload
            if isinstance(recall_debug, dict):
                recall_debug.setdefault("profile", selected_profile)
                recall_debug.setdefault("profile_source", profile_source)
                recall_debug.setdefault("profile_override_source", recall_policy.get("profile_override_source"))
                recall_debug.setdefault("recall_gating_reason", recall_policy.get("recall_gating_reason"))
            if recall_required and recall_count == 0:
                if diagnostic:
                    logger.info(
                        "required recall empty; failing fast session_id=%s trace_id=%s",
                        ctx.get("session_id"),
                        ctx.get("trace_id"),
                    )
                record_assembled_grammar(
                    ctx,
                    status="fail",
                    final_text_present=False,
                    reasoning_present=False,
                    thinking_source="none",
                )
                return PlanExecutionResult(
                    verb_name=plan.verb_name,
                    request_id=req.args.request_id,
                    status="fail",
                    blocked=False,
                    blocked_reason=None,
                    steps=step_results,
                    mode=mode,
                    final_text=None,
                    memory_used=memory_used,
                    recall_debug=recall_debug,
                    error="recall_required_but_empty",
                )
            if recall_step.status != "success":
                overall_status = "fail" if recall_required else "partial"
                soft_failure = not recall_required
                if not recall_required:
                    ctx["recall_soft_failed"] = True
                    ctx["memory_used"] = False
                    logger.info(
                        "recall_fail_open_continuing corr=%s verb=%s recall_error=%s",
                        correlation_id,
                        plan.verb_name,
                        recall_step.error,
                    )
                if recall_required:
                    record_assembled_grammar(
                        ctx,
                        status=overall_status,
                        final_text_present=False,
                        reasoning_present=False,
                        thinking_source="none",
                    )
                    return PlanExecutionResult(
                        verb_name=plan.verb_name,
                        request_id=req.args.request_id,
                        status=overall_status,
                        blocked=False,
                        blocked_reason=None,
                        steps=step_results,
                        mode=mode,
                        final_text=None,
                        memory_used=memory_used,
                        recall_debug=recall_debug,
                        error=recall_step.error,
                    )

        else:
            if inline_recall:
                recall_debug = {
                    "skipped": "inline_recall_step_present",
                    "recall_gating_reason": recall_policy["recall_gating_reason"],
                }
                logger.info(
                    "Recall skipped; inline RecallService step present",
                    extra={"correlation_id": correlation_id, "recall_cfg": recall_cfg, "diagnostic": diagnostic},
                )
            else:
                recall_debug = {
                    "skipped": recall_reason,
                    "recall_gating_reason": recall_policy["recall_gating_reason"],
                }
                ctx["memory_used"] = False
                logger.info(
                    "Recall skipped by gating (%s) recall_gating_reason=%s",
                    recall_reason,
                    recall_policy["recall_gating_reason"],
                    extra={"correlation_id": correlation_id, "recall_cfg": recall_cfg, "diagnostic": diagnostic},
                )

        for step in sorted(plan.steps, key=lambda s: s.order):
            if step.verb_name and not is_active(step.verb_name, node_name=settings.node_name):
                logger.warning("Inactive step verb blocked corr_id=%s verb=%s", correlation_id, step.verb_name)
                step_results.append(
                    StepExecutionResult(
                        status="fail",
                        verb_name=step.verb_name,
                        step_name=step.step_name,
                        order=step.order,
                        result={"error": "inactive_verb", "verb": step.verb_name, "node": settings.node_name},
                        latency_ms=0,
                        node=settings.node_name,
                        logs=[f"reject <- inactive verb {step.verb_name}"],
                        error=f"inactive_verb:{step.verb_name}",
                    )
                )
                overall_status = "fail"
                break
            ctx["prior_step_results"] = [res.model_dump(mode="json") for res in step_results]
            scoped = ctx.setdefault("prior_step_results_by_corr", {})
            if isinstance(scoped, dict):
                scoped[correlation_id] = list(ctx["prior_step_results"])
            grammar_collector = get_or_create_collector(
                ctx,
                correlation_id=correlation_id,
                node_name=settings.node_name,
                code_version=settings.service_version,
            )
            grammar_collector.record_step_started(
                order=step.order,
                step_name=step.step_name,
                verb_name=step.verb_name or plan.verb_name or "unknown",
                services=list(step.services or []),
            )
            step_res = await call_step_services(
                bus,
                source=source,
                step=step,
                ctx=ctx,
                correlation_id=correlation_id,
                diagnostic=diagnostic,
            )
            step_results.append(step_res)
            if step_res.status == "success":
                keys = sorted(step_res.result.keys()) if isinstance(step_res.result, dict) else []
                grammar_collector.record_step_completed(
                    order=step.order,
                    step_name=step.step_name,
                    latency_ms=step_res.latency_ms,
                    result_service_keys=keys,
                )
            else:
                grammar_collector.record_step_failed(
                    order=step.order,
                    step_name=step.step_name,
                    error_kind=short_error_kind(step_res.error),
                )
            if isinstance(step_res.result, dict) and "RecallService" in step_res.result:
                recall_debug = step_res.result.get("RecallService", {})
                memory_used = step_res.status == "success"
                ctx["memory_used"] = memory_used

            if (
                settings.chat_pcr_enabled
                and settings.chat_pcr_post_stance_recall
                and str(plan.verb_name or "").strip().lower() in ("chat_general", "chat_quick")
                and step.step_name == "synthesize_chat_stance_brief"
                and step_res.status == "success"
            ):
                _pcr_phase3, phase3_recall_step, phase3_debug = await run_pcr_phase3(
                    bus,
                    source=source,
                    ctx=ctx,
                    correlation_id=correlation_id,
                    recall_cfg=recall_cfg,
                )
                if phase3_recall_step is not None:
                    step_results.append(phase3_recall_step)
                    grammar_collector.record_step_started(
                        order=step.order,
                        step_name=phase3_recall_step.step_name or "pcr_belief_recall",
                        verb_name=phase3_recall_step.verb_name or "recall",
                        services=["RecallService"],
                    )
                    if phase3_recall_step.status == "success":
                        keys = (
                            sorted(phase3_recall_step.result.keys())
                            if isinstance(phase3_recall_step.result, dict)
                            else []
                        )
                        grammar_collector.record_step_completed(
                            order=step.order,
                            step_name=phase3_recall_step.step_name or "pcr_belief_recall",
                            latency_ms=phase3_recall_step.latency_ms,
                            result_service_keys=keys,
                        )
                    else:
                        grammar_collector.record_step_failed(
                            order=step.order,
                            step_name=phase3_recall_step.step_name or "pcr_belief_recall",
                            error_kind=short_error_kind(phase3_recall_step.error),
                        )
                    memory_used = phase3_recall_step.status == "success"
                if isinstance(phase3_debug, dict) and isinstance(recall_debug, dict):
                    recall_debug.update(phase3_debug)

            if (
                settings.orion_unified_grounding_enabled
                and str(plan.verb_name or "").strip().lower() == "stance_react"
                and step.step_name == "llm_stance_react"
                and step_res.status == "success"
            ):
                stance_text, _ = _extract_final_text([step_res], verb_name=plan.verb_name)
                grounding_capsule = await assemble_stance_grounding(
                    bus,
                    source=source,
                    ctx=ctx,
                    correlation_id=correlation_id,
                    recall_cfg=recall_cfg,
                    stance_step_text=stance_text or "",
                )
                if grounding_capsule is not None:
                    ctx["grounding_capsule"] = grounding_capsule.model_dump(mode="json")

            if step_res.status != "success":
                overall_status = "partial" if len(step_results) > 1 else "fail"
                break

        final_text, final_text_diag = _extract_final_text(step_results, verb_name=plan.verb_name)
        if plan.verb_name == "chat_general" and isinstance(final_text, str) and final_text.strip():
            final_text, identity_diag = _apply_chat_general_identity_boundary_guard(
                final_text,
                verb_name=plan.verb_name,
            )
            stance_brief = (
                ctx.get("chat_stance_brief") if isinstance(ctx.get("chat_stance_brief"), dict) else None
            )
            final_text, recital_stripped = strip_identity_recital_leadin(
                final_text,
                str(ctx.get("user_message") or ""),
                chat_stance_brief=stance_brief,
            )
            if recital_stripped:
                identity_diag["identity_recital_leadin_stripped"] = True
            final_text, closer_stripped = strip_transactional_closers(
                final_text,
                chat_stance_brief=stance_brief,
            )
            if closer_stripped:
                identity_diag["transactional_closer_stripped"] = True
            final_text_diag.update(identity_diag)
            if (
                identity_diag.get("identity_boundary_applied")
                or recital_stripped
                or closer_stripped
            ):
                final_text_diag["result_len"] = len(final_text)
                final_text_diag["final_len"] = len(final_text)
        logger.info(
            "final_text_assembly corr_id=%s verb=%s source_service=%s source_field=%s think_tags_detected=%s think_full_block_detected=%s think_close_tag_only_detected=%s think_stripping_applied=%s planning_stripping_applied=%s planning_lines_dropped=%s planning_candidate_rejected=%s planning_candidate_rejection_reason=%s provider_has_reasoning_content=%s provider_has_reasoning_trace=%s provider_reasoning_available=%s inline_think_extracted=%s provider_completion_tokens=%s provider_finish_reason=%s truncation_detected=%s structured_output_sanitized=%s structured_output_rejected=%s structured_json_extraction_attempted=%s candidates=%s rejected_candidates=%s candidate_fields_considered=%s identity_boundary_applied=%s identity_boundary_violations=%s raw_len=%s clean_len=%s final_len=%s result_len=%s",
            correlation_id,
            plan.verb_name,
            final_text_diag.get("source_service"),
            final_text_diag.get("source_field"),
            final_text_diag.get("think_tags_detected"),
            final_text_diag.get("think_full_block_detected"),
            final_text_diag.get("think_close_tag_only_detected"),
            final_text_diag.get("think_stripping_applied"),
            final_text_diag.get("planning_stripping_applied"),
            final_text_diag.get("planning_lines_dropped"),
            final_text_diag.get("planning_candidate_rejected"),
            final_text_diag.get("planning_candidate_rejection_reason"),
            final_text_diag.get("provider_has_reasoning_content"),
            final_text_diag.get("provider_has_reasoning_trace"),
            final_text_diag.get("provider_reasoning_available"),
            final_text_diag.get("inline_think_extracted"),
            final_text_diag.get("provider_completion_tokens"),
            final_text_diag.get("provider_finish_reason"),
            final_text_diag.get("truncation_detected"),
            final_text_diag.get("structured_output_sanitized"),
            final_text_diag.get("structured_output_rejected"),
            final_text_diag.get("structured_json_extraction_attempted"),
            final_text_diag.get("candidate_count"),
            final_text_diag.get("rejected_candidate_count"),
            final_text_diag.get("candidate_fields_considered"),
            final_text_diag.get("identity_boundary_applied"),
            final_text_diag.get("identity_boundary_violations"),
            final_text_diag.get("raw_len"),
            final_text_diag.get("clean_len"),
            final_text_diag.get("final_len"),
            final_text_diag.get("result_len"),
        )
        ctx["runtime_response_diagnostics"] = {
            "provider_finish_reason": final_text_diag.get("provider_finish_reason"),
            "provider_completion_tokens": final_text_diag.get("provider_completion_tokens"),
            "truncation_detected": bool(final_text_diag.get("truncation_detected")),
            "planning_candidate_rejected": bool(final_text_diag.get("planning_candidate_rejected")),
            "status": overall_status,
            "world_context_capsule_loaded": bool(ctx.get("world_context_capsule_loaded")),
            "capsule_age_hours": ctx.get("capsule_age_hours"),
            "capsule_topic_count": len((ctx.get("world_context_capsule") or {}).get("salient_topics") or [])
            if isinstance(ctx.get("world_context_capsule"), dict)
            else 0,
            "capsule_filtered_reason": ctx.get("capsule_filtered_reason"),
            "stance_world_context_items_used": ctx.get("stance_world_context_items_used"),
            "politics_context_suppressed": bool(ctx.get("politics_context_suppressed")),
            "identity_boundary_applied": bool(final_text_diag.get("identity_boundary_applied")),
            "identity_boundary_violations": final_text_diag.get("identity_boundary_violations"),
        }
        if overall_status == "success" and soft_failure:
            overall_status = "partial"
        if _should_fail_empty_runtime_skill_output(
            overall_status=overall_status,
            verb_name=plan.verb_name,
            final_text=final_text,
        ):
            overall_status = "fail"
            logger.error(
                "runtime_skill_empty_terminal_output corr_id=%s verb=%s status=fail",
                correlation_id,
                plan.verb_name,
            )
        if _should_fail_empty_structured_verb_output(
            overall_status=overall_status,
            verb_name=plan.verb_name,
            final_text=final_text,
            final_text_diag=final_text_diag,
        ):
            overall_status = "fail"
            preview = str(final_text_diag.get("structured_rejection_preview") or "")[:240]
            logger.error(
                "===MEMGRAPH_SUGGEST_TRACE=== structured_empty_terminal_output corr_id=%s verb=%s "
                "structured_output_rejected=%s preview=%r candidates=%s rejected=%s",
                correlation_id,
                plan.verb_name,
                bool(final_text_diag.get("structured_output_rejected")),
                preview,
                final_text_diag.get("candidate_count"),
                final_text_diag.get("rejected_candidate_count"),
            )

        if depth == 1:
            logger.info("depth1_complete corr_id=%s verb=%s elapsed=%s", correlation_id, plan.verb_name, sum([s.latency_ms for s in step_results]))
        metacog_traces = _collect_metacog_traces(
            step_results,
            correlation_id=correlation_id,
            session_id=str(ctx.get("session_id")) if ctx.get("session_id") else None,
        )
        reasoning_content, inline_think_content, thinking_source, reasoning_trace = _extract_reasoning_payload(
            step_results,
            think_close_tag_only_detected=bool(final_text_diag.get("think_close_tag_only_detected")),
            prior_step_results=list(ctx.get("prior_step_results") or []),
            correlation_id=correlation_id,
            canonical_verb_name="chat_general" if plan.verb_name == "chat_general" else None,
            canonical_step_name="llm_chat_general" if plan.verb_name == "chat_general" else None,
        )
        if reasoning_trace is None and metacog_traces:
            first_trace = metacog_traces[0]
            reasoning_trace = first_trace.model_dump(mode="json")
            if not reasoning_content:
                reasoning_content = str(first_trace.content or "").strip() or None
                if thinking_source == "none" and reasoning_content:
                    thinking_source = "provider_reasoning"
        if thinking_source == "none" and inline_think_content:
            thinking_source = "inline_think_close_tag_only" if bool(final_text_diag.get("think_close_tag_only_detected")) else "inline_think_full_block"
        logger.info(
            "exec_reasoning_selection corr_id=%s provider_reasoning_available=%s reasoning_content_len=%s inline_think_content_len=%s thinking_source=%s final_clean_answer_len=%s",
            correlation_id,
            bool(reasoning_content),
            len(reasoning_content) if isinstance(reasoning_content, str) else 0,
            len(inline_think_content) if isinstance(inline_think_content, str) else 0,
            thinking_source,
            len(final_text or ""),
        )
        if plan.verb_name == "chat_general":
            wrote_chat_history = bool(str(inline_think_content or "").strip())
            logger.info(
                "chat_general_thought_capture corr=%s step=llm_chat_general think_len=%s source=%s wrote_chat_history=%s",
                correlation_id,
                len(str(inline_think_content or "").strip()),
                thinking_source,
                wrote_chat_history,
            )
            if isinstance(final_text, str):
                ctx["chat_general_final_text_clean"] = final_text
        ctx["reasoning_content"] = reasoning_content
        ctx["inline_think_content"] = inline_think_content
        ctx["thinking_source"] = thinking_source
        if isinstance(reasoning_trace, dict):
            ctx["reasoning_trace"] = reasoning_trace
        logger.info(
            "cortex_exec_metacog_attached corr_id=%s verb=%s traces=%s",
            correlation_id,
            plan.verb_name,
            len(metacog_traces),
        )
        if _thought_debug_enabled():
            logger.info(
                "THOUGHT_DEBUG_EXEC stage=metacog_attached corr=%s verb=%s traces=%s final_text_len=%s fallback_to_final_text=%s",
                correlation_id,
                plan.verb_name,
                len(metacog_traces),
                _debug_len(final_text),
                len(metacog_traces) == 0,
            )
        _harvest_llm_uncertainty_from_steps(ctx, step_results)
        metadata = _autonomy_payload_from_ctx(ctx)
        metadata.update(_situation_grounding_metadata_from_ctx(ctx))
        if final_text_diag.get("structured_output_rejected"):
            metadata["structured_output_rejected"] = True
            if final_text_diag.get("structured_rejection_preview"):
                metadata["structured_rejection_preview"] = str(final_text_diag["structured_rejection_preview"])[:500]
        if isinstance(ctx.get("grounding_capsule"), dict):
            metadata["grounding_capsule"] = ctx["grounding_capsule"]
        mark_orion_turn(str(ctx.get("session_id") or "global"))

        record_assembled_grammar(
            ctx,
            status=overall_status,
            final_text_present=bool((final_text or "").strip()),
            reasoning_present=bool(reasoning_content or reasoning_trace or metacog_traces),
            thinking_source=str(thinking_source or "none"),
        )

        if settings.publish_reasoning_telemetry:
            try:
                reasoning_call = build_reasoning_call(
                    correlation_id=correlation_id,
                    verb=plan.verb_name,
                    mode=str(mode or "unknown"),
                    node_id=settings.node_name,
                    turn_id=str(ctx.get("turn_id") or ctx.get("message_id") or ctx.get("messageId") or "") or None,
                    # No plan-run signal for the enable_thinking template flag is
                    # available here (set only in pre_turn_appraisal). Report False.
                    thinking_enabled=False,
                    diagnostics=final_text_diag,
                    completion_tokens=final_text_diag.get("provider_completion_tokens"),
                    # prompt_tokens is not surfaced in _extract_final_text diagnostics.
                    prompt_tokens=None,
                    # No provider exposes a separate thinking-token count today.
                    thinking_tokens=None,
                )
                await publish_reasoning_call(
                    bus,
                    source=source,
                    channel=settings.channel_reasoning_call,
                    call=reasoning_call,
                )
            except Exception:
                logger.warning(
                    "reasoning_telemetry_emit_failed corr_id=%s verb=%s",
                    correlation_id,
                    plan.verb_name,
                    exc_info=True,
                )

        return PlanExecutionResult(
            verb_name=plan.verb_name,
            request_id=req.args.request_id,
            status=overall_status,
            blocked=False,
            blocked_reason=None,
            steps=step_results,
            mode=mode,
            final_text=final_text or None,
            reasoning_content=reasoning_content,
            inline_think_content=inline_think_content,
            thinking_source=thinking_source,
            reasoning_trace=reasoning_trace,
            memory_used=memory_used,
            recall_debug=recall_debug,
            metacog_traces=metacog_traces,
            metadata=metadata,
            error=None if overall_status == "success" else (step_results[-1].error or "empty_runtime_terminal_output"),
        )


# Backward-compat alias: earlier patches referenced PlanRouter.
PlanRouter = PlanRunner
