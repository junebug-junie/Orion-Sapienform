"""Memory graph suggest: grounded Quick primary, Brain escalation on hard failures."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Literal, Optional, Tuple
from unittest.mock import AsyncMock
from uuid import uuid4

from pydantic import ValidationError

from orion.memory_graph.approve import preview_validate_only
from orion.memory_graph.dto import SuggestDraftV1
from orion.memory_graph.suggest_validate import (
    parse_json_object,
    repair_role_grounded_suggest_draft,
    role_grounded_extraction_expected,
    validate_for_escalation,
)
from orion.schemas.cortex.contracts import CortexChatRequest, CortexChatResult

from scripts.cortex_memory_graph_text import hub_memory_graph_suggest_text
from scripts.cortex_request_builder import (
    HubRequestValidationError,
    build_chat_request,
    build_continuity_messages,
    validate_single_verb_override,
)
from scripts.memory_graph_suggest_timeout import (
    cortex_rpc_timeout_sec,
    hub_client_fetch_timeout_ms,
    memory_graph_suggest_server_budget_sec,
    resolve_memory_graph_suggest_timeouts,
)
from scripts.memory_graph_structured_output import (
    extract_gateway_structured_diagnostics,
    memory_graph_suggest_llm_options,
    resolve_memory_graph_structured_output_method,
)
from scripts.mutation_cognition_context import build_mutation_cognition_context

logger = logging.getLogger("orion-hub.memory_graph_suggest")

RouteName = Literal["brain", "quick"]


def _normalize_route(raw: str | None, default: RouteName) -> RouteName:
    s = str(raw or "").strip().lower()
    if s == "quick":
        return "quick"
    if s == "brain":
        return "brain"
    return default


def _apply_llm_route(options: Dict[str, Any], route: RouteName) -> Dict[str, Any]:
    """Map suggest escalation labels to LLM gateway lanes (always cheap quick, not UI brain/chat)."""
    out = dict(options or {})
    # MEMORY_GRAPH_SUGGEST_ESCALATION_ROUTE=brain is a retry label only — not Hub UI brain mode.
    out["llm_route"] = "quick"
    return out


def _short_exc(exc: BaseException, *, limit: int = 240) -> str:
    s = f"{type(exc).__name__}: {exc}"
    return s if len(s) <= limit else s[: limit - 3] + "..."


def _extract_attempt_diagnostics(resp: CortexChatResult, text: str) -> Dict[str, Any]:
    cr = resp.cortex_result
    meta = cr.metadata if isinstance(cr.metadata, dict) else {}
    finish_reason = meta.get("provider_finish_reason") or meta.get("finish_reason")
    completion_tokens = meta.get("provider_completion_tokens") or meta.get("completion_tokens")
    model_used = meta.get("model") or meta.get("model_name")

    if finish_reason is None or completion_tokens is None:
        for step in cr.steps or []:
            detail = getattr(step, "detail", None) or {}
            if not isinstance(detail, dict):
                continue
            raw = detail.get("raw") if isinstance(detail.get("raw"), dict) else {}
            usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
            choices = raw.get("choices") if isinstance(raw.get("choices"), list) else []
            first = choices[0] if choices and isinstance(choices[0], dict) else {}
            finish_reason = finish_reason or first.get("finish_reason")
            completion_tokens = completion_tokens or usage.get("completion_tokens")

    return {
        "finish_reason": finish_reason,
        "completion_tokens": completion_tokens,
        "model_used": model_used,
        "content_len": len(text or ""),
    }


async def _call_cortex(
    cortex_client: Any,
    req: CortexChatRequest,
    *,
    timeout_sec: float,
    settings: Any,
    route: RouteName,
) -> Tuple[Optional[CortexChatResult], Optional[str], Dict[str, Any]]:
    corr = str(uuid4())
    rpc_timeout = cortex_rpc_timeout_sec(timeout_sec, settings)
    timing: Dict[str, Any] = {
        "configured_timeout_sec": float(timeout_sec),
        "bus_rpc_timeout_sec": float(rpc_timeout),
        "timeout_layer": "hub_asyncio_wait_for",
        "target_service": "cortex_gateway",
        "route": route,
    }
    t0 = time.monotonic()
    try:
        chat_fn = cortex_client.chat
        if isinstance(chat_fn, AsyncMock):
            chat_coro = chat_fn(req, correlation_id=corr)
        else:
            chat_coro = chat_fn(req, correlation_id=corr, rpc_timeout_sec=rpc_timeout)
        resp = await asyncio.wait_for(chat_coro, timeout=timeout_sec)
        timing["elapsed_sec"] = round(time.monotonic() - t0, 3)
        timing["reached_cortex"] = True
        return resp, None, timing
    except TimeoutError:
        timing["elapsed_sec"] = round(time.monotonic() - t0, 3)
        timing["error_type"] = "TimeoutError"
        timing["error_summary"] = "hub_wait_for_timeout"
        return None, "hub_wait_for_timeout", timing
    except Exception as exc:  # noqa: BLE001
        timing["elapsed_sec"] = round(time.monotonic() - t0, 3)
        timing["error_type"] = type(exc).__name__
        timing["error_summary"] = _short_exc(exc, limit=400)
        logger.warning("memory_graph_suggest_cortex_error corr=%s error=%s", corr, exc)
        return None, timing["error_summary"], timing


def _cortex_text_and_status(resp: CortexChatResult) -> Tuple[str, bool, str]:
    cr = resp.cortex_result
    text, _ = hub_memory_graph_suggest_text(resp)
    ok_flag = bool(getattr(cr, "ok", False))
    status = str(getattr(cr, "status", "") or "")
    if not str(text).strip():
        return text, False, "empty_final_text"
    if not ok_flag:
        return text, False, status or "not_ok"
    return text, True, status or "ok"


def _parse_suggest_draft_from_text(
    text: str,
    *,
    utterance_text: str,
    finish_reason: Optional[str] = None,
) -> Tuple[Optional[SuggestDraftV1], bool, List[str], Optional[str]]:
    """Return (draft, should_escalate, validation_errors, parse_error_code)."""
    data, parse_err = parse_json_object(text)
    if data is None:
        raw = (text or "").strip()
        if (
            parse_err == "no_json_object"
            and raw.startswith("{")
            and str(finish_reason or "").strip().lower() == "length"
        ):
            return None, True, ["json_truncated"], "json_truncated"
        return None, True, [parse_err or "parse_failed"], parse_err

    if role_grounded_extraction_expected(utterance_text):
        data = repair_role_grounded_suggest_draft(data, utterance_text=utterance_text)

    from orion.memory_graph.draft_sanitize import sanitize_suggest_draft_dict

    data = sanitize_suggest_draft_dict(data)

    should_escalate, validation_errors = validate_for_escalation(data, utterance_text=utterance_text)
    if should_escalate:
        return None, True, validation_errors, None

    try:
        return SuggestDraftV1.model_validate(data), False, validation_errors, None
    except ValidationError as e:
        return None, True, [_short_exc(e, limit=400)], "pydantic_validation"


async def suggest_with_escalation(
    *,
    cortex_client: Any,
    payload: Dict[str, Any],
    session_id: str,
    user_id: Optional[str],
    settings: Any,
    mutation_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Try grounded Quick, then Brain on hard failures; RDF-validate without persisting.

    Returns a dict suitable for JSONResponse (no HTTPException).
    """
    inactive = validate_single_verb_override(
        {**payload, "verbs": ["memory_graph_suggest"]},
        node_name=str(getattr(settings, "NODE_NAME", "orion-hub")),
        prompt=str((payload.get("messages") or [{}])[-1].get("content") or ""),
    )
    if inactive:
        return {
            "ok": False,
            "error": inactive.get("error"),
            "detail": inactive,
            "attempts": [
                {
                    "route": None,
                    "index": 0,
                    "phase": "verb_inactive",
                    "error_summary": str(inactive.get("error") or "inactive_verb"),
                }
            ],
        }

    primary = _normalize_route(
        getattr(settings, "MEMORY_GRAPH_SUGGEST_PRIMARY_ROUTE", None), "quick"
    )
    escalation = _normalize_route(
        getattr(settings, "MEMORY_GRAPH_SUGGEST_ESCALATION_ROUTE", None), "brain"
    )
    enable_escalation = bool(
        getattr(settings, "MEMORY_GRAPH_SUGGEST_ENABLE_ESCALATION", True)
    )
    include_grounding = bool(
        getattr(settings, "MEMORY_GRAPH_SUGGEST_INCLUDE_GROUNDING", True)
    )
    verb_timeout_sec, t_quick, t_brain, verb_timeout_ms = resolve_memory_graph_suggest_timeouts(
        settings, escalation_enabled=enable_escalation
    )

    def timeout_for(route: RouteName) -> float:
        return t_brain if route == "brain" else t_quick

    user_messages = payload.get("messages") or []
    if not isinstance(user_messages, list) or not user_messages:
        return {
            "ok": False,
            "error": "missing_messages",
            "attempts": [
                {
                    "route": None,
                    "index": 0,
                    "phase": "request",
                    "error_summary": "messages[] missing or empty",
                }
            ],
        }
    user_prompt = str(user_messages[-1].get("content", "") or "")

    context_turns = int(
        payload.get("context_turns")
        or getattr(settings, "MEMORY_GRAPH_SUGGEST_CONTEXT_TURNS", None)
        or getattr(settings, "HUB_CONTEXT_TURNS", 10)
    )
    continuity = build_continuity_messages(
        history=user_messages,
        latest_user_prompt=user_prompt,
        turns=context_turns,
    )
    routed = dict(payload)
    routed["no_write"] = True
    routed["verbs"] = ["memory_graph_suggest"]
    if include_grounding:
        routed.setdefault("use_recall", True)
    mc = mutation_context if mutation_context is not None else build_mutation_cognition_context()
    routed["mutation_cognition_context"] = mc

    diagnostic = bool(payload.get("diagnostic")) or bool(
        isinstance(payload.get("options"), dict) and (payload.get("options") or {}).get("diagnostic")
    )

    routes_to_try: List[Tuple[RouteName, float]] = [(primary, timeout_for(primary))]
    if enable_escalation and escalation != primary:
        routes_to_try.append((escalation, timeout_for(escalation)))

    attempts_meta: List[Dict[str, Any]] = []
    route_used: Optional[RouteName] = None
    suggest_timeout_budget: Dict[str, Any] = {
        "verb_timeout_ms": verb_timeout_ms,
        "verb_timeout_sec": verb_timeout_sec,
        "quick_timeout_sec": t_quick,
        "brain_timeout_sec": t_brain,
        "escalation_enabled": enable_escalation,
        "max_server_sec": memory_graph_suggest_server_budget_sec(
            settings, escalation_enabled=enable_escalation
        ),
        "client_fetch_timeout_ms": hub_client_fetch_timeout_ms(
            settings, escalation_enabled=enable_escalation
        ),
        "source": "orion/cognition/verbs/memory_graph_suggest.yaml",
    }

    for idx, (route, timeout_sec) in enumerate(routes_to_try):
        attempt: Dict[str, Any] = {
            "route": route,
            "timeout_sec": timeout_sec,
            "index": idx,
            "grounding_included": include_grounding,
        }
        try:
            req, _route_debug, _ = build_chat_request(
                payload=routed,
                session_id=session_id,
                user_id=user_id,
                trace_id=None,
                default_mode="quick",
                auto_default_enabled=bool(getattr(settings, "HUB_AUTO_DEFAULT_ENABLED", False)),
                source_label="hub_memory_graph_suggest",
                prompt=user_prompt,
                messages=continuity,
            )
        except HubRequestValidationError as exc:
            return {
                "ok": False,
                "error": str(exc),
                "error_code": getattr(exc, "code", None),
                "attempts": [
                    {
                        "route": route,
                        "index": idx,
                        "phase": "build_request",
                        "error_summary": str(exc),
                    }
                ],
            }

        opts = _apply_llm_route(dict(req.options or {}), route)
        if include_grounding:
            opts["memory_graph_include_grounding"] = True
        structured_opts = memory_graph_suggest_llm_options(settings, diagnostic=diagnostic)
        opts.update(structured_opts)
        opts["skip_brain_reply_context"] = True
        opts["structured_output_method_requested"] = resolve_memory_graph_structured_output_method(
            settings
        )
        req = req.model_copy(update={"options": opts})

        resp, cortex_err, timing = await _call_cortex(
            cortex_client,
            req,
            timeout_sec=timeout_sec,
            settings=settings,
            route=route,
        )
        attempt.update(timing)
        if resp is None:
            err_code = cortex_err or "cortex_unavailable"
            attempt.update(
                {
                    "phase": "cortex",
                    "error_summary": err_code,
                    "validation_errors": [err_code],
                    "fallback_draft_loaded": False,
                }
            )
            attempts_meta.append(attempt)
            continue

        text, text_diag = hub_memory_graph_suggest_text(resp)
        _, cortex_ok, st = _cortex_text_and_status(resp)
        if text:
            cortex_ok = True
            st = "ok"
        attempt.update(_extract_attempt_diagnostics(resp, text))
        attempt["text_extraction"] = text_diag
        if text_diag.get("structured_rejection_preview"):
            attempt["structured_rejection_preview"] = text_diag["structured_rejection_preview"]
        logger.info(
            "===MEMGRAPH_SUGGEST_TRACE=== hub_extract corr=%s route=%s final_text_len=%s source=%s candidates=%s",
            getattr(resp.cortex_result, "correlation_id", None),
            route,
            text_diag.get("final_text_len"),
            text_diag.get("selected_text_source"),
            text_diag.get("candidate_fields"),
        )
        attempt["cortex_ok"] = cortex_ok
        attempt["cortex_status"] = st
        if diagnostic:
            attempt["raw_text"] = text

        if not cortex_ok:
            err_parts: List[str] = []
            cr = resp.cortex_result
            if isinstance(getattr(cr, "error", None), dict):
                err_parts.append(str(cr.error))
            if getattr(cr, "status", None):
                err_parts.append(str(cr.status))
            attempt["phase"] = "cortex"
            attempt["error_summary"] = " | ".join(err_parts) if err_parts else "cortex_not_ok"
            attempt["validation_errors"] = [attempt["error_summary"]]
            attempts_meta.append(attempt)
            continue

        draft, should_escalate, validation_errors, parse_err = _parse_suggest_draft_from_text(
            text,
            utterance_text=user_prompt,
            finish_reason=attempt.get("finish_reason"),
        )
        attempt["validation_errors"] = validation_errors
        structured_diag = extract_gateway_structured_diagnostics(resp)
        structured_diag["parse_ok"] = parse_err is None and bool(str(text or "").strip())
        structured_diag["pydantic_validation_ok"] = draft is not None
        structured_diag["structured_output_method_requested"] = opts.get(
            "structured_output_method_requested"
        ) or opts.get("structured_output_method")
        attempt["structured_output"] = structured_diag

        if draft is None:
            attempt["phase"] = parse_err or "parse"
            attempt["error_summary"] = "; ".join(validation_errors[:6]) or "parse_failed"
            attempts_meta.append(attempt)
            if should_escalate:
                continue
            return {
                "ok": False,
                "error": "memory_graph_suggest_failed",
                "attempts": attempts_meta,
                "route_used": None,
                "validation_errors": validation_errors,
            }

        ok_graph, violations, preview = preview_validate_only(draft)

        if not ok_graph:
            attempt["phase"] = "rdf_validate"
            attempt["error_summary"] = "; ".join(violations[:6]) + ("…" if len(violations) > 6 else "")
            attempts_meta.append(attempt)
            out: Dict[str, Any] = {
                "ok": False,
                "violations": violations,
                "validation_warnings": violations,
                "preview": preview,
                "draft": draft.model_dump(mode="json", by_alias=True),
                "appendix_c_json": draft.model_dump_json(by_alias=True),
                "route_used": route,
                "attempts": attempts_meta,
                "grounding_included": include_grounding,
            }
            if diagnostic:
                out["diagnostic_raw"] = text
            return out

        attempt["phase"] = "success"
        attempt["error_summary"] = None
        attempts_meta.append(attempt)
        route_used = route
        out_ok: Dict[str, Any] = {
            "ok": True,
            "violations": [],
            "validation_warnings": validation_errors,
            "preview": preview,
            "draft": draft.model_dump(mode="json", by_alias=True),
            "appendix_c_json": draft.model_dump_json(by_alias=True),
            "route_used": route,
            "attempts": attempts_meta,
            "grounding_included": include_grounding,
            "suggest_timeout_budget": suggest_timeout_budget,
            "structured_output_diagnostics": structured_diag,
        }
        if diagnostic:
            out_ok["diagnostic_raw"] = text
        return out_ok

    all_errors: List[str] = []
    for att in attempts_meta:
        for err in att.get("validation_errors") or []:
            if err and err not in all_errors:
                all_errors.append(str(err))
    return {
        "ok": False,
        "error": "memory_graph_suggest_failed",
        "attempts": attempts_meta,
        "route_used": route_used,
        "grounding_included": include_grounding,
        "validation_errors": all_errors,
        "suggest_timeout_budget": suggest_timeout_budget,
        "fallback_draft_loaded": True,
    }


async def run_memory_graph_suggest_with_fallback(
    *,
    cortex_client: Any,
    payload: Dict[str, Any],
    session_id: str,
    user_id: Optional[str],
    settings: Any,
    mutation_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Backward-compatible alias for suggest_with_escalation."""
    out = await suggest_with_escalation(
        cortex_client=cortex_client,
        payload=payload,
        session_id=session_id,
        user_id=user_id,
        settings=settings,
        mutation_context=mutation_context,
    )
    if "suggest_route_used" not in out and "route_used" in out:
        out["suggest_route_used"] = out.get("route_used")
    if "suggest_attempts" not in out and "attempts" in out:
        out["suggest_attempts"] = out.get("attempts")
    return out
