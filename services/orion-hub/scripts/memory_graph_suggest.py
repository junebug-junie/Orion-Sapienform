"""Memory graph Appendix C suggest: cortex primary route then optional quick fallback."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Literal, Optional, Tuple
from uuid import uuid4

from pydantic import ValidationError

from orion.memory_graph.approve import preview_validate_only
from orion.memory_graph.dto import SuggestDraftV1
from orion.memory_graph.json_extract import extract_first_json_object_text
from orion.schemas.cortex.contracts import CortexChatRequest, CortexChatResult

from scripts.cortex_chat_display import hub_effective_chat_text
from scripts.cortex_request_builder import (
    HubRequestValidationError,
    build_chat_request,
    build_continuity_messages,
    validate_single_verb_override,
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
    out = dict(options or {})
    if route == "quick":
        out["llm_route"] = "quick"
    else:
        out.pop("llm_route", None)
    return out


def _short_exc(exc: BaseException, *, limit: int = 240) -> str:
    s = f"{type(exc).__name__}: {exc}"
    return s if len(s) <= limit else s[: limit - 3] + "..."


def _parse_suggest_draft_from_text(text: str) -> Tuple[Optional[SuggestDraftV1], Optional[str], Optional[str]]:
    """Return (draft, error_code, error_summary)."""
    raw = (text or "").strip()
    if not raw:
        return None, "empty_output", "empty model output"
    blob = extract_first_json_object_text(raw)
    if not blob:
        return None, "no_json_object", "no JSON object found in model output"
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as e:
        return None, "invalid_json", _short_exc(e, limit=200)
    if not isinstance(data, dict):
        return None, "invalid_json", "parsed JSON is not an object"
    try:
        return SuggestDraftV1.model_validate(data), None, None
    except ValidationError as e:
        return None, "pydantic_validation", _short_exc(e, limit=400)


async def _call_cortex(
    cortex_client: Any,
    req: CortexChatRequest,
    *,
    timeout_sec: float,
) -> Tuple[Optional[CortexChatResult], Optional[str]]:
    corr = str(uuid4())
    try:
        resp = await asyncio.wait_for(
            cortex_client.chat(req, correlation_id=corr),
            timeout=timeout_sec,
        )
        return resp, None
    except TimeoutError:
        return None, "hub_wait_for_timeout"
    except Exception as exc:  # noqa: BLE001 — surface as attempt error
        logger.warning("memory_graph_suggest_cortex_error corr=%s error=%s", corr, exc)
        return None, _short_exc(exc, limit=400)


def _cortex_text_and_status(resp: CortexChatResult) -> Tuple[str, bool, str]:
    """effective text, whether we treat cortex as usable, status label."""
    cr = resp.cortex_result
    text = hub_effective_chat_text(resp)
    ok_flag = bool(getattr(cr, "ok", False))
    status = str(getattr(cr, "status", "") or "")
    if not ok_flag:
        return text, False, status or "not_ok"
    if not str(text).strip():
        return text, False, status or "empty_final_text"
    return text, True, status or "ok"


async def run_memory_graph_suggest_with_fallback(
    *,
    cortex_client: Any,
    payload: Dict[str, Any],
    session_id: str,
    user_id: Optional[str],
    settings: Any,
    mutation_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Try primary/fallback cortex routes; parse Appendix C JSON; RDF-validate without persisting.

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
            "suggest_attempts": [
                {
                    "route": None,
                    "index": 0,
                    "phase": "verb_inactive",
                    "error_summary": str(inactive.get("error") or "inactive_verb"),
                }
            ],
        }

    primary = _normalize_route(getattr(settings, "MEMORY_GRAPH_SUGGEST_PRIMARY_ROUTE", None), "brain")
    fallback = _normalize_route(getattr(settings, "MEMORY_GRAPH_SUGGEST_FALLBACK_ROUTE", None), "quick")
    enable_fb = bool(getattr(settings, "MEMORY_GRAPH_SUGGEST_ENABLE_FALLBACK", True))
    t_brain = float(getattr(settings, "MEMORY_GRAPH_SUGGEST_BRAIN_TIMEOUT_SEC", 180.0))
    t_quick = float(getattr(settings, "MEMORY_GRAPH_SUGGEST_QUICK_TIMEOUT_SEC", 120.0))

    def timeout_for(route: RouteName) -> float:
        return t_brain if route == "brain" else t_quick

    user_messages = payload.get("messages") or []
    if not isinstance(user_messages, list) or not user_messages:
        return {
            "ok": False,
            "error": "missing_messages",
            "suggest_attempts": [
                {
                    "route": None,
                    "index": 0,
                    "phase": "request",
                    "error_summary": "messages[] missing or empty",
                }
            ],
        }
    user_prompt = str(user_messages[-1].get("content", "") or "")

    context_turns = int(payload.get("context_turns") or getattr(settings, "HUB_CONTEXT_TURNS", 10))
    continuity = build_continuity_messages(
        history=user_messages,
        latest_user_prompt=user_prompt,
        turns=context_turns,
    )
    routed = dict(payload)
    routed["no_write"] = True
    routed["verbs"] = ["memory_graph_suggest"]
    mc = mutation_context if mutation_context is not None else build_mutation_cognition_context()
    routed["mutation_cognition_context"] = mc

    diagnostic = bool(payload.get("diagnostic")) or bool(
        isinstance(payload.get("options"), dict) and (payload.get("options") or {}).get("diagnostic")
    )

    attempts_meta: List[Dict[str, Any]] = []
    routes_to_try: List[Tuple[RouteName, float]] = [(primary, timeout_for(primary))]
    if enable_fb and fallback != primary:
        routes_to_try.append((fallback, timeout_for(fallback)))

    last_violations: List[str] = []
    last_draft: Optional[SuggestDraftV1] = None
    last_route: Optional[RouteName] = None
    last_preview: Optional[Dict[str, Any]] = None

    for idx, (route, timeout_sec) in enumerate(routes_to_try):
        attempt: Dict[str, Any] = {
            "route": route,
            "timeout_sec": timeout_sec,
            "index": idx,
        }
        try:
            req, _route_debug, _ = build_chat_request(
                payload=routed,
                session_id=session_id,
                user_id=user_id,
                trace_id=None,
                default_mode="brain",
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
                "suggest_attempts": [
                    {
                        "route": route,
                        "index": idx,
                        "phase": "build_request",
                        "error_summary": str(exc),
                    }
                ],
            }

        opts = _apply_llm_route(dict(req.options or {}), route)
        req = req.model_copy(update={"options": opts})

        resp, cortex_err = await _call_cortex(cortex_client, req, timeout_sec=timeout_sec)
        if resp is None:
            attempt["phase"] = "cortex"
            attempt["error_summary"] = cortex_err or "cortex_unavailable"
            attempt["raw_length"] = 0
            attempts_meta.append(attempt)
            continue

        text, cortex_ok, st = _cortex_text_and_status(resp)
        attempt["cortex_ok"] = cortex_ok
        attempt["cortex_status"] = st
        attempt["raw_length"] = len(text or "")
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
            attempts_meta.append(attempt)
            continue

        draft, err_code, err_sum = _parse_suggest_draft_from_text(text)
        if draft is None:
            attempt["phase"] = err_code or "parse"
            attempt["error_summary"] = err_sum or "parse_failed"
            attempts_meta.append(attempt)
            continue

        ok_graph, violations, preview = preview_validate_only(draft)
        last_draft = draft
        last_route = route
        last_preview = preview
        last_violations = violations

        if not ok_graph:
            attempt["phase"] = "rdf_validate"
            attempt["error_summary"] = "; ".join(violations[:6]) + ("…" if len(violations) > 6 else "")
            attempt["violations"] = violations
            attempts_meta.append(attempt)
            out: Dict[str, Any] = {
                "ok": False,
                "violations": violations,
                "preview": preview,
                "draft": draft.model_dump(mode="json", by_alias=True),
                "appendix_c_json": draft.model_dump_json(by_alias=True),
                "suggest_route_used": route,
                "suggest_attempts": attempts_meta,
            }
            if diagnostic:
                out["diagnostic_raw"] = text
            return out

        attempt["phase"] = "success"
        attempt["error_summary"] = None
        attempts_meta.append(attempt)
        out_ok: Dict[str, Any] = {
            "ok": True,
            "violations": [],
            "preview": preview,
            "draft": draft.model_dump(mode="json", by_alias=True),
            "appendix_c_json": draft.model_dump_json(by_alias=True),
            "suggest_route_used": route,
            "suggest_attempts": attempts_meta,
        }
        if diagnostic:
            out_ok["diagnostic_raw"] = text
        return out_ok

    # All attempts failed at cortex/parse level (never got a pydantic-valid draft, or only violations handled above)
    return {
        "ok": False,
        "error": "memory_graph_suggest_exhausted",
        "violations": last_violations,
        "preview": last_preview,
        "draft": last_draft.model_dump(mode="json", by_alias=True) if last_draft else None,
        "appendix_c_json": last_draft.model_dump_json(by_alias=True) if last_draft else None,
        "suggest_route_used": None,
        "suggest_attempts": attempts_meta,
    }
