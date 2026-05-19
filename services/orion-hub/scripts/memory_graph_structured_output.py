"""Structured-output options for memory_graph_suggest (llama.cpp probe-selected method)."""

from __future__ import annotations

import os
from typing import Any, Dict

from orion.memory_graph.schema_contract import compact_suggest_draft_json_schema


def resolve_memory_graph_structured_output_method(settings: Any) -> str:
    raw = (
        str(getattr(settings, "MEMORY_GRAPH_SUGGEST_STRUCTURED_OUTPUT_METHOD", "") or "").strip()
        or os.getenv("MEMORY_GRAPH_SUGGEST_STRUCTURED_OUTPUT_METHOD", "").strip()
        or os.getenv("LLM_STRUCTURED_OUTPUT_METHOD", "").strip()
        or "none"
    )
    if raw == "auto":
        return (
            os.getenv("MEMORY_GRAPH_SUGGEST_STRUCTURED_OUTPUT_METHOD", "").strip()
            or os.getenv("LLM_STRUCTURED_OUTPUT_METHOD", "").strip()
            or "none"
        )
    return raw or "none"


def memory_graph_suggest_llm_options(settings: Any, *, diagnostic: bool) -> Dict[str, Any]:
    """Options merged into CortexChatRequest for structured SuggestDraftV1 extraction."""
    method = resolve_memory_graph_structured_output_method(settings)
    max_tokens = max(
        1600,
        int(getattr(settings, "MEMORY_GRAPH_SUGGEST_MAX_TOKENS", 0) or 0),
    )
    opts: Dict[str, Any] = {
        "structured_output_schema_name": "SuggestDraftV1",
        "structured_output_schema": compact_suggest_draft_json_schema(),
        "structured_output_method": method,
        "structured_output_thinking_policy": "disabled_for_artifact",
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }
    if diagnostic:
        opts["diagnostic"] = True
    return opts


def extract_gateway_structured_diagnostics(resp: Any) -> Dict[str, Any]:
    """Pull structured_output_diagnostics from cortex LLM gateway step raw payload."""
    out: Dict[str, Any] = {
        "structured_output_requested": False,
        "structured_output_method_requested": None,
        "structured_output_method_effective": None,
        "schema_name": None,
        "thinking_policy": None,
        "gateway_response_format_shape": None,
        "thinking_disabled_requested": None,
        "inline_think_extracted": False,
        "reasoning_content_present": False,
    }
    cr = getattr(resp, "cortex_result", None)
    if cr is None:
        return out
    for step in getattr(cr, "steps", None) or []:
        detail = getattr(step, "detail", None) or {}
        if not isinstance(detail, dict):
            continue
        raw = detail.get("raw") if isinstance(detail.get("raw"), dict) else {}
        diag = raw.get("structured_output_diagnostics")
        if not isinstance(diag, dict):
            meta = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
            diag = meta.get("structured_output_diagnostics") if isinstance(meta, dict) else None
        if isinstance(diag, dict) and diag:
            out["structured_output_requested"] = bool(diag.get("structured_output_requested"))
            out["structured_output_method_effective"] = diag.get("structured_output_method")
            out["schema_name"] = diag.get("structured_output_schema_name")
            out["thinking_policy"] = diag.get("thinking_policy")
            out["gateway_response_format_shape"] = diag.get("response_format_shape")
            out["thinking_disabled_requested"] = diag.get("thinking_disabled_requested")
            break
        if str(getattr(step, "service", "") or "").lower() == "llmgatewayservice":
            content = detail.get("content") or detail.get("text") or ""
            if isinstance(content, str) and (
                "<think>" in content or "</think>" in content
            ):
                out["inline_think_extracted"] = True
    meta = getattr(cr, "metadata", None) or {}
    if isinstance(meta, dict):
        out["reasoning_content_present"] = bool(
            meta.get("provider_reasoning_available") or meta.get("reasoning_content")
        )
        out["inline_think_extracted"] = out["inline_think_extracted"] or bool(
            meta.get("inline_think_extracted")
        )
    return out
