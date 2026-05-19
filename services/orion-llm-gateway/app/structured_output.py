"""Build llama.cpp / OpenAI-compatible response_format payloads from a named method."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

METHOD_NONE = "none"
METHOD_JSON_OBJECT_ONLY = "json_object_only"
METHOD_JSON_OBJECT_SCHEMA = "json_object_schema"
METHOD_JSON_SCHEMA_SCHEMA = "json_schema_schema"
METHOD_JSON_SCHEMA_NESTED = "json_schema_nested_schema"
METHOD_OPENAI_JSON_SCHEMA_WRAPPER = "openai_json_schema_wrapper"

STRUCTURED_OUTPUT_METHODS = frozenset(
    {
        METHOD_NONE,
        METHOD_JSON_OBJECT_ONLY,
        METHOD_JSON_OBJECT_SCHEMA,
        METHOD_JSON_SCHEMA_SCHEMA,
        METHOD_JSON_SCHEMA_NESTED,
        METHOD_OPENAI_JSON_SCHEMA_WRAPPER,
    }
)

SCHEMA_METHODS = frozenset(
    {
        METHOD_JSON_OBJECT_SCHEMA,
        METHOD_JSON_SCHEMA_SCHEMA,
        METHOD_JSON_SCHEMA_NESTED,
        METHOD_OPENAI_JSON_SCHEMA_WRAPPER,
    }
)


def resolve_structured_output_method(
    options: Optional[Dict[str, Any]],
    *,
    env_default: Optional[str] = None,
) -> str:
    """Pick method: options.structured_output_method → env → none."""
    opts = options if isinstance(options, dict) else {}
    raw = opts.get("structured_output_method")
    if raw is not None and str(raw).strip():
        method = str(raw).strip()
        if method == "auto":
            method = (
                os.getenv("MEMORY_GRAPH_SUGGEST_STRUCTURED_OUTPUT_METHOD", "").strip()
                or os.getenv("LLM_STRUCTURED_OUTPUT_METHOD", "").strip()
                or METHOD_NONE
            )
        return method if method in STRUCTURED_OUTPUT_METHODS else METHOD_NONE
    env_val = (env_default or os.getenv("LLM_STRUCTURED_OUTPUT_METHOD", "") or "").strip()
    if env_val in STRUCTURED_OUTPUT_METHODS:
        return env_val
    return METHOD_NONE


def build_response_format(
    method: str,
    schema: Dict[str, Any],
    *,
    schema_name: str = "StructuredOutput",
) -> Optional[Dict[str, Any]]:
    """Return response_format dict for the given method, or None for none/unknown."""
    if method == METHOD_NONE or not method:
        return None
    if method == METHOD_JSON_OBJECT_ONLY:
        return {"type": "json_object"}
    if method == METHOD_JSON_OBJECT_SCHEMA:
        return {"type": "json_object", "schema": schema}
    if method == METHOD_JSON_SCHEMA_SCHEMA:
        return {"type": "json_schema", "schema": schema}
    if method == METHOD_JSON_SCHEMA_NESTED:
        return {"type": "json_schema", "json_schema": {"schema": schema}}
    if method == METHOD_OPENAI_JSON_SCHEMA_WRAPPER:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema,
            },
        }
    return None


def response_format_shape_label(response_format: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(response_format, dict):
        return None
    rf_type = str(response_format.get("type") or "")
    if rf_type == "json_object":
        return "json_object_schema" if "schema" in response_format else "json_object_only"
    if rf_type == "json_schema":
        inner = response_format.get("json_schema")
        if isinstance(inner, dict):
            if "strict" in inner and "name" in inner:
                return "openai_json_schema_wrapper"
            if "schema" in inner:
                return "json_schema_nested_schema"
        if "schema" in response_format:
            return "json_schema_schema"
    return rf_type or "unknown"


def apply_structured_output_to_payload(
    opts: Dict[str, Any],
    *,
    backend_name: str,
    env_default: Optional[str] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Mutate opts in place for structured output + thinking policy.
    Returns (opts, diagnostics).
    """
    diagnostics: Dict[str, Any] = {
        "structured_output_requested": False,
        "structured_output_method": None,
        "structured_output_schema_name": None,
        "response_format_shape": None,
        "thinking_policy": None,
        "chat_template_kwargs_sent": None,
        "backend_name": backend_name,
    }

    schema = opts.get("structured_output_schema")
    if not isinstance(schema, dict) or not schema:
        if isinstance(opts.get("response_format"), dict):
            diagnostics["response_format_shape"] = response_format_shape_label(opts.get("response_format"))
        return opts, diagnostics

    diagnostics["structured_output_requested"] = True
    schema_name = str(opts.get("structured_output_schema_name") or "StructuredOutput")
    diagnostics["structured_output_schema_name"] = schema_name

    method = resolve_structured_output_method(opts, env_default=env_default)
    diagnostics["structured_output_method"] = method

    explicit_rf = opts.get("response_format") if isinstance(opts.get("response_format"), dict) else None
    built_rf = build_response_format(method, schema, schema_name=schema_name)
    if built_rf is not None:
        opts["response_format"] = built_rf
    elif explicit_rf is None:
        opts.pop("response_format", None)

    diagnostics["response_format_shape"] = response_format_shape_label(opts.get("response_format"))

    thinking_policy = str(opts.get("structured_output_thinking_policy") or "").strip()
    diagnostics["thinking_policy"] = thinking_policy or None

    if (
        thinking_policy == "disabled_for_artifact"
        and backend_name in ("llamacpp", "llama-cola")
    ):
        ctk = opts.get("chat_template_kwargs")
        if not isinstance(ctk, dict):
            ctk = {}
        else:
            ctk = dict(ctk)
        if ctk.get("enable_thinking") is not False:
            ctk["enable_thinking"] = False
            opts["chat_template_kwargs"] = ctk
        diagnostics["thinking_disabled_requested"] = True

    if isinstance(opts.get("chat_template_kwargs"), dict):
        diagnostics["chat_template_kwargs_sent"] = dict(opts["chat_template_kwargs"])

    required_keys = list(schema.get("required") or [])[:12]
    diagnostics["structured_output_schema_required_keys"] = required_keys

    return opts, diagnostics
