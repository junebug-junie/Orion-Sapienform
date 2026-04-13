"""Standard payload for finalize_response tool (shared, testable)."""

from __future__ import annotations

import json
from typing import Any

from .delivery_grounding import build_delivery_grounding_context, extract_trace_preferred_output


def answer_contract_expects_findings_rendering(answer_contract: dict[str, Any] | None) -> bool:
    """
    Findings-only prompts are for evidence acquisition shapes.
    Personal / conceptual coaching must not receive an empty synthetic bundle (it steers models to refuse help).
    """
    if not isinstance(answer_contract, dict):
        return False
    kind = answer_contract.get("request_kind")
    if kind == "personal":
        return False
    if kind in ("repo_technical", "runtime_debug", "mixed"):
        return True
    return bool(
        answer_contract.get("requires_repo_grounding") or answer_contract.get("requires_runtime_grounding")
    )


def build_finalize_tool_input(
    *,
    user_text: str,
    trace_snapshot: list,
    output_mode: str | None,
    response_profile: str | None,
    findings_bundle: dict[str, Any] | None = None,
    answer_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    grounding = build_delivery_grounding_context(user_text=user_text, output_mode=output_mode)
    trace_preferred_output, trace_used = extract_trace_preferred_output(trace_snapshot)
    out: dict[str, Any] = {
        "original_request": user_text,
        "request": user_text,
        "text": user_text,
        "trace": json.dumps([dict(s) for s in trace_snapshot], default=str)[:12000],
        "prior_trace": str(trace_snapshot),
        "output_mode": output_mode or "direct_answer",
        "response_profile": response_profile or "direct_answer",
        "trace_preferred_output": trace_preferred_output,
        "finalization_source_trace_used": trace_used,
        **grounding,
    }
    if isinstance(findings_bundle, dict) and answer_contract_expects_findings_rendering(answer_contract):
        out["findings_bundle"] = findings_bundle
        out["findings_bundle_json"] = json.dumps(findings_bundle, ensure_ascii=False, default=str)[:12000]
    if isinstance(answer_contract, dict):
        out["answer_contract"] = answer_contract
        out["preferred_render_style"] = answer_contract.get("preferred_render_style") or "answer"
    return out
