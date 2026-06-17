"""Mandatory finalize pass: FindingsBundle + AnswerContract → user-visible RenderedAnswer text."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orion.cognition
from jinja2 import Environment, FileSystemLoader, select_autoescape

from orion.cognition.finalize_payload import build_finalize_tool_input
from orion.cognition.output_mode_classifier import classify_output_mode
from orion.schemas.cognition.answer_contract import AnswerContract, RenderedAnswer
from orion.schemas.context_exec import ContextExecRequestV1

from .llm_tools import llm_chat_route
from .settings import ContextExecSettings

logger = logging.getLogger("orion-context-exec.finalize_pass")

_COGNITION_ROOT = Path(orion.cognition.__file__).resolve().parent
_PROMPTS_DIR = _COGNITION_ROOT / "prompts"


@dataclass(frozen=True)
class FinalizePassResult:
    rendered: RenderedAnswer | None
    model_finalize_used: bool
    fallback_used: bool
    fallback_reason: str | None
    text: str


def _settings(cfg: ContextExecSettings | None = None) -> ContextExecSettings:
    if cfg is not None:
        return cfg
    from .settings import settings as live

    return live


def _jinja_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_PROMPTS_DIR)),
        autoescape=select_autoescape(disabled_extensions=("j2",)),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _contract_dict(request: ContextExecRequestV1) -> dict[str, Any]:
    if request.answer_contract is not None:
        return request.answer_contract.model_dump(mode="json")
    from orion.cognition.answer_contract_normalize import heuristic_answer_contract

    return heuristic_answer_contract(request.text or "").model_dump(mode="json")


def _findings_bundle_dict(
    request: ContextExecRequestV1,
    artifact: dict[str, Any],
    findings_bundle: Any,
) -> dict[str, Any] | None:
    if findings_bundle is not None and hasattr(findings_bundle, "model_dump"):
        return findings_bundle.model_dump(mode="json")
    from .artifact_builder import synthesize_findings_bundle

    fb = synthesize_findings_bundle(request, artifact, schema_valid=True)
    return fb.model_dump(mode="json") if fb is not None else None


def _render_finalize_prompt(
    *,
    request: ContextExecRequestV1,
    artifact: dict[str, Any],
    findings_bundle: Any,
    operator_report_text: str,
) -> str:
    ac = _contract_dict(request)
    fb = _findings_bundle_dict(request, artifact, findings_bundle)
    omd = classify_output_mode(request.text or "")
    tool_input = build_finalize_tool_input(
        user_text=request.text or "",
        trace_snapshot=[
            {
                "tool_id": "investigation_v2",
                "summary": operator_report_text[:4000],
                "artifact_answer_status": artifact.get("answer_status"),
            }
        ],
        output_mode=omd.output_mode,
        response_profile=omd.response_profile,
        findings_bundle=fb,
        answer_contract=ac,
    )
    env = _jinja_env()
    template_name = (
        "render_from_findings_finalize_response.j2"
        if fb and ac.get("request_kind") in ("repo_technical", "runtime_debug", "mixed")
        else "finalize_response_prompt.j2"
    )
    template = env.get_template(template_name)
    return template.render(**tool_input)


def _fallback_text(
    *,
    request: ContextExecRequestV1,
    artifact: dict[str, Any],
    reason: str,
) -> str:
    ac = request.answer_contract
    kind = ac.request_kind if ac else "conceptual"
    if kind in ("personal", "conceptual"):
        return (
            "I hear you — that last reply didn't land well. "
            "Tell me what felt shallow (missing depth, wrong tone, or off-topic) "
            "and I'll answer directly."
        )
    summary = str(artifact.get("summary") or "").strip()
    if summary:
        return summary
    return f"Investigation complete ({reason})."


async def run_finalize_pass(
    *,
    request: ContextExecRequestV1,
    artifact: dict[str, Any],
    findings_bundle: Any,
    operator_report_text: str,
    bus: Any,
    route_used: str,
    cfg: ContextExecSettings | None = None,
) -> FinalizePassResult:
    """Produce user-visible discourse; operator report stays in artifact sidecar."""
    cfg = _settings(cfg)
    if not cfg.context_exec_finalize_enabled:
        text = _fallback_text(request=request, artifact=artifact, reason="finalize_disabled")
        return FinalizePassResult(
            rendered=None,
            model_finalize_used=False,
            fallback_used=True,
            fallback_reason="finalize_disabled",
            text=text,
        )

    prompt = _render_finalize_prompt(
        request=request,
        artifact=artifact,
        findings_bundle=findings_bundle,
        operator_report_text=operator_report_text,
    )
    llm_result = await llm_chat_route(
        bus,
        prompt=prompt,
        route=route_used,
        correlation_id=request.correlation_id,
        session_id=request.session_id,
        user_id=request.user_id,
        schema="context_exec_finalize_v1",
    )
    if not llm_result.get("ok"):
        reason = f"finalize unavailable: {llm_result.get('error') or 'llm_failed'}"
        logger.warning("finalize_pass failed: %s", reason)
        text = _fallback_text(request=request, artifact=artifact, reason=reason)
        return FinalizePassResult(
            rendered=None,
            model_finalize_used=False,
            fallback_used=True,
            fallback_reason=reason,
            text=text,
        )

    content = str(llm_result.get("content") or "").strip()
    if not content:
        reason = "finalize empty content"
        text = _fallback_text(request=request, artifact=artifact, reason=reason)
        return FinalizePassResult(
            rendered=None,
            model_finalize_used=False,
            fallback_used=True,
            fallback_reason=reason,
            text=text,
        )

    grounded = str(artifact.get("answer_status") or "grounded_partial")
    if grounded == "answered_grounded":
        status = "grounded_complete"
    elif grounded in ("partial_grounding", "dependency_unavailable"):
        status = "grounded_partial"
    else:
        status = "insufficient_grounding"

    rendered = RenderedAnswer(
        text=content,
        grounded_status=status,
    )
    return FinalizePassResult(
        rendered=rendered,
        model_finalize_used=True,
        fallback_used=False,
        fallback_reason=None,
        text=content,
    )
