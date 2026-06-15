"""Route-bound read-only Agent synthesis pass for context-exec."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from orion.core.bus.async_service import OrionBusAsync
from orion.schemas.context_exec import (
    ContextExecMode,
    ContextExecOperatorSummaryV1,
    ContextExecRequestV1,
    ContextExecSafetySummaryV1,
)

from .llm_profile_resolver import LLMProfileSelection
from .llm_tools import llm_chat_route
from .settings import ContextExecSettings

logger = logging.getLogger("orion-context-exec.agent_synthesis")

SYNTHESIS_MODES: frozenset[str] = frozenset(
    {
        "belief_provenance",
        "trace_autopsy",
        "repo_impact_analysis",
        "patch_proposal",
        "memory_correction_proposal",
    }
)

_PATH_LIKE = re.compile(
    r"(?:/[\w./_-]+|[\w.-]+\.(?:py|js|ts|tsx|json|yaml|yml|md|sh|toml|ini|env)(?:\b|$))"
)
_MEMORY_ID = re.compile(r"(?:memory:|mem_|card_)[\w:-]+", re.I)


@dataclass(frozen=True)
class AgentSynthesisResult:
    operator_summary: ContextExecOperatorSummaryV1 | None
    model_synthesis_used: bool
    fallback_used: bool
    fallback_reason: str | None
    synthesis_summary: str | None = None


def _settings(cfg: ContextExecSettings | None = None) -> ContextExecSettings:
    if cfg is not None:
        return cfg
    from .settings import settings as live

    return live


def _flatten_strings(value: Any, out: list[str]) -> None:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            out.append(stripped)
    elif isinstance(value, dict):
        for item in value.values():
            _flatten_strings(item, out)
    elif isinstance(value, list):
        for item in value:
            _flatten_strings(item, out)


def _ground_corpus(artifact: dict[str, Any], request_text: str) -> str:
    parts: list[str] = [request_text]
    _flatten_strings(artifact, parts)
    return "\n".join(parts).lower()


def _extract_path_tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in _PATH_LIKE.finditer(text)}


def _extract_memory_id_tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in _MEMORY_ID.finditer(text)}


def _synthesis_is_grounded(summary: str, corpus: str) -> bool:
    summary_lower = summary.lower()
    for token in _extract_path_tokens(summary):
        if token not in corpus and token.rsplit("/", 1)[-1] not in corpus:
            return False
    for token in _extract_memory_id_tokens(summary):
        if token not in corpus:
            return False
    return True


def _permissions_summary(request: ContextExecRequestV1) -> dict[str, bool]:
    p = request.permissions
    return {
        "read_memory": p.read_memory,
        "read_recall": p.read_recall,
        "read_repo": p.read_repo,
        "read_redis_traces": p.read_redis_traces,
        "write_memory": p.write_memory,
        "write_repo": p.write_repo,
        "network_enabled": p.network_enabled,
        "shell_enabled": p.shell_enabled,
    }


def _default_title(mode: ContextExecMode) -> str:
    titles = {
        "belief_provenance": "Belief provenance complete",
        "trace_autopsy": "Trace autopsy complete",
        "repo_impact_analysis": "Repo impact analysis complete",
        "patch_proposal": "Patch proposal drafted",
        "memory_correction_proposal": "Memory correction proposal drafted",
    }
    return titles.get(mode, "Agent investigation complete")


def _deterministic_summary(mode: ContextExecMode, artifact: dict[str, Any]) -> str:
    if mode == "belief_provenance":
        return (
            f"Claim '{artifact.get('claim', 'unknown')}' status={artifact.get('status', 'unknown')}. "
            f"Likely origin: {artifact.get('likely_origin') or 'unknown'}."
        )
    if mode == "trace_autopsy":
        return (
            f"Target {artifact.get('target', 'unknown')}: "
            f"root cause {artifact.get('root_cause') or 'unknown'}."
        )
    if mode == "repo_impact_analysis":
        paths = artifact.get("affected_paths") or []
        return f"Impact status={artifact.get('status', 'unknown')}. Affected paths: {len(paths)}."
    if mode in {"patch_proposal", "memory_correction_proposal"}:
        return str(artifact.get("summary") or artifact.get("title") or "Proposal drafted for review.")
    return str(artifact.get("summary") or "Investigation complete.")


def _build_synthesis_prompt(
    request: ContextExecRequestV1,
    artifact: dict[str, Any],
    *,
    max_chars: int,
) -> str:
    payload = {
        "user_text": request.text[:800],
        "mode": request.mode,
        "permissions": _permissions_summary(request),
        "artifact": artifact,
        "safety_constraints": [
            "read-only synthesis",
            "no shell",
            "no file writes",
            "no memory writes",
            "no proposal execution",
            "do not invent evidence, file paths, memory ids, or trace causes",
        ],
        "output_contract": {
            "title": "short operator title",
            "summary": "grounded operator summary using only artifact facts",
        },
    }
    raw = json.dumps(payload, ensure_ascii=False)
    if len(raw) > max_chars:
        payload["artifact"] = {"truncated": True, "artifact_type": artifact.get("artifact_type")}
        raw = json.dumps(payload, ensure_ascii=False)
    return (
        "You are a read-only context-exec synthesis pass. Improve the operator-facing summary "
        "using ONLY facts present in artifact and user_text. Return JSON only with keys "
        "title and summary. Do not invent evidence.\n\n"
        f"{raw[:max_chars]}"
    )


def _parse_synthesis_json(content: str) -> dict[str, str] | None:
    text = (content or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(parsed, dict):
        return None
    title = str(parsed.get("title") or "").strip()
    summary = str(parsed.get("summary") or "").strip()
    if not summary:
        return None
    return {"title": title or "Agent run complete", "summary": summary}


def build_operator_summary(
    *,
    mode: ContextExecMode,
    route_used: str,
    artifact: dict[str, Any],
    runtime_debug: dict[str, Any],
    model_synthesis_used: bool,
    summary_text: str,
    title: str | None = None,
) -> ContextExecOperatorSummaryV1:
    proposal_id = artifact.get("proposal_id") or runtime_debug.get("proposal_id")
    proposal_status = (
        artifact.get("review_status")
        or runtime_debug.get("ledger_status")
        or runtime_debug.get("proposal_status")
    )
    triage_action = runtime_debug.get("triage_action")
    requires_human = bool(
        artifact.get("requires_human_approval", True)
        if artifact.get("proposal_type")
        else runtime_debug.get("requires_human_approval", True)
    )
    return ContextExecOperatorSummaryV1(
        title=title or _default_title(mode),
        summary=summary_text,
        agent_mode=mode,
        route_used=route_used,
        model_synthesis_used=model_synthesis_used,
        proposal_id=str(proposal_id) if proposal_id else None,
        proposal_status=str(proposal_status) if proposal_status else None,
        triage_action=str(triage_action) if triage_action else None,
        safety=ContextExecSafetySummaryV1(
            mutation_allowed=False,
            mutation_performed=False,
            requires_human_approval=requires_human,
        ),
    )


async def run_agent_synthesis(
    *,
    request: ContextExecRequestV1,
    artifact: dict[str, Any],
    profile_selection: LLMProfileSelection,
    runtime_debug: dict[str, Any],
    bus: OrionBusAsync | None,
    cfg: ContextExecSettings | None = None,
) -> AgentSynthesisResult:
    """Route-bound synthesis after deterministic organ collection."""
    cfg = _settings(cfg)
    route_used = profile_selection.route_used
    deterministic = _deterministic_summary(request.mode, artifact)

    if request.mode not in SYNTHESIS_MODES:
        summary = build_operator_summary(
            mode=request.mode,
            route_used=route_used,
            artifact=artifact,
            runtime_debug=runtime_debug,
            model_synthesis_used=False,
            summary_text=deterministic,
        )
        return AgentSynthesisResult(
            operator_summary=summary,
            model_synthesis_used=False,
            fallback_used=False,
            fallback_reason=None,
        )

    if not cfg.context_exec_agent_synthesis_enabled:
        summary = build_operator_summary(
            mode=request.mode,
            route_used=route_used,
            artifact=artifact,
            runtime_debug=runtime_debug,
            model_synthesis_used=False,
            summary_text=deterministic,
        )
        return AgentSynthesisResult(
            operator_summary=summary,
            model_synthesis_used=False,
            fallback_used=False,
            fallback_reason=None,
        )

    prompt = _build_synthesis_prompt(
        request,
        artifact,
        max_chars=int(cfg.context_exec_agent_synthesis_max_chars),
    )
    llm_result = await llm_chat_route(
        bus,
        prompt=prompt,
        route=route_used,
        correlation_id=request.correlation_id,
        session_id=request.session_id,
        user_id=request.user_id,
        schema="agent_synthesis_v1",
    )

    if not llm_result.get("ok"):
        reason = f"synthesis unavailable: {llm_result.get('error') or 'llm_failed'}"
        if cfg.context_exec_agent_synthesis_required:
            summary = build_operator_summary(
                mode=request.mode,
                route_used=route_used,
                artifact=artifact,
                runtime_debug=runtime_debug,
                model_synthesis_used=False,
                summary_text=deterministic,
            )
            return AgentSynthesisResult(
                operator_summary=summary,
                model_synthesis_used=False,
                fallback_used=True,
                fallback_reason=reason,
            )
        summary = build_operator_summary(
            mode=request.mode,
            route_used=route_used,
            artifact=artifact,
            runtime_debug=runtime_debug,
            model_synthesis_used=False,
            summary_text=deterministic,
        )
        return AgentSynthesisResult(
            operator_summary=summary,
            model_synthesis_used=False,
            fallback_used=True,
            fallback_reason=reason,
        )

    parsed = _parse_synthesis_json(str(llm_result.get("content") or ""))
    if parsed is None:
        reason = "synthesis rejected: invalid_json"
        summary = build_operator_summary(
            mode=request.mode,
            route_used=route_used,
            artifact=artifact,
            runtime_debug=runtime_debug,
            model_synthesis_used=False,
            summary_text=deterministic,
        )
        return AgentSynthesisResult(
            operator_summary=summary,
            model_synthesis_used=False,
            fallback_used=True,
            fallback_reason=reason,
        )

    corpus = _ground_corpus(artifact, request.text)
    if not _synthesis_is_grounded(parsed["summary"], corpus):
        reason = "synthesis rejected: ungrounded"
        summary = build_operator_summary(
            mode=request.mode,
            route_used=route_used,
            artifact=artifact,
            runtime_debug=runtime_debug,
            model_synthesis_used=False,
            summary_text=deterministic,
        )
        return AgentSynthesisResult(
            operator_summary=summary,
            model_synthesis_used=False,
            fallback_used=True,
            fallback_reason=reason,
        )

    summary = build_operator_summary(
        mode=request.mode,
        route_used=route_used,
        artifact=artifact,
        runtime_debug=runtime_debug,
        model_synthesis_used=True,
        summary_text=parsed["summary"],
        title=parsed.get("title"),
    )
    return AgentSynthesisResult(
        operator_summary=summary,
        model_synthesis_used=True,
        fallback_used=False,
        fallback_reason=None,
        synthesis_summary=parsed["summary"],
    )
