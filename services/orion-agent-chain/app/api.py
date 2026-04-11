# services/orion-agent-chain/app/api.py
from __future__ import annotations

import json
import logging
import os
import asyncio
import uuid
from time import perf_counter
from pathlib import Path
from typing import Any, List

from fastapi import APIRouter, HTTPException
from orion.core.bus.async_service import OrionBusAsync

from .planner_rpc import call_planner_react
from .settings import settings
from .tool_executor import ToolExecutor
from .tool_registry import ToolRegistry
from orion.schemas.agents.schemas import AgentChainRequest, AgentChainResult, ToolDef
from orion.schemas.agents.bound_capability import (
    BoundCapabilityExecutionFailureV1,
    BoundCapabilityExecutionRequestV1,
    BoundCapabilityExecutionResultV1,
    CapabilityRecoveryDecisionV1,
    CapabilityRecoveryReasonV1,
)
from orion.cognition.output_mode_classifier import classify_output_mode
from orion.cognition.quality_evaluator import detect_generic_delivery_drift, should_rewrite_for_instructional
from orion.cognition.runtime_pack_merge import ensure_delivery_pack_in_packs
from orion.cognition.agent_chain_guards import (
    consecutive_tool_count,
    plan_action_saturated,
    triage_must_finalize,
)
from orion.cognition.finalize_payload import build_finalize_tool_input
from orion.cognition.delivery_grounding import build_delivery_grounding_context

logger = logging.getLogger("agent-chain.api")


def _cognition_base() -> Path:
    p = os.environ.get("COGNITION_BASE_DIR") or settings.cognition_base_dir
    return Path(p)


TOOL_REGISTRY = ToolRegistry(base_dir=_cognition_base())
router = APIRouter()


async def _maybe_rewrite_meta_plan(
    text: str,
    body: AgentChainRequest,
    tool_executor: ToolExecutor,
    parent_corr_id: str | None,
    logger_inst: "logging.Logger",
    output_mode: str | None,
    response_profile: str | None = None,
    trace_snapshot: list | None = None,
) -> tuple[str, bool]:
    """If text looks like meta-plan for instructional mode, rewrite via finalize_response."""
    grounding = build_delivery_grounding_context(user_text=body.text, output_mode=output_mode)
    should_rewrite, rewrite_reason = should_rewrite_for_instructional(
        text,
        output_mode,
        request_text=body.text,
        grounding_mode=grounding.get("delivery_grounding_mode"),
    )
    if not should_rewrite:
        return text, False
    logger_inst.info(
        "[agent-chain] quality_evaluator_rewrite=1 output_mode=%s reason=%s grounding=%s",
        output_mode,
        rewrite_reason,
        grounding.get("delivery_grounding_mode"),
    )
    try:
        fin_result = await tool_executor.execute_llm_verb(
            "finalize_response",
            {
                "original_request": body.text,
                "request": f"Rewrite the following as a concrete, architecture-grounded answer. Do not use meta-planning language or silently swap architectures:\n\n{text[:4000]}",
                "trace": text[:8000],
                "output_mode": output_mode or "direct_answer",
                "response_profile": response_profile or "direct_answer",
                "trace_preferred_output": text[:8000],
                **grounding,
            },
            parent_correlation_id=parent_corr_id,
        )
        return str(fin_result.get("llm_output") or text), True
    except Exception as e:
        logger_inst.warning("[agent-chain] quality rewrite failed: %s", e)
        return text, False


def _effective_output_modes(body: AgentChainRequest) -> tuple[str | None, str | None]:
    om = getattr(body, "output_mode", None) or None
    rp = getattr(body, "response_profile", None) or None
    if not om or not rp:
        omd = classify_output_mode(body.text or "")
        om = om or omd.output_mode
        rp = rp or omd.response_profile
    return om, rp


def _resolve_tools(body: AgentChainRequest, *, output_mode: str | None) -> tuple[List[ToolDef], list[str]]:
    if body.tools:
        return [ToolDef(**t) for t in body.tools], []
    pack_names = ensure_delivery_pack_in_packs(
        body.packs,
        output_mode=output_mode,
        user_text=body.text or "",
    )
    local_tools = TOOL_REGISTRY.tools_for_packs(pack_names)
    return (
        [ToolDef(**(t.dict() if hasattr(t, "dict") else t)) for t in local_tools],
        pack_names,
    )


def _finalize_tool_input(
    body: AgentChainRequest,
    trace_snapshot: list,
    *,
    output_mode: str | None,
    response_profile: str | None,
) -> dict[str, Any]:
    return build_finalize_tool_input(
        user_text=body.text,
        trace_snapshot=trace_snapshot,
        output_mode=output_mode,
        response_profile=response_profile,
    )


def _ground_tool_input(
    *,
    tool_id: str,
    tool_input: dict[str, Any],
    body: AgentChainRequest,
    trace_snapshot: list,
    output_mode: str | None,
    response_profile: str | None,
) -> dict[str, Any]:
    grounded = dict(tool_input or {})
    if tool_id not in {
        "answer_direct",
        "finalize_response",
        "write_guide",
        "write_tutorial",
        "write_runbook",
        "write_recommendation",
        "compare_options",
        "synthesize_patterns",
        "generate_code_scaffold",
    }:
        return grounded
    grounding = build_delivery_grounding_context(user_text=body.text, output_mode=output_mode)
    grounded.setdefault("output_mode", output_mode or "direct_answer")
    grounded.setdefault("response_profile", response_profile or "direct_answer")
    grounded.setdefault("request", body.text)
    grounded.setdefault("text", body.text)
    grounded.setdefault("original_request", body.text)
    grounded.setdefault("trace", json.dumps([dict(s) for s in trace_snapshot], default=str)[:12000])
    grounded.update({k: v for k, v in grounding.items() if k not in grounded})
    finalize_payload = build_finalize_tool_input(
        user_text=body.text,
        trace_snapshot=trace_snapshot,
        output_mode=output_mode,
        response_profile=response_profile,
    )
    for key in ("trace_preferred_output", "finalization_source_trace_used"):
        if key in finalize_payload and key not in grounded:
            grounded[key] = finalize_payload[key]
    return grounded


def _delivery_override_for_plan_action_repeat(
    output_mode: str | None,
) -> str:
    if output_mode == "code_delivery":
        return "generate_code_scaffold"
    if output_mode in {"comparative_analysis", "decision_support"}:
        return "compare_options" if output_mode == "comparative_analysis" else "write_recommendation"
    return "write_guide"


def _select_non_triage_tool(tools: List[ToolDef], preferred_order: List[str] | None = None) -> str | None:
    preferred = preferred_order or ["plan_action", "evaluate", "analyze_text"]
    tool_ids = [t.tool_id for t in tools if t.tool_id != "triage"]
    for candidate in preferred:
        if candidate in tool_ids:
            return candidate
    return tool_ids[0] if tool_ids else None


def _best_effort_text(*, last_thought: str, last_observation: Any, reason: str) -> str:
    if isinstance(last_observation, dict):
        obs = last_observation.get("llm_output") or last_observation.get("text") or last_observation.get("content")
        if isinstance(obs, str) and obs.strip():
            return obs
    if isinstance(last_thought, str) and last_thought.strip():
        return last_thought
    return reason


def _usable_finalized_text(observation: Any) -> str:
    if isinstance(observation, dict):
        for key in ("llm_output", "text", "content"):
            value = observation.get(key)
            if isinstance(value, str) and value.strip():
                return value
    if isinstance(observation, str) and observation.strip():
        return observation
    return ""


def _coverage_threshold_met(*, tools_called: list[str], trace_snapshot: list[dict[str, Any]]) -> bool:
    seen = set(tools_called)
    if "triage" in seen:
        seen.add("triage_like")
    if "plan_action" in seen:
        seen.add("planning_like")
    if {"evaluate", "summarize_context", "analyze_text"} & seen:
        seen.add("analysis_like")
    if {"write_recommendation", "write_guide", "compare_options", "finalize_response"} & seen:
        seen.add("delivery_like")
    if len({"triage_like", "planning_like", "analysis_like", "delivery_like"} & seen) >= 3:
        return True
    return len(trace_snapshot) >= 4 and len(seen) >= 3


def _pick_finalization_tool(tool_ids: list[str]) -> str | None:
    preferred = [
        "finalize_response",
        "write_recommendation",
        "summarize_context",
        "evaluate",
        "write_guide",
    ]
    for tool_id in preferred:
        if tool_id in tool_ids:
            return tool_id
    return None


def _resolve_delegate_tool_id(
    requested_tool_id: str | None,
    *,
    available_tools: List[ToolDef],
    output_mode: str | None,
) -> tuple[str | None, str]:
    requested = str(requested_tool_id or "").strip()
    if not requested:
        return None, "missing"

    available_ids = [t.tool_id for t in available_tools]
    if requested in set(available_ids):
        return requested, "exact"

    alias_map = {
        "analyze_conversation": "analyze_text",
        "gather_info": "plan_action",
    }
    alias_target = alias_map.get(requested)
    if alias_target and alias_target in set(available_ids):
        return alias_target, "alias"

    lowered = requested.lower()
    for tool_id in available_ids:
        tid = str(tool_id or "").lower()
        if tid == lowered:
            return tool_id, "casefold"
        if lowered in tid or tid in lowered:
            return tool_id, "fuzzy"

    fallback = _select_non_triage_tool(
        available_tools,
        preferred_order=["analyze_text", "plan_action", "evaluate"],
    )
    if fallback:
        return fallback, "fallback_non_triage"
    return _pick_finalization_tool(available_ids) or _delivery_override_for_plan_action_repeat(output_mode), "fallback_finalize"


def _extract_bound_capability_contract(body: AgentChainRequest) -> BoundCapabilityExecutionRequestV1 | None:
    if body.bound_capability_execution is not None:
        return body.bound_capability_execution
    tools = body.tools if isinstance(body.tools, list) else []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        candidate = tool.get("bound_execution")
        if isinstance(candidate, dict):
            logger.warning("[agent-chain] selected_verb_lost_contract_violation=1 legacy_bound_execution_field=tools[0].bound_execution")
            return BoundCapabilityExecutionRequestV1.model_validate(candidate)
    return None


def _replan_allowed_for_bound_recovery(
    contract: BoundCapabilityExecutionRequestV1,
    reason: CapabilityRecoveryReasonV1,
) -> bool:
    recovery = contract.recovery
    if recovery is None:
        return False
    if not bool(recovery.allow_replan):
        return False
    allowed_reasons: set[CapabilityRecoveryReasonV1] = {
        CapabilityRecoveryReasonV1.selected_verb_missing,
        CapabilityRecoveryReasonV1.invalid_bound_input,
        CapabilityRecoveryReasonV1.no_compatible_capability,
        CapabilityRecoveryReasonV1.policy_blocked,
        CapabilityRecoveryReasonV1.capability_executor_unavailable,
        CapabilityRecoveryReasonV1.internal_contract_error,
    }
    return reason in allowed_reasons




def _bound_policy_state(contract: BoundCapabilityExecutionRequestV1) -> dict[str, Any]:
    meta = contract.policy_metadata if isinstance(contract.policy_metadata, dict) else {}
    no_write_active = bool(meta.get("no_write_active"))
    tool_execution_policy = str(meta.get("tool_execution_policy") or "").strip().lower()
    action_execution_policy = str(meta.get("action_execution_policy") or "").strip().lower()
    execution_forbidden = bool(no_write_active or tool_execution_policy == "none" or action_execution_policy == "none")
    reasons: list[str] = []
    if no_write_active:
        reasons.append("no_write_active=true")
    if tool_execution_policy == "none":
        reasons.append("tool_execution_policy=none")
    if action_execution_policy == "none":
        reasons.append("action_execution_policy=none")
    return {
        "no_write_active": no_write_active,
        "tool_execution_policy": tool_execution_policy or None,
        "action_execution_policy": action_execution_policy or None,
        "execution_forbidden": execution_forbidden,
        "blocked_reason": ";".join(reasons) if reasons else None,
    }


def _bound_policy_blocked_result(*, body: AgentChainRequest, contract: BoundCapabilityExecutionRequestV1, dbg: dict[str, Any]) -> AgentChainResult:
    policy_state = _bound_policy_state(contract)
    blocked_reason = str(policy_state.get("blocked_reason") or "execution_forbidden_policy")
    selected_verb = str(contract.selected_verb or "unknown")
    detail = f"Execution blocked by policy: {blocked_reason}"
    failure = BoundCapabilityExecutionFailureV1(
        reason=CapabilityRecoveryReasonV1.policy_blocked,
        selected_verb=selected_verb,
        detail=detail,
        recovery=CapabilityRecoveryDecisionV1(
            reason=CapabilityRecoveryReasonV1.policy_blocked,
            allow_replan=False,
            replanned=False,
            detail="blocked_agent_chain_entry",
        ),
        observation={
            "selected_verb": selected_verb,
            "selected_tool_would_have_been": selected_verb,
            "no_write_active": bool(policy_state.get("no_write_active")),
            "tool_execution_policy": policy_state.get("tool_execution_policy"),
            "action_execution_policy": policy_state.get("action_execution_policy"),
            "execution_blocked_reason": blocked_reason,
            "path": "blocked_agent_chain_entry",
            "dispatch_skipped": False,
            "reply_emitted": True,
        },
    )
    dbg["bound_capability_policy_state"] = dict(policy_state)
    dbg["bound_capability_agent_chain_blocked"] = True
    dbg["bound_capability_blocked_reply_emitted"] = True
    dbg["bound_capability_timeout_prevented"] = True
    logger.warning(
        "[agent-chain] bound_capability_agent_chain_blocked=1 selected_verb=%s no_write_active=%s tool_execution_policy=%s action_execution_policy=%s blocked_reason=%s dispatch_skipped=false reply_emitted=true",
        selected_verb,
        policy_state.get("no_write_active"),
        policy_state.get("tool_execution_policy"),
        policy_state.get("action_execution_policy"),
        blocked_reason,
    )
    logger.info(
        "[agent-chain] bound_capability_blocked_reply_emitted=1 selected_verb=%s path=blocked_agent_chain_entry",
        selected_verb,
    )
    return AgentChainResult(
        mode=body.mode,
        text=(
            f"Execution is disabled by runtime policy, so I’m not running '{selected_verb}'. "
            "I can provide a safe preview, but no capability was executed."
        ),
        structured={
            "finalization_reason": "bound_capability_policy_blocked",
            "bound_capability": failure.model_dump(mode="json"),
        },
        planner_raw={"runtime_debug": dbg, "trace": []},
        runtime_debug=dbg,
    )


def _is_semantic_mismatch(*, selected_verb: str, user_text: str, action_input: dict[str, Any] | None) -> tuple[bool, list[str]]:
    ask = f"{str(user_text or '').lower()} {json.dumps(action_input or {}, ensure_ascii=False, default=str).lower()}"
    asks_runtime_cleanup = ("docker" in ask and "container" in ask and ("cleanup" in ask or "prune" in ask or "stopped" in ask))
    verb = str(selected_verb or "").strip().lower()
    reasons: list[str] = []
    if asks_runtime_cleanup and any(token in verb for token in ("summarize", "recent_changes", "change_intel")):
        reasons.append("runtime_cleanup_request_conflicts_with_change_summary_verb")
    return bool(reasons), reasons


def _bound_terminal_failure(
    *,
    body: AgentChainRequest,
    dbg: dict[str, Any],
    selected_verb: str,
    reason: CapabilityRecoveryReasonV1,
    detail: str,
    path: str,
    failure_category: str,
    capability_decision: dict[str, Any] | None = None,
    observation: dict[str, Any] | None = None,
    selected_tool_would_have_been: str | None = None,
) -> AgentChainResult:
    payload_observation = dict(observation or {})
    payload_observation.setdefault("path", path)
    payload_observation.setdefault("selected_verb", selected_verb)
    payload_observation.setdefault("selected_tool_would_have_been", selected_tool_would_have_been or selected_verb)
    payload_observation.setdefault("failure_category", failure_category)
    payload_observation["reply_emitted"] = True
    failure = BoundCapabilityExecutionFailureV1(
        reason=reason,
        selected_verb=selected_verb,
        detail=detail,
        recovery=CapabilityRecoveryDecisionV1(reason=reason, allow_replan=False, replanned=False, detail=path),
        capability_decision=capability_decision or {},
        observation=payload_observation,
    )
    dbg["bound_capability_terminal_path"] = path
    dbg["bound_capability_reply_emitted"] = True
    logger.info("[agent-chain] bound_capability_terminal_reply_emitted selected_verb=%s path=%s", selected_verb, path)
    return AgentChainResult(
        mode=body.mode,
        text=f"Bound capability execution failed: {detail}",
        structured={"finalization_reason": "bound_capability_fail_closed", "bound_capability": failure.model_dump(mode="json")},
        planner_raw={"runtime_debug": dbg, "trace": []},
        runtime_debug=dbg,
    )


def _bound_execution_timeout_seconds() -> float:
    """
    Bound-capability direct execution guardrail.
    Keep this strictly below the broader agent-chain / exec hop timeout so
    we can always emit a terminal bound-capability reply instead of letting
    the parent RPC time out first.
    """
    default_timeout = float(settings.default_timeout_seconds or 90)
    # Bound execution wraps ToolExecutor.execute_llm_verb(), which for capability-backed
    # verbs performs a nested cortex-orch RPC using settings.default_timeout_seconds.
    # The guardrail here must exceed that inner timeout so we don't fail closed early
    # while the concrete skill invocation is still legitimately in flight.
    return max(12.0, min(180.0, default_timeout + 15.0))


async def _execute_bound_capability_request(
    *,
    body: AgentChainRequest,
    dbg: dict[str, Any],
    tools: List[ToolDef],
    tool_executor: ToolExecutor,
    parent_corr_id: str,
    bound_contract: BoundCapabilityExecutionRequestV1,
    action_input: dict[str, Any],
) -> AgentChainResult | None:
    selected_verb = str(bound_contract.selected_verb or "").strip()
    semantic_mismatch, mismatch_reasons = _is_semantic_mismatch(
        selected_verb=selected_verb,
        user_text=body.text,
        action_input=action_input,
    )
    if semantic_mismatch:
        logger.warning(
            "[agent-chain] bound_capability_semantic_mismatch selected_verb=%s reasons=%s",
            selected_verb,
            mismatch_reasons,
        )
        return _bound_terminal_failure(
            body=body,
            dbg=dbg,
            selected_verb=selected_verb,
            reason=CapabilityRecoveryReasonV1.policy_blocked,
            detail=f"semantic mismatch: {','.join(mismatch_reasons)}",
            path="bound_direct_semantic_mismatch",
            failure_category="semantic_mismatch",
            observation={"mismatch_reasons": mismatch_reasons, "capability_resolution_status": "skipped"},
        )
    policy_state = _bound_policy_state(bound_contract)
    dbg["bound_capability_policy_state"] = dict(policy_state)
    logger.info(
        "[agent-chain] bound_capability_policy_state selected_verb=%s no_write_active=%s tool_execution_policy=%s action_execution_policy=%s execution_forbidden=%s",
        selected_verb,
        policy_state.get("no_write_active"),
        policy_state.get("tool_execution_policy"),
        policy_state.get("action_execution_policy"),
        policy_state.get("execution_forbidden"),
    )
    if policy_state.get("execution_forbidden"):
        return _bound_policy_blocked_result(body=body, contract=bound_contract, dbg=dbg)

    candidate = next((t for t in tools if t.tool_id == selected_verb), None)
    is_capability = bool(
        candidate
        and (
            str(candidate.execution_mode or "").strip().lower() == "capability_backed"
            or bool(candidate.requires_capability_selector)
        )
    )
    logger.info(
        "[agent-chain] bound_capability_candidate selected_verb=%s candidate_exists=%s capability_backed=%s execution_mode=%s requires_selector=%s",
        selected_verb,
        bool(candidate),
        is_capability,
        str(candidate.execution_mode) if candidate else None,
        bool(candidate.requires_capability_selector) if candidate else False,
    )
    if not is_capability:
        return _bound_terminal_failure(
            body=body,
            dbg=dbg,
            selected_verb=selected_verb,
            reason=CapabilityRecoveryReasonV1.policy_blocked,
            detail=f"selected verb '{selected_verb}' is not capability-backed.",
            path="bound_direct_no_compatible_capability",
            failure_category="no_compatible_capability",
            observation={"capability_resolution_status": "selected_verb_not_capability"},
        )
    exec_started = perf_counter()
    try:
        execution_timeout = _bound_execution_timeout_seconds()
        logger.info(
            "[agent-chain] bound_capability_pre_tool_executor selected_verb=%s correlation_id=%s timeout_sec=%.2f elapsed_ms=%.1f",
            selected_verb,
            parent_corr_id,
            execution_timeout,
            0.0,
        )
        observation = await asyncio.wait_for(
            tool_executor.execute_llm_verb(
                selected_verb,
                action_input,
                parent_correlation_id=parent_corr_id,
            ),
            timeout=execution_timeout,
        )
        logger.info(
            "[agent-chain] bound_capability_post_tool_executor selected_verb=%s correlation_id=%s elapsed_ms=%.1f has_observation=%s selected_skill=%s",
            selected_verb,
            parent_corr_id,
            (perf_counter() - exec_started) * 1000.0,
            bool(isinstance(observation, dict)),
            (observation or {}).get("selected_skill") if isinstance(observation, dict) else None,
        )
    except (asyncio.TimeoutError, TimeoutError):
        logger.error(
            "[agent-chain] bound_capability_execution_timeout selected_verb=%s correlation_id=%s timeout_sec=%.2f elapsed_ms=%.1f has_observation=%s",
            selected_verb,
            parent_corr_id,
            execution_timeout,
            (perf_counter() - exec_started) * 1000.0,
            False,
        )
        return _bound_terminal_failure(
            body=body,
            dbg=dbg,
            selected_verb=selected_verb,
            reason=CapabilityRecoveryReasonV1.capability_executor_unavailable,
            detail=f"capability execution timed out after {execution_timeout:.2f}s",
            path="bound_direct_timeout",
            failure_category="execution_timeout",
            observation={"capability_resolution_status": "executor_timeout"},
        )
    except FileNotFoundError:
        reason = CapabilityRecoveryReasonV1.selected_verb_missing
        dbg["bound_execution_recovery_reason"] = reason.value
        if _replan_allowed_for_bound_recovery(bound_contract, reason):
            logger.warning("[agent-chain] bound_capability_recovery_replan=1 reason=%s", reason.value)
            dbg["bound_execution_replanned"] = True
            bound_contract.recovery.reason = reason
            bound_contract.recovery.replanned = True
            return None
        return _bound_terminal_failure(
            body=body,
            dbg=dbg,
            selected_verb=selected_verb,
            reason=reason,
            detail=f"verb '{selected_verb}' no longer exists.",
            path="bound_direct_resolution_failed",
            failure_category="resolution_failed",
            observation={"capability_resolution_status": "selected_verb_missing"},
        )
    except Exception as e:
        logger.exception(
            "[agent-chain] bound_capability_resolution_failed selected_verb=%s correlation_id=%s elapsed_ms=%.1f has_observation=%s",
            selected_verb,
            parent_corr_id,
            (perf_counter() - exec_started) * 1000.0,
            False,
        )
        return _bound_terminal_failure(
            body=body,
            dbg=dbg,
            selected_verb=selected_verb,
            reason=CapabilityRecoveryReasonV1.capability_executor_unavailable,
            detail=str(e),
            path="bound_direct_internal_error",
            failure_category="internal_error",
            observation={"capability_resolution_status": "executor_exception"},
        )

    selected_skill = observation.get("selected_skill") if isinstance(observation, dict) else None
    if not selected_skill:
        failure_reason = (
            (observation or {}).get("capability_decision", {}).get("resolution_failure")
            if isinstance(observation, dict)
            else None
        )
        rejection_reasons = (
            (observation or {}).get("capability_decision", {}).get("rejection_reasons")
            if isinstance(observation, dict)
            else None
        )
        logger.error(
            "[agent-chain] bound_capability_no_compatible_capability selected_verb=%s resolution_failure=%s rejection_reasons=%s",
            selected_verb,
            failure_reason,
            rejection_reasons,
        )
        return _bound_terminal_failure(
            body=body,
            dbg=dbg,
            selected_verb=selected_verb,
            reason=CapabilityRecoveryReasonV1.no_compatible_capability,
            detail="no compatible capability skill was resolved.",
            path="bound_direct_no_compatible_capability",
            failure_category="no_compatible_capability",
            capability_decision=(observation or {}).get("capability_decision", {}) if isinstance(observation, dict) else {},
            observation={
                "capability_resolution_status": "no_skill",
                "selected_tool_would_have_been": selected_verb,
                "resolution_failure": failure_reason,
                "rejection_reasons": rejection_reasons or [],
            },
        )
    raw_payload_ref = (observation or {}).get("raw_payload_ref", {}) if isinstance(observation, dict) else {}
    if str(raw_payload_ref.get("status") or "").strip().lower() == "empty_terminal_output":
        return _bound_terminal_failure(
            body=body,
            dbg=dbg,
            selected_verb=selected_verb,
            reason=CapabilityRecoveryReasonV1.capability_executor_unavailable,
            detail=f"{selected_skill} returned empty terminal output.",
            path="bound_direct_empty_terminal_output",
            failure_category="empty_terminal_output",
            capability_decision=(observation or {}).get("capability_decision", {}) if isinstance(observation, dict) else {},
            observation={
                "capability_resolution_status": "selected_skill_empty_output",
                "selected_tool_would_have_been": selected_verb,
                "selected_skill": selected_skill,
            },
        )
    if raw_payload_ref.get("ok") is False or raw_payload_ref.get("domain_negative"):
        friendly = str((observation or {}).get("execution_summary") or "").strip()
        detail = friendly or str(raw_payload_ref.get("domain_reason") or "skill_domain_negative")
        failure = BoundCapabilityExecutionFailureV1(
            reason=CapabilityRecoveryReasonV1.capability_executor_unavailable,
            selected_verb=selected_verb,
            detail=detail,
            recovery=CapabilityRecoveryDecisionV1(
                reason=CapabilityRecoveryReasonV1.capability_executor_unavailable,
                allow_replan=False,
                replanned=False,
                detail="bound_direct_skill_domain_negative",
            ),
            capability_decision=(observation or {}).get("capability_decision", {}) if isinstance(observation, dict) else {},
            observation={
                "capability_resolution_status": "skill_domain_negative",
                "selected_tool_would_have_been": selected_verb,
                "selected_skill": selected_skill,
                "raw_payload_ref": raw_payload_ref,
                "failure_category": "skill_domain_negative",
            },
        )
        dbg["bound_capability_terminal_path"] = "bound_direct_skill_domain_negative"
        dbg["bound_capability_reply_emitted"] = True
        logger.info(
            "[agent-chain] bound_capability_terminal_reply_emitted selected_verb=%s path=bound_direct_skill_domain_negative selected_skill=%s",
            selected_verb,
            selected_skill,
        )
        return AgentChainResult(
            mode=body.mode,
            text=friendly or f"Bound capability execution failed: {detail}",
            structured={"finalization_reason": "bound_capability_domain_negative", "bound_capability": failure.model_dump(mode="json")},
            planner_raw={"runtime_debug": dbg, "trace": []},
            runtime_debug={**dbg, "bound_capability_reply_emitted": True},
        )
    dbg["bound_execution_completed"] = True
    logger.info(
        "[agent-chain] bound_capability_direct_execute=1 selected_verb=%s selected_skill=%s selected_verb_preserved=1",
        selected_verb,
        selected_skill,
    )
    result_obj = BoundCapabilityExecutionResultV1(
        selected_verb=selected_verb,
        normalized_action_input=action_input,
        selected_skill_family=(observation or {}).get("selected_skill_family") if isinstance(observation, dict) else None,
        selected_skill=selected_skill,
        policy_metadata=bound_contract.policy_metadata,
        capability_decision=(observation or {}).get("capability_decision", {}) if isinstance(observation, dict) else {},
        structured_skill_output=observation if isinstance(observation, dict) else {},
        execution_path="direct_execute",
        recovery=CapabilityRecoveryDecisionV1(
            reason=CapabilityRecoveryReasonV1.internal_contract_error,
            allow_replan=False,
            replanned=False,
            detail="not_used",
        ),
    )
    logger.info(
        "[agent-chain] bound_capability_terminal_reply_emitted selected_verb=%s path=bound_direct_success",
        selected_verb,
    )
    return AgentChainResult(
        mode=body.mode,
        text=str(observation.get("execution_summary") or "Capability executed."),
        structured={"finalization_reason": "bound_capability_execution", "bound_capability": result_obj.model_dump(mode="json")},
        planner_raw={"runtime_debug": dbg, "trace": []},
        runtime_debug={**dbg, "bound_capability_reply_emitted": True, "bound_capability_terminal_path": "bound_direct_success"},
    )


async def execute_agent_chain(
    body: AgentChainRequest,
    *,
    correlation_id: str | None = None,
    rpc_bus: OrionBusAsync | None = None,
) -> AgentChainResult:
    parent_corr_id = str(correlation_id or uuid.uuid4())
    output_mode, response_profile = _effective_output_modes(body)
    grounding = build_delivery_grounding_context(user_text=body.text, output_mode=output_mode)
    tools, pack_names = _resolve_tools(body, output_mode=output_mode)
    tool_ids = [t.tool_id for t in tools]
    dbg: dict[str, Any] = {
        "output_mode": output_mode,
        "response_profile": response_profile,
        "delivery_grounding_mode": grounding.get("delivery_grounding_mode"),
        "packs": pack_names,
        "resolved_tool_ids": tool_ids,
        "triage_blocked_post_step0": False,
        "repeated_tool_breaker": False,
        "repeated_plan_action_escalation": False,
        "finalize_response_invoked": False,
        "quality_evaluator_rewrite": False,
        "generic_drift_detected": False,
        "finalization_source_trace_used": False,
        "finalization_reason": None,
        "suppressed_plan_action_count": 0,
        "invalid_tool_remap_count": 0,
        "invalid_tool_last": None,
    }
    logger.info(
        "[agent-chain] wiring corr=%s output_mode=%s profile=%s packs=%s tools=%s",
        parent_corr_id,
        output_mode,
        response_profile,
        pack_names,
        tool_ids[:25],
    )

    base_toolset = [t.model_dump() for t in tools]
    planner_payload: dict[str, Any] = {
        "request_id": str(uuid.uuid4()),
        "parent_correlation_id": parent_corr_id,
        "caller": "agent-chain",
        "goal": {
            "type": "chat",
            "description": body.goal_description or f"Agentic mode={body.mode}: {body.text}",
        },
        "context": {
            "conversation_history": [m.model_dump() for m in (body.messages or [])]
            or [{"role": "user", "content": body.text}],
            "orion_state_snapshot": {},
            "external_facts": {
                "text": body.text,
                "output_mode": output_mode,
                "response_profile": response_profile,
                **grounding,
            },
        },
        "toolset": base_toolset,
        "trace": [],
        "limits": {
            "max_steps": settings.default_max_steps,
            "timeout_seconds": settings.default_timeout_seconds,
        },
        "preferences": {
            "style": "neutral",
            "delegate_tool_execution": True,
            "return_trace": True,
        },
    }

    owns_bus = rpc_bus is None
    working_bus = rpc_bus or OrionBusAsync(url=settings.orion_bus_url)
    if owns_bus:
        await working_bus.connect()
    tool_executor = ToolExecutor(working_bus, base_dir=str(_cognition_base()))

    tools_called: list[str] = []
    try:
        try:
            bound_contract = _extract_bound_capability_contract(body)
        except Exception as exc:
            reason = CapabilityRecoveryReasonV1.internal_contract_error
            failure = BoundCapabilityExecutionFailureV1(
                reason=reason,
                selected_verb="unknown",
                detail=str(exc),
                recovery=CapabilityRecoveryDecisionV1(reason=reason, allow_replan=False, replanned=False),
            )
            logger.error("[agent-chain] bound_capability_fail_closed=1 reason=%s", reason.value)
            return AgentChainResult(
                mode=body.mode,
                text=f"Bound capability execution failed: {exc}",
                structured={"bound_capability": failure.model_dump(mode="json")},
                planner_raw={"runtime_debug": dbg, "trace": []},
                runtime_debug=dbg,
            )
        if isinstance(bound_contract, BoundCapabilityExecutionRequestV1):
            dbg["bound_execution_received"] = True
            logger.info(
                "[agent-chain] bound_capability_request_received=1 selected_verb=%s planner_corr=%s",
                bound_contract.selected_verb,
                bound_contract.planner_correlation_id,
            )
            selected_verb = str(bound_contract.selected_verb or "").strip()
            action_input = bound_contract.normalized_action_input
            if not isinstance(action_input, dict):
                reason = CapabilityRecoveryReasonV1.invalid_bound_input
                dbg["bound_execution_recovery_reason"] = reason.value
                if _replan_allowed_for_bound_recovery(bound_contract, reason):
                    logger.warning("[agent-chain] bound_capability_recovery_replan=1 reason=%s", reason.value)
                    dbg["bound_execution_replanned"] = True
                    bound_contract.recovery.reason = reason
                    bound_contract.recovery.replanned = True
                else:
                    return _bound_terminal_failure(
                        body=body,
                        dbg=dbg,
                        selected_verb=selected_verb or "unknown",
                        reason=reason,
                        detail="normalized_action_input must be an object.",
                        path="bound_direct_internal_error",
                        failure_category="invalid_bound_input",
                    )
            elif not selected_verb:
                reason = CapabilityRecoveryReasonV1.selected_verb_missing
                dbg["bound_execution_recovery_reason"] = reason.value
                if _replan_allowed_for_bound_recovery(bound_contract, reason):
                    logger.warning("[agent-chain] bound_capability_recovery_replan=1 reason=%s", reason.value)
                    dbg["bound_execution_replanned"] = True
                    bound_contract.recovery.reason = reason
                    bound_contract.recovery.replanned = True
                else:
                    return _bound_terminal_failure(
                        body=body,
                        dbg=dbg,
                        selected_verb="unknown",
                        reason=reason,
                        detail="selected semantic verb is missing.",
                        path="bound_direct_internal_error",
                        failure_category="selected_verb_missing",
                    )
            else:
                bound_result = await _execute_bound_capability_request(
                    body=body,
                    dbg=dbg,
                    tools=tools,
                    tool_executor=tool_executor,
                    parent_corr_id=parent_corr_id,
                    bound_contract=bound_contract,
                    action_input=action_input,
                )
                if bound_result is not None:
                    return bound_result


        for step_idx in range(settings.default_max_steps):
            consecutive_plan_actions = consecutive_tool_count(tools_called=tools_called, candidate="plan_action")
            if step_idx > 0:
                planner_visible_tools = [t for t in base_toolset if t.get("tool_id") != "triage"]
            else:
                planner_visible_tools = list(base_toolset)
            if consecutive_plan_actions >= 2:
                planner_visible_tools = [t for t in planner_visible_tools if t.get("tool_id") != "plan_action"]
            planner_payload["toolset"] = planner_visible_tools
            visible_tool_ids = [t.get("tool_id") for t in planner_visible_tools][:12]
            logger.info(
                "[agent-chain] planner step=%s parent_corr=%s planner_visible_tools=%s repeated_plan_action_count=%s",
                step_idx,
                parent_corr_id,
                visible_tool_ids,
                consecutive_plan_actions,
            )
            raw_resp = await call_planner_react(
                planner_payload,
                parent_correlation_id=parent_corr_id,
                rpc_bus=working_bus,
            )

            if not isinstance(raw_resp, dict):
                raise RuntimeError(f"Invalid Planner Response: {raw_resp}")
            if raw_resp.get("status") != "ok":
                raise RuntimeError(f"Planner Failed: {raw_resp.get('error')}")

            final = raw_resp.get("final_answer") or {}
            text = final.get("content") or ""
            structured = final.get("structured") or {}
            if text or structured:
                if not text and structured:
                    text = json.dumps(structured, indent=2)
                pre_rewrite_drift, _ = detect_generic_delivery_drift(
                    text,
                    request_text=body.text,
                    grounding_mode=dbg.get("delivery_grounding_mode"),
                )
                text, rewrote = await _maybe_rewrite_meta_plan(
                    text,
                    body,
                    tool_executor,
                    parent_corr_id,
                    logger,
                    output_mode,
                    response_profile,
                    trace_snapshot=planner_payload.get("trace") or [],
                )
                if rewrote:
                    dbg["finalize_response_invoked"] = True
                drifted, _ = detect_generic_delivery_drift(
                    text,
                    request_text=body.text,
                    grounding_mode=dbg.get("delivery_grounding_mode"),
                )
                dbg["generic_drift_detected"] = dbg["generic_drift_detected"] or pre_rewrite_drift or drifted
                if pre_rewrite_drift or drifted:
                    logger.info("[agent-chain] generic_drift_detected=1 output_mode=%s", output_mode)
                dbg["quality_evaluator_rewrite"] = rewrote
                dbg["finalization_reason"] = dbg.get("finalization_reason") or "planner_finish"
                raw_resp = {**raw_resp, "runtime_debug": dbg}
                return AgentChainResult(
                    mode=body.mode,
                    text=text,
                    structured={**structured, "finalization_reason": dbg["finalization_reason"]},
                    planner_raw=raw_resp,
                    runtime_debug=dbg,
                )

            trace = list(raw_resp.get("trace") or [])
            if not trace:
                raise RuntimeError("Planner returned no trace/action in delegate mode")

            last = trace[-1] if isinstance(trace[-1], dict) else {}
            action = last.get("action") or {}
            tool_id = action.get("tool_id")
            tool_input = action.get("input") or {}
            last_thought = str(last.get("thought") or "")
            logger.info("[agent-chain] planner_action tool_id=%s input_keys=%s", tool_id, sorted(tool_input.keys()) if isinstance(tool_input, dict) else [])

            # Triage impossible once trace has prior completed steps (hard cap, not suggestive)
            prior_trace_len = len(planner_payload.get("trace") or [])
            if _coverage_threshold_met(
                tools_called=tools_called,
                trace_snapshot=planner_payload.get("trace") or [],
            ):
                logger.info(
                    "[agent-chain] finalization_trigger reason=coverage_threshold_met step=%s",
                    step_idx,
                )
                dbg["finalization_reason"] = "coverage_threshold_met"
                tool_id = _pick_finalization_tool(tool_ids) or "finalize_response"
                tool_input = _finalize_tool_input(
                    body,
                    planner_payload.get("trace") or [],
                    output_mode=output_mode,
                    response_profile=response_profile,
                )
                if tool_id != "finalize_response":
                    tool_input = {"request": body.text, "text": body.text, "goal": body.text, **tool_input}
            if triage_must_finalize(tool_id=str(tool_id or ""), step_idx=step_idx, prior_trace_len=prior_trace_len):
                logger.info(
                    "[agent-chain] triage_blocked_post_step0=1 step=%s prior_trace_len=%s -> remap_non_triage",
                    step_idx,
                    prior_trace_len,
                )
                dbg["triage_blocked_post_step0"] = True
                remap = _select_non_triage_tool(tools)
                if remap:
                    tool_id = remap
                    if remap in {"plan_action", "evaluate", "analyze_text"}:
                        tool_input = {"text": body.text, "request": body.text, "goal": body.text}
                else:
                    tool_id = "finalize_response"
                    trace_snapshot = planner_payload.get("trace") or []
                    tool_input = _finalize_tool_input(
                        body, trace_snapshot, output_mode=output_mode, response_profile=response_profile
                    )
                    dbg["finalize_response_invoked"] = True

            if not tool_id:
                stop_reason = str(raw_resp.get("stop_reason") or "")
                continue_reason = str(raw_resp.get("continue_reason") or "")
                # Planner may conclude without emitting a delegate action in this step.
                if stop_reason in {"final_answer", "continue"} or continue_reason != "action_present":
                    fallback = ""
                    if isinstance(last.get("thought"), str):
                        fallback = last.get("thought") or ""
                    if not fallback and isinstance(last.get("observation"), dict):
                        fallback = str(last.get("observation", {}).get("llm_output") or "")
                    if not fallback:
                        fallback = "Planner completed without explicit action; no final answer content provided."

                    # plan_action leakage: if last executed tool was plan_action, don't use its output as final
                    prev_tool = None
                    for s in reversed(planner_payload.get("trace") or []):
                        a = (s or {}).get("action") or {}
                        if isinstance(a, dict) and a.get("tool_id"):
                            prev_tool = a.get("tool_id")
                            break
                    if prev_tool == "plan_action" and fallback:
                        logger.info("[agent-chain] plan_action leakage guard finalize_response_invoked=1")
                        try:
                            trace_snapshot = planner_payload.get("trace") or []
                            fin_result = await tool_executor.execute_llm_verb(
                                "finalize_response",
                                _finalize_tool_input(
                                    body,
                                    trace_snapshot,
                                    output_mode=output_mode,
                                    response_profile=response_profile,
                                ),
                                parent_correlation_id=parent_corr_id,
                            )
                            fallback = str(fin_result.get("llm_output") or fallback)
                            dbg["finalize_response_invoked"] = True
                        except Exception as e:
                            logger.warning("[agent-chain] finalize_response failed: %s", e)

                    fallback, rewrote = await _maybe_rewrite_meta_plan(
                        fallback,
                        body,
                        tool_executor,
                        parent_corr_id,
                        logger,
                        output_mode,
                        response_profile,
                        trace_snapshot=planner_payload.get("trace") or [],
                    )
                    dbg["quality_evaluator_rewrite"] = dbg["quality_evaluator_rewrite"] or rewrote
                    if rewrote:
                        dbg["finalize_response_invoked"] = True
                    dbg["finalization_reason"] = dbg.get("finalization_reason") or "fallback_finalization"
                    raw_resp = {**raw_resp, "runtime_debug": dbg}
                    return AgentChainResult(
                        mode=body.mode,
                        text=fallback,
                        structured={"finalization_reason": dbg["finalization_reason"]},
                        planner_raw=raw_resp,
                        runtime_debug=dbg,
                    )
                raise RuntimeError("Planner delegate response missing action.tool_id")

            # Allow up to two consecutive plan_action calls; suppress the third and force synthesis-capable tool.
            if plan_action_saturated(tool_id=str(tool_id or ""), tools_called=tools_called):
                override = _pick_finalization_tool(tool_ids) or _delivery_override_for_plan_action_repeat(output_mode)
                logger.info(
                    "[agent-chain] repeated_plan_action_suppressed=1 consecutive=%s -> tool_id=%s output_mode=%s",
                    consecutive_tool_count(tools_called=tools_called, candidate="plan_action"),
                    override,
                    output_mode,
                )
                dbg["repeated_plan_action_escalation"] = True
                dbg["suppressed_plan_action_count"] = int(dbg.get("suppressed_plan_action_count") or 0) + 1
                dbg["finalization_reason"] = dbg.get("finalization_reason") or "repeated_plan_action"
                tool_id = override
                tool_input = _finalize_tool_input(
                    body,
                    planner_payload.get("trace") or [],
                    output_mode=output_mode,
                    response_profile=response_profile,
                )
                if tool_id != "finalize_response":
                    tool_input = {"request": body.text, "text": body.text, "goal": body.text, **tool_input}

            # Repeated same-tool loop breaker
            if tools_called and tools_called[-1] == tool_id and str(tool_id) != "plan_action":
                logger.info("[agent-chain] repeated_tool_breaker=1 tool=%s -> finalize_response", tool_id)
                dbg["repeated_tool_breaker"] = True
                tool_id = "finalize_response"
                trace_snapshot = planner_payload.get("trace") or []
                tool_input = _finalize_tool_input(
                    body, trace_snapshot, output_mode=output_mode, response_profile=response_profile
                )
                dbg["finalize_response_invoked"] = True

            if tool_id == "finalize_response":
                dbg["finalize_response_invoked"] = True
            resolved_tool_id, resolution_source = _resolve_delegate_tool_id(
                tool_id,
                available_tools=tools,
                output_mode=output_mode,
            )
            if resolved_tool_id != tool_id:
                logger.warning(
                    "[agent-chain] invalid_delegate_tool remapped requested=%s resolved=%s source=%s",
                    tool_id,
                    resolved_tool_id,
                    resolution_source,
                )
                dbg["invalid_tool_remap_count"] = int(dbg.get("invalid_tool_remap_count") or 0) + 1
                dbg["invalid_tool_last"] = {
                    "requested": str(tool_id or ""),
                    "resolved": str(resolved_tool_id or ""),
                    "source": resolution_source,
                }
            tool_id = resolved_tool_id
            if not tool_id:
                raise RuntimeError("Planner selected invalid delegate tool and no fallback tool is available")
            tool_input = _ground_tool_input(
                tool_id=str(tool_id),
                tool_input=tool_input if isinstance(tool_input, dict) else {},
                body=body,
                trace_snapshot=planner_payload.get("trace") or [],
                output_mode=output_mode,
                response_profile=response_profile,
            )
            if tool_input.get("finalization_source_trace_used") is True:
                dbg["finalization_source_trace_used"] = True
                logger.info(
                    "[agent-chain] finalization_source_trace_used=1 tool_id=%s output_mode=%s",
                    tool_id,
                    output_mode,
                )

            chosen_tool = next((t for t in tools if t.tool_id == str(tool_id)), None)
            is_capability_selected = bool(
                chosen_tool
                and (
                    str(chosen_tool.execution_mode or "").strip().lower() == "capability_backed"
                    or bool(chosen_tool.requires_capability_selector)
                )
            )
            if is_capability_selected:
                logger.info("[agent-chain] planner_selected_capability_handoff tool_id=%s", tool_id)
                synthetic_bound = BoundCapabilityExecutionRequestV1(
                    selected_verb=str(tool_id),
                    normalized_action_input=tool_input if isinstance(tool_input, dict) else {},
                    planner_correlation_id=parent_corr_id,
                    planner_metadata={"source": "planner_action", "step_index": step_idx},
                    selected_tool_metadata={
                        "tool_id": str(tool_id),
                        "execution_mode": str(chosen_tool.execution_mode or ""),
                        "requires_capability_selector": bool(chosen_tool.requires_capability_selector),
                    },
                    policy_metadata={},
                    recovery=CapabilityRecoveryDecisionV1(
                        reason=CapabilityRecoveryReasonV1.internal_contract_error,
                        allow_replan=False,
                        replanned=False,
                        detail="planner_selected_capability_fail_closed",
                    ),
                )
                logger.info("[agent-chain] planner_selected_capability_bound_entry tool_id=%s", tool_id)
                bound_result = await _execute_bound_capability_request(
                    body=body,
                    dbg=dbg,
                    tools=tools,
                    tool_executor=tool_executor,
                    parent_corr_id=parent_corr_id,
                    bound_contract=synthetic_bound,
                    action_input=synthetic_bound.normalized_action_input,
                )
                if bound_result is not None:
                    logger.info("[agent-chain] planner_selected_capability_terminal_reply_emitted tool_id=%s", tool_id)
                    return bound_result
                logger.error("[agent-chain] planner_selected_capability_fail_closed tool_id=%s", tool_id)
                return _bound_terminal_failure(
                    body=body,
                    dbg=dbg,
                    selected_verb=str(tool_id),
                    reason=CapabilityRecoveryReasonV1.internal_contract_error,
                    detail="planner-selected capability execution did not terminalize.",
                    path="planner_selected_non_terminal",
                    failure_category="internal_error",
                    observation={"capability_resolution_status": "non_terminal"},
                )

            observation = await tool_executor.execute_llm_verb(
                tool_id,
                tool_input if isinstance(tool_input, dict) else {},
                parent_correlation_id=parent_corr_id,
            )
            tools_called.append(tool_id)
            if tool_id == "finalize_response":
                final_text = _usable_finalized_text(observation)
                if final_text:
                    dbg["quality_evaluator_rewrite"] = True
                    final_text, rewrote = await _maybe_rewrite_meta_plan(
                        final_text,
                        body,
                        tool_executor,
                        parent_corr_id,
                        logger,
                        output_mode,
                        response_profile,
                        trace_snapshot=planner_payload.get("trace") or [],
                    )
                    dbg["quality_evaluator_rewrite"] = dbg["quality_evaluator_rewrite"] or rewrote
                    drifted, _ = detect_generic_delivery_drift(
                        final_text,
                        request_text=body.text,
                        grounding_mode=dbg.get("delivery_grounding_mode"),
                    )
                    dbg["generic_drift_detected"] = dbg["generic_drift_detected"] or drifted
                    if drifted:
                        logger.info("[agent-chain] generic_drift_detected=1 output_mode=%s", output_mode)
                    return AgentChainResult(
                        mode=body.mode,
                        text=final_text,
                        structured={"finalization_reason": dbg.get("finalization_reason") or "planner_finish"},
                        planner_raw={"runtime_debug": dbg, "trace": planner_payload.get("trace") or []},
                        runtime_debug=dbg,
                    )
            last["observation"] = observation
            last_observation = observation
            trace[-1] = last
            planner_payload["trace"] = trace

        # Step cap: best-effort finalization instead of raw error
        logger.info("[agent-chain] step_cap finalize_response_invoked=1 corr=%s", parent_corr_id)
        dbg["finalize_response_invoked"] = True
        dbg["finalization_reason"] = dbg.get("finalization_reason") or "step_cap_best_effort"
        try:
            trace_snapshot = planner_payload.get("trace") or []
            fin_result = await tool_executor.execute_llm_verb(
                "finalize_response",
                _finalize_tool_input(
                    body, trace_snapshot, output_mode=output_mode, response_profile=response_profile
                ),
                parent_correlation_id=parent_corr_id,
            )
            final_text = str(fin_result.get("llm_output") or "Max steps reached; synthesis unavailable.")
        except Exception as e:
            logger.warning("[agent-chain] finalize_response at step cap failed: %s", e)
            final_text = "Max steps reached. Please try a more focused request."
        final_text, rewrote = await _maybe_rewrite_meta_plan(
            final_text,
            body,
            tool_executor,
            parent_corr_id,
            logger,
            output_mode,
            response_profile,
            trace_snapshot=planner_payload.get("trace") or [],
        )
        dbg["quality_evaluator_rewrite"] = dbg["quality_evaluator_rewrite"] or rewrote
        return AgentChainResult(
            mode=body.mode,
            text=final_text,
            structured={"finalization_reason": dbg["finalization_reason"]},
            planner_raw={"runtime_debug": dbg},
            runtime_debug=dbg,
        )
    finally:
        if owns_bus:
            await working_bus.close()


@router.post("/run", response_model=AgentChainResult)
async def run_chain(body: AgentChainRequest) -> AgentChainResult:
    if not body.text.strip():
        raise HTTPException(400, "text required")
    try:
        return await execute_agent_chain(body)
    except Exception as e:
        logger.exception("Agent Chain Error")
        raise HTTPException(500, str(e))
