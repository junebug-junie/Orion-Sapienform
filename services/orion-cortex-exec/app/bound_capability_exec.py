"""Bound capability execution via nested cortex RPC (moved from orion-agent-chain)."""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Dict
from uuid import uuid4

import orion

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, LLMMessage, ServiceRef
from orion.schemas.agents.bound_capability import BoundCapabilityExecutionRequestV1
from orion.schemas.cortex.contracts import CortexClientContext, CortexClientRequest, CortexClientResult, RecallDirective
from orion.schemas.cortex.schemas import StepExecutionResult

from .actions_skill_registry import ActionsSkillRegistry
from .capability_bridge import (
    CapabilityDecision,
    humanize_domain_failure,
    normalize_capability_observation,
    resolve_capability_decision,
    skill_domain_failure_reason_from_final_text,
)
from .settings import settings

logger = logging.getLogger("orion.cortex.exec.bound_capability")

_CORTEX_REQUEST_CHANNEL = "orion:cortex:request"
_VERBS_DIR = Path(orion.__file__).resolve().parent / "cognition" / "verbs"


def _bound_capability_service_payload(
    *,
    observation: Dict[str, Any],
    decision: CapabilityDecision,
    ok: bool,
    terminal_path: str,
    text: str | None = None,
) -> Dict[str, Any]:
    final_text = text if text is not None else str(observation.get("execution_summary") or "")
    bound: Dict[str, Any] = {
        "selected_verb": decision.verb,
        "selected_skill": decision.selected_skill,
        "path": terminal_path,
    }
    raw_ref = observation.get("raw_payload_ref")
    if isinstance(raw_ref, dict) and raw_ref.get("domain_reason"):
        bound["reason"] = raw_ref.get("domain_reason")
    if not ok and terminal_path == "bound_direct_timeout":
        bound.setdefault("reason", "capability_executor_unavailable")
    return {
        "text": final_text,
        "final_text": final_text,
        "structured": {
            "finalization_reason": "bound_capability_execution" if ok else "bound_capability_fail_closed",
            "bound_capability": bound,
        },
        "runtime_debug": {
            "bound_capability_terminal_path": terminal_path,
            "bound_capability_reply_emitted": True,
        },
    }


def _step_from_observation(
    *,
    observation: Dict[str, Any],
    decision: CapabilityDecision,
    ok: bool,
    terminal_path: str,
    tool_id: str,
    latency_ms: int,
    text: str | None = None,
    error: str | None = None,
) -> StepExecutionResult:
    payload = _bound_capability_service_payload(
        observation=observation,
        decision=decision,
        ok=ok,
        terminal_path=terminal_path,
        text=text,
    )
    return StepExecutionResult(
        status="success" if ok else "fail",
        verb_name=tool_id,
        step_name="bound_capability_execution",
        order=0,
        result={"ContextExecService": payload},
        latency_ms=latency_ms,
        node=settings.node_name,
        logs=[f"{'ok' if ok else 'fail'} <- bound_capability path={terminal_path}"],
        error=error,
    )


async def execute_bound_capability(
    bus: OrionBusAsync,
    *,
    source: ServiceRef,
    bound: BoundCapabilityExecutionRequestV1,
    ctx: Dict[str, Any],
    correlation_id: str,
    verb_cfg: Any,
) -> StepExecutionResult:
    tool_id = str(bound.selected_verb or "")
    tool_input = bound.normalized_action_input if isinstance(bound.normalized_action_input, dict) else {}
    registry = ActionsSkillRegistry(verbs_dir=_VERBS_DIR)
    decision = resolve_capability_decision(
        verb=tool_id,
        preferred_skill_families=list(getattr(verb_cfg, "preferred_skill_families", None) or []),
        registry=registry,
    )
    if not decision.selected_skill:
        observation = normalize_capability_observation(
            decision=decision,
            execution_summary="No compatible orion-actions skill available.",
            raw_payload={"status": "unavailable", "ok": False},
        )
        return _step_from_observation(
            observation=observation,
            decision=decision,
            ok=False,
            terminal_path="bound_direct_timeout",
            tool_id=tool_id,
            latency_ms=0,
            text="Bound capability execution failed: no compatible skill available.",
            error="capability_executor_unavailable",
        )

    corr = str(uuid4())
    reply_channel = f"orion:cortex:result:{corr}"
    bridge_user_text = str(
        tool_input.get("text")
        or tool_input.get("request")
        or ctx.get("raw_user_text")
        or _last_user_message(ctx)
        or ""
    )
    bridge_skill_args: Dict[str, Any] = {}
    lim = tool_input.get("limit")
    try:
        if lim is not None and str(lim).strip() != "":
            bridge_skill_args["limit"] = max(1, min(500, int(lim)))
    except (TypeError, ValueError):
        pass
    tz = tool_input.get("timezone")
    if isinstance(tz, str) and tz.strip():
        bridge_skill_args["timezone"] = tz.strip()

    req = CortexClientRequest(
        mode="brain",
        route_intent="none",
        verb=decision.selected_skill,
        packs=["executive_pack"],
        options={"policy_dispatch_only": True, "source": "cortex_exec_capability_bridge"},
        recall=RecallDirective(enabled=False, required=False, mode="hybrid"),
        context=CortexClientContext(
            messages=[LLMMessage(role="user", content=bridge_user_text)],
            raw_user_text=bridge_user_text,
            user_message=bridge_user_text,
            session_id=str(ctx.get("session_id") or "cortex-exec-capability"),
            user_id=str(ctx.get("user_id") or "cortex-exec"),
            trace_id=correlation_id,
            metadata={
                "capability_decision": decision.model_dump(mode="json"),
                "capability_bridge": True,
                "requested_verb": tool_id,
                "selected_skill_family": decision.skill_family,
                "selected_skill": decision.selected_skill,
                "risk_class": decision.policy.get("risk_class"),
                "requires_confirmation": bool(decision.policy.get("confirmation_required")),
                "execute_opt_in": bool(decision.policy.get("execute_opt_in")),
                "observational": bool(decision.observational),
                "capability_bridge_user_text": bridge_user_text,
                **({"capability_bridge_skill_args": bridge_skill_args} if bridge_skill_args else {}),
            },
        ),
    )
    env = BaseEnvelope(
        kind="cortex.orch.request",
        source=source,
        correlation_id=corr,
        reply_to=reply_channel,
        payload=req.model_dump(mode="json"),
    )
    t0 = perf_counter()
    logger.info(
        "bound_capability_nested_cortex verb=%s skill=%s parent=%s",
        tool_id,
        decision.selected_skill,
        correlation_id,
    )
    try:
        msg = await bus.rpc_request(
            _CORTEX_REQUEST_CHANNEL,
            env,
            reply_channel=reply_channel,
            timeout_sec=float(settings.step_timeout_ms) / 1000.0,
        )
    except Exception as exc:
        latency_ms = int((perf_counter() - t0) * 1000)
        observation = normalize_capability_observation(
            decision=decision,
            execution_summary=f"Bound capability execution failed: {exc}",
            raw_payload={"status": "error", "ok": False, "error": str(exc)},
        )
        return _step_from_observation(
            observation=observation,
            decision=decision,
            ok=False,
            terminal_path="bound_direct_timeout",
            tool_id=tool_id,
            latency_ms=latency_ms,
            text=f"Bound capability execution failed: {exc}",
            error=f"capability_executor_unavailable:{tool_id}",
        )

    decoded = bus.codec.decode(msg.get("data"))
    if not decoded.ok:
        raise RuntimeError(f"Capability bridge decode failed: {decoded.error}")
    payload = decoded.envelope.payload if isinstance(decoded.envelope.payload, dict) else {}
    result = CortexClientResult.model_validate(payload)
    final_text = str(result.final_text or "").strip()
    raw_payload = {
        "status": result.status,
        "ok": result.ok,
        "final_text": result.final_text,
    }
    latency_ms = int((perf_counter() - t0) * 1000)

    if "empty_terminal_output" in final_text.lower():
        observation = normalize_capability_observation(
            decision=decision,
            execution_summary=(
                f"Execution failed: {decision.selected_skill} returned synthetic empty-output fallback text."
            ),
            raw_payload={"status": "empty_terminal_output", "ok": False, "final_text": result.final_text},
        )
        return _step_from_observation(
            observation=observation,
            decision=decision,
            ok=False,
            terminal_path="bound_direct_timeout",
            tool_id=tool_id,
            latency_ms=latency_ms,
            text="Bound capability execution failed: empty terminal output.",
            error="empty_terminal_output",
        )

    if result.ok and not final_text:
        observation = normalize_capability_observation(
            decision=decision,
            execution_summary=f"Execution failed: {decision.selected_skill} returned empty terminal output.",
            raw_payload={"status": "empty_terminal_output", "ok": False, "final_text": ""},
        )
        return _step_from_observation(
            observation=observation,
            decision=decision,
            ok=False,
            terminal_path="bound_direct_timeout",
            tool_id=tool_id,
            latency_ms=latency_ms,
            text="Bound capability execution failed: empty terminal output.",
            error="empty_terminal_output",
        )

    domain_reason = skill_domain_failure_reason_from_final_text(final_text)
    if domain_reason:
        human = humanize_domain_failure(decision.selected_skill or "", domain_reason)
        observation = normalize_capability_observation(
            decision=decision,
            execution_summary=human,
            raw_payload={
                "status": "fail",
                "ok": False,
                "final_text": result.final_text,
                "domain_negative": True,
                "domain_reason": domain_reason,
                "nested_ok": result.ok,
                "nested_status": result.status,
            },
        )
        return _step_from_observation(
            observation=observation,
            decision=decision,
            ok=False,
            terminal_path="bound_direct_skill_domain_negative",
            tool_id=tool_id,
            latency_ms=latency_ms,
            text=human,
            error=domain_reason,
        )

    if not result.ok:
        dr = skill_domain_failure_reason_from_final_text(final_text)
        summary = (
            humanize_domain_failure(decision.selected_skill or "", dr)
            if dr
            else f"Bound capability execution failed: {decision.selected_skill} status={result.status}"
        )
        observation = normalize_capability_observation(
            decision=decision,
            execution_summary=summary,
            raw_payload={**raw_payload, "ok": False, "domain_negative": bool(dr)},
        )
        return _step_from_observation(
            observation=observation,
            decision=decision,
            ok=False,
            terminal_path="bound_direct_timeout",
            tool_id=tool_id,
            latency_ms=latency_ms,
            text=summary,
            error=f"capability_failed:{tool_id}",
        )

    summary = f"Executed {decision.selected_skill}: status={result.status} ok={result.ok}"
    observation = normalize_capability_observation(
        decision=decision,
        execution_summary=summary,
        raw_payload=raw_payload,
    )
    display_text = final_text or summary
    return _step_from_observation(
        observation=observation,
        decision=decision,
        ok=True,
        terminal_path="bound_direct_success",
        tool_id=tool_id,
        latency_ms=latency_ms,
        text=display_text,
    )


def _last_user_message(ctx: Dict[str, Any]) -> str:
    messages = ctx.get("messages") or []
    for item in reversed(messages):
        if isinstance(item, dict) and str(item.get("role") or "").lower() == "user":
            return str(item.get("content") or "")
    return str(ctx.get("raw_user_text") or ctx.get("user_message") or "")
