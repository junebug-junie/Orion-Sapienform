# services/orion-agent-chain/app/api.py
from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, List

from fastapi import APIRouter, HTTPException
from orion.core.bus.async_service import OrionBusAsync

from .planner_rpc import call_planner_react
from .settings import settings
from .tool_executor import ToolExecutor
from .tool_registry import ToolRegistry
from orion.schemas.agents.schemas import AgentChainRequest, AgentChainResult, ToolDef
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
