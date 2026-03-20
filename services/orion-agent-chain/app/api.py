# services/orion-agent-chain/app/api.py
from __future__ import annotations

import json
import logging
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
from orion.cognition.quality_evaluator import should_rewrite_for_instructional

logger = logging.getLogger("agent-chain.api")
TOOL_REGISTRY = ToolRegistry(base_dir=Path("/app/orion/cognition"))
router = APIRouter()


async def _maybe_rewrite_meta_plan(
    text: str,
    body: AgentChainRequest,
    tool_executor: ToolExecutor,
    parent_corr_id: str | None,
    logger_inst: "logging.Logger",
) -> str:
    """If text looks like meta-plan for instructional mode, rewrite via finalize_response."""
    should_rewrite, _ = should_rewrite_for_instructional(text, getattr(body, "output_mode", None))
    if not should_rewrite:
        return text
    logger_inst.info("[agent-chain] quality evaluator flagged meta-plan, invoking finalize_response rewrite")
    try:
        fin_result = await tool_executor.execute_llm_verb(
            "finalize_response",
            {
                "original_request": body.text,
                "request": f"Rewrite the following as concrete, actionable instructions. Do not use meta-planning language:\n\n{text[:4000]}",
                "trace": text[:8000],
            },
            parent_correlation_id=parent_corr_id,
        )
        return str(fin_result.get("llm_output") or text)
    except Exception as e:
        logger_inst.warning("[agent-chain] quality rewrite failed: %s", e)
        return text


def _external_facts(body: AgentChainRequest) -> dict:
    out = {"text": body.text}
    if getattr(body, "output_mode", None):
        out["output_mode"] = body.output_mode
    if getattr(body, "response_profile", None):
        out["response_profile"] = body.response_profile
    return out


def _resolve_tools(body: AgentChainRequest) -> List[ToolDef]:
    if body.tools:
        return [ToolDef(**t) for t in body.tools]
    pack_names = body.packs or ["executive_pack", "memory_pack"]
    local_tools = TOOL_REGISTRY.tools_for_packs(pack_names)
    return [ToolDef(**(t.dict() if hasattr(t, "dict") else t)) for t in local_tools]


async def execute_agent_chain(
    body: AgentChainRequest,
    *,
    correlation_id: str | None = None,
    rpc_bus: OrionBusAsync | None = None,
) -> AgentChainResult:
    parent_corr_id = str(correlation_id or uuid.uuid4())
    tools = _resolve_tools(body)
    logger.info("[agent-chain] resolved_tools count=%s tool_ids=%s", len(tools), [t.tool_id for t in tools])

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
            "external_facts": _external_facts(body),
        },
        "toolset": [t.model_dump() for t in tools],
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
    tool_executor = ToolExecutor(working_bus)

    tools_called: list[str] = []
    try:
        for step_idx in range(settings.default_max_steps):
            logger.info("[agent-chain] planner step=%s parent_corr=%s", step_idx, parent_corr_id)
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
                text = await _maybe_rewrite_meta_plan(text, body, tool_executor, parent_corr_id, logger)
                return AgentChainResult(mode=body.mode, text=text, structured=structured, planner_raw=raw_resp)

            trace = list(raw_resp.get("trace") or [])
            if not trace:
                raise RuntimeError("Planner returned no trace/action in delegate mode")

            last = trace[-1] if isinstance(trace[-1], dict) else {}
            action = last.get("action") or {}
            tool_id = action.get("tool_id")
            tool_input = action.get("input") or {}
            logger.info("[agent-chain] planner_action tool_id=%s input_keys=%s", tool_id, sorted(tool_input.keys()) if isinstance(tool_input, dict) else [])

            # Triage capped to step 0 only
            if tool_id == "triage" and step_idx > 0:
                logger.info("[agent-chain] triage blocked at step %s, forcing finalize_response", step_idx)
                tool_id = "finalize_response"
                trace_snapshot = planner_payload.get("trace") or []
                tool_input = {
                    "original_request": body.text,
                    "request": body.text,
                    "trace": json.dumps([dict(s) for s in trace_snapshot], default=str)[:12000],
                    "prior_trace": str(trace_snapshot),
                }

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
                        logger.info("[agent-chain] plan_action leakage guard, invoking finalize_response")
                        try:
                            trace_snapshot = planner_payload.get("trace") or []
                            fin_result = await tool_executor.execute_llm_verb(
                                "finalize_response",
                                {
                                    "original_request": body.text,
                                    "request": body.text,
                                    "trace": json.dumps([dict(s) for s in trace_snapshot], default=str)[:12000],
                                    "prior_trace": str(trace_snapshot),
                                },
                                parent_correlation_id=parent_corr_id,
                            )
                            fallback = str(fin_result.get("llm_output") or fallback)
                        except Exception as e:
                            logger.warning("[agent-chain] finalize_response failed: %s", e)

                    return AgentChainResult(
                        mode=body.mode,
                        text=fallback,
                        structured={},
                        planner_raw=raw_resp,
                    )
                raise RuntimeError("Planner delegate response missing action.tool_id")

            # Repeated same-tool loop breaker
            if tools_called and tools_called[-1] == tool_id:
                logger.info("[agent-chain] repeated tool %s detected, forcing finalize_response", tool_id)
                tool_id = "finalize_response"
                trace_snapshot = planner_payload.get("trace") or []
                tool_input = {
                    "original_request": body.text,
                    "request": body.text,
                    "trace": json.dumps([dict(s) for s in trace_snapshot], default=str)[:12000],
                    "prior_trace": str(trace_snapshot),
                }

            observation = await tool_executor.execute_llm_verb(
                tool_id,
                tool_input if isinstance(tool_input, dict) else {},
                parent_correlation_id=parent_corr_id,
            )
            tools_called.append(tool_id)
            last["observation"] = observation
            trace[-1] = last
            planner_payload["trace"] = trace

        # Step cap: best-effort finalization instead of raw error
        logger.info("[agent-chain] max steps reached, invoking finalize_response")
        try:
            trace_snapshot = planner_payload.get("trace") or []
            fin_result = await tool_executor.execute_llm_verb(
                "finalize_response",
                {
                    "original_request": body.text,
                    "request": body.text,
                    "trace": json.dumps([dict(s) for s in trace_snapshot], default=str)[:12000],
                    "prior_trace": str(trace_snapshot),
                },
                parent_correlation_id=parent_corr_id,
            )
            final_text = str(fin_result.get("llm_output") or "Max steps reached; synthesis unavailable.")
        except Exception as e:
            logger.warning("[agent-chain] finalize_response at step cap failed: %s", e)
            final_text = "Max steps reached. Please try a more focused request."
        return AgentChainResult(mode=body.mode, text=final_text, structured={}, planner_raw={})
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
