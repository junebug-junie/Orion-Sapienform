# services/orion-cortex-orch/app/orchestrator.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import yaml
from pydantic import BaseModel, ConfigDict, Field

import orion  # used to locate installed package path for cognition/verbs

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.cortex.schemas import (
    ExecutionPlan, 
    ExecutionStep, 
    PlanExecutionRequest, 
    PlanExecutionArgs
)
from .clients import CortexExecClient
from orion.schemas.cortex.contracts import CortexClientRequest, RecallDirective

logger = logging.getLogger("orion.cortex.orch")

# Locate cognition directories
ORION_PKG_DIR = Path(orion.__file__).resolve().parent
VERBS_DIR = ORION_PKG_DIR / "cognition" / "verbs"
PROMPTS_DIR = ORION_PKG_DIR / "cognition" / "prompts"  # <--- Added Prompts Directory


def _load_verb_yaml(verb_name: str) -> dict:
    path = VERBS_DIR / f"{verb_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No verb YAML found for '{verb_name}' at {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_prompt_content(template_ref: Optional[str]) -> Optional[str]:
    """
    If template_ref looks like a file (ends in .j2), load its content.
    Otherwise return it as-is (assuming it's a raw string or None).
    """
    if not template_ref:
        return None
    
    if template_ref.strip().endswith(".j2"):
        prompt_path = PROMPTS_DIR / template_ref.strip()
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        else:
            logger.warning(f"Prompt template file not found: {prompt_path}")
            # Fallback: return the filename so at least something happens
            return template_ref
            
    return template_ref


def build_plan_for_verb(verb_name: str, *, mode: str = "brain") -> ExecutionPlan:
    data = _load_verb_yaml(verb_name)

    # Defaults
    timeout_ms = int(data.get("timeout_ms", 120000) or 120000)
    default_services = list(data.get("services") or [])
    
    # [FIX] Load the raw content if it's a file reference
    raw_template_ref = str(data.get("prompt_template") or "")
    default_prompt = _load_prompt_content(raw_template_ref)

    steps: List[ExecutionStep] = []
    raw_steps = data.get("steps") or data.get("plan") # handle 'plan' alias in yaml
    
    if isinstance(raw_steps, list) and raw_steps:
        for i, s in enumerate(raw_steps):
            # Resolve step-level prompt if provided, else use default
            step_template_ref = str(s.get("prompt_template") or "")
            step_prompt = _load_prompt_content(step_template_ref) if step_template_ref else default_prompt

            steps.append(
                ExecutionStep(
                    verb_name=verb_name,
                    step_name=str(s.get("name") or f"step_{i}"),
                    description=str(s.get("description") or ""),
                    order=int(s.get("order", i)),
                    services=list(s.get("services") or default_services),
                    prompt_template=step_prompt,
                    requires_gpu=bool(s.get("requires_gpu", False)),
                    requires_memory=bool(s.get("requires_memory", False)),
                    timeout_ms=int(s.get("timeout_ms", timeout_ms) or timeout_ms),
                )
            )
    else:
        # Single-step inference
        steps.append(
            ExecutionStep(
                verb_name=verb_name,
                step_name=verb_name,
                description=str(data.get("description") or ""),
                order=0,
                services=default_services,
                prompt_template=default_prompt,
                requires_gpu=bool(data.get("requires_gpu", False)),
                requires_memory=bool(data.get("requires_memory", False)),
                timeout_ms=timeout_ms,
            )
        )

    return ExecutionPlan(
        verb_name=verb_name,
        label=str(data.get("label") or verb_name),
        description=str(data.get("description") or ""),
        category=str(data.get("category") or "general"),
        priority=str(data.get("priority") or "normal"),
        interruptible=bool(data.get("interruptible", True)),
        can_interrupt_others=bool(data.get("can_interrupt_others", False)),
        timeout_ms=timeout_ms,
        max_recursion_depth=int(data.get("max_recursion_depth", 2) or 2),
        steps=steps,
        metadata={"verb_yaml": f"{verb_name}.yaml", "mode": mode},
    )


def build_agent_plan(verb_name: str) -> ExecutionPlan:
    """Single-step agent plan that defers to AgentChainService."""
    return ExecutionPlan(
        verb_name=verb_name,
        label=f"{verb_name}-agent",
        description="Agent chain execution via planner-react",
        category="agentic",
        priority="normal",
        interruptible=True,
        can_interrupt_others=False,
        timeout_ms=300000,
        max_recursion_depth=1,
        steps=[
            ExecutionStep(
                verb_name=verb_name,
                step_name="agent_chain",
                description="Delegate to AgentChainService (ReAct)",
                order=0,
                services=["AgentChainService"],
                prompt_template=None,
                requires_gpu=False,
                requires_memory=True,
                timeout_ms=300000,
            )
        ],
        metadata={"mode": "agent"},
    )


def build_council_plan(verb_name: str) -> ExecutionPlan:
    """Stub council plan; routed to CouncilService."""
    return ExecutionPlan(
        verb_name=verb_name,
        label=f"{verb_name}-council",
        description="Council supervisor stub",
        category="council",
        priority="normal",
        interruptible=False,
        can_interrupt_others=False,
        timeout_ms=300000,
        max_recursion_depth=1,
        steps=[
            ExecutionStep(
                verb_name=verb_name,
                step_name="council_supervisor",
                description="Council supervisor placeholder",
                order=0,
                services=["CouncilService"],
                prompt_template=None,
                requires_gpu=False,
                requires_memory=True,
                timeout_ms=120000,
            )
        ],
        metadata={"mode": "council"},
    )


def _build_plan_for_mode(req: CortexClientRequest) -> ExecutionPlan:
    if req.mode == "agent":
        return build_agent_plan(req.verb)
    if req.mode == "council":
        return build_council_plan(req.verb)
    return build_plan_for_verb(req.verb, mode=req.mode)


def _build_context(req: CortexClientRequest) -> Dict[str, Any]:
    return {
        "messages": [m.model_dump(mode="json") for m in req.context.messages],
        "session_id": req.context.session_id,
        "user_id": req.context.user_id,
        "trace_id": req.context.trace_id,
        "metadata": req.context.metadata,
        "packs": req.packs,
        "mode": req.mode,
    }


def _plan_args(req: CortexClientRequest, correlation_id: str) -> PlanExecutionArgs:
    recall: RecallDirective = req.recall
    return PlanExecutionArgs(
        request_id=req.context.trace_id or correlation_id,
        user_id=req.context.user_id,
        trigger_source="cortex-orch",
        extra={
            "mode": req.mode,
            "packs": req.packs,
            "recall": recall.model_dump(),
            "options": req.options,
            "trace_id": req.context.trace_id or correlation_id,
            "session_id": req.context.session_id,
            "verb": req.verb,
        },
    )


async def call_cortex_exec(
    bus: OrionBusAsync,
    *,
    source: ServiceRef,
    exec_request_channel: str,
    client_request: CortexClientRequest,
    correlation_id: str,
    timeout_sec: float = 900.0,
) -> Dict[str, Any]:
    plan = _build_plan_for_mode(client_request)
    context = _build_context(client_request)
    args = _plan_args(client_request, correlation_id)

    request_object = PlanExecutionRequest(plan=plan, args=args, context=context)

    client = CortexExecClient(bus, exec_request_channel)
    return await client.execute_plan(
        source=source,
        req=request_object,
        correlation_id=correlation_id,
        timeout_sec=timeout_sec,
    )
