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
# Import shared schemas
from orion.schemas.cortex.schemas import (
    ExecutionPlan,
    ExecutionStep,
    PlanExecutionRequest,
    PlanExecutionArgs,
)
from .clients import CortexExecClient

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


def build_plan_for_verb(verb_name: str) -> ExecutionPlan:
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
        metadata={"verb_yaml": f"{verb_name}.yaml"},
    )


def build_plan_for_mode(mode: str) -> ExecutionPlan:
    """
    Canonical plan builder for top-level modes.
    """
    m = (mode or "").lower()
    if m in ("brain", "chat_general", "chat"):
        prompt = _load_prompt_content("chat_general.j2")
        return ExecutionPlan(
            verb_name="brain.chat_general",
            label="Brain Chat",
            description="Brain path: optional recall + single LLM gateway hop.",
            category="brain",
            priority="normal",
            interruptible=True,
            can_interrupt_others=False,
            timeout_ms=180000,
            max_recursion_depth=0,
            steps=[
                ExecutionStep(
                    verb_name="brain.chat_general",
                    step_name="recall_aug",
                    description="Optional recall augmentation",
                    order=0,
                    services=["RecallService"],
                    prompt_template=None,
                    requires_gpu=False,
                    requires_memory=True,
                    timeout_ms=5000,
                ),
                ExecutionStep(
                    verb_name="brain.chat_general",
                    step_name="llm_chat",
                    description="Primary chat via LLM Gateway",
                    order=1,
                    services=["LLMGatewayService"],
                    prompt_template=prompt,
                    requires_gpu=True,
                    requires_memory=True,
                    timeout_ms=120000,
                ),
            ],
            metadata={"mode": "brain"},
        )

    if m in ("agent", "agent_chat"):
        return ExecutionPlan(
            verb_name="agent.chat",
            label="Agent Chain",
            description="Agent mode: optional recall, then agent-chain worker.",
            category="agent",
            priority="normal",
            interruptible=True,
            can_interrupt_others=False,
            timeout_ms=180000,
            max_recursion_depth=0,
            steps=[
                ExecutionStep(
                    verb_name="agent.chat",
                    step_name="recall_aug",
                    description="Optional recall augmentation",
                    order=0,
                    services=["RecallService"],
                    prompt_template=None,
                    requires_gpu=False,
                    requires_memory=True,
                    timeout_ms=5000,
                ),
                ExecutionStep(
                    verb_name="agent.chat",
                    step_name="agent_chain",
                    description="Agent-chain / planner-react worker",
                    order=1,
                    services=["AgentChainService"],
                    prompt_template=None,
                    requires_gpu=True,
                    requires_memory=True,
                    timeout_ms=120000,
                ),
            ],
            metadata={"mode": "agent"},
        )

    if m in ("council", "council_supervisor"):
        return ExecutionPlan(
            verb_name="council.supervisor",
            label="Council Supervisor",
            description="Council stub: optional recall then council deliberation.",
            category="council",
            priority="normal",
            interruptible=True,
            can_interrupt_others=False,
            timeout_ms=180000,
            max_recursion_depth=0,
            steps=[
                ExecutionStep(
                    verb_name="council.supervisor",
                    step_name="recall_aug",
                    description="Optional recall augmentation",
                    order=0,
                    services=["RecallService"],
                    prompt_template=None,
                    requires_gpu=False,
                    requires_memory=True,
                    timeout_ms=5000,
                ),
                ExecutionStep(
                    verb_name="council.supervisor",
                    step_name="council_supervisor",
                    description="Council deliberation",
                    order=1,
                    services=["AgentCouncilService"],
                    prompt_template=None,
                    requires_gpu=True,
                    requires_memory=True,
                    timeout_ms=120000,
                ),
            ],
            metadata={"mode": "council"},
        )

    return build_plan_for_verb(mode)


async def call_cortex_exec(
    bus: OrionBusAsync,
    *,
    source: ServiceRef,
    exec_request_channel: str,
    verb_name: str,
    args: Dict[str, Any],
    context: Dict[str, Any],
    correlation_id: str,
    timeout_sec: float = 900.0,
) -> Dict[str, Any]:

    # 1. Logic: Build the Plan Object (mode-aware)
    plan = build_plan_for_mode(verb_name)

    # 2. Logic: Build the Request Object (Strict Type Check)
    request_object = PlanExecutionRequest(
        plan=plan,
        args=PlanExecutionArgs(
            request_id=str(args.get("request_id") or args.get("id") or correlation_id),
            user_id=args.get("user_id"),
            trigger_source=str(args.get("trigger_source") or "cortex-orch"),
            extra=args
        ),
        context={**(context or {}), **args}
    )

    # 3. Transport: Delegate to Client
    client = CortexExecClient(bus, exec_request_channel)
    
    return await client.execute_plan(
        source=source,
        req=request_object,
        correlation_id=correlation_id,
        timeout_sec=timeout_sec
    )
