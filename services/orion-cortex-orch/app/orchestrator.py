# services/orion-cortex-orch/app/orchestrator.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
import orion

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import ServiceRef

from orion.schemas.cortex.schemas import (
    ExecutionPlan,
    ExecutionStep,
    PlanExecutionRequest,
    PlanExecutionArgs
)
from .clients import CortexExecClient

logger = logging.getLogger("orion.cortex.orch")

ORION_PKG_DIR = Path(orion.__file__).resolve().parent
VERBS_DIR = ORION_PKG_DIR / "cognition" / "verbs"


def _load_verb_yaml(verb_name: str) -> dict:
    path = VERBS_DIR / f"{verb_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No verb YAML found for '{verb_name}' at {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_plan_for_verb(verb_name: str) -> ExecutionPlan:
    data = _load_verb_yaml(verb_name)

    # Defaults
    timeout_ms = int(data.get("timeout_ms", 120000) or 120000)
    default_services = list(data.get("services") or [])
    default_prompt = str(data.get("prompt_template") or "")

    steps: List[ExecutionStep] = []
    raw_steps = data.get("steps")

    if isinstance(raw_steps, list) and raw_steps:
        for i, s in enumerate(raw_steps):
            steps.append(
                ExecutionStep(
                    verb_name=verb_name,
                    step_name=str(s.get("name") or f"step_{i}"),
                    description=str(s.get("description") or ""),
                    order=int(s.get("order", i)),
                    services=list(s.get("services") or default_services),
                    prompt_template=str(s.get("prompt_template") or default_prompt) or None,
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
                prompt_template=default_prompt or None,
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
        timeout_ms=timeout_ms,
        steps=steps,
        metadata={"verb_yaml": f"{verb_name}.yaml"},
    )


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

    # 1. Logic: Build the Plan Object
    plan = build_plan_for_verb(verb_name)

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
