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

logger = logging.getLogger("orion.cortex.orch")


# Locate cognition/verbs from the installed `orion` package
ORION_PKG_DIR = Path(orion.__file__).resolve().parent
VERBS_DIR = ORION_PKG_DIR / "cognition" / "verbs"


# ─────────────────────────────────────────────────────────────
# Public API models (hub -> cortex-orch)
# ─────────────────────────────────────────────────────────────

class CortexOrchInput(BaseModel):
    """Hub-facing request shape (legacy + future-proof).

    Hub historically publishes a raw dict (not an envelope) with keys:
      {verb_name, args, reply_channel, correlation_id}

    We validate it here and normalize into an execution request for cortex-exec.
    """

    model_config = ConfigDict(extra="ignore")

    verb_name: str
    args: Dict[str, Any] = Field(default_factory=dict)
    origin_node: str = "unknown"
    context: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────
# Execution plan schema (sent to cortex-exec)
# NOTE: kept local for now; we'll promote into orion.core.cortex later.
# ─────────────────────────────────────────────────────────────

class ExecutionStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verb_name: str
    step_name: str
    description: str = ""
    order: int
    services: List[str]
    prompt_template: Optional[str] = None
    requires_gpu: bool = False
    requires_memory: bool = False
    timeout_ms: int = 120000


class ExecutionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verb_name: str
    label: str = ""
    description: str = ""
    category: str = "general"
    priority: str = "normal"
    interruptible: bool = True
    can_interrupt_others: bool = False
    timeout_ms: int = 120000
    max_recursion_depth: int = 2
    steps: List[ExecutionStep]
    blocked: bool = False
    blocked_reason: Optional[str] = None
    metadata: Dict[str, str] = Field(default_factory=dict)


def _load_verb_yaml(verb_name: str) -> dict:
    path = VERBS_DIR / f"{verb_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No verb YAML found for '{verb_name}' at {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_plan_for_verb(verb_name: str) -> ExecutionPlan:
    data = _load_verb_yaml(verb_name)

    label = str(data.get("label") or data.get("name") or verb_name)
    description = str(data.get("description") or "")
    category = str(data.get("category") or "general")
    priority = str(data.get("priority") or "normal")
    interruptible = bool(data.get("interruptible", True))
    can_interrupt_others = bool(data.get("can_interrupt_others", False))
    timeout_ms = int(data.get("timeout_ms", 120000) or 120000)
    max_recursion_depth = int(data.get("max_recursion_depth", 2) or 2)

    default_services = list(data.get("services") or [])
    default_prompt_template = str(data.get("prompt_template") or "")
    default_requires_gpu = bool(data.get("requires_gpu", False))
    default_requires_memory = bool(data.get("requires_memory", False))

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
                    prompt_template=str(s.get("prompt_template") or default_prompt_template) or None,
                    requires_gpu=bool(s.get("requires_gpu", default_requires_gpu)),
                    requires_memory=bool(s.get("requires_memory", default_requires_memory)),
                    timeout_ms=int(s.get("timeout_ms", timeout_ms) or timeout_ms),
                )
            )
    else:
        # single-step verb
        steps.append(
            ExecutionStep(
                verb_name=verb_name,
                step_name=verb_name,
                description=description,
                order=0,
                services=default_services,
                prompt_template=default_prompt_template or None,
                requires_gpu=default_requires_gpu,
                requires_memory=default_requires_memory,
                timeout_ms=timeout_ms,
            )
        )

    return ExecutionPlan(
        verb_name=verb_name,
        label=label,
        description=description,
        category=category,
        priority=priority,
        interruptible=interruptible,
        can_interrupt_others=can_interrupt_others,
        timeout_ms=timeout_ms,
        max_recursion_depth=max_recursion_depth,
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
    """Build plan + RPC to cortex-exec, returning raw dict payload."""

    plan = build_plan_for_verb(verb_name)

    reply = f"orion-cortex-exec:result:{uuid4()}"

    exec_payload = {
        "plan": plan.model_dump(mode="json"),
        "args": {
            "request_id": args.get("request_id") or args.get("id") or correlation_id,
            "user_id": args.get("user_id"),
            "trigger_source": args.get("trigger_source") or "cortex-orch",
            "extra": args,
        },
        "context": {**(context or {}), **args},
    }

    env = BaseEnvelope(
        kind="cortex.exec.request",
        source=source,
        correlation_id=correlation_id,
        reply_to=reply,
        payload=exec_payload,
    )

    msg = await bus.rpc_request(exec_request_channel, env, reply_channel=reply, timeout_sec=timeout_sec)
    decoded = bus.codec.decode(msg.get("data"))
    if not decoded.ok:
        return {"ok": False, "error": decoded.error or "decode_failed"}
    if isinstance(decoded.envelope.payload, dict):
        return decoded.envelope.payload
    return decoded.envelope.model_dump(mode="json")
