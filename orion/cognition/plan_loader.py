from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml

import orion
from orion.schemas.cortex.schemas import ExecutionPlan
from orion.schemas.cortex.types import ExecutionStep

ORION_PKG_DIR = Path(orion.__file__).resolve().parent
VERBS_DIR = ORION_PKG_DIR / "cognition" / "verbs"
PROMPTS_DIR = ORION_PKG_DIR / "cognition" / "prompts"


def load_verb_yaml(verb_name: str) -> dict[str, Any]:
    path = VERBS_DIR / f"{verb_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No verb YAML found for '{verb_name}' at {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_prompt_template(template_ref: Optional[str]) -> Optional[str]:
    if not template_ref:
        return None

    value = str(template_ref).strip()
    if not value:
        return None

    if value.endswith(".j2"):
        prompt_path = PROMPTS_DIR / value
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
    return value


def build_plan_for_verb(verb_name: str, *, mode: str = "brain") -> ExecutionPlan:
    data = load_verb_yaml(verb_name)

    timeout_ms = int(data.get("timeout_ms", 120000) or 120000)
    default_services = list(data.get("services") or [])
    verb_recall_profile = data.get("recall_profile")
    default_prompt = load_prompt_template(str(data.get("prompt_template") or ""))

    steps: list[ExecutionStep] = []
    raw_steps = data.get("steps") or data.get("plan")

    if isinstance(raw_steps, list) and raw_steps:
        for i, step in enumerate(raw_steps):
            step_template_ref = str(step.get("prompt_template") or "")
            step_prompt = load_prompt_template(step_template_ref) if step_template_ref else default_prompt
            steps.append(
                ExecutionStep(
                    verb_name=verb_name,
                    step_name=str(step.get("name") or f"step_{i}"),
                    description=str(step.get("description") or ""),
                    order=int(step.get("order", i)),
                    services=list(step.get("services") or default_services),
                    prompt_template=step_prompt,
                    requires_gpu=bool(step.get("requires_gpu", False)),
                    requires_memory=bool(step.get("requires_memory", False)),
                    timeout_ms=int(step.get("timeout_ms", timeout_ms) or timeout_ms),
                    recall_profile=step.get("recall_profile"),
                )
            )
    else:
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
                recall_profile=data.get("recall_profile"),
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
        metadata={
            "verb_yaml": f"{verb_name}.yaml",
            "mode": mode,
            "recall_profile": str(verb_recall_profile) if verb_recall_profile else "",
        },
    )
