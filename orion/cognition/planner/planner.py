from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import logging
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Core data models
# -------------------------------------------------------------------


@dataclass
class SystemState:
    """
    Lightweight representation of Orion's current mode/context.
    You can extend this later as needed.
    """

    name: str = "Idle"
    mode: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionStep:
    """
    One step in an execution plan derived from a verb YAML.
    """

    verb_name: str
    step_name: str
    description: str
    order: int
    services: List[str] = field(default_factory=list)
    prompt_template: Optional[str] = None
    requires_gpu: bool = False
    requires_memory: bool = False
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionPlan:
    """
    Structured plan for executing a single cognitive verb.
    Mirrors your original fields closely so RDF, etc. can hook into it later.
    """

    verb_name: str
    label: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    priority: Optional[str] = None
    interruptible: bool = True
    can_interrupt_others: bool = False
    timeout_ms: int = 0
    max_recursion_depth: int = 0
    steps: List[ExecutionStep] = field(default_factory=list)
    blocked: bool = False
    blocked_reason: Optional[str] = None
    system_state: Optional[SystemState] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verb_name": self.verb_name,
            "label": self.label,
            "description": self.description,
            "category": self.category,
            "priority": self.priority,
            "interruptible": self.interruptible,
            "can_interrupt_others": self.can_interrupt_others,
            "timeout_ms": self.timeout_ms,
            "max_recursion_depth": self.max_recursion_depth,
            "blocked": self.blocked,
            "blocked_reason": self.blocked_reason,
            "system_state": self.system_state,
            "steps": [
                {
                    "verb_name": s.verb_name,
                    "step_name": s.step_name,
                    "description": s.description,
                    "order": s.order,
                    "services": s.services,
                    "prompt_template": s.prompt_template,
                    "requires_gpu": s.requires_gpu,
                    "requires_memory": s.requires_memory,
                    "raw": s.raw,
                }
                for s in self.steps
            ],
            "metadata": self.metadata,
        }


# -------------------------------------------------------------------
# Semantic Planner
# -------------------------------------------------------------------


class SemanticPlanner:
    """
    Semantic Planner

    This version:
    - Reads verb definitions directly from verbs/*.yaml
    - Applies a simple safety rule check based on `safety_rules` in YAML
    - Constructs an ExecutionPlan (steps + metadata) for the router
    - Provides a prompt renderer backed by Jinja2 templates
    """

    def __init__(self, verbs_dir: Path, prompts_dir: Path):
        self.verbs_dir = Path(verbs_dir)
        self.prompts_dir = Path(prompts_dir)

        if not self.verbs_dir.exists():
            raise FileNotFoundError(f"verbs directory not found: {self.verbs_dir}")
        if not self.prompts_dir.exists():
            logger.warning("prompts directory does not exist: %s", self.prompts_dir)

        self.env = Environment(
            loader=FileSystemLoader(str(self.prompts_dir)),
            autoescape=select_autoescape(enabled_extensions=("j2",)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    # -------- Internal helpers -------- #

    def _load_verb_config(self, verb_name: str) -> Dict[str, Any]:
        path = self.verbs_dir / f"{verb_name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Verb config not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Normalize name if missing
        data.setdefault("name", verb_name)
        return data

    def _evaluate_safety(
        self,
        verb_def: Dict[str, Any],
        system_state: Optional[SystemState],
    ) -> Tuple[bool, Optional[str]]:
        """
        Simple generic safety rule evaluation.

        Expects safety_rules in YAML like:
          safety_rules:
            - rule_name: "block_in_state_X"
              applies_when_state: "X"
        """
        if system_state is None:
            return False, None

        rules = verb_def.get("safety_rules", []) or []
        for rule in rules:
            applies_state = rule.get("applies_when_state")
            if applies_state and applies_state == system_state.name:
                reason = (
                    f"Verb '{verb_def.get('name')}' blocked by safety rule "
                    f"'{rule.get('rule_name', 'unnamed')}' in system state "
                    f"'{system_state.name}'."
                )
                return True, reason

        return False, None

    def _build_steps(self, verb_def: Dict[str, Any]) -> List[ExecutionStep]:
        """
        Turn the verb_def['plan'] list into ordered ExecutionSteps.
        """
        plan_defs = verb_def.get("plan", []) or []
        verb_name = verb_def.get("name", "")

        # Sort steps by 'order'
        sorted_defs = sorted(plan_defs, key=lambda s: s.get("order", 0))

        steps: List[ExecutionStep] = []
        default_prompt_template = verb_def.get("prompt_template")

        for step_def in sorted_defs:
            step_prompt = step_def.get("prompt_template") or default_prompt_template

            step = ExecutionStep(
                verb_name=verb_name,
                step_name=step_def.get("name", ""),
                description=step_def.get("description", ""),
                order=int(step_def.get("order", 0)),
                services=step_def.get("services", []) or [],
                prompt_template=step_prompt,
                requires_gpu=bool(step_def.get("requires_gpu", False)),
                requires_memory=bool(step_def.get("requires_memory", False)),
                raw=step_def,
            )
            steps.append(step)

        return steps

    # -------- Public API -------- #

    def build_plan(
        self,
        verb_name: str,
        system_state: Optional[SystemState] = None,
        current_recursion_depth: int = 0,
    ) -> ExecutionPlan:
        """
        Build an ExecutionPlan for a given verb and system state.

        This intentionally mirrors the original signature so you can
        drop it into the Execution Cortex later.
        """
        verb_def = self._load_verb_config(verb_name)

        # Recursion control (YAML can define max_recursion_depth)
        max_depth = int(verb_def.get("max_recursion_depth", 0) or 0)
        if max_depth and current_recursion_depth > max_depth:
            raise ValueError(
                f"Recursion depth {current_recursion_depth} exceeds "
                f"max_recursion_depth={max_depth} for verb '{verb_def.get('name')}'."
            )

        blocked, reason = self._evaluate_safety(verb_def, system_state)
        steps = self._build_steps(verb_def)

        plan = ExecutionPlan(
            verb_name=verb_def.get("name"),
            label=verb_def.get("label"),
            description=verb_def.get("description"),
            category=verb_def.get("category"),
            priority=verb_def.get("priority"),
            interruptible=bool(verb_def.get("interruptible", True)),
            can_interrupt_others=bool(verb_def.get("can_interrupt_others", False)),
            timeout_ms=int(verb_def.get("timeout_ms", 0) or 0),
            max_recursion_depth=max_depth,
            steps=steps,
            blocked=blocked,
            blocked_reason=reason,
            system_state=system_state,
            metadata={"raw": verb_def},
        )

        return plan

    def render_prompt(self, prompt_template: str, context: Dict[str, Any]) -> str:
        """Render a Jinja2 prompt template with the given context."""
        template = self.env.get_template(prompt_template)
        return template.render(**context)


__all__ = [
    "SemanticPlanner",
    "SystemState",
    "ExecutionPlan",
    "ExecutionStep",
]
