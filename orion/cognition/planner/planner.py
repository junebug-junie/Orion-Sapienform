# orion-cognition/planner/planner.py

from typing import Optional, List
from pathlib import Path

from .models import (
    VerbConfig,
    SystemState,
    ExecutionPlan,
    ExecutionStep,
)
from .loader import VerbRegistry
from .prompt_renderer import PromptRenderer


class SemanticPlanner:
    """
    Semantic Planner
    -----------------
    - Reads Verb definitions from JSON (via VerbRegistry)
    - Applies safety rules based on SystemState
    - Constructs an ExecutionPlan (steps + metadata) for the router

    Router responsibility:
        - Actually execute steps in order
        - Route each step to the correct service/node
    """

    def __init__(self, verbs_dir: Path, prompts_dir: Path):
        self.registry = VerbRegistry(verbs_dir)
        self.prompt_renderer = PromptRenderer(prompts_dir)

    # -------- Public API -------- #

    def build_plan(
        self,
        verb_name: str,
        system_state: Optional[SystemState] = None,
        current_recursion_depth: int = 0,
    ) -> ExecutionPlan:
        """
        Build an ExecutionPlan for a given verb and system state.

        Raises:
            - KeyError if verb not found
            - ValueError if recursion depth exceeded
        """
        verb = self.registry.get(verb_name)

        # Enforce recursion constraints
        self._check_recursion_depth(verb, current_recursion_depth)

        # Evaluate safety rules
        blocked, reason = self._evaluate_safety(verb, system_state)

        # Build steps (even if blocked, so caller can inspect plan)
        steps = self._build_steps(verb)

        return ExecutionPlan(
            verb_name=verb.name,
            label=verb.display_label(),
            description=verb.description,
            category=verb.category,
            priority=verb.priority,
            interruptible=verb.interruptible,
            can_interrupt_others=verb.can_interrupt_others,
            timeout_ms=verb.timeout_ms,
            max_recursion_depth=verb.max_recursion_depth,
            steps=steps,
            blocked=blocked,
            blocked_reason=reason,
            system_state=system_state,
            metadata={},
        )

    # -------- Internal helpers -------- #

    def _check_recursion_depth(
        self,
        verb: VerbConfig,
        current_recursion_depth: int,
    ) -> None:
        """
        Ensure we don't exceed the verb's max_recursion_depth.
        """
        if current_recursion_depth > verb.max_recursion_depth:
            raise ValueError(
                f"Recursion depth {current_recursion_depth} exceeds "
                f"max_recursion_depth={verb.max_recursion_depth} "
                f"for verb '{verb.name}'."
            )

    def _evaluate_safety(
        self,
        verb: VerbConfig,
        system_state: Optional[SystemState],
    ) -> tuple[bool, Optional[str]]:
        """
        Evaluate safety rules for a verb against the current system state.

        Returns:
            (blocked: bool, reason: Optional[str])
        """
        if not system_state:
            return False, None

        # Simple, generic rule: any safety rule with applies_when_state
        # that matches the current state => block the verb.
        for rule in verb.safety_rules:
            if rule.applies_when_state and rule.applies_when_state == system_state.name:
                reason = (
                    f"Verb '{verb.name}' blocked by safety rule "
                    f"'{rule.rule_name}' in system state '{system_state.name}'."
                )
                return True, reason

        return False, None

    def _build_steps(self, verb: VerbConfig) -> List[ExecutionStep]:
        """
        Turn the VerbConfig.plan into ordered ExecutionSteps.
        """
        # Sort steps by 'order' just in case
        sorted_steps = sorted(verb.plan, key=lambda s: s.order)

        steps: List[ExecutionStep] = []

        for step_cfg in sorted_steps:
            # If step doesn't specify a prompt_template, inherit the verb's.
            prompt_template = step_cfg.prompt_template or verb.prompt_template

            step = ExecutionStep(
                verb_name=verb.name,
                step_name=step_cfg.name,
                description=step_cfg.description,
                order=step_cfg.order,
                services=step_cfg.services,
                prompt_template=prompt_template,
                requires_gpu=step_cfg.requires_gpu,
                requires_memory=step_cfg.requires_memory,
            )
            steps.append(step)

        return steps

    def render_prompt(
        self,
        prompt_template: str,
        context: dict
    ) -> str:
        return self.prompt_renderer.render(prompt_template, context)
