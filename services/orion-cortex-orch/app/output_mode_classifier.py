"""Output mode classifier for answer depth routing.

Classifies user requests into output_mode and response_profile so that
delivery-oriented requests route toward concrete artifacts (guides, tutorials,
code) instead of executive scaffolding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from orion.schemas.cortex.contracts import (
    OutputMode,
    OutputModeDecisionV1,
    ResponseProfile,
)

if TYPE_CHECKING:
    pass

# Patterns that indicate direct answer / tutorial / implementation requests
INSTRUCTION_TERMS = {
    "how do i",
    "how to",
    "how can i",
    "how does",
    "instructions",
    "steps to",
    "guide",
    "tutorial",
    "walkthrough",
    "deploy",
    "deployment",
    "setup",
    "install",
    "configure",
    "configured",
    "get started",
    "getting started",
}

COMPARE_TERMS = {
    "compare",
    "comparison",
    "vs ",
    " versus ",
    "difference between",
    "different from",
}

DECISION_TERMS = {
    "decide",
    "deciding",
    "should i",
    "whether to",
    "help me choose",
    "recommend",
}

CODE_TERMS = {
    "code",
    "scaffold",
    "snippet",
    "implementation",
    "write the",
    "generate code",
}

DEBUG_TERMS = {
    "debug",
    "troubleshoot",
    "diagnose",
    "fix",
    "error",
    "exception",
}

REFLECTIVE_TERMS = {
    "explain the architecture",
    "architecture tradeoffs",
    "tradeoffs",
    "trade-offs",
    "design decision",
    "why does",
    "synthesize",
}

PROJECT_PLANNING_TERMS = {
    "plan",
    "roadmap",
    "project plan",
    "sprint",
    "milestones",
}

# Meta-plan phrases that indicate we should NOT match (user is asking about planning)
META_PLANNING_TERMS = {
    "plan to",
    "plan for",
}


def _normalize(text: str) -> str:
    return (text or "").strip().lower()


def classify_output_mode(user_text: str) -> OutputModeDecisionV1:
    """Classify user request into output_mode and response_profile."""
    text = _normalize(user_text)
    if not text:
        return OutputModeDecisionV1(
            output_mode="direct_answer",
            response_profile="direct_answer",
            direct_answer_bypass_used=False,
        )

    # Comparative analysis
    if any(term in text for term in COMPARE_TERMS):
        return OutputModeDecisionV1(
            output_mode="comparative_analysis",
            response_profile="reflective_depth",
            direct_answer_bypass_used=True,
        )

    # Decision support
    if any(term in text for term in DECISION_TERMS):
        return OutputModeDecisionV1(
            output_mode="decision_support",
            response_profile="reflective_depth",
            direct_answer_bypass_used=True,
        )

    # Debug / diagnosis
    if any(term in text for term in DEBUG_TERMS) and ("how" in text or "?" in text):
        return OutputModeDecisionV1(
            output_mode="debug_diagnosis",
            response_profile="technical_delivery",
            direct_answer_bypass_used=True,
        )

    # Code delivery
    if any(term in text for term in CODE_TERMS):
        return OutputModeDecisionV1(
            output_mode="code_delivery",
            response_profile="technical_delivery",
            direct_answer_bypass_used=True,
        )

    # Reflective / architect
    if any(term in text for term in REFLECTIVE_TERMS):
        return OutputModeDecisionV1(
            output_mode="reflective_depth",
            response_profile="architect",
            direct_answer_bypass_used=True,
        )

    # Implementation guide / tutorial (how-to, deploy, setup)
    if any(term in text for term in INSTRUCTION_TERMS):
        return OutputModeDecisionV1(
            output_mode="implementation_guide",
            response_profile="technical_delivery",
            direct_answer_bypass_used=True,
        )

    # Project planning (explicit planning request, not meta)
    if any(term in text for term in PROJECT_PLANNING_TERMS) and not any(
        m in text for m in META_PLANNING_TERMS
    ):
        return OutputModeDecisionV1(
            output_mode="project_planning",
            response_profile="technical_delivery",
            direct_answer_bypass_used=False,
        )

    # Simple question (short, has ?)
    if len(text) <= 120 and "?" in text:
        return OutputModeDecisionV1(
            output_mode="direct_answer",
            response_profile="direct_answer",
            direct_answer_bypass_used=True,
        )

    # Explain (direct answer)
    if text.startswith("explain") or text.startswith("what is") or text.startswith("why "):
        return OutputModeDecisionV1(
            output_mode="direct_answer",
            response_profile="direct_answer",
            direct_answer_bypass_used=True,
        )

    return OutputModeDecisionV1(
        output_mode="direct_answer",
        response_profile="direct_answer",
        direct_answer_bypass_used=False,
    )
