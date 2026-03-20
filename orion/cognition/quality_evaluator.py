"""Lightweight answer-quality evaluator.

Flags meta-plan or shallow outputs that should be rewritten before returning.
"""

from __future__ import annotations

import re
from typing import Tuple

# Phrases that indicate meta-plan / executive scaffolding rather than actual answer
META_PLAN_PHRASES = [
    r"gather\s+requirements",
    r"create\s+a\s+guide",
    r"review\s+and\s+refine",
    r"test\s+deployment",
    r"implement\s+the\s+solution",
    r"break\s+down\s+into\s+steps",
    r"outline\s+the\s+process",
    r"define\s+the\s+approach",
    r"analyze\s+the\s+requirements",
    r"develop\s+a\s+plan",
    r"clarified\s+goal",  # plan_action scaffold
    r"^\s*steps:\s*$",
]

META_PLAN_PATTERN = re.compile(
    "|".join(f"({p})" for p in META_PLAN_PHRASES),
    re.IGNORECASE,
)


def looks_like_meta_plan(text: str) -> bool:
    """Return True if output appears to be meta-planning rather than concrete answer."""
    if not text or not isinstance(text, str):
        return False
    return bool(META_PLAN_PATTERN.search(text))


def should_rewrite_for_instructional(
    text: str,
    output_mode: str | None,
) -> Tuple[bool, str]:
    """
    Return (should_rewrite, reason).
    For tutorial/implementation output modes, flag meta-plan for rewrite.
    """
    if not text:
        return False, ""
    instructional_modes = {
        "implementation_guide",
        "tutorial",
        "code_delivery",
        "direct_answer",
        "comparative_analysis",
        "decision_support",
        "reflective_depth",
        "debug_diagnosis",
        "project_planning",
    }
    if output_mode not in instructional_modes:
        return False, ""
    if looks_like_meta_plan(text):
        return True, "meta_plan_detected"
    # Short generic outputs that lack specifics
    if len(text.strip()) < 120 and not any(c in text for c in ["`", "1.", "2.", "Step", "```"]):
        return False, ""
    return False, ""
