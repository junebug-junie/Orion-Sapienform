"""Re-export shared classifier (implementation lives in orion.cognition)."""

from orion.cognition.output_mode_classifier import (  # noqa: F401
    classify_output_mode,
    preferred_render_style_from_classifier,
    INSTRUCTION_TERMS,
    COMPARE_TERMS,
    DECISION_TERMS,
    CODE_TERMS,
)
