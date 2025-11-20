from __future__ import annotations

from pathlib import Path

from .planner import SemanticPlanner, SystemState, ExecutionPlan


def create_default_planner(base_dir: Path | None = None) -> SemanticPlanner:
    """
    Convenience helper to construct a SemanticPlanner using the standard
    verbs/ and prompts/ directories relative to the cognition package.

    If base_dir is None, we infer it as the package root:
        orion/cognition/
    """
    if base_dir is None:
        # planner/__init__.py -> planner/ -> cognition/
        base_dir = Path(__file__).resolve().parents[1]

    verbs_dir = base_dir / "verbs"
    prompts_dir = base_dir / "prompts"

    return SemanticPlanner(verbs_dir=verbs_dir, prompts_dir=prompts_dir)


__all__ = [
    "SemanticPlanner",
    "SystemState",
    "ExecutionPlan",
    "create_default_planner",
]
