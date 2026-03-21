from __future__ import annotations

from app.recall_utils import delivery_safe_recall_decision
from orion.schemas.cortex.schemas import ExecutionStep


def _memory_step() -> ExecutionStep:
    return ExecutionStep(
        verb_name="agent_runtime",
        step_name="planner_react",
        description="planner",
        order=0,
        services=["PlannerReactService"],
        requires_memory=True,
    )


def test_delivery_oriented_asks_disable_default_reflective_recall() -> None:
    decision = delivery_safe_recall_decision(
        {"enabled": True},
        [_memory_step()],
        output_mode="implementation_guide",
        verb_profile=None,
    )

    assert decision["run_recall"] is False
    assert decision["reason"] == "delivery_safe_default_disabled"
    assert decision["recall_gating_reason"] == "delivery_safe_default_disabled"


def test_delivery_required_recall_switches_to_assist_light_profile() -> None:
    decision = delivery_safe_recall_decision(
        {"enabled": True, "required": True},
        [_memory_step()],
        output_mode="implementation_guide",
        verb_profile=None,
    )

    assert decision["run_recall"] is True
    assert decision["profile"] == "assist.light.v1"
    assert decision["profile_source"] == "delivery_safe_default"


def test_reflective_modes_keep_normal_recall_behavior() -> None:
    decision = delivery_safe_recall_decision(
        {"enabled": True},
        [_memory_step()],
        output_mode="reflective_depth",
        verb_profile=None,
    )

    assert decision["run_recall"] is True
    assert decision["profile"] == "reflect.v1"
    assert decision["reason"] == "enabled"
