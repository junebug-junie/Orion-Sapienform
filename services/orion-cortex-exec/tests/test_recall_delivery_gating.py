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


def test_client_recall_on_runs_without_requires_memory_steps() -> None:
    from app.recall_utils import should_run_recall

    step = ExecutionStep(
        verb_name="chat_quick",
        step_name="llm_chat_quick",
        description="chat",
        order=0,
        services=["LLMGatewayService"],
        requires_memory=False,
    )
    assert should_run_recall({"enabled": True}, [step]) == (True, "enabled_client_explicit")


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


def test_concrete_ops_query_disables_default_reflective_recall() -> None:
    decision = delivery_safe_recall_decision(
        {"enabled": True},
        [_memory_step()],
        output_mode="direct_answer",
        verb_profile=None,
        user_text="Need runtime estimate for V100 on APC UPS battery backup and power draw.",
    )

    assert decision["run_recall"] is False
    assert decision["profile"] == "assist.light.v1"
    assert decision["reason"] == "concrete_ops_default_disabled"


def test_agent_mode_uses_chat_general_profile_when_inherited_reflect_arrives() -> None:
    decision = delivery_safe_recall_decision(
        {"enabled": True, "profile": "reflect.v1", "mode": "hybrid"},
        [_memory_step()],
        output_mode="direct_answer",
        verb_profile=None,
        runtime_mode="agent",
        user_text="yep",
    )

    assert decision["run_recall"] is True
    assert decision["profile"] == "chat.general.v1"
    assert decision["profile_source"] == "runtime_mode"
