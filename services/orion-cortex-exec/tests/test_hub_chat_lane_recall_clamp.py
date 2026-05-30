from __future__ import annotations

from app.recall_utils import (
    apply_hub_chat_lane_recall_clamp,
    delivery_safe_recall_decision,
    hub_chat_lane_from_ctx,
)
from orion.schemas.cortex.schemas import ExecutionStep


def _chat_general_step() -> ExecutionStep:
    return ExecutionStep(
        verb_name="chat_general",
        step_name="llm_chat_general",
        description="",
        order=0,
        services=["LLMGatewayService"],
        requires_memory=True,
    )


def test_hub_chat_lane_from_ctx_reads_nested_surface_context() -> None:
    lane = hub_chat_lane_from_ctx(
        {
            "metadata": {
                "surface_context": {"hub_chat_lane": "grounded_small"},
            }
        }
    )
    assert lane == "grounded_small"


def test_apply_hub_chat_lane_clamp_grounded_small_to_assist_light() -> None:
    profile, source = apply_hub_chat_lane_recall_clamp(
        recall_cfg={"profile": "recall.v1"},
        profile="chat.general.v1",
        profile_source="verb",
        hub_chat_lane="grounded_small",
    )
    assert profile == "assist.light.v1"
    assert source == "hub_chat_lane_grounded_small"


def test_apply_hub_chat_lane_clamp_respects_profile_explicit() -> None:
    profile, source = apply_hub_chat_lane_recall_clamp(
        recall_cfg={"profile": "reflect.v1", "profile_explicit": True},
        profile="reflect.v1",
        profile_source="explicit",
        hub_chat_lane="grounded_small",
    )
    assert profile == "reflect.v1"
    assert source == "explicit"


def test_delivery_safe_recall_decision_uses_lane_clamp_for_chat_general() -> None:
    decision = delivery_safe_recall_decision(
        {"enabled": True, "profile": "recall.v1"},
        [_chat_general_step()],
        verb_profile="chat.general.v1",
        plan_verb_name="chat_general",
        hub_chat_lane="grounded_small",
    )
    assert decision["profile"] == "assist.light.v1"
    assert decision["profile_source"] == "hub_chat_lane_grounded_small"


def test_delivery_safe_recall_decision_brain_lane_maps_to_recall_v1() -> None:
    decision = delivery_safe_recall_decision(
        {"enabled": True},
        [_chat_general_step()],
        verb_profile="chat.general.v1",
        plan_verb_name="chat_general",
        hub_chat_lane="brain",
    )
    assert decision["profile"] == "recall.v1"
    assert decision["profile_source"] == "hub_chat_lane_brain"
