from __future__ import annotations

from types import SimpleNamespace

import pytest

from app import chat_stance
from app.chat_stance import _project_self_state_from_beliefs
from orion.substrate.relational.beliefs import AnchorBeliefSliceV1, UnifiedRelationalBeliefSetV1


def _node(label: str, metadata: dict) -> SimpleNamespace:
    return SimpleNamespace(node_kind="concept", label=label, metadata=metadata)


def _beliefs_with_self_nodes(overall_condition: str, *, pressure_score: float = 0.2) -> SimpleNamespace:
    orion_items = [
        _node(
            "self:overall_condition",
            {"overall_condition": overall_condition, "trajectory_condition": "stable", "prediction_error": 0.1},
        ),
        _node(
            "self:execution_pressure",
            {"self_dimension_id": "execution_pressure", "score": pressure_score, "trajectory": 0.0, "prediction_error": 0.1},
        ),
    ]
    anchor_slice = SimpleNamespace(concepts=orion_items, tensions=[], goals=[], drives=[], snapshots=[], events=[], degraded=False, tier_outcomes=[])
    return SimpleNamespace(anchors={"orion": anchor_slice})


@pytest.mark.parametrize("condition", ["quiet", "steady"])
def test_quiet_and_steady_produce_no_hazard(condition):
    beliefs = _beliefs_with_self_nodes(condition)
    result = _project_self_state_from_beliefs(beliefs, {})
    assert result is None or not result.get("hazards")


@pytest.mark.parametrize("condition", ["strained", "unstable"])
def test_strained_and_unstable_produce_hazard(condition):
    beliefs = _beliefs_with_self_nodes(condition)
    result = _project_self_state_from_beliefs(beliefs, {})
    assert result is not None
    assert result["overall_condition"] == condition
    assert result.get("hazards")


def test_high_single_dimension_pressure_produces_hazard_even_when_overall_steady():
    beliefs = _beliefs_with_self_nodes("steady", pressure_score=0.95)
    result = _project_self_state_from_beliefs(beliefs, {})
    assert result is not None
    assert any("execution_pressure" in h for h in result.get("hazards", []))


def test_none_beliefs_returns_none():
    assert _project_self_state_from_beliefs(None, {}) is None


def test_no_self_nodes_returns_none():
    anchor_slice = SimpleNamespace(concepts=[], tensions=[], goals=[], drives=[], snapshots=[], events=[], degraded=False, tier_outcomes=[])
    beliefs = SimpleNamespace(anchors={"orion": anchor_slice})
    assert _project_self_state_from_beliefs(beliefs, {}) is None


def _real_beliefs_with_self_nodes(overall_condition: str) -> UnifiedRelationalBeliefSetV1:
    """Build a real UnifiedRelationalBeliefSetV1 (not a SimpleNamespace) for the
    integration test, since build_chat_stance_inputs touches beliefs.cold_anchors /
    beliefs.degraded_producers / beliefs.lineage directly."""
    orion_items = [
        _node(
            "self:overall_condition",
            {"overall_condition": overall_condition, "trajectory_condition": "stable", "prediction_error": 0.1},
        ),
    ]
    anchor_slice = AnchorBeliefSliceV1(anchor="orion", concepts=orion_items)
    return UnifiedRelationalBeliefSetV1(anchors={"orion": anchor_slice})


@pytest.mark.asyncio
async def test_build_chat_stance_inputs_folds_self_state_hazard_when_strained(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat_stance, "_unified_beliefs_for_stance", lambda ctx: _real_beliefs_with_self_nodes("strained"))

    ctx = {"user_message": "hello"}
    built = await chat_stance.build_chat_stance_inputs(ctx)

    assert any(h.startswith("self_state overall_condition=strained") for h in built["social"]["hazards"])
    assert ctx["chat_self_state_condition"] == "strained"


@pytest.mark.asyncio
async def test_build_chat_stance_inputs_no_self_state_hazard_when_steady(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat_stance, "_unified_beliefs_for_stance", lambda ctx: _real_beliefs_with_self_nodes("steady"))

    ctx = {"user_message": "hello"}
    built = await chat_stance.build_chat_stance_inputs(ctx)

    assert not any(h.startswith("self_state overall_condition=") for h in built["social"]["hazards"])
