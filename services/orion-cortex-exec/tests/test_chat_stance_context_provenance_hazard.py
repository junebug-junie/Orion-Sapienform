from __future__ import annotations

from types import SimpleNamespace

import pytest

from app import chat_stance
from app.chat_stance import _project_context_provenance_hazard
from orion.substrate.relational.beliefs import AnchorBeliefSliceV1, UnifiedRelationalBeliefSetV1


def test_no_hazard_when_no_live_runtime_keys_present():
    ctx = {"user_message": "hello", "orion_identity_summary": ["I am Oríon."]}
    assert _project_context_provenance_hazard(ctx) is None


def test_hazard_fires_when_live_runtime_key_present():
    ctx = {"self_state": {"overall_condition": "steady"}, "attention_broadcast": {"selected_action_type": "watch"}}
    hazard = _project_context_provenance_hazard(ctx)
    assert hazard is not None
    assert "live_runtime_projection" in hazard
    assert len(hazard) <= 140  # must survive _unique()'s hard truncation


def test_hazard_none_when_live_runtime_key_present_but_empty():
    # An empty dict from a `computed_value or {}` fallback carries no real
    # evidence of live computation -- must not be classified as live.
    ctx = {"attention_broadcast": {}, "pad_frame": []}
    assert _project_context_provenance_hazard(ctx) is None


def test_hazard_ignores_plumbing_and_unregistered_keys():
    ctx = {"trace_id": "t-1", "correlation_id": "c-1", "some_unclassified_key": "x"}
    assert _project_context_provenance_hazard(ctx) is None


def _empty_beliefs() -> UnifiedRelationalBeliefSetV1:
    anchor_slice = AnchorBeliefSliceV1(anchor="orion", concepts=[])
    return UnifiedRelationalBeliefSetV1(anchors={"orion": anchor_slice})


@pytest.mark.asyncio
async def test_build_chat_stance_inputs_folds_provenance_hazard_when_live_key_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(chat_stance, "_unified_beliefs_for_stance", lambda ctx: _empty_beliefs())

    ctx = {"user_message": "hello", "self_state": {"overall_condition": "steady"}}
    built = await chat_stance.build_chat_stance_inputs(ctx)

    assert any(h.startswith("context_provenance:") for h in built["social"]["hazards"])


@pytest.mark.asyncio
async def test_build_chat_stance_inputs_no_provenance_hazard_without_live_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(chat_stance, "_unified_beliefs_for_stance", lambda ctx: _empty_beliefs())

    ctx = {"user_message": "hello"}
    built = await chat_stance.build_chat_stance_inputs(ctx)

    assert not any(h.startswith("context_provenance:") for h in built["social"]["hazards"])


def _beliefs_with_strained_self_state() -> UnifiedRelationalBeliefSetV1:
    node = SimpleNamespace(
        node_kind="concept",
        label="self:overall_condition",
        metadata={"overall_condition": "strained", "trajectory_condition": "stable", "prediction_error": 0.1},
    )
    anchor_slice = AnchorBeliefSliceV1(anchor="orion", concepts=[node])
    return UnifiedRelationalBeliefSetV1(anchors={"orion": anchor_slice})


@pytest.mark.asyncio
async def test_self_state_and_provenance_hazards_both_survive_when_social_hazards_fill_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: an earlier version prepended only the provenance hazard
    ahead of a list that could already be full, which could evict the
    self_state severity hazard folded in just before it. Both are standing
    safety/epistemic signals from this function's own reasoning and must
    both survive ahead of lower-stakes social hazards when the cap bites."""
    monkeypatch.setattr(chat_stance, "_unified_beliefs_for_stance", lambda ctx: _beliefs_with_strained_self_state())
    filler_hazards = [f"filler_social_hazard_{i}" for i in range(8)]
    monkeypatch.setattr(
        chat_stance,
        "_project_social_from_beliefs",
        lambda beliefs, ctx: ({"hazards": filler_hazards}, {}),
    )

    ctx = {"user_message": "hello", "self_state": {"overall_condition": "strained"}}
    built = await chat_stance.build_chat_stance_inputs(ctx)

    hazards = built["social"]["hazards"]
    assert len(hazards) == 8  # cap still enforced
    assert any(h.startswith("self_state overall_condition=strained") for h in hazards)
    assert any(h.startswith("context_provenance:") for h in hazards)
