from __future__ import annotations

import pytest

from app import chat_stance
from app.chat_stance import _project_context_provenance_hazard
from orion.substrate.relational.beliefs import AnchorBeliefSliceV1, UnifiedRelationalBeliefSetV1


def test_no_hazard_when_no_live_runtime_keys_present():
    ctx = {"user_message": "hello", "orion_identity_summary": ["I am Oríon."]}
    assert _project_context_provenance_hazard(ctx) is None


def test_hazard_names_live_runtime_keys_present():
    ctx = {"self_state": {"overall_condition": "steady"}, "attention_broadcast": {}}
    hazard = _project_context_provenance_hazard(ctx)
    assert hazard is not None
    assert "attention_broadcast" in hazard
    assert "self_state" in hazard
    assert "live" in hazard


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
