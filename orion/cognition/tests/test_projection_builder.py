from __future__ import annotations

from orion.cognition import projection_builder
from orion.cognition.projection_builder import (
    DEFAULT_PROJECTION_ANCHORS,
    build_projection_unification_registry,
    unified_beliefs_for_chat_stance,
)
from orion.substrate.relational.beliefs import AnchorBeliefSliceV1, UnifiedRelationalBeliefSetV1


def test_projection_registry_matches_expected_chat_stance_producers() -> None:
    registry = build_projection_unification_registry()
    producer_ids = [producer.producer_id for producer in registry.producers]

    assert producer_ids == [
        "identity_yaml",
        "self_study",
        "autonomy",
        "concept_induction",
        "spark",
        # self-model (higher-order rung) + substrate "felt state" reducer lanes
        "self_state",
        "biometrics",
        "execution",
        "transport",
        "orionmem",
        "recall",
        "social",
    ]


def test_unified_beliefs_for_chat_stance_uses_shared_context_path(monkeypatch) -> None:
    calls: list[dict] = []
    expected = UnifiedRelationalBeliefSetV1(
        anchors={anchor: AnchorBeliefSliceV1(anchor=anchor) for anchor in DEFAULT_PROJECTION_ANCHORS},
        lineage=["test:shared_projection_builder"],
    )

    def fake_unified_beliefs_for_context(ctx, **kwargs):
        calls.append({"ctx": ctx, "kwargs": kwargs})
        return expected

    monkeypatch.setattr(projection_builder, "unified_beliefs_for_context", fake_unified_beliefs_for_context)

    result = unified_beliefs_for_chat_stance({"verb": "chat_general", "correlation_id": "corr-1"}, timeout_sec=1.25)

    assert result is expected
    assert calls
    assert tuple(calls[0]["kwargs"]["anchors"]) == DEFAULT_PROJECTION_ANCHORS
    assert calls[0]["kwargs"]["timeout_sec"] == 1.25
    assert calls[0]["kwargs"]["publish_tier_outcomes"] is True
    assert calls[0]["ctx"]["correlation_id"] == "corr-1"
