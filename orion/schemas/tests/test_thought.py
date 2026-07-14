from __future__ import annotations

from datetime import datetime, timezone

from orion.schemas.thought import AutonomySliceV1, StanceHarnessSliceV1, ThoughtEventV1


def _base_thought(**overrides: object) -> ThoughtEventV1:
    kwargs = dict(
        event_id="t-1",
        correlation_id="c-1",
        session_id=None,
        created_at=datetime.now(timezone.utc),
        imperative="Stay present; one situated question.",
        tone="warm",
        strain_refs=["loop-1"],
        evidence_refs=["loop-1"],
        stance_harness_slice=StanceHarnessSliceV1(
            task_mode="reflective_dialogue",
            conversation_frame="reflective",
            response_priorities=["companion_presence"],
            answer_strategy="RelationalHoldSpace",
        ),
    )
    kwargs.update(overrides)
    return ThoughtEventV1(**kwargs)


def test_thought_event_validates_without_autonomy_slice() -> None:
    """Back-compat: existing constructors that omit autonomy_slice still validate."""
    thought = _base_thought()
    assert thought.autonomy_slice is None
    assert thought.grounding_capsule is None


def test_thought_event_accepts_autonomy_slice() -> None:
    slice_ = AutonomySliceV1(
        dominant_drive="curiosity",
        active_tensions=["novelty_vs_stability", "autonomy_vs_connection"],
        pressure_trend="rising",
        confidence=0.72,
    )
    thought = _base_thought(autonomy_slice=slice_)
    assert thought.autonomy_slice is not None
    assert thought.autonomy_slice.dominant_drive == "curiosity"
    assert thought.autonomy_slice.active_tensions == [
        "novelty_vs_stability",
        "autonomy_vs_connection",
    ]


def test_autonomy_slice_v1_defaults() -> None:
    slice_ = AutonomySliceV1()
    assert slice_.schema_version == "autonomy.slice.v1"
    assert slice_.dominant_drive is None
    assert slice_.active_tensions == []
    assert slice_.pressure_trend is None
    assert slice_.confidence is None
    assert slice_.recent_actions == []


def test_autonomy_slice_v1_round_trips_through_json() -> None:
    slice_ = AutonomySliceV1(
        dominant_drive="stability",
        active_tensions=["load_vs_recovery"],
        pressure_trend="falling",
        confidence=0.41,
    )
    dumped = slice_.model_dump(mode="json")
    assert dumped == {
        "schema_version": "autonomy.slice.v1",
        "dominant_drive": "stability",
        "active_tensions": ["load_vs_recovery"],
        "pressure_trend": "falling",
        "confidence": 0.41,
        "recent_actions": [],
    }
    restored = AutonomySliceV1.model_validate(dumped)
    assert restored == slice_


def test_autonomy_slice_v1_recent_actions_round_trips_through_json() -> None:
    """Acceptance check for the P4 stance_react dispatch-evidence patch: a
    payload carrying the new recent_actions field survives
    dict -> JSON -> dict -> model unchanged, proving router.py's metadata
    map-on and orion-thought/app/bus_listener.py's _extract_autonomy_slice
    need zero code changes to carry this field end to end.
    """
    slice_ = AutonomySliceV1(
        dominant_drive="coherence",
        active_tensions=["unresolved_thread"],
        pressure_trend="rising",
        confidence=0.63,
        recent_actions=["inspect: checked substrate graph health"],
    )
    dumped = slice_.model_dump(mode="json")
    assert dumped == {
        "schema_version": "autonomy.slice.v1",
        "dominant_drive": "coherence",
        "active_tensions": ["unresolved_thread"],
        "pressure_trend": "rising",
        "confidence": 0.63,
        "recent_actions": ["inspect: checked substrate graph health"],
    }
    restored = AutonomySliceV1.model_validate(dumped)
    assert restored == slice_
    assert restored.recent_actions == ["inspect: checked substrate graph health"]


def test_thought_event_with_autonomy_slice_round_trips_through_json() -> None:
    thought = _base_thought(
        autonomy_slice=AutonomySliceV1(
            dominant_drive="curiosity",
            active_tensions=["novelty_vs_stability"],
            pressure_trend="stable",
            confidence=0.5,
        )
    )
    dumped = thought.model_dump(mode="json")
    restored = ThoughtEventV1.model_validate(dumped)
    assert restored.autonomy_slice == thought.autonomy_slice
