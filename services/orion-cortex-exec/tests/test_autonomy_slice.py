from __future__ import annotations

from app.autonomy_slice import build_autonomy_slice
from orion.autonomy.models import AutonomyStateDeltaV1, AutonomyStateV2


def _state_v2(**overrides) -> dict:
    base = dict(
        subject="orion",
        model_layer="graph",
        entity_id="orion",
        dominant_drive="coherence",
        active_drives=["coherence", "continuity"],
        drive_pressures={"coherence": 0.72, "continuity": 0.41},
        tension_kinds=["drive_competition.coherence_continuity", "unresolved_thread", "identity_drift"],
        goal_headlines=[],
        source="reducer",
        confidence=0.63,
    )
    base.update(overrides)
    return AutonomyStateV2.model_validate(base).model_dump(mode="json")


def _delta(**overrides) -> dict:
    base = dict(subject="orion", new_tensions=["fresh_tension"], resolved_tensions=[])
    base.update(overrides)
    return AutonomyStateDeltaV1.model_validate(base).model_dump(mode="json")


def test_build_autonomy_slice_from_populated_state() -> None:
    ctx = {
        "chat_autonomy_state_v2": _state_v2(),
        "chat_autonomy_state_delta": _delta(),
        "chat_autonomy_movement_debug": {
            "pressures_before": {"coherence": 0.5, "continuity": 0.4},
            "pressures_after": {"coherence": 0.72, "continuity": 0.41},
        },
    }
    slice_ = build_autonomy_slice(ctx)
    assert slice_ is not None
    assert slice_.schema_version == "autonomy.slice.v1"
    assert slice_.dominant_drive == "coherence"
    # tension_kinds has 3 entries; must be capped, never exceed 3.
    assert slice_.active_tensions == [
        "drive_competition.coherence_continuity",
        "unresolved_thread",
        "identity_drift",
    ]
    assert len(slice_.active_tensions) <= 3
    assert slice_.pressure_trend == "rising"
    assert slice_.confidence == 0.63


def test_active_tensions_hard_capped_at_three_even_with_more_tension_kinds() -> None:
    ctx = {
        "chat_autonomy_state_v2": _state_v2(
            tension_kinds=["a", "b", "c", "d", "e"],
        ),
    }
    slice_ = build_autonomy_slice(ctx)
    assert slice_ is not None
    assert slice_.active_tensions == ["a", "b", "c"]


def test_falls_back_to_delta_new_tensions_when_tension_kinds_empty() -> None:
    ctx = {
        "chat_autonomy_state_v2": _state_v2(tension_kinds=[], dominant_drive=None),
        "chat_autonomy_state_delta": _delta(new_tensions=["just_minted"]),
    }
    slice_ = build_autonomy_slice(ctx)
    assert slice_ is not None
    assert slice_.active_tensions == ["just_minted"]


def test_pressure_trend_none_when_before_pressures_missing_first_turn() -> None:
    ctx = {
        "chat_autonomy_state_v2": _state_v2(),
        "chat_autonomy_movement_debug": {
            "pressures_before": None,
            "pressures_after": {"coherence": 0.72},
        },
    }
    slice_ = build_autonomy_slice(ctx)
    assert slice_ is not None
    assert slice_.pressure_trend is None


def test_pressure_trend_stable_when_movement_below_epsilon() -> None:
    ctx = {
        "chat_autonomy_state_v2": _state_v2(),
        "chat_autonomy_movement_debug": {
            "pressures_before": {"coherence": 0.72, "continuity": 0.41},
            "pressures_after": {"coherence": 0.725, "continuity": 0.405},
        },
    }
    slice_ = build_autonomy_slice(ctx)
    assert slice_ is not None
    assert slice_.pressure_trend == "stable"


def test_returns_none_when_state_absent() -> None:
    assert build_autonomy_slice({}) is None
    assert build_autonomy_slice({"chat_autonomy_state_v2": None}) is None
    assert build_autonomy_slice({"chat_autonomy_state_v2": {}}) is None


def test_returns_none_when_state_present_but_no_meaningful_signal() -> None:
    # dominant_drive/tensions empty; confidence explicitly None (not the model's
    # own 0.5 default) to exercise the true all-empty branch.
    ctx = {
        "chat_autonomy_state_v2": {
            "subject": "orion",
            "dominant_drive": None,
            "tension_kinds": [],
            "confidence": None,
        },
    }
    assert build_autonomy_slice(ctx) is None


def test_does_not_raise_on_malformed_ctx() -> None:
    assert build_autonomy_slice({"chat_autonomy_state_v2": "not-a-dict"}) is None
    assert build_autonomy_slice({"chat_autonomy_state_v2": _state_v2(), "chat_autonomy_movement_debug": "garbage"}) is not None
