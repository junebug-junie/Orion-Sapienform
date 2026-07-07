from __future__ import annotations

from orion.embodiment.drive_map import DriveMapThresholds, map_drive_state_to_intent
from orion.core.schemas.drives import DriveStateV1
from orion.core.schemas.drives import ArtifactProvenance


def _drive(pressures: dict[str, float]) -> DriveStateV1:
    return DriveStateV1(
        subject="orion", model_layer="drive", entity_id="orion", kind="memory.drives.state.v1",
        provenance=ArtifactProvenance(intake_channel="test"), pressures=pressures,
    )


def test_high_social_escalates_to_start_conversation():
    intent = map_drive_state_to_intent(
        _drive({"social": 0.85}), correlation_id="tick-1", in_conversation=False,
    )
    assert intent is not None
    assert intent.kind == "start_conversation"
    assert intent.source == "involuntary"
    assert intent.reason.strip()


def test_high_social_while_in_conversation_does_not_initiate():
    intent = map_drive_state_to_intent(
        _drive({"social": 0.85}), correlation_id="t", in_conversation=True,
    )
    assert intent is None or intent.kind != "start_conversation"


def test_dominant_social_approaches():
    intent = map_drive_state_to_intent(_drive({"social": 0.6, "predictive": 0.1}), correlation_id="t")
    assert intent is not None and intent.kind == "approach_player"


def test_dominant_curiosity_wanders():
    intent = map_drive_state_to_intent(_drive({"predictive": 0.6, "social": 0.1}), correlation_id="t")
    assert intent is not None and intent.kind == "wander"


def test_all_low_is_idle():
    intent = map_drive_state_to_intent(_drive({"social": 0.02, "predictive": 0.02}), correlation_id="t")
    assert intent is not None and intent.kind == "idle"


def test_empty_pressures_is_none():
    assert map_drive_state_to_intent(_drive({}), correlation_id="t") is None
