"""Tests for VisionSceneInterpretationV1 compact JSON schema contract."""

from __future__ import annotations

from orion.schemas.vision_interpretation_contract import (
    compact_vision_scene_interpretation_json_schema,
)


def test_compact_schema_has_required_top_level_keys():
    schema = compact_vision_scene_interpretation_json_schema()
    required = set(schema.get("required") or [])
    assert "window_id" in required
    assert "scene_summary" in required
    assert "event_candidates" in required
    assert "salient_observations" in required


def test_salient_observation_item_requires_observation_field():
    schema = compact_vision_scene_interpretation_json_schema()
    salient = schema["properties"]["salient_observations"]["items"]
    assert "observation" in salient["required"]
    assert "narrative" not in salient.get("properties", {})


def test_event_candidate_item_requires_event_type_and_narrative():
    schema = compact_vision_scene_interpretation_json_schema()
    event = schema["properties"]["event_candidates"]["items"]
    assert set(event["required"]) == {"event_type", "narrative"}
