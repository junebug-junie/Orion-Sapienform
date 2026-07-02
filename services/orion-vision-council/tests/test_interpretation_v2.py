"""Council V2 scene interpretation parsing and projection tests (no live LLM/Redis/SQL)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from app.interpretation import (  # noqa: E402
    InterpretationParseOutcome,
    parse_llm_content,
    project_interpretation_to_events,
    try_legacy_fallback,
)
from orion.schemas.registry import _REGISTRY, resolve  # noqa: E402
from orion.schemas.vision import (  # noqa: E402
    VisionEventBundleItem,
    VisionEventPayload,
    VisionSceneInterpretationV1,
    VisionWindowPayload,
)


def _window(**kwargs) -> VisionWindowPayload:
    base = dict(
        window_id="w1",
        start_ts=1.0,
        end_ts=31.0,
        summary={"top_labels": ["person"], "captions": ["a person walks"], "detection_count": 1},
        artifact_ids=["art-1", "art-2"],
        stream_id="stream-1",
        camera_id="cam-1",
    )
    base.update(kwargs)
    return VisionWindowPayload(**base)


def _parse(content: str, window: VisionWindowPayload) -> InterpretationParseOutcome:
    return parse_llm_content(content, window)


def _valid_interpretation_json(window: VisionWindowPayload) -> str:
    payload = {
        "schema_version": "1.0",
        "window_id": window.window_id,
        "stream_id": window.stream_id,
        "camera_id": window.camera_id,
        "scene_summary": "A person walks through the frame.",
        "scene_state": {},
        "entities": [],
        "relations": [],
        "salient_observations": [],
        "uncertainties": [],
        "task_relevance": [],
        "event_candidates": [
            {
                "event_type": "movement",
                "narrative": "Person enters from the left.",
                "entities": ["person"],
                "tags": ["motion"],
                "confidence": 0.85,
                "salience": 0.7,
                "evidence_refs": ["art-1"],
            }
        ],
        "memory_delta_candidates": [],
        "grammar_projection": None,
        "evidence_refs": ["art-1", "art-2"],
    }
    return json.dumps(payload)


def test_strict_v2_still_parses_as_strict_v2():
    window = _window()
    outcome = _parse(_valid_interpretation_json(window), window)

    assert outcome.parse_mode == "strict_v2"
    assert outcome.interpretation is not None
    assert isinstance(outcome.interpretation, VisionSceneInterpretationV1)
    assert outcome.interpretation.scene_summary == "A person walks through the frame."
    assert outcome.salvage_warnings == []


def test_salient_observations_event_type_narrative_salvages_to_observation():
    window = _window()
    content = json.dumps(
        {
            "window_id": window.window_id,
            "scene_summary": "A door and a screen are visible.",
            "salient_observations": [
                {
                    "event_type": "object_seen",
                    "narrative": "A door and a screen are visible.",
                    "confidence": 0.8,
                    "salience": 0.6,
                    "tags": ["door", "screen"],
                }
            ],
            "uncertainties": [],
            "event_candidates": [],
        }
    )

    outcome = _parse(content, window)

    assert outcome.parse_mode == "salvaged_v2"
    assert outcome.interpretation is not None
    assert outcome.interpretation.scene_summary == "A door and a screen are visible."
    obs = outcome.interpretation.salient_observations[0]
    assert obs.observation == "A door and a screen are visible."
    assert obs.confidence == pytest.approx(0.8)
    assert obs.tags == ["door", "screen"]
    assert any("narrative" in w for w in outcome.salvage_warnings)


def test_uncertainties_string_list_salvages_to_objects():
    window = _window()
    content = json.dumps(
        {
            "window_id": window.window_id,
            "scene_summary": "Uncertain scene.",
            "salient_observations": [],
            "uncertainties": ["door", "screen"],
            "event_candidates": [
                {
                    "event_type": "observation",
                    "narrative": "Something visible.",
                }
            ],
        }
    )

    outcome = _parse(content, window)

    assert outcome.parse_mode == "salvaged_v2"
    assert outcome.interpretation is not None
    uncertainties = outcome.interpretation.uncertainties
    assert len(uncertainties) == 2
    assert uncertainties[0].uncertainty == "door"
    assert uncertainties[1].uncertainty == "screen"


def test_uncertainties_uncertainty_type_salvages_to_uncertainty():
    window = _window()
    content = json.dumps(
        {
            "window_id": window.window_id,
            "scene_summary": "Ambiguous context.",
            "salient_observations": [],
            "uncertainties": [{"uncertainty_type": "unknown context", "reason": "camera angle"}],
            "event_candidates": [
                {
                    "event_type": "observation",
                    "narrative": "Scene is unclear.",
                }
            ],
        }
    )

    outcome = _parse(content, window)

    assert outcome.parse_mode == "salvaged_v2"
    assert outcome.interpretation is not None
    assert outcome.interpretation.uncertainties[0].uncertainty == "unknown context"
    assert outcome.interpretation.uncertainties[0].reason == "camera angle"


def test_scene_summary_preserved_when_nested_fields_malformed():
    window = _window()
    content = json.dumps(
        {
            "window_id": window.window_id,
            "scene_summary": "A door and a screen are visible.",
            "salient_observations": [{"event_type": "object_seen", "narrative": "door visible"}],
            "uncertainties": ["door"],
            "event_candidates": [],
        }
    )

    outcome = _parse(content, window)

    assert outcome.parse_mode == "salvaged_v2"
    assert outcome.interpretation is not None
    assert outcome.interpretation.scene_summary == "A door and a screen are visible."


def test_valid_event_candidates_preserved_while_other_nested_malformed():
    window = _window()
    content = json.dumps(
        {
            "window_id": window.window_id,
            "scene_summary": "Person at door.",
            "salient_observations": [{"event_type": "bad", "narrative": "ignored for events"}],
            "uncertainties": ["lighting"],
            "event_candidates": [
                {
                    "event_type": "presence",
                    "narrative": "Someone stands at the door.",
                    "confidence": 0.9,
                    "salience": 0.8,
                    "tags": ["person"],
                }
            ],
        }
    )

    outcome = _parse(content, window)

    assert outcome.parse_mode == "salvaged_v2"
    assert outcome.interpretation is not None
    assert len(outcome.interpretation.event_candidates) == 1
    assert outcome.interpretation.event_candidates[0].event_type == "presence"
    assert outcome.interpretation.event_candidates[0].narrative == "Someone stands at the door."


def test_event_candidates_synthesized_from_salient_observations_when_missing():
    window = _window()
    content = json.dumps(
        {
            "window_id": window.window_id,
            "scene_summary": "Door and screen visible.",
            "salient_observations": [
                {
                    "event_type": "object_seen",
                    "narrative": "A door and a screen are visible.",
                    "confidence": 0.8,
                    "salience": 0.6,
                    "tags": ["door", "screen"],
                }
            ],
            "uncertainties": [],
            "event_candidates": [],
        }
    )

    outcome = _parse(content, window)

    assert outcome.parse_mode == "salvaged_v2"
    assert outcome.interpretation is not None
    assert len(outcome.interpretation.event_candidates) == 1
    candidate = outcome.interpretation.event_candidates[0]
    assert candidate.event_type == "visual_observation"
    assert candidate.narrative == "A door and a screen are visible."
    assert candidate.tags == ["door", "screen"]
    assert any("synthesized" in w for w in outcome.salvage_warnings)


def test_missing_scene_summary_salvages_not_legacy():
    window = _window()
    content = json.dumps(
        {
            "window_id": window.window_id,
            "salient_observations": [],
            "uncertainties": [],
            "event_candidates": [],
        }
    )

    outcome = _parse(content, window)

    assert outcome.parse_mode == "salvaged_v2"
    assert outcome.interpretation is not None
    assert outcome.interpretation.scene_summary == "a person walks"
    assert outcome.interpretation.event_candidates == []


def test_scene_summary_with_top_level_event_fields_salvages():
    window = _window()
    content = json.dumps(
        {
            "window_id": window.window_id,
            "scene_summary": "Person visible at door.",
            "event_type": "presence",
            "narrative": "Someone stands at the door.",
            "salient_observations": [],
            "uncertainties": ["lighting"],
            "event_candidates": [],
        }
    )

    outcome = _parse(content, window)

    assert outcome.parse_mode == "salvaged_v2"
    assert outcome.interpretation is not None
    assert outcome.interpretation.scene_summary == "Person visible at door."
    assert len(outcome.interpretation.event_candidates) == 1
    assert outcome.interpretation.event_candidates[0].event_type == "presence"
    assert outcome.interpretation.event_candidates[0].narrative == "Someone stands at the door."


def test_flat_event_dict_uses_legacy_not_salvaged_v2():
    window = _window()
    content = json.dumps({"event_type": "solo", "narrative": "solo event"})

    outcome = _parse(content, window)

    assert outcome.parse_mode == "legacy_events_dict"
    assert outcome.interpretation is not None
    assert len(outcome.interpretation.event_candidates) == 1
    assert outcome.interpretation.event_candidates[0].event_type == "solo"
    assert outcome.interpretation.event_candidates[0].narrative == "solo event"


def test_raw_events_dict_uses_legacy_path_not_salvaged_v2():
    window = _window()
    content = json.dumps(
        {
            "events": [
                {
                    "event_type": "presence",
                    "narrative": "Someone is visible near the door.",
                    "entities": ["person"],
                    "tags": ["entry"],
                    "confidence": 0.6,
                    "salience": 0.5,
                }
            ]
        }
    )

    outcome = _parse(content, window)

    assert outcome.parse_mode == "legacy_events_dict"
    assert outcome.interpretation is not None
    assert outcome.interpretation.scene_summary == "Someone is visible near the door."


def test_raw_event_list_uses_legacy_list_path():
    window = _window()
    content = json.dumps(
        [
            {
                "event_type": "activity",
                "narrative": "Person sits at the table.",
                "confidence": 0.75,
            }
        ]
    )

    outcome = _parse(content, window)

    assert outcome.parse_mode == "legacy_events_list"
    assert outcome.interpretation is not None
    assert outcome.interpretation.scene_summary == "Person sits at the table."


def test_garbage_json_does_not_crash():
    window = _window()

    outcome = _parse("not json at all", window)
    assert outcome.interpretation is None
    assert outcome.parse_mode == "parse_failed"

    outcome = _parse("{broken", window)
    assert outcome.interpretation is None
    assert outcome.parse_mode == "parse_failed"

    assert try_legacy_fallback("also not json", window) is None


def test_evidence_refs_default_to_window_artifact_ids_on_salvage():
    window = _window(artifact_ids=["fallback-a", "fallback-b"])
    content = json.dumps(
        {
            "window_id": window.window_id,
            "scene_summary": "Salvaged scene.",
            "salient_observations": [
                {"event_type": "object_seen", "narrative": "Object visible."},
            ],
            "uncertainties": [],
            "event_candidates": [],
        }
    )

    outcome = _parse(content, window)
    assert outcome.interpretation is not None

    payload = project_interpretation_to_events(outcome.interpretation, window)
    assert len(payload.events) == 1
    assert payload.events[0].evidence_refs == ["fallback-a", "fallback-b"]


def test_debug_parse_mode_available_from_outcome():
    window = _window()
    outcome = _parse(_valid_interpretation_json(window), window)

    record = outcome.interpretation.model_dump() if outcome.interpretation else {}
    record["parse_mode"] = outcome.parse_mode
    record["salvage_warnings"] = outcome.salvage_warnings

    assert record["parse_mode"] == "strict_v2"
    assert "salvage_warnings" in record


def test_project_event_candidates_to_vision_event_payload():
    window = _window()
    outcome = _parse(_valid_interpretation_json(window), window)
    assert outcome.interpretation is not None

    payload = project_interpretation_to_events(outcome.interpretation, window)

    assert isinstance(payload, VisionEventPayload)
    assert len(payload.events) == 1
    event = payload.events[0]
    assert event.event_type == "movement"
    assert event.narrative == "Person enters from the left."
    assert event.entities == ["person"]
    assert event.tags == ["motion"]
    assert event.confidence == pytest.approx(0.85)
    assert event.salience == pytest.approx(0.7)
    assert event.evidence_refs == ["art-1"]
    assert event.event_id


def test_parse_interpretation_wrapper():
    window = _window()
    inner = json.loads(_valid_interpretation_json(window))
    content = json.dumps({"interpretation": inner})

    outcome = _parse(content, window)

    assert outcome.parse_mode == "strict_v2"
    assert outcome.interpretation is not None
    assert outcome.interpretation.raw_model_output is not None
    assert "interpretation" in outcome.interpretation.raw_model_output


def test_hybrid_events_key_with_scene_summary_coerces_event_candidates():
    window = _window()
    content = json.dumps(
        {
            "schema_version": "1.0",
            "window_id": window.window_id,
            "scene_summary": "Hybrid legacy events field.",
            "events": [
                {
                    "event_type": "presence",
                    "narrative": "Someone is visible near the door.",
                    "entities": ["person"],
                    "tags": ["entry"],
                    "confidence": 0.6,
                    "salience": 0.5,
                }
            ],
        }
    )

    outcome = _parse(content, window)

    assert outcome.parse_mode == "strict_v2"
    assert outcome.interpretation is not None
    assert outcome.interpretation.scene_summary == "Hybrid legacy events field."
    assert len(outcome.interpretation.event_candidates) == 1
    assert outcome.interpretation.event_candidates[0].event_type == "presence"


def test_empty_event_candidates_projection_returns_empty_payload():
    window = _window()
    interpretation = VisionSceneInterpretationV1(
        window_id=window.window_id,
        scene_summary="No events detected.",
        event_candidates=[],
    )

    payload = project_interpretation_to_events(interpretation, window)

    assert payload.events == []


def test_markdown_fenced_json_parses():
    window = _window()
    fenced = f"```json\n{_valid_interpretation_json(window)}\n```"

    outcome = _parse(fenced, window)

    assert outcome.interpretation is not None
    assert outcome.interpretation.event_candidates[0].event_type == "movement"


def test_evidence_refs_default_to_window_artifact_ids_on_projection():
    window = _window(artifact_ids=["fallback-a", "fallback-b"])
    interpretation = VisionSceneInterpretationV1(
        window_id=window.window_id,
        stream_id=window.stream_id,
        camera_id=window.camera_id,
        scene_summary="Test scene",
        event_candidates=[
            {
                "event_type": "observation",
                "narrative": "No explicit evidence on candidate.",
                "evidence_refs": [],
            }
        ],
    )

    payload = project_interpretation_to_events(interpretation, window)

    assert len(payload.events) == 1
    assert payload.events[0].evidence_refs == ["fallback-a", "fallback-b"]


def test_vision_event_payload_shape_compatible():
    window = _window()
    outcome = _parse(_valid_interpretation_json(window), window)
    assert outcome.interpretation is not None

    payload = project_interpretation_to_events(outcome.interpretation, window)

    assert hasattr(payload, "events")
    assert isinstance(payload.events, list)
    assert len(payload.events) >= 1

    required_fields = {
        "event_id",
        "event_type",
        "narrative",
        "entities",
        "tags",
        "confidence",
        "salience",
        "evidence_refs",
    }
    for event in payload.events:
        assert isinstance(event, VisionEventBundleItem)
        assert required_fields.issubset(set(VisionEventBundleItem.model_fields))
        for field in required_fields:
            assert hasattr(event, field)
            assert getattr(event, field) is not None or field in {"entities", "tags", "evidence_refs"}


def test_registry_imports_vision_scene_interpretation_v1():
    assert "VisionSceneInterpretationV1" in _REGISTRY
    schema_cls = resolve("VisionSceneInterpretationV1")
    assert schema_cls is VisionSceneInterpretationV1
    assert _REGISTRY["VisionSceneInterpretationV1"] is VisionSceneInterpretationV1

    instance = schema_cls(
        window_id="w-reg",
        scene_summary="Registry smoke test",
    )
    assert instance.schema_version == "1.0"
