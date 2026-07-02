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


def test_parse_valid_vision_scene_interpretation_v1_json():
    window = _window()
    interpretation = parse_llm_content(_valid_interpretation_json(window), window)

    assert interpretation is not None
    assert isinstance(interpretation, VisionSceneInterpretationV1)
    assert interpretation.window_id == "w1"
    assert interpretation.scene_summary == "A person walks through the frame."
    assert len(interpretation.event_candidates) == 1
    assert interpretation.event_candidates[0].event_type == "movement"
    assert interpretation.event_candidates[0].narrative == "Person enters from the left."


def test_project_event_candidates_to_vision_event_payload():
    window = _window()
    interpretation = parse_llm_content(_valid_interpretation_json(window), window)
    assert interpretation is not None

    payload = project_interpretation_to_events(interpretation, window)

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

    interpretation = parse_llm_content(content, window)

    assert interpretation is not None
    assert interpretation.scene_summary == "A person walks through the frame."
    assert interpretation.raw_model_output is not None
    assert "interpretation" in interpretation.raw_model_output


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

    interpretation = parse_llm_content(content, window)

    assert interpretation is not None
    assert interpretation.scene_summary == "Hybrid legacy events field."
    assert len(interpretation.event_candidates) == 1
    assert interpretation.event_candidates[0].event_type == "presence"


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

    interpretation = parse_llm_content(fenced, window)

    assert interpretation is not None
    assert interpretation.event_candidates[0].event_type == "movement"


def test_fallback_events_dict_to_minimal_interpretation():
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

    interpretation = parse_llm_content(content, window)

    assert interpretation is not None
    assert interpretation.window_id == "w1"
    assert interpretation.stream_id == "stream-1"
    assert interpretation.camera_id == "cam-1"
    assert interpretation.scene_summary == "Someone is visible near the door."
    assert len(interpretation.event_candidates) == 1
    assert interpretation.event_candidates[0].event_type == "presence"
    assert interpretation.evidence_refs == ["art-1", "art-2"]


def test_fallback_raw_event_list_to_minimal_interpretation():
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

    interpretation = parse_llm_content(content, window)

    assert interpretation is not None
    assert interpretation.scene_summary == "Person sits at the table."
    assert len(interpretation.event_candidates) == 1
    assert interpretation.event_candidates[0].event_type == "activity"

    legacy = try_legacy_fallback(content, window)
    assert legacy is not None
    assert legacy.scene_summary == "Person sits at the table."


def test_invalid_json_returns_none_without_crashing():
    window = _window()

    assert parse_llm_content("not json at all", window) is None
    assert parse_llm_content("{broken", window) is None
    assert try_legacy_fallback("also not json", window) is None


def test_evidence_refs_default_to_window_artifact_ids():
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
    interpretation = parse_llm_content(_valid_interpretation_json(window), window)
    assert interpretation is not None

    payload = project_interpretation_to_events(interpretation, window)

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
