"""Compact JSON Schema for VisionSceneInterpretationV1 (llama.cpp json_object+schema)."""

from __future__ import annotations

from typing import Any

_SALIENT_OBSERVATION = {
    "type": "object",
    "additionalProperties": False,
    "required": ["observation"],
    "properties": {
        "observation": {"type": "string"},
        "salience": {"type": "number"},
        "confidence": {"type": "number"},
        "evidence_refs": {"type": "array", "items": {"type": "string"}},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
}

_EVENT_CANDIDATE = {
    "type": "object",
    "additionalProperties": False,
    "required": ["event_type", "narrative"],
    "properties": {
        "event_type": {"type": "string"},
        "narrative": {"type": "string"},
        "entities": {"type": "array", "items": {"type": "string"}},
        "tags": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
        "salience": {"type": "number"},
        "evidence_refs": {"type": "array", "items": {"type": "string"}},
    },
}

_UNCERTAINTY = {
    "type": "object",
    "additionalProperties": False,
    "required": ["uncertainty"],
    "properties": {
        "uncertainty": {"type": "string"},
        "reason": {"type": "string"},
        "confidence": {"type": "number"},
        "evidence_refs": {"type": "array", "items": {"type": "string"}},
    },
}


def compact_vision_scene_interpretation_json_schema() -> dict[str, Any]:
    """Inline schema without $ref — suitable for Atlas metacog json_object+schema."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "window_id",
            "scene_summary",
            "salient_observations",
            "uncertainties",
            "event_candidates",
            "evidence_refs",
        ],
        "properties": {
            "schema_version": {"type": "string", "enum": ["1.0"]},
            "window_id": {"type": "string"},
            "stream_id": {"type": "string"},
            "camera_id": {"type": "string"},
            "scene_summary": {"type": "string"},
            "scene_state": {"type": "object"},
            "entities": {"type": "array", "items": {"type": "object"}},
            "relations": {"type": "array", "items": {"type": "object"}},
            "salient_observations": {"type": "array", "items": _SALIENT_OBSERVATION},
            "uncertainties": {"type": "array", "items": _UNCERTAINTY},
            "task_relevance": {"type": "array", "items": {"type": "object"}},
            "event_candidates": {"type": "array", "items": _EVENT_CANDIDATE},
            "memory_delta_candidates": {"type": "array", "items": {"type": "object"}},
            "grammar_projection": {"type": ["object", "null"]},
            "evidence_refs": {"type": "array", "items": {"type": "string"}},
        },
    }
