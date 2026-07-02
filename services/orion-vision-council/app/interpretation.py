"""
V2 scene interpretation: prompt building, LLM JSON parsing, and event projection.
Pure helpers — no bus or Redis dependencies.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from loguru import logger

from orion.schemas.vision import (
    VisionEventBundleItem,
    VisionEventCandidateV1,
    VisionEventPayload,
    VisionSceneInterpretationV1,
    VisionWindowPayload,
)


def _strip_markdown_fences(content: str) -> str:
    text = content.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    return text.strip()


def build_interpretation_prompt(window: VisionWindowPayload) -> str:
    summary = window.summary or {}
    context: dict[str, Any] = {
        "window_id": window.window_id,
        "stream_id": window.stream_id,
        "camera_id": window.camera_id,
        "start_ts": window.start_ts,
        "end_ts": window.end_ts,
        "summary": {
            "top_labels": summary.get("top_labels", []),
            "object_counts": summary.get("object_counts", {}),
            "captions": summary.get("captions", []),
            "detection_count": summary.get("detection_count", summary.get("item_count", 0)),
        },
        "artifact_ids": window.artifact_ids or [],
    }
    if window.artifact_uris:
        context["artifact_uris"] = window.artifact_uris
    if window.freshness is not None:
        context["freshness"] = window.freshness
    if window.meta is not None:
        context["meta"] = window.meta

    schema_hint = {
        "schema_version": "1.0",
        "window_id": window.window_id,
        "stream_id": window.stream_id,
        "camera_id": window.camera_id,
        "scene_summary": "concise factual summary of what is visible",
        "scene_state": {},
        "entities": [],
        "relations": [],
        "salient_observations": [],
        "uncertainties": [],
        "task_relevance": [],
        "event_candidates": [
            {
                "event_type": "string",
                "narrative": "string",
                "entities": [],
                "tags": [],
                "confidence": 0.0,
                "salience": 0.0,
                "evidence_refs": [],
            }
        ],
        "memory_delta_candidates": [],
        "grammar_projection": None,
        "evidence_refs": [],
    }

    return (
        "Analyze this visual window and return exactly one JSON object matching "
        "VisionSceneInterpretationV1.\n\n"
        "Window context:\n"
        f"{json.dumps(context, indent=2, default=str)}\n\n"
        "Required output shape (use empty arrays instead of omitting list fields):\n"
        f"{json.dumps(schema_hint, indent=2)}\n\n"
        "Rules:\n"
        "- Output strict JSON only. No markdown fences, commentary, or prose.\n"
        "- Use empty arrays [] for missing list fields; do not omit required keys.\n"
        "- Do not invent identities, names, or facts not supported by the context.\n"
        "- Preserve uncertainty when visual evidence is weak; record it in uncertainties.\n"
        "- Prefer concrete observations over vague impressions.\n"
        "- Set evidence_refs on observations, entities, and event_candidates using "
        "artifact_ids from the window whenever possible.\n"
        "- Include scene_summary and event_candidates; other fields may be empty arrays.\n"
    )


def _dict_to_event_candidate(evt: dict[str, Any]) -> VisionEventCandidateV1:
    return VisionEventCandidateV1(
        event_type=str(evt.get("event_type", "unknown")),
        narrative=str(evt.get("narrative", "")),
        entities=list(evt.get("entities") or []),
        tags=list(evt.get("tags") or []),
        confidence=float(evt.get("confidence", 0.5)),
        salience=float(evt.get("salience", 0.5)),
        evidence_refs=list(evt.get("evidence_refs") or []),
    )


def _events_list_to_minimal_interpretation(
    events: list[dict[str, Any]],
    window: VisionWindowPayload,
) -> VisionSceneInterpretationV1:
    scene_summary = "Events from legacy format"
    if events:
        first_narrative = events[0].get("narrative")
        if first_narrative:
            scene_summary = str(first_narrative)

    return VisionSceneInterpretationV1(
        window_id=window.window_id,
        stream_id=window.stream_id,
        camera_id=window.camera_id,
        scene_summary=scene_summary,
        event_candidates=[_dict_to_event_candidate(evt) for evt in events],
        evidence_refs=list(window.artifact_ids or []),
    )


def parse_llm_content(content: str, window: VisionWindowPayload) -> VisionSceneInterpretationV1 | None:
    try:
        text = _strip_markdown_fences(content)
        data = json.loads(text)

        if isinstance(data, dict) and "interpretation" in data:
            data = data["interpretation"]

        if isinstance(data, list):
            return _events_list_to_minimal_interpretation(data, window)

        if isinstance(data, dict):
            if set(data.keys()) == {"events"} and isinstance(data.get("events"), list):
                return _events_list_to_minimal_interpretation(data["events"], window)

            if not data.get("window_id"):
                data = {**data, "window_id": window.window_id}
            return VisionSceneInterpretationV1.model_validate(data)

        logger.error(f"[COUNCIL] Unexpected LLM JSON type: {type(data).__name__}")
        return None
    except Exception as e:
        logger.error(f"[COUNCIL] Failed to parse interpretation JSON: {e} | Content: {content!r}")
        return None


def project_interpretation_to_events(
    interpretation: VisionSceneInterpretationV1,
    window: VisionWindowPayload,
) -> VisionEventPayload:
    fallback_refs = list(window.artifact_ids or [])
    bundle_items: list[VisionEventBundleItem] = []

    for candidate in interpretation.event_candidates:
        evidence_refs = list(candidate.evidence_refs) if candidate.evidence_refs else fallback_refs
        bundle_items.append(
            VisionEventBundleItem(
                event_id=str(uuid.uuid4()),
                event_type=candidate.event_type,
                narrative=candidate.narrative,
                entities=list(candidate.entities),
                tags=list(candidate.tags),
                confidence=candidate.confidence,
                salience=candidate.salience,
                evidence_refs=evidence_refs,
            )
        )

    return VisionEventPayload(events=bundle_items)


def try_legacy_fallback(
    content: str,
    window: VisionWindowPayload,
) -> VisionSceneInterpretationV1 | None:
    """Parse legacy flat event-list JSON when full interpretation parsing fails."""
    try:
        text = _strip_markdown_fences(content)
        data = json.loads(text)
        events: list[dict[str, Any]] | None = None
        if isinstance(data, list):
            events = data
        elif isinstance(data, dict) and isinstance(data.get("events"), list):
            events = data["events"]
        elif isinstance(data, dict):
            events = [data]
        if not events:
            return None
        return _events_list_to_minimal_interpretation(events, window)
    except Exception as e:
        logger.error(f"[COUNCIL] Legacy fallback parse failed: {e} | Content: {content!r}")
        return None


def parse_and_project(
    content: str,
    window: VisionWindowPayload,
) -> tuple[VisionSceneInterpretationV1 | None, VisionEventPayload | None]:
    interpretation = parse_llm_content(content, window)
    if interpretation is None:
        interpretation = try_legacy_fallback(content, window)
    if interpretation is None:
        return None, None
    return interpretation, project_interpretation_to_events(interpretation, window)
