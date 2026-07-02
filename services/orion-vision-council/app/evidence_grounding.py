from __future__ import annotations

import re

from orion.schemas.vision import (
    VisionEventCandidateV1,
    VisionSceneInterpretationV1,
    VisionWindowPayload,
)

ACTIVITY_PATTERN = re.compile(
    r"\b(watching|reading|using|talking|listening|playing|browsing)\b", re.I
)
PERSON_PATTERN = re.compile(r"\bperson|someone|human\b", re.I)


def _hard_labels(window: VisionWindowPayload) -> set[str]:
    ev = (window.summary or {}).get("evidence") or {}
    return {str(x).lower() for x in (ev.get("hard_labels") or [])}


def edge_person_hits(window: VisionWindowPayload) -> int:
    ev = (window.summary or {}).get("evidence") or {}
    try:
        return int(ev.get("edge_person_hits") or 0)
    except (TypeError, ValueError):
        return 0


def enforce_evidence_grounding(
    interpretation: VisionSceneInterpretationV1,
    window: VisionWindowPayload,
) -> tuple[VisionSceneInterpretationV1, list[str]]:
    hard = _hard_labels(window)
    notes: list[str] = []
    kept: list[VisionEventCandidateV1] = []
    for cand in interpretation.event_candidates:
        narrative = cand.narrative or ""
        mentions_person = bool(PERSON_PATTERN.search(narrative))
        mentions_activity = bool(ACTIVITY_PATTERN.search(narrative))
        if mentions_person and "person" not in hard:
            notes.append(f"dropped:{cand.event_type}:person_not_in_hard_labels")
            continue
        if mentions_activity and "person" not in hard:
            notes.append(f"dropped:{cand.event_type}:activity_without_person")
            continue
        updated = cand
        if mentions_activity and "person" in hard and cand.confidence > 0.4:
            if (window.summary or {}).get("captions"):
                updated = cand.model_copy(update={"confidence": 0.4, "tags": [*cand.tags, "caption_inferred"]})
                notes.append(f"capped_confidence:{cand.event_type}")
        kept.append(updated)
    return interpretation.model_copy(update={"event_candidates": kept}), notes


def build_person_presence_fallback(window: VisionWindowPayload) -> VisionSceneInterpretationV1:
    refs = list(window.artifact_ids or [])
    return VisionSceneInterpretationV1(
        window_id=window.window_id,
        stream_id=window.stream_id,
        camera_id=window.camera_id,
        scene_summary="A person was detected on camera.",
        event_candidates=[
            VisionEventCandidateV1(
                event_type="person_presence",
                narrative="A person was detected on camera.",
                entities=["person"],
                tags=["edge_yolo"],
                confidence=0.85,
                salience=0.7,
                evidence_refs=refs,
            )
        ],
        evidence_refs=refs,
    )
