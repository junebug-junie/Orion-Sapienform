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
PERSON_PATTERN = re.compile(r"\b(person|someone|human)\b", re.I)
CAPTION_STOPLIST = frozenset(
    {"youtube", "google", "video", "watching", "describe", "image", "webcam", "com"}
)


def _hard_labels(window: VisionWindowPayload) -> set[str]:
    ev = (window.summary or {}).get("evidence") or {}
    return {str(x).lower() for x in (ev.get("hard_labels") or [])}


def _soft_labels(window: VisionWindowPayload) -> set[str]:
    ev = (window.summary or {}).get("evidence") or {}
    return {str(x).lower() for x in (ev.get("soft_labels") or [])}


def _narrative_slop_tokens(text: str) -> set[str]:
    tokens = {t.strip(".,!?").lower() for t in text.split() if t.strip()}
    return tokens & CAPTION_STOPLIST


def _activity_claim_has_caption_slop(
    narrative: str,
    window: VisionWindowPayload,
    *,
    mentions_activity: bool,
) -> bool:
    if not mentions_activity:
        return False
    hard = _hard_labels(window)
    soft_slop = _soft_labels(window) & CAPTION_STOPLIST
    if "person" in hard:
        return bool(soft_slop)
    return bool(_narrative_slop_tokens(narrative) | soft_slop)


def host_person_hits(window: VisionWindowPayload) -> int:
    ev = (window.summary or {}).get("evidence") or {}
    try:
        return int(ev.get("host_person_hits") or 0)
    except (TypeError, ValueError):
        return 0


def _text_mentions_person(text: str) -> bool:
    return bool(PERSON_PATTERN.search(text or ""))


def _grounded_scene_summary_from_labels(hard: set[str]) -> str:
    if not hard:
        return "No salient objects detected above threshold."
    labels = sorted(hard)
    if len(labels) == 1:
        return f"{labels[0].capitalize()} visible in the frame."
    head = ", ".join(label.capitalize() for label in labels[:-1])
    return f"{head} and {labels[-1].capitalize()} visible in the frame."


def _filter_person_entity_names(names: list[str], hard: set[str]) -> list[str]:
    if "person" in hard:
        return names
    return [name for name in names if not _text_mentions_person(name)]


def _scrub_non_event_person_fields(
    interpretation: VisionSceneInterpretationV1,
    hard: set[str],
    notes: list[str],
) -> dict[str, object]:
    updates: dict[str, object] = {}
    if _text_mentions_person(interpretation.scene_summary) and "person" not in hard:
        updates["scene_summary"] = _grounded_scene_summary_from_labels(hard)
        notes.append("scrubbed:scene_summary:person_not_in_hard_labels")

    kept_observations = []
    for obs in interpretation.salient_observations:
        if _text_mentions_person(obs.observation) and "person" not in hard:
            notes.append("dropped:salient_observation:person_not_in_hard_labels")
            continue
        kept_observations.append(obs)
    if len(kept_observations) != len(interpretation.salient_observations):
        updates["salient_observations"] = kept_observations

    kept_entities = []
    for entity in interpretation.entities:
        if _text_mentions_person(entity.label) and "person" not in hard:
            notes.append("dropped:entity:person_not_in_hard_labels")
            continue
        kept_entities.append(entity)
    if len(kept_entities) != len(interpretation.entities):
        updates["entities"] = kept_entities

    kept_relations = []
    for relation in interpretation.relations:
        if "person" not in hard and (
            _text_mentions_person(relation.subject) or _text_mentions_person(relation.object)
        ):
            notes.append("dropped:relation:person_not_in_hard_labels")
            continue
        kept_relations.append(relation)
    if len(kept_relations) != len(interpretation.relations):
        updates["relations"] = kept_relations

    kept_memory = []
    for delta in interpretation.memory_delta_candidates:
        if _text_mentions_person(delta.claim) and "person" not in hard:
            notes.append("dropped:memory_delta:person_not_in_hard_labels")
            continue
        kept_memory.append(delta)
    if len(kept_memory) != len(interpretation.memory_delta_candidates):
        updates["memory_delta_candidates"] = kept_memory

    return updates


def enforce_evidence_grounding(
    interpretation: VisionSceneInterpretationV1,
    window: VisionWindowPayload,
) -> tuple[VisionSceneInterpretationV1, list[str]]:
    hard = _hard_labels(window)
    notes: list[str] = []
    kept: list[VisionEventCandidateV1] = []
    for cand in interpretation.event_candidates:
        narrative = cand.narrative or ""
        mentions_person = _text_mentions_person(narrative)
        mentions_activity = bool(ACTIVITY_PATTERN.search(narrative))
        if mentions_person and "person" not in hard:
            notes.append(f"dropped:{cand.event_type}:person_not_in_hard_labels")
            continue
        if mentions_activity and "person" not in hard:
            notes.append(f"dropped:{cand.event_type}:activity_without_person")
            continue
        if _activity_claim_has_caption_slop(
            narrative, window, mentions_activity=mentions_activity
        ):
            notes.append(f"dropped:{cand.event_type}:caption_slop")
            continue
        updated = cand
        if mentions_activity and "person" in hard and cand.confidence > 0.4:
            if (window.summary or {}).get("captions"):
                updated = cand.model_copy(
                    update={"confidence": 0.4, "tags": [*cand.tags, "caption_inferred"]}
                )
                notes.append(f"capped_confidence:{cand.event_type}")
        filtered_entities = _filter_person_entity_names(list(updated.entities), hard)
        if filtered_entities != list(updated.entities):
            updated = updated.model_copy(update={"entities": filtered_entities})
            notes.append(f"scrubbed:entities:{updated.event_type}")
        kept.append(updated)

    updates = _scrub_non_event_person_fields(interpretation, hard, notes)
    updates["event_candidates"] = kept
    return interpretation.model_copy(update=updates), notes


def _events_mention_person(candidates: list[VisionEventCandidateV1]) -> bool:
    return any(
        c.event_type == "person_presence" or _text_mentions_person(c.narrative or "")
        for c in candidates
    )


def ensure_grounded_person_presence(
    interpretation: VisionSceneInterpretationV1,
    window: VisionWindowPayload,
) -> tuple[VisionSceneInterpretationV1, list[str]]:
    """Inject person_presence when detections prove person but the LLM omitted them."""
    hard = _hard_labels(window)
    if "person" not in hard and host_person_hits(window) <= 0:
        return interpretation, []
    if _events_mention_person(list(interpretation.event_candidates)):
        return interpretation, []

    refs = list(window.artifact_ids or [])
    person_event = VisionEventCandidateV1(
        event_type="person_presence",
        narrative="A person was detected on camera.",
        entities=["person"],
        tags=["host_detect"],
        confidence=0.85,
        salience=0.7,
        evidence_refs=refs,
    )
    updates: dict[str, object] = {
        "event_candidates": [*interpretation.event_candidates, person_event],
    }
    if not _text_mentions_person(interpretation.scene_summary):
        updates["scene_summary"] = _grounded_scene_summary_from_labels(hard | {"person"})
    return interpretation.model_copy(update=updates), ["injected:person_presence:grounded_evidence"]


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
                tags=["host_detect"],
                confidence=0.85,
                salience=0.7,
                evidence_refs=refs,
            )
        ],
        evidence_refs=refs,
    )
