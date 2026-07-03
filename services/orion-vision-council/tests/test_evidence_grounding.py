import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.schemas.vision import VisionSceneInterpretationV1, VisionWindowPayload

from app.evidence_grounding import (
    ACTIVITY_PATTERN,
    build_person_presence_fallback,
    ensure_grounded_person_presence,
    enforce_evidence_grounding,
)


def _window(**summary) -> VisionWindowPayload:
    base = {
        "object_counts": {},
        "top_labels": [],
        "captions": summary.get("captions", []),
        "item_count": 1,
        "detection_count": 0,
        "evidence": summary.get("evidence", {}),
    }
    return VisionWindowPayload(
        window_id="w1",
        start_ts=1.0,
        end_ts=2.0,
        summary=base,
        artifact_ids=["art-edge-1"],
    )


def test_enforce_drops_youtube_activity_without_hard_person() -> None:
    window = _window(
        captions=["describe this image. youtube"],
        evidence={"hard_labels": ["door", "screen"], "edge_person_hits": 0, "soft_labels": ["youtube"]},
    )
    interpretation = VisionSceneInterpretationV1(
        window_id="w1",
        scene_summary="Someone watching YouTube",
        event_candidates=[
            {
                "event_type": "human_activity",
                "narrative": "A person is watching a YouTube video on a screen.",
                "entities": ["person"],
                "tags": [],
                "confidence": 0.9,
                "salience": 0.8,
                "evidence_refs": ["art-edge-1"],
            }
        ],
    )
    grounded, notes = enforce_evidence_grounding(interpretation, window)
    assert grounded.event_candidates == []
    assert notes


def test_person_presence_fallback_uses_host_detect_tag() -> None:
    window = _window(evidence={"hard_labels": ["person"], "host_person_hits": 2, "edge_person_hits": 0})
    fb = build_person_presence_fallback(window)
    assert fb.event_candidates[0].tags == ["host_detect"]


def test_activity_pattern_matches_watching() -> None:
    assert ACTIVITY_PATTERN.search("watching a video")


def test_enforce_drops_activity_caption_slop_when_soft_labels_slop() -> None:
    window = _window(
        captions=["youtube stream on screen"],
        evidence={
            "hard_labels": ["person", "screen"],
            "edge_person_hits": 1,
            "soft_labels": ["youtube"],
        },
    )
    interpretation = VisionSceneInterpretationV1(
        window_id="w1",
        scene_summary="Person watching",
        event_candidates=[
            {
                "event_type": "human_activity",
                "narrative": "A person is watching a video on the screen.",
                "entities": ["person"],
                "tags": [],
                "confidence": 0.9,
                "salience": 0.8,
                "evidence_refs": ["art-edge-1"],
            }
        ],
    )
    grounded, notes = enforce_evidence_grounding(interpretation, window)
    assert grounded.event_candidates == []
    assert any("caption_slop" in note for note in notes)


def test_enforce_scrubs_scene_summary_when_person_not_in_hard_labels() -> None:
    window = _window(
        evidence={"hard_labels": ["door", "screen"], "edge_person_hits": 0, "host_person_hits": 0},
    )
    interpretation = VisionSceneInterpretationV1(
        window_id="w1",
        scene_summary="A person is standing near the door.",
        event_candidates=[
            {
                "event_type": "visual_observation",
                "narrative": "A door and screen are visible.",
                "entities": ["door", "screen"],
                "tags": [],
                "confidence": 0.8,
                "salience": 0.7,
                "evidence_refs": ["art-edge-1"],
            }
        ],
    )
    grounded, notes = enforce_evidence_grounding(interpretation, window)
    assert grounded.scene_summary == "Door and Screen visible in the frame."
    assert any("scrubbed:scene_summary" in note for note in notes)


def test_enforce_drops_salient_observation_with_ungrounded_person() -> None:
    window = _window(
        evidence={"hard_labels": ["door"], "edge_person_hits": 0, "host_person_hits": 0},
    )
    interpretation = VisionSceneInterpretationV1(
        window_id="w1",
        scene_summary="Door visible.",
        salient_observations=[
            {
                "observation": "Someone may be near the door.",
                "confidence": 0.7,
                "salience": 0.6,
                "evidence_refs": ["art-edge-1"],
            },
            {
                "observation": "The door is closed.",
                "confidence": 0.9,
                "salience": 0.5,
                "evidence_refs": ["art-edge-1"],
            },
        ],
        event_candidates=[],
    )
    grounded, notes = enforce_evidence_grounding(interpretation, window)
    assert len(grounded.salient_observations) == 1
    assert grounded.salient_observations[0].observation == "The door is closed."
    assert any("dropped:salient_observation" in note for note in notes)


def test_enforce_scrubs_person_from_event_entities() -> None:
    window = _window(
        evidence={"hard_labels": ["door", "screen"], "edge_person_hits": 0, "host_person_hits": 0},
    )
    interpretation = VisionSceneInterpretationV1(
        window_id="w1",
        scene_summary="Door and screen visible.",
        event_candidates=[
            {
                "event_type": "visual_observation",
                "narrative": "A door and screen are visible.",
                "entities": ["person", "door", "screen"],
                "tags": [],
                "confidence": 0.8,
                "salience": 0.7,
                "evidence_refs": ["art-edge-1"],
            }
        ],
    )
    grounded, notes = enforce_evidence_grounding(interpretation, window)
    assert grounded.event_candidates[0].entities == ["door", "screen"]
    assert any("scrubbed:entities" in note for note in notes)


def test_ensure_grounded_person_presence_injects_when_llm_omits_person() -> None:
    window = _window(
        evidence={
            "hard_labels": ["door", "person", "screen"],
            "host_person_hits": 5,
            "edge_person_hits": 0,
        },
    )
    interpretation = VisionSceneInterpretationV1(
        window_id="w1",
        scene_summary="Multiple screens and doors are visible.",
        event_candidates=[
            {
                "event_type": "visual_observation",
                "narrative": "Two doors are identified in the scene.",
                "entities": ["door"],
                "tags": [],
                "confidence": 0.8,
                "salience": 0.7,
                "evidence_refs": ["art-edge-1"],
            },
        ],
    )
    updated, notes = ensure_grounded_person_presence(interpretation, window)
    assert len(updated.event_candidates) == 2
    assert updated.event_candidates[-1].event_type == "person_presence"
    assert "person" in updated.scene_summary.lower()
    assert notes == ["injected:person_presence:grounded_evidence"]


def test_enforce_caps_activity_confidence_when_captions_present() -> None:
    window = _window(
        captions=["A person near the door."],
        evidence={"hard_labels": ["person", "door"], "edge_person_hits": 1, "host_person_hits": 1},
    )
    interpretation = VisionSceneInterpretationV1(
        window_id="w1",
        scene_summary="Activity near door",
        event_candidates=[
            {
                "event_type": "human_activity",
                "narrative": "A person is watching the door.",
                "entities": ["person"],
                "tags": [],
                "confidence": 0.9,
                "salience": 0.8,
                "evidence_refs": ["art-edge-1"],
            }
        ],
    )
    grounded, notes = enforce_evidence_grounding(interpretation, window)
    assert grounded.event_candidates[0].confidence == 0.4
    assert "caption_inferred" in grounded.event_candidates[0].tags
    assert any("capped_confidence" in n for n in notes)
