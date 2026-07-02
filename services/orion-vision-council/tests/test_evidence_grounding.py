import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.schemas.vision import VisionSceneInterpretationV1, VisionWindowPayload

from app.evidence_grounding import (
    ACTIVITY_PATTERN,
    build_person_presence_fallback,
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


def test_person_presence_fallback_on_edge_hits() -> None:
    window = _window(evidence={"hard_labels": ["person"], "edge_person_hits": 2, "host_person_hits": 0})
    fb = build_person_presence_fallback(window)
    assert fb.event_candidates[0].event_type == "person_presence"
    assert "youtube" not in fb.event_candidates[0].narrative.lower()


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
