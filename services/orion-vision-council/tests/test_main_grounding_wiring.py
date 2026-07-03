import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.schemas.vision import VisionSceneInterpretationV1, VisionWindowPayload

from app.interpretation import InterpretationParseOutcome
from app.main import CouncilService


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


def _youtube_interpretation() -> VisionSceneInterpretationV1:
    return VisionSceneInterpretationV1(
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


def test_finalize_drops_youtube_activity_without_hard_person() -> None:
    svc = CouncilService()
    window = _window(
        captions=["describe this image. youtube"],
        evidence={"hard_labels": ["door", "screen"], "edge_person_hits": 0, "soft_labels": ["youtube"]},
    )
    interpretation, outcome = svc._finalize_interpretation(
        _youtube_interpretation(),
        InterpretationParseOutcome(interpretation=_youtube_interpretation(), parse_mode="strict_v2"),
        window,
    )
    assert interpretation is not None
    assert interpretation.event_candidates == []
    assert outcome.parse_mode == "strict_v2"


def test_finalize_preserves_strict_v2_when_host_fallback_after_grounding() -> None:
    svc = CouncilService()
    window = _window(
        captions=["describe this image. youtube"],
        evidence={"hard_labels": ["door", "screen"], "edge_person_hits": 0, "host_person_hits": 2},
    )
    interpretation, outcome = svc._finalize_interpretation(
        _youtube_interpretation(),
        InterpretationParseOutcome(interpretation=_youtube_interpretation(), parse_mode="strict_v2"),
        window,
    )
    assert interpretation is not None
    assert interpretation.event_candidates[0].event_type == "person_presence"
    assert outcome.parse_mode == "strict_v2"
    assert "host_fallback_after_grounding" in outcome.salvage_warnings


def test_finalize_host_fallback_on_parse_failure() -> None:
    svc = CouncilService()
    window = _window(evidence={"hard_labels": ["person"], "host_person_hits": 2, "edge_person_hits": 0})
    interpretation, outcome = svc._finalize_interpretation(
        None,
        InterpretationParseOutcome(interpretation=None, parse_mode="parse_failed"),
        window,
    )
    assert interpretation is not None
    assert interpretation.event_candidates[0].event_type == "person_presence"
    assert outcome.parse_mode == "host_fallback"
