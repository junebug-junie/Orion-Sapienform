"""Every identity_face check must leave a trace, including "no face" and a
weak match (2026-10-08: 26 checks in one office session, zero evidence of
what any of them said)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.schemas.vision import VisionArtifactOutputs, VisionArtifactPayload

from app.projection import identity_verdict_summary


def _art(candidates):
    return VisionArtifactPayload(
        artifact_id="a",
        correlation_id="c",
        task_type="identity_face",
        device="cuda:0",
        inputs={"stream_id": "cam0"},
        outputs=VisionArtifactOutputs(
            identities={"candidates": candidates, "enrolled_subject": "juniper", "gallery_enrolled": True}
        ),
        timing={},
        model_fingerprints={},
    )


def test_no_face_is_its_own_outcome():
    v = identity_verdict_summary(_art([]))
    assert v["outcome"] == "no_face" and v["faces"] == 0


def test_not_enrolled_is_not_reported_as_unsure():
    v = identity_verdict_summary(_art([{"subject": "unknown", "state": "unsure", "reason": "not_enrolled"}]))
    assert v["outcome"] == "not_enrolled"


def test_weak_match_keeps_its_similarity():
    v = identity_verdict_summary(
        _art([{"subject": "unknown", "state": "unsure", "similarity": 0.21, "detect_confidence": 0.93}])
    )
    assert (v["outcome"], v["similarity"], v["detect_confidence"]) == ("unsure", 0.21, 0.93)


def test_best_of_several_faces_wins():
    v = identity_verdict_summary(
        _art(
            [
                {"subject": "unknown", "state": "unsure", "similarity": 0.1},
                {"subject": "juniper", "state": "probable", "similarity": 0.7},
            ]
        )
    )
    assert (v["outcome"], v["faces"], v["similarity"]) == ("probable", 2, 0.7)
