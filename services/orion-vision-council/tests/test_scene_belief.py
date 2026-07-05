from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.schemas.vision import VisionWindowPayload

from app.evidence_transition import EvidenceTransitionTracker, snapshot_from_window


def _window(
    *,
    hard_labels: list[str] | None = None,
    believed_hard_labels: list[str] | None = None,
    stream_id: str = "cam0",
) -> VisionWindowPayload:
    evidence: dict = {
        "hard_labels": hard_labels or [],
        "soft_labels": [],
        "host_person_hits": 0,
        "caption_count": 0,
    }
    if believed_hard_labels is not None:
        evidence["believed_hard_labels"] = believed_hard_labels
        evidence["belief"] = {"schema": "scene_belief.v1"}
    return VisionWindowPayload(
        window_id="w1",
        start_ts=1.0,
        end_ts=2.0,
        stream_id=stream_id,
        summary={
            "object_counts": {},
            "top_labels": [],
            "captions": [],
            "item_count": 1,
            "detection_count": 0,
            "evidence": evidence,
        },
        artifact_ids=["art-1"],
    )


def test_snapshot_prefers_believed_hard_labels() -> None:
    window = _window(hard_labels=[], believed_hard_labels=["door", "screen"])
    snap = snapshot_from_window(window)
    assert snap.hard_labels == frozenset({"door", "screen"})
    assert snap.person_present is False


def test_snapshot_falls_back_to_hard_labels_without_belief() -> None:
    window = _window(hard_labels=["chair"])
    snap = snapshot_from_window(window)
    assert snap.hard_labels == frozenset({"chair"})


def test_stable_scene_when_observed_flickers_but_belief_stable() -> None:
    tracker = EvidenceTransitionTracker()
    stable = snapshot_from_window(
        _window(hard_labels=["door", "screen"], believed_hard_labels=["door", "screen"])
    )
    tracker.record_interpretation(stream_key="cam0", snapshot=stable, now=100.0)
    flicker = snapshot_from_window(
        _window(hard_labels=[], believed_hard_labels=["door", "screen"])
    )
    decision = tracker.evaluate(
        stream_key="cam0",
        snapshot=flicker,
        now=110.0,
        max_refresh_sec=0.0,
    )
    assert decision.interpret is False
    assert decision.reason == "stable_scene"


def test_salient_labels_changed_on_belief_transition() -> None:
    tracker = EvidenceTransitionTracker()
    door = snapshot_from_window(_window(hard_labels=["door"], believed_hard_labels=["door"]))
    tracker.record_interpretation(stream_key="cam0", snapshot=door, now=100.0)
    chair = snapshot_from_window(_window(hard_labels=["chair"], believed_hard_labels=["chair"]))
    decision = tracker.evaluate(
        stream_key="cam0",
        snapshot=chair,
        now=110.0,
        max_refresh_sec=0.0,
    )
    assert decision.interpret is True
    assert decision.reason == "salient_labels_changed"
