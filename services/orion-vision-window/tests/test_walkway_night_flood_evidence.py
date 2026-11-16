"""#3387: what the window tells the council so a parked car flickering in the
dark is not read as a changing scene.

Live 2026-11-15 (walkway, 24 h): 2,035 belief flips, 1,333 of them after four
or more windows in a row with no detection at all. The belief tracker dropped
``vehicle``, re-entered it from empty windows alone, and dropped it again.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.schemas.vision import VisionArtifactOutputs, VisionArtifactPayload, VisionObject  # noqa: E402

from app.caption_gate import CaptionGate  # noqa: E402
from app.main import WindowService  # noqa: E402
from app.projection import summarize_items  # noqa: E402
from app.scene_belief import SceneBeliefTracker  # noqa: E402

PARKED = [1233.0, 466.0, 1347.0, 536.0]     # the far parked car, live box
PASSING = [300.0, 900.0, 900.0, 1300.0]
LIVE = dict(vote_n=3, enter_votes=3, exit_votes=0)  # live WINDOW_BELIEF_* values


def _detect(objects, *, stream="walkway", aid="det") -> VisionArtifactPayload:
    return VisionArtifactPayload(
        artifact_id=aid, correlation_id="c", task_type="retina_fast", device="cuda:0",
        inputs={"stream_id": stream},
        outputs=VisionArtifactOutputs(objects=[VisionObject(label=l, score=s, box_xyxy=b) for l, s, b in objects]),
        timing={}, model_fingerprints={},
    )


# --- belief --------------------------------------------------------------------------------------


def test_belief_does_not_oscillate_on_an_empty_scene() -> None:
    tracker = SceneBeliefTracker(**LIVE)
    for _ in range(3):
        tracker.observe(frozenset({"vehicle"}))
    assert tracker.believed_labels == frozenset({"vehicle"})
    history = [tracker.observe(frozenset()).believed_labels for _ in range(12)]
    # Two empty windows keep the belief, the third drops it, and it stays dropped.
    assert history[:2] == [frozenset({"vehicle"})] * 2
    assert history[2:] == [frozenset()] * 10
    flips = sum(1 for a, b in zip(history, history[1:]) if a != b)
    assert flips == 1


def test_one_dropped_frame_still_does_not_break_a_belief() -> None:
    """The flicker fix the empty-slot rule exists for is kept."""
    tracker = SceneBeliefTracker(**LIVE)
    for observed in ({"vehicle"}, set(), {"vehicle"}):
        result = tracker.observe(frozenset(observed))
    assert result.believed_labels == frozenset({"vehicle"})


def test_a_label_seen_again_re_enters() -> None:
    tracker = SceneBeliefTracker(**LIVE)
    for _ in range(3):
        tracker.observe(frozenset({"vehicle"}))
    for _ in range(5):
        tracker.observe(frozenset())
    assert tracker.believed_labels == frozenset()
    assert tracker.observe(frozenset({"vehicle"})).believed_labels == frozenset({"vehicle"})


# --- detector scores -----------------------------------------------------------------------------


def test_evidence_carries_the_best_detector_score_per_hard_label() -> None:
    now = time.time()
    items = [
        (_detect([("vehicle", 0.2712, PARKED), ("vehicle", 0.31, PASSING), ("person", 0.2, PASSING)], aid="a"), now),
        (_detect([("vehicle", 0.4466, PARKED), ("Dog", 0.6, PASSING)], aid="b"), now + 1),
    ]
    evidence = summarize_items(items)["evidence"]
    assert evidence["hard_labels"] == ["dog", "vehicle"]          # person 0.2 is under the 0.25 threshold
    assert evidence["label_scores"] == {"dog": 0.6, "vehicle": 0.4466}


def test_no_detection_means_no_scores() -> None:
    assert summarize_items([(_detect([]), time.time())])["evidence"]["label_scores"] == {}


# --- fixture-only labels -------------------------------------------------------------------------


def _gate() -> CaptionGate:
    return CaptionGate(streams="walkway", labels="person,vehicle", min_score=0.25, max_age_sec=20.0,
                       stationary_min_sec=1800.0, stationary_iou=0.5, stationary_max_gap_sec=300.0)


def test_parked_car_becomes_a_fixture_only_label_and_a_passing_car_clears_it() -> None:
    gate, t0 = _gate(), 1_000_000.0
    # Seen every 60 s for 31 minutes in the same box.
    for i in range(32):
        gate.apply("walkway", [(_detect([("vehicle", 0.3, PARKED)]), t0 + 60 * i)])
        if i < 30:
            assert gate.fixture_only_labels("walkway") == [], i   # not held 1800 s yet
    assert gate.fixture_only_labels("walkway") == ["vehicle"]

    # Lost for an hour (night), then re-found in the same place: still only a fixture.
    t1 = t0 + 60 * 31 + 3600
    gate.apply("walkway", [(_detect([]), t1 - 1800)])
    assert gate.fixture_only_labels("walkway") == ["vehicle"]
    gate.apply("walkway", [(_detect([("vehicle", 0.27, PARKED)]), t1)])
    assert gate.fixture_only_labels("walkway") == ["vehicle"]

    # A second vehicle somewhere else: the label is no longer fixtures only.
    gate.apply("walkway", [(_detect([("vehicle", 0.27, PARKED), ("vehicle", 0.5, PASSING)]), t1 + 5)])
    assert gate.fixture_only_labels("walkway") == []
    # It drove on; the parked car alone is fixtures only again.
    gate.apply("walkway", [(_detect([("vehicle", 0.27, PARKED)]), t1 + 400)])
    assert gate.fixture_only_labels("walkway") == ["vehicle"]


def test_streams_without_fixture_tracking_report_none() -> None:
    gate = _gate()
    assert gate.fixture_only_labels("cam0") == []
    assert gate.fixture_only_labels("") == []


@pytest.mark.asyncio
async def test_flush_publishes_fixture_only_labels_and_scores_to_the_council() -> None:
    svc = WindowService()
    t0 = time.time() - 4000
    with patch.object(svc.bus, "publish", new_callable=AsyncMock):
        for i in range(32):
            await svc._flush_and_publish(
                stream_id="walkway",
                buffered=[{"artifact": _detect([("vehicle", 0.2931, PARKED)], aid=f"a{i}"), "ts": t0 + 60 * i,
                           "env": None}],
                correlation_id=None, causality_chain=[])
    evidence = svc._live_by_stream["walkway"].summary["evidence"]
    assert evidence["fixture_only_labels"] == ["vehicle"]
    assert evidence["label_scores"] == {"vehicle": 0.2931}
    assert evidence["believed_hard_labels"] == ["vehicle"]   # belief enrichment kept both keys
