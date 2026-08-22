"""`object_counts` must estimate what is in the room, not how many frames ran.

Regression for a live defect found 2026-08-21. `summarize_items` summed every
detection across the window, so the value scaled with `item_count`. Measured on
cam0 over 20 consecutive windows of a static room: `item_count=1` -> `chair: 2`
(~6.4 detections total), `item_count=2` -> `chair: 4` (~13.6). Exactly 2x.

That reached Orion. Consecutive `vision_events` narratives of the same untouched
room read "Two chairs, two doors, a desk" then "Four chairs, two doors, two
desks" then back again -- the furniture appearing to double and halve every few
minutes. It would have made any object-permanence work measure the frame rate.
"""

from __future__ import annotations

from orion.schemas.vision import (
    VisionArtifactOutputs,
    VisionArtifactPayload,
    VisionObject,
)

from app.projection import summarize_items


def _artifact(labels: list[str]) -> VisionArtifactPayload:
    """Real payload, not a stub -- matches test_projection.py's helper.

    `_skip_edge_artifact` filters on real fields, so a hand-rolled fake could
    pass this test while the live path skipped the artifact entirely.
    """
    return VisionArtifactPayload(
        artifact_id="a1",
        correlation_id="c1",
        task_type="detect",
        device="cam-1",
        inputs={},
        outputs=VisionArtifactOutputs(
            objects=[
                VisionObject(label=x, score=0.9, box_xyxy=[0.0, 0.0, 1.0, 1.0])
                for x in labels
            ]
        ),
        timing={},
        model_fingerprints={},
    )


def _items(frames: list[list[str]]):
    return [(_artifact(f), float(i)) for i, f in enumerate(frames)]


def test_count_is_stable_as_frames_per_window_change() -> None:
    """The room has two chairs whether the window caught 1, 2 or 5 frames.

    This is the exact live scenario, hand-built: a static room with 2 chairs and
    1 desk visible in every frame.
    """
    room = ["chair", "chair", "desk"]
    for n_frames in (1, 2, 3, 5):
        summary = summarize_items(_items([room] * n_frames))
        assert summary["object_counts"]["chair"] == 2, (
            f"{n_frames} frames of a 2-chair room reported "
            f"{summary['object_counts']['chair']} chairs -- the count is "
            "tracking the frame rate, not the room"
        )
        assert summary["object_counts"]["desk"] == 1
        assert summary["item_count"] == n_frames


def test_the_old_summing_behaviour_would_fail_this() -> None:
    """Pins the specific arithmetic, so a revert to summing is unambiguous.

    Old behaviour on 2 frames of a 2-chair room: chair=4. New: chair=2.
    """
    summary = summarize_items(_items([["chair", "chair"]] * 2))
    assert summary["object_counts"]["chair"] == 2
    assert summary["object_counts"]["chair"] != 4, "summed across frames again"


def test_raw_detection_tally_is_still_available_under_an_honest_name() -> None:
    """The sum is not wrong, it is just not an object count. Keep it, name it."""
    summary = summarize_items(_items([["chair", "chair"]] * 3))
    assert summary["object_counts"]["chair"] == 2      # two chairs
    assert summary["label_detections"]["chair"] == 6   # six detections fired
    assert summary["detection_count"] == 6


def test_max_not_mean_so_a_missed_frame_does_not_drag_the_estimate_down() -> None:
    """A detector that misses an object in one frame must not halve the count.

    Two chairs visible in frame 1, detector finds only one in frame 2.
    Mean would say 1.5 -> 1. Max says 2, which is correct.
    """
    summary = summarize_items(_items([["chair", "chair"], ["chair"]]))
    assert summary["object_counts"]["chair"] == 2


def test_top_labels_ranks_by_the_corrected_count() -> None:
    summary = summarize_items(_items([["chair", "chair", "chair", "door"]] * 4))
    assert summary["top_labels"][0] == ("chair", 3)


def test_empty_window_is_not_an_error() -> None:
    summary = summarize_items([])
    assert summary["object_counts"] == {}
    assert summary["detection_count"] == 0
    assert summary["item_count"] == 0
