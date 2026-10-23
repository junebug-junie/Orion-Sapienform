"""People-count estimators and decision 6's "clearly a person" glimpse rule (spec PR #2763)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.vision.walkway_count import (
    CLEAR_GLIMPSE_MIN_BOX_H_FRAC,
    CLEAR_GLIMPSE_MIN_SCORE,
    ESTIMATORS,
    clear_detection,
    clear_glimpse,
    estimate_all,
    presence_runs,
    story_count,
    within_range,
)

T0 = datetime(2026, 10, 20, 18, 0, tzinfo=timezone.utc)
FRAME_H = 1920.0


def _det(score=0.5, h=300.0, zone="sidewalk", clipped=False, iid=None, label="person"):
    return {"label": label, "score": score, "box": [10.0, 500.0, 90.0, 500.0 + h], "zone": zone,
            "clipped": clipped, "individual_id": iid}


def _story(reason, at=0.0, dur=0.0, dets=None, kind="person", iid=None, frame_h=None):
    a = T0 + timedelta(seconds=at)
    return {"story_id": f"{reason}{at}", "kind": kind, "drop_reason": reason, "started_at": a,
            "ended_at": a + timedelta(seconds=dur), "evidence": dets if dets is not None else [_det()],
            "individual_id": iid, "frame_h": frame_h}


def test_a_detection_is_clear_only_at_or_above_both_thresholds() -> None:
    h_min = CLEAR_GLIMPSE_MIN_BOX_H_FRAC * FRAME_H  # 120 px
    assert clear_detection(_det(score=CLEAR_GLIMPSE_MIN_SCORE, h=h_min))
    assert not clear_detection(_det(score=CLEAR_GLIMPSE_MIN_SCORE - 0.001, h=h_min))
    assert not clear_detection(_det(score=0.9, h=h_min - 1))
    # The live "wrong" stories: score 0.36-0.40, boxes 55-77 px.
    assert not clear_detection(_det(score=0.397, h=77.0))


def test_clipped_outside_every_zone_prop_or_not_a_person_is_never_clear() -> None:
    assert not clear_detection(_det(clipped=True))
    assert not clear_detection(_det(zone=None))
    assert not clear_detection(_det(iid="prop1"), prop_ids=["prop1"])
    assert not clear_detection(_det(label="dog"))
    assert not clear_detection({**_det(), "box": [1, 2, 3]})
    assert not clear_detection({**_det(), "score": None})


def test_box_height_is_a_share_of_the_story_frame() -> None:
    # 120 px is clear in a 1920 px frame but not in a 3840 px one.
    assert clear_detection(_det(h=120.0), frame_h=1920)
    assert not clear_detection(_det(h=120.0), frame_h=3840)


def test_clear_glimpse_needs_a_person_glimpse_with_one_clear_detection() -> None:
    assert clear_glimpse(_story("glimpse", dets=[_det(score=0.36), _det(score=0.5)]))
    assert not clear_glimpse(_story("glimpse", dets=[_det(score=0.36)]))
    assert not clear_glimpse(_story(None))           # published is not a glimpse
    assert not clear_glimpse(_story("glimpse", kind="dog"))
    assert not clear_glimpse(_story("glimpse", iid="prop1"), prop_ids=["prop1"])
    assert not clear_glimpse(_story("glimpse", dets=[]))


def test_story_count_counts_tracks_and_isolated_clear_glimpses_only() -> None:
    stories = [
        _story(None, at=0, dur=20),                         # a walker
        _story("glimpse", at=25),                           # flicker 5 s after the walker: covered
        _story("glimpse", at=300),                          # someone the tracker saw once: counted
        _story("glimpse", at=306),                          # same moment again: covered by the one above
        _story("glimpse", at=900, dets=[_det(score=0.37)]),  # unclear
        _story("prop", at=1200, dur=600),                   # the skeleton
        _story("standing", at=2000, dur=90),
    ]
    sc = story_count(stories)
    assert (sc.multi_frame, sc.clear_glimpses, sc.covered_glimpses, sc.unclear_glimpses) == (2, 1, 2, 1)
    assert sc.total == 3


def test_presence_runs_split_on_gaps_over_30_seconds() -> None:
    w = [(T0, 1), (T0 + timedelta(seconds=30), 2), (T0 + timedelta(seconds=61), 1), (T0 + timedelta(seconds=70), 0)]
    assert presence_runs(w) == [(T0, T0 + timedelta(seconds=30), 2),
                                (T0 + timedelta(seconds=61), T0 + timedelta(seconds=61), 1)]
    assert presence_runs([]) == []


def test_empty_hour_reads_zero_for_every_estimator() -> None:
    assert estimate_all(clusters=0, most_at_once=0, stories=[], windows=[]) == {k: 0 for k in ESTIMATORS}


def test_a_person_standing_two_minutes_stays_one() -> None:
    start = T0 + timedelta(minutes=5)
    standing = _story("standing", at=300, dur=120, dets=[_det() for _ in range(25)])
    flicker = [_story("glimpse", at=300 + s, dets=[_det(score=0.7)]) for s in (15, 60, 119)]
    windows = [(start + timedelta(seconds=5 * i), 1) for i in range(25)]
    est = estimate_all(clusters=5, most_at_once=1, stories=[standing, *flicker], windows=windows)
    for k in ("stories", "stories_in_range", "stories_multi_only", "run_peaks", "runs", "most_at_once"):
        assert est[k] == 1, k
    assert est["clusters"] == 5  # why clusters cannot be the estimate


def test_a_story_estimate_stays_inside_the_range() -> None:
    assert within_range(2, 3, 14) == 3     # never below most at once
    assert within_range(20, 3, 14) == 14   # never above the clusters
    assert within_range(7, 3, 14) == 7
    assert within_range(0, 0, 0) == 0
    assert within_range(1, 3, 2) == 3      # high below low: the low wins (as count_range raises it)
