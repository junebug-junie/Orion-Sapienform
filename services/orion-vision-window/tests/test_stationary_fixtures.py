"""A person box that never moves (the patio skeleton, 2026-10-15) does not
make the walkway "present"; a moving person does; room cameras are untouched."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.fixtures import FixtureTracker

SKELETON = [2060.0, 877.0, 2274.0, 1227.0]
TRUNCATED = [2080.0, 877.0, 2272.0, 1136.0]  # a real low-IoU frame of the same prop (IoU ~0.68)
WALKER = [[200.0 + 150 * k, 900.0, 320.0 + 150 * k, 1400.0] for k in range(10)]


def _tracker(**kw):
    kw.setdefault("streams", ["walkway"])
    return FixtureTracker(min_sec=1800, iou=0.5, max_gap_sec=300, **kw)


def _feed_skeleton(tr, start=0.0, minutes=35, stream="walkway"):
    verdict = None
    for k in range(int(minutes * 3)):  # one window every 20 s
        t = start + 20.0 * k
        box = TRUNCATED if k % 17 == 0 else SKELETON
        verdict = tr.observe_window(stream, [(t, [box])])
    return verdict


def test_skeleton_alone_stops_making_the_walkway_present_after_thirty_minutes() -> None:
    tr = _tracker()
    believed = frozenset({"person", "chair"})
    assert _feed_skeleton(tr, minutes=20) is False
    assert tr.presence_labels("walkway", believed) == believed
    assert _feed_skeleton(tr, start=1200.0, minutes=15) is True
    assert tr.presence_labels("walkway", believed) == frozenset({"chair"})


def test_a_moving_person_next_to_the_skeleton_is_present() -> None:
    tr = _tracker()
    _feed_skeleton(tr, minutes=35)
    t = 35 * 60.0
    assert tr.observe_window("walkway", [(t, [SKELETON, WALKER[0]])]) is False
    assert "person" in tr.presence_labels("walkway", frozenset({"person"}))


def test_skeleton_that_moves_is_no_longer_a_fixture() -> None:
    tr = _tracker()
    _feed_skeleton(tr, minutes=35)
    assert tr.observe_window("walkway", [(35 * 60.0, [[1500.0, 700.0, 1700.0, 1300.0]])]) is False


def test_window_without_people_keeps_the_last_verdict() -> None:
    tr = _tracker()
    _feed_skeleton(tr, minutes=35)
    assert tr.observe_window("walkway", [(35 * 60.0, [])]) is None
    assert tr.fixture_only("walkway") is True


def test_fixture_survives_a_night_gap_in_the_same_place() -> None:
    tr = _tracker()
    _feed_skeleton(tr, minutes=35)
    assert tr.observe_window("walkway", [(35 * 60.0 + 7200, [SKELETON])]) is True


def test_room_cameras_are_never_filtered() -> None:
    tr = _tracker()
    assert _feed_skeleton(tr, minutes=120, stream="cam0") is None
    assert tr.presence_labels("cam0", frozenset({"person"})) == frozenset({"person"})


def test_disabled_with_zero_min_sec() -> None:
    tr = FixtureTracker(streams=["walkway"], min_sec=0, iou=0.5, max_gap_sec=300)
    assert _feed_skeleton(tr) is None
