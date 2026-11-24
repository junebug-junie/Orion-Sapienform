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


# --- room cameras (the office, #3614) -----------------------------------------


def _room():
    return FixtureTracker(streams=["walkway"], min_sec=1800, iou=0.5, max_gap_sec=300,
                          room_streams="cam0", room_min_sec=7200)


def test_room_box_pile_still_for_four_hours_is_a_prop():
    """2026-11-15..16: boxes on the office floor read as `person` in one box
    for 4-5 h. Past the room hold (2 h) they no longer make the office present."""
    tr = _room()
    assert _feed_skeleton(tr, minutes=110, stream="cam0") is False
    assert _feed_skeleton(tr, start=110 * 60.0, minutes=20, stream="cam0") is True
    assert tr.presence_labels("cam0", frozenset({"person", "chair"})) == frozenset({"chair"})


def test_room_person_who_sat_still_is_present_again_after_leaving():
    """The review's lifecycle: Juniper sits still 2.5 h (her seat becomes a
    prop), leaves for an hour, comes back to the same box. On a room camera a
    gap ends the run, so she counts as present at once."""
    tr = _room()
    assert _feed_skeleton(tr, minutes=150, stream="cam0") is True
    back = 150 * 60.0 + 3600.0
    assert tr.observe_window("cam0", [(back, [SKELETON])]) is False
    assert tr.presence_labels("cam0", frozenset({"person"})) == frozenset({"person"})


def test_outdoor_prop_still_survives_the_same_gap():
    tr = _room()
    assert _feed_skeleton(tr, minutes=35, stream="walkway") is True
    assert tr.observe_window("walkway", [(35 * 60.0 + 3600.0, [SKELETON])]) is True


def test_room_face_restarts_the_run():
    """A box with a face is a person: the run starts over, so a person who
    sits still but shows their face is never a prop."""
    tr = _room()
    assert _feed_skeleton(tr, minutes=150, stream="cam0") is True
    t = 150 * 60.0 + 20.0
    assert tr.observe_window("cam0", [(t, [SKELETON], [True])]) is False
    assert tr.observe_window("cam0", [(t + 20.0, [SKELETON], [False])]) is False


def test_face_does_not_restart_an_outdoor_prop():
    tr = _room()
    assert _feed_skeleton(tr, minutes=35, stream="walkway") is True
    assert tr.observe_window("walkway", [(35 * 60.0 + 20.0, [SKELETON], [True])]) is True


def test_room_person_at_the_desk_for_an_hour_and_a_half_is_a_person():
    tr = _room()
    assert _feed_skeleton(tr, minutes=90, stream="cam0") is False
