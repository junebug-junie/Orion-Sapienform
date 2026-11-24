from orion.vision.stationary import StillRun, advance, box_iou, is_stationary

SK = [2060.0, 877.0, 2274.0, 1227.0]


def test_iou_basics():
    assert box_iou(SK, SK) == 1.0
    assert box_iou(SK, [0, 0, 10, 10]) == 0.0
    # The real truncated frame of the patio prop (2026-10-15) still overlaps.
    assert 0.6 < box_iou(SK, [2080.0, 877.0, 2272.0, 1136.0]) < 0.7


def test_run_extends_while_still_and_restarts_on_move():
    run = None
    for t in range(0, 1801, 20):
        run = advance(run, SK, float(t), iou_threshold=0.5, max_gap_sec=300)
    assert run.since == 0.0 and is_stationary(run, 1800)
    moved = advance(run, [1500, 700, 1700, 1300], 1820.0, iou_threshold=0.5, max_gap_sec=300)
    assert moved.since == 1820.0 and not is_stationary(moved, 1800)


def test_gap_restarts_unless_caller_waives_it():
    run = StillRun(tuple(SK), 0.0, 100.0)
    assert advance(run, SK, 1000.0, iou_threshold=0.5, max_gap_sec=300).since == 1000.0
    assert advance(run, SK, 1000.0, iou_threshold=0.5, max_gap_sec=None).since == 0.0


def test_older_observation_leaves_run_alone():
    run = StillRun(tuple(SK), 0.0, 100.0)
    assert advance(run, [0, 0, 5, 5], 50.0) is run


def test_zero_min_sec_never_stationary():
    assert not is_stationary(StillRun(tuple(SK), 0.0, 1e6), 0)


def test_stream_rules_room_cameras_get_their_own_hold():
    from orion.vision.stationary import StreamRule, stream_rules

    rules = stream_rules("walkway, Doorbell", 1800, "cam0", 7200)
    assert rules == {"walkway": StreamRule(1800.0), "doorbell": StreamRule(1800.0),
                     "cam0": StreamRule(7200.0, room=True)}
    assert stream_rules("", 1800, "", 7200) == {}
    assert stream_rules("walkway", 0, "cam0", 0) == {}  # a hold <= 0 turns it off
    # A room listed in both lists is a room.
    assert stream_rules("cam0", 1800, ["cam0"], 7200)["cam0"].room


def test_marked_outdoor_prop_survives_gaps_a_room_one_does_not():
    from orion.vision.stationary import StreamRule

    held = StillRun(tuple(SK), 0.0, 7300.0)
    assert StreamRule(1800).gap_sec(held, 300) is None
    assert StreamRule(7200, room=True).gap_sec(held, 300) == 300
    assert StreamRule(1800).gap_sec(StillRun(tuple(SK), 0.0, 60.0), 300) == 300


def test_restart_starts_a_new_run_in_place():
    run = StillRun(tuple(SK), 0.0, 5000.0)
    nxt = advance(run, SK, 5020.0, iou_threshold=0.5, max_gap_sec=300, restart=True)
    assert nxt.since == 5020.0 and nxt.last == 5020.0
