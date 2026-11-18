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
