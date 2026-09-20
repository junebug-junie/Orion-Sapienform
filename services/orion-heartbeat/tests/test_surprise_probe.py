"""Pre-registered fail criteria for surprise / situation-change.

docs/research/preregistration/2026-09-20-heartbeat-surprise.md
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.substrate.surprise_probe import (
    decide_surprise,
    keep_sessions,
    l2,
    ols_slope,
    pick_controls,
    score_side,
    score_window,
    sessionize,
    window_drop,
)


TZ = timezone.utc


def _ts(minutes: int) -> datetime:
    return datetime(2026, 9, 18, 12, 0, tzinfo=TZ) + timedelta(minutes=minutes)


def test_sessionize_splits_on_thirty_minute_gap() -> None:
    stamps = [_ts(0), _ts(2), _ts(4), _ts(40), _ts(41)]
    spans = sessionize(stamps)
    assert len(spans) == 2
    assert spans[0].hub_atoms == 3
    assert spans[1].hub_atoms == 2


def test_keep_sessions_requires_duration_and_hub_count() -> None:
    short = sessionize([_ts(0), _ts(1), _ts(2), _ts(3), _ts(4), _ts(5), _ts(6), _ts(7)])
    long_enough = sessionize(
        [_ts(0), _ts(2), _ts(4), _ts(6), _ts(8), _ts(10), _ts(12), _ts(14)]
    )
    assert keep_sessions(short) == []
    kept = keep_sessions(long_enough)
    assert len(kept) == 1
    assert kept[0].hub_atoms == 8


def test_ols_slope_negative_on_decline() -> None:
    assert ols_slope([9.0, 7.0, 5.0, 3.0, 1.0]) < 0
    assert ols_slope([1.0, 3.0, 5.0, 7.0, 9.0]) > 0
    assert ols_slope([2.0, 2.0, 2.0, 2.0]) == 0.0


def test_drop_is_head_minus_tail() -> None:
    values = [6.0, 6.0, 6.0, 3.0, 3.0, 1.0, 1.0, 1.0]
    assert abs(window_drop(values) - 5.0) < 1e-12


def test_score_window_none_if_too_short() -> None:
    assert score_window(kind="session", surprises=[0.2] * 7) is None
    scored = score_window(kind="session", surprises=[0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    assert scored is not None
    assert scored.negative_slope is True


def _side(slopes: list[float], drops: list[float], kind: str = "session"):
    windows = []
    for slope, drop in zip(slopes, drops):
        # 8 declining or flat points shaped to hit the requested slope sign
        if slope < 0:
            surprises = [1.0 - 0.05 * i for i in range(8)]
        elif slope > 0:
            surprises = [0.1 + 0.05 * i for i in range(8)]
        else:
            surprises = [0.4] * 8
        scored = score_window(kind=kind, surprises=surprises)
        assert scored is not None
        windows.append(scored)
    # overwrite slope/drop via score_side on constructed WindowScores
    from app.substrate.surprise_probe import WindowScore

    forced = [
        WindowScore(
            kind=kind,
            slope=s,
            drop=d,
            n_surprise=8,
            negative_slope=s < 0,
        )
        for s, d in zip(slopes, drops)
    ]
    return score_side(forced)


def test_side_settles_when_median_neg_and_three_quarters() -> None:
    side = _side([-0.2] * 6 + [0.1] * 2, [0.3] * 8)
    assert side.settles is True
    assert side.n == 8


def test_side_does_not_settle_at_half() -> None:
    side = _side([-0.2] * 4 + [0.1] * 4, [0.1] * 8)
    assert side.settles is False


def test_unverified_below_eight_contexts() -> None:
    sess = _side([-0.2] * 7, [0.4] * 7)
    ctrl = _side([0.1] * 8, [0.0] * 8)
    decided = decide_surprise(sessions=sess, controls=ctrl)
    assert decided.reason.startswith("UNVERIFIED")
    assert decided.holds is False


def test_saturation_when_both_sides_settle() -> None:
    sess = _side([-0.2] * 8, [0.4] * 8)
    ctrl = _side([-0.1] * 8, [0.1] * 8)
    decided = decide_surprise(sessions=sess, controls=ctrl)
    assert "saturation" in decided.reason
    assert decided.holds is False
    assert decided.thermometer is False


def test_holds_when_sessions_settle_and_drop_beats_control() -> None:
    sess = _side([-0.2] * 8, [0.20] * 8)
    ctrl = _side([0.05] * 8, [0.05] * 8)
    decided = decide_surprise(sessions=sess, controls=ctrl)
    assert decided.holds is True
    assert decided.thermometer is False


def test_mixed_when_settle_but_drop_too_small() -> None:
    sess = _side([-0.2] * 8, [0.06] * 8)
    ctrl = _side([0.05] * 8, [0.05] * 8)
    decided = decide_surprise(sessions=sess, controls=ctrl)
    assert decided.mixed is True
    assert "not louder" in decided.reason


def test_thermometer_when_neither_clause_holds() -> None:
    sess = _side([0.1] * 8, [0.04] * 8)
    ctrl = _side([0.05] * 8, [0.03] * 8)
    decided = decide_surprise(sessions=sess, controls=ctrl)
    assert decided.thermometer is True


def test_l2_and_control_clearance() -> None:
    assert abs(l2([0.0, 3.0], [4.0, 0.0]) - 5.0) < 1e-12
    sessions = keep_sessions(
        sessionize([_ts(0), _ts(2), _ts(4), _ts(6), _ts(8), _ts(10), _ts(12), _ts(14)])
    )
    # candidate inside the session is rejected; one far away is kept
    far = _ts(80)
    near = _ts(5)
    picked = pick_controls(sessions=sessions, window_starts=[near, far], want=2)
    assert far in picked
    assert near not in picked
