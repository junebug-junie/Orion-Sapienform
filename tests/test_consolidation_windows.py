from datetime import datetime, timedelta, timezone

from orion.consolidation.windows import compute_consolidation_window, stable_consolidation_frame_id

NOW = datetime(2026, 5, 25, 15, 37, tzinfo=timezone.utc)


def test_window_lookback_60_minutes() -> None:
    start, end = compute_consolidation_window(now=NOW, lookback_minutes=60)
    assert end == NOW
    assert start == NOW - timedelta(minutes=60)


def test_stable_frame_id() -> None:
    start = datetime(2026, 5, 25, 14, 30, tzinfo=timezone.utc)
    end = datetime(2026, 5, 25, 15, 30, tzinfo=timezone.utc)
    fid = stable_consolidation_frame_id(
        window_start=start,
        window_end=end,
        policy_id="consolidation_policy.v1",
    )
    assert fid == "consolidation.frame:2026-05-25T14:30:00+00:00:2026-05-25T15:30:00+00:00:consolidation_policy.v1"
