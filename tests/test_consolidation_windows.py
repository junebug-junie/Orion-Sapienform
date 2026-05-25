from datetime import datetime, timedelta, timezone

from orion.consolidation.windows import compute_consolidation_window, stable_consolidation_frame_id

NOW = datetime(2026, 5, 25, 15, 37, tzinfo=timezone.utc)


def test_window_lookback_60_minutes_aligned_to_hour() -> None:
    start, end = compute_consolidation_window(now=NOW, lookback_minutes=60)
    assert end == datetime(2026, 5, 25, 15, 0, tzinfo=timezone.utc)
    assert start == datetime(2026, 5, 25, 14, 0, tzinfo=timezone.utc)


def test_same_bucket_produces_identical_frame_id() -> None:
    t1 = datetime(2026, 5, 25, 15, 10, tzinfo=timezone.utc)
    t2 = datetime(2026, 5, 25, 15, 45, tzinfo=timezone.utc)
    s1, e1 = compute_consolidation_window(now=t1, lookback_minutes=60)
    s2, e2 = compute_consolidation_window(now=t2, lookback_minutes=60)
    assert (s1, e1) == (s2, e2)
    fid1 = stable_consolidation_frame_id(
        window_start=s1, window_end=e1, policy_id="consolidation_policy.v1"
    )
    fid2 = stable_consolidation_frame_id(
        window_start=s2, window_end=e2, policy_id="consolidation_policy.v1"
    )
    assert fid1 == fid2


def test_stable_frame_id() -> None:
    start = datetime(2026, 5, 25, 14, 30, tzinfo=timezone.utc)
    end = datetime(2026, 5, 25, 15, 30, tzinfo=timezone.utc)
    fid = stable_consolidation_frame_id(
        window_start=start,
        window_end=end,
        policy_id="consolidation_policy.v1",
    )
    assert fid == "consolidation.frame:2026-05-25T14:30:00+00:00:2026-05-25T15:30:00+00:00:consolidation_policy.v1"
