from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.core.bus.velocity_keys import (
    VELOCITY_KEY_PREFIX,
    velocity_key,
    velocity_minute_bucket,
    velocity_window_keys,
)


def test_velocity_minute_bucket_truncates_seconds() -> None:
    dt = datetime(2026, 7, 23, 14, 5, 59, tzinfo=timezone.utc)
    assert velocity_minute_bucket(dt) == "20260723T1405Z"


def test_velocity_minute_bucket_normalizes_to_utc() -> None:
    # Same instant, expressed with a non-UTC offset -- must bucket identically.
    dt_utc = datetime(2026, 7, 23, 14, 5, 0, tzinfo=timezone.utc)
    dt_offset = dt_utc.astimezone(timezone(timedelta(hours=-5)))
    assert velocity_minute_bucket(dt_offset) == velocity_minute_bucket(dt_utc)


def test_velocity_key_format() -> None:
    dt = datetime(2026, 7, 23, 14, 5, 0, tzinfo=timezone.utc)
    assert velocity_key("orion:core:events", dt) == (
        f"{VELOCITY_KEY_PREFIX}:orion:core:events:20260723T1405Z"
    )


def test_velocity_window_keys_zero_or_negative_window() -> None:
    now = datetime(2026, 7, 23, 14, 5, 0, tzinfo=timezone.utc)
    assert velocity_window_keys("ch", now=now, window_minutes=0) == []
    assert velocity_window_keys("ch", now=now, window_minutes=-1) == []


def test_velocity_window_keys_covers_trailing_minutes_including_now() -> None:
    now = datetime(2026, 7, 23, 14, 5, 30, tzinfo=timezone.utc)
    keys = velocity_window_keys("ch", now=now, window_minutes=3)
    assert keys == [
        "orion:bus:velocity:ch:20260723T1405Z",
        "orion:bus:velocity:ch:20260723T1404Z",
        "orion:bus:velocity:ch:20260723T1403Z",
    ]
