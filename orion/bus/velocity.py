from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from orion.core.bus.velocity_keys import (
    DEFAULT_BUCKET_TTL_SEC,
    VELOCITY_KEY_PREFIX,
    velocity_key,
    velocity_minute_bucket,
    velocity_window_keys,
)

__all__ = [
    "DEFAULT_BUCKET_TTL_SEC",
    "VELOCITY_KEY_PREFIX",
    "velocity_key",
    "velocity_minute_bucket",
    "velocity_window_keys",
    "read_channel_velocity",
]


async def read_channel_velocity(
    redis: Any,
    channel: str,
    *,
    window_minutes: int = 5,
    now: datetime | None = None,
) -> float:
    """Messages/sec for `channel`, averaged over the trailing `window_minutes`.

    Best-effort read: a Redis error, or any individual bucket being missing
    or expired, counts as zero for that bucket rather than raising. This
    mirrors the write side's fail-open behavior in
    OrionBusAsync._record_velocity() (orion/core/bus/async_service.py) --
    broken telemetry must never look like a broken bus.

    window_minutes should stay <= DEFAULT_BUCKET_TTL_SEC / 60 (10 minutes at
    the current default) or older buckets will have already expired,
    understating the rate rather than erroring.
    """
    if window_minutes <= 0:
        return 0.0
    keys = velocity_window_keys(
        channel, now=now or datetime.now(timezone.utc), window_minutes=window_minutes
    )
    try:
        raw = await redis.mget(keys)
    except Exception:
        return 0.0
    total = 0
    for val in raw or []:
        if val is None:
            continue
        try:
            total += int(val)
        except (TypeError, ValueError):
            continue
    return total / (window_minutes * 60.0)
