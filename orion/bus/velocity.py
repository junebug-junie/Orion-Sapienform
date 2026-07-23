from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
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
    "scan_active_channels",
]

_BUCKET_SUFFIX_RE = re.compile(r":(\d{8}T\d{4}Z)$")


def _parse_velocity_key(key: str) -> tuple[str, str] | None:
    """Split "orion:bus:velocity:{channel}:{bucket}" back into (channel,
    bucket). Channel names themselves contain colons, so this anchors on the
    fixed-format minute-bucket suffix rather than naive splitting."""
    prefix = f"{VELOCITY_KEY_PREFIX}:"
    if not key.startswith(prefix):
        return None
    remainder = key[len(prefix):]
    match = _BUCKET_SUFFIX_RE.search(remainder)
    if not match:
        return None
    channel = remainder[: match.start()]
    if not channel:
        return None
    return channel, match.group(1)


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


async def scan_active_channels(
    redis: Any,
    *,
    window_minutes: int = 5,
    now: datetime | None = None,
) -> dict[str, float]:
    """Discover every channel currently emitting velocity data and its
    trailing-window rate (msgs/sec).

    Unlike read_channel_velocity(), which requires the caller to already
    know the channel name, this SCANs the velocity key namespace itself --
    the only way to find channels that were never in the static catalog
    (Phase 2's undeclared_active) or to confirm which cataloged channels
    have zero live keys at all (declared_silent).

    Fail-open like the rest of this module: a SCAN or read error returns an
    empty result rather than raising.

    Callers get back only channels with a real (>=1) counted message in the
    window -- INCR-backed keys never hold a zero or negative value, so a
    channel present in the returned dict is always genuinely active. This is
    an implicit contract with compute_census() (orion/bus/census.py), which
    treats key presence, not value, as "active".

    SCAN cost is bounded by the full velocity-key namespace (up to
    DEFAULT_BUCKET_TTL_SEC/60 minutes of buckets per channel that has
    recently published), not by window_minutes -- a 5-minute census still
    scans and discards up to ~10 minutes of stale buckets. Fine at current
    key cardinality (~200-300 total); revisit if this ends up on a tight
    polling loop.
    """
    if window_minutes <= 0:
        return {}
    now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    allowed_buckets = {
        velocity_minute_bucket(now_utc - timedelta(minutes=offset))
        for offset in range(window_minutes)
    }

    try:
        matched_keys: list[str] = []
        channel_by_key: dict[str, str] = {}
        async for raw_key in redis.scan_iter(match=f"{VELOCITY_KEY_PREFIX}:*", count=200):
            key = raw_key.decode() if isinstance(raw_key, bytes) else raw_key
            parsed = _parse_velocity_key(key)
            if parsed is None:
                continue
            channel, bucket = parsed
            if bucket not in allowed_buckets:
                continue
            matched_keys.append(key)
            channel_by_key[key] = channel

        if not matched_keys:
            return {}
        raw_values = await redis.mget(matched_keys)
    except Exception:
        return {}

    totals: dict[str, int] = {}
    for key, val in zip(matched_keys, raw_values or []):
        if val is None:
            continue
        try:
            count = int(val)
        except (TypeError, ValueError):
            continue
        channel = channel_by_key[key]
        totals[channel] = totals.get(channel, 0) + count

    window_sec = window_minutes * 60.0
    return {channel: total / window_sec for channel, total in totals.items()}
