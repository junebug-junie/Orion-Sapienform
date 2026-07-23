from __future__ import annotations

from datetime import datetime, timedelta, timezone

# Redis key namespace for per-channel publish-velocity counters. Not a bus
# channel and not a schema-registered contract -- this is transport-internal
# telemetry about the bus itself, incremented at the OrionBusAsync.publish()
# seam (async_service.py, same package) and read back from
# orion.bus.velocity.read_channel_velocity(). See
# docs/superpowers/specs/2026-07-23-bus-channel-velocity-census-design.md.
#
# Lives in orion.core.bus (not orion.bus) so async_service.py's write-side
# instrumentation never has to import out of its own package -- orion.bus.*
# depends on orion.core.bus.* elsewhere in the repo (e.g. bus_observer.py),
# never the reverse; keeping the key format here preserves that direction
# instead of introducing the first orion.core.bus -> orion.bus edge.
VELOCITY_KEY_PREFIX = "orion:bus:velocity"

# Buckets outlive the longest window read_channel_velocity() is expected to
# sum over, with headroom. A bucket that expires mid-read just reads as
# zero for that minute (undercount, never overcount) -- the safe failure
# direction for best-effort telemetry that must never block a real publish.
DEFAULT_BUCKET_TTL_SEC = 600


def velocity_minute_bucket(dt: datetime) -> str:
    """Caller must pass a tz-aware datetime -- a naive value is interpreted
    as local system time by astimezone(), which would silently mis-bucket."""
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%MZ")


def velocity_key(channel: str, dt: datetime) -> str:
    return f"{VELOCITY_KEY_PREFIX}:{channel}:{velocity_minute_bucket(dt)}"


def velocity_window_keys(channel: str, *, now: datetime, window_minutes: int) -> list[str]:
    """Keys for the trailing `window_minutes` one-minute buckets ending at (and
    including) `now`'s own bucket. Pure and Redis-free -- the correctness of
    the bucketing scheme is testable without a live client.

    Callers must keep window_minutes * 60 <= DEFAULT_BUCKET_TTL_SEC: a wider
    window silently understates the rate (expired buckets read as zero, same
    as a Redis error) rather than raising."""
    if window_minutes <= 0:
        return []
    now_utc = now.astimezone(timezone.utc)
    return [
        velocity_key(channel, now_utc - timedelta(minutes=offset))
        for offset in range(window_minutes)
    ]
