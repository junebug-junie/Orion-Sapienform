from __future__ import annotations

from redis.asyncio import Redis


def _inflight_key(prefix: str, trace_id: str) -> str:
    return f"{prefix}:inflight:{trace_id}"


def _done_key(prefix: str, trace_id: str) -> str:
    return f"{prefix}:done:{trace_id}"


async def is_done(redis: Redis, *, prefix: str, trace_id: str) -> bool:
    v = await redis.get(_done_key(prefix, trace_id))
    return v is not None and v != b""


async def try_claim_inflight(
    redis: Redis,
    *,
    prefix: str,
    trace_id: str,
    owner: str,
    ttl_sec: int,
) -> bool:
    """
    Returns True if this instance claimed inflight work (SETNX ok).
    Returns False if another holder already claimed.
    """
    key = _inflight_key(prefix, trace_id)
    ok = await redis.set(key, owner, nx=True, ex=int(ttl_sec))
    return bool(ok)


async def mark_done(
    redis: Redis,
    *,
    prefix: str,
    trace_id: str,
    status: str,
    ttl_sec: int,
) -> None:
    await redis.set(_done_key(prefix, trace_id), status, ex=int(ttl_sec))


async def release_inflight(redis: Redis, *, prefix: str, trace_id: str) -> None:
    await redis.delete(_inflight_key(prefix, trace_id))
