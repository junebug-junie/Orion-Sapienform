# services/orion-spark-introspector/app/introspection_guard.py
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from redis import asyncio as aioredis

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = logging.getLogger("orion-spark-introspector")

_client: Redis | None = None
_client_lock = asyncio.Lock()


def _done_key(prefix: str, trace_id: str) -> str:
    return f"{prefix}:done:{trace_id}"


def _inflight_key(prefix: str, trace_id: str) -> str:
    return f"{prefix}:inflight:{trace_id}"


def _redis_url(settings) -> str:
    return (settings.spark_introspection_redis_url or settings.orion_bus_url or "").strip()


async def get_redis_client(settings) -> Redis | None:
    """
    Shared async Redis client for idempotency keys.
    Returns None if idempotency disabled or connect/ping fails.
    """
    global _client
    if not settings.spark_introspection_idempotency_enable:
        return None
    url = _redis_url(settings)
    if not url:
        return None
    async with _client_lock:
        if _client is not None:
            try:
                await asyncio.wait_for(_client.ping(), timeout=2.0)
                return _client
            except Exception:
                try:
                    await _client.close()
                except Exception:
                    pass
                _client = None
        try:
            r = aioredis.from_url(url, decode_responses=True)
            await asyncio.wait_for(r.ping(), timeout=2.0)
            _client = r
            return _client
        except Exception as exc:
            logger.warning("spark_introspection_redis_unavailable error=%s", exc)
            return None


async def close_redis_client() -> None:
    global _client
    async with _client_lock:
        if _client is not None:
            try:
                await _client.close()
            except Exception:
                pass
            _client = None


async def is_done(redis: Redis | None, *, settings, trace_id: str) -> bool:
    """
    Return whether a done marker exists for ``trace_id``.

    On Redis read errors, returns False (fail-open for this read). Heavy work still
    relies on :func:`try_claim_inflight` (fail-closed when idempotency is enabled) to
    avoid duplicate execution across workers.
    """
    if redis is None:
        return False
    key = _done_key(settings.spark_introspection_key_prefix, trace_id)
    try:
        return bool(await redis.exists(key))
    except Exception as exc:
        logger.warning("introspection_guard is_done_failed trace_id=%s error=%s", trace_id, exc)
        return False


async def try_claim_inflight(redis: Redis | None, *, settings, trace_id: str, owner: str) -> bool:
    """
    SETNX inflight key. Returns True if this worker owns the heavy slot for trace_id.

    When idempotency is enabled, Redis must be reachable: ``redis is None`` or SET errors
    return False (fail closed) to avoid duplicate heavy introspection across workers.
    """
    if not settings.spark_introspection_idempotency_enable:
        return True
    if redis is None:
        logger.warning(
            "introspection_guard try_claim_denied trace_id=%s reason=redis_unavailable",
            trace_id,
        )
        return False
    key = _inflight_key(settings.spark_introspection_key_prefix, trace_id)
    ttl = max(1, int(settings.spark_introspection_inflight_ttl_sec))
    try:
        ok = await redis.set(key, owner, nx=True, ex=ttl)
        return bool(ok)
    except Exception as exc:
        logger.warning("introspection_guard try_claim_failed trace_id=%s error=%s", trace_id, exc)
        return False


async def mark_done(redis: Redis | None, *, settings, trace_id: str) -> None:
    if redis is None:
        return
    key = _done_key(settings.spark_introspection_key_prefix, trace_id)
    ttl = max(1, int(settings.spark_introspection_done_ttl_sec))
    try:
        await redis.set(key, "1", ex=ttl)
    except Exception as exc:
        logger.warning("introspection_guard mark_done_failed trace_id=%s error=%s", trace_id, exc)


async def release_inflight(redis: Redis | None, *, settings, trace_id: str) -> None:
    if redis is None:
        return
    key = _inflight_key(settings.spark_introspection_key_prefix, trace_id)
    try:
        await redis.delete(key)
    except Exception as exc:
        logger.warning("introspection_guard release_failed trace_id=%s error=%s", trace_id, exc)
