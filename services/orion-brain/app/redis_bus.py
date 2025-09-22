import time, uuid
from typing import Dict, Any, Optional
from redis.asyncio import Redis
from .config import (
    REDIS_URL, SERVICE_NAME,
    EVENTS_ENABLE, EVENTS_STREAM,
    BUS_OUT_ENABLE, BUS_OUT_STREAM,
)

_redis: Optional[Redis] = None

async def _get_redis() -> Optional[Redis]:
    global _redis
    if not REDIS_URL:
        return None
    if _redis is None:
        _redis = Redis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
        try:
            await _redis.ping()
        except Exception:
            _redis = None
    return _redis

def _now_ms() -> int:
    return int(time.time() * 1000)

def _uuid() -> str:
    return str(uuid.uuid4())

async def publish_event(kind: str, fields: Dict[str, Any]) -> None:
    if not EVENTS_ENABLE: return
    r = await _get_redis()
    if not r: return
    payload = {"sv": SERVICE_NAME, "kind": kind, "ts": _now_ms(), **fields}
    await r.xadd(EVENTS_STREAM, payload, maxlen=100_000, approximate=True)

async def publish_bus_out(topic: str, content: Dict[str, Any]) -> None:
    if not BUS_OUT_ENABLE: return
    r = await _get_redis()
    if not r: return
    payload = {"topic": topic, "ts": _now_ms(), **content}
    await r.xadd(BUS_OUT_STREAM, payload, maxlen=200_000, approximate=True)
