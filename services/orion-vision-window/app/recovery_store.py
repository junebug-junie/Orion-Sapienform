"""
Bounded Redis recovery index for vision-window (design §4.2, §4.3).
Stores JSON envelopes and cursors only — no raw media.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import redis.asyncio as aioredis
from loguru import logger


def _decode_envelope(raw: str) -> Dict[str, Any]:
    return json.loads(raw)


class RecoveryStore:
    def __init__(self, url: str, *, ttl_sec: int, max_n: int) -> None:
        self._url = url
        self.ttl_sec = ttl_sec
        self.max_n = max_n
        self._r: Optional[aioredis.Redis] = None

    @property
    def enabled(self) -> bool:
        return self._r is not None

    async def connect(self) -> bool:
        try:
            self._r = aioredis.from_url(self._url, decode_responses=True)
            await self._r.ping()
            logger.info("[WINDOW] Recovery Redis connected")
            return True
        except Exception as e:
            logger.error(f"[WINDOW] Recovery Redis connect failed: {e}")
            self._r = None
            return False

    async def close(self) -> None:
        if self._r:
            await self._r.aclose()
            self._r = None

    def _k(self, *parts: str) -> str:
        return ":".join(parts)

    async def persist_snapshot(
        self,
        stream_id: str,
        envelope: Dict[str, Any],
        cursor: str,
    ) -> bool:
        if not self._r:
            return False
        try:
            blob = json.dumps(envelope, separators=(",", ":"))
            p = self._r.pipeline(transaction=True)
            latest = self._k("vision-window", "stream", stream_id, "latest")
            last_n = self._k("vision-window", "stream", stream_id, "last_n")
            cur_k = self._k("vision-window", "stream", stream_id, "cursor")
            g_last = self._k("vision-window", "global", "last_n")
            g_cur = self._k("vision-window", "global", "cursor")
            health = self._k("vision-window", "health", "last_ingest")

            p.set(latest, blob, ex=self.ttl_sec)
            p.lpush(last_n, blob)
            p.ltrim(last_n, 0, self.max_n - 1)
            p.expire(last_n, self.ttl_sec)
            p.set(cur_k, cursor, ex=self.ttl_sec)

            g_blob = json.dumps({**envelope, "_stream_id": stream_id}, separators=(",", ":"))
            p.lpush(g_last, g_blob)
            p.ltrim(g_last, 0, self.max_n - 1)
            p.expire(g_last, self.ttl_sec)
            p.set(g_cur, cursor, ex=self.ttl_sec)
            p.set(health, str(envelope.get("end_ts", "")), ex=self.ttl_sec)
            await p.execute()
            return True
        except Exception as e:
            logger.warning(f"[WINDOW] Recovery write failed: {e}")
            return False

    async def read_latest(self, stream_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not self._r:
            return None
        try:
            if stream_id:
                raw = await self._r.get(self._k("vision-window", "stream", stream_id, "latest"))
            else:
                raw = await self._r.lindex(self._k("vision-window", "global", "last_n"), 0)
            if not raw:
                return None
            return _decode_envelope(raw)
        except Exception as e:
            logger.warning(f"[WINDOW] Recovery read_latest failed: {e}")
            return None

    async def read_last_n(self, stream_id: Optional[str], limit: int) -> List[Dict[str, Any]]:
        if not self._r:
            return []
        lim = max(1, min(limit, self.max_n))
        try:
            key = (
                self._k("vision-window", "stream", stream_id, "last_n")
                if stream_id
                else self._k("vision-window", "global", "last_n")
            )
            raw_list = await self._r.lrange(key, 0, lim - 1)
            return [_decode_envelope(x) for x in raw_list if x]
        except Exception as e:
            logger.warning(f"[WINDOW] Recovery read_last_n failed: {e}")
            return []

    async def read_cursors_bounds(self, stream_id: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        """Return (latest_cursor, earliest_cursor) from last_n list if parseable."""
        rows = await self.read_last_n(stream_id, self.max_n)
        cursors = [r.get("cursor") for r in rows if r.get("cursor")]
        if not cursors:
            cur_k = (
                self._k("vision-window", "stream", stream_id, "cursor")
                if stream_id
                else self._k("vision-window", "global", "cursor")
            )
            try:
                if self._r:
                    only = await self._r.get(cur_k)
                    if only:
                        return only, only
            except Exception:
                pass
            return None, None
        return cursors[0], cursors[-1]

    async def ping(self) -> bool:
        if not self._r:
            return False
        try:
            return bool(await self._r.ping())
        except Exception:
            return False
