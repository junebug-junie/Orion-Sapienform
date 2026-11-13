"""Small in-memory stand-ins for the broker's Redis and bus (tests only)."""
from __future__ import annotations

from typing import Any, Optional


class FakeRedis:
    def __init__(self, *, fail: bool = False) -> None:
        self.kv: dict[str, Any] = {}
        self.lists: dict[str, list[str]] = {}
        self.ttl: dict[str, int] = {}
        self.fail = fail

    def _check(self) -> None:
        if self.fail:
            raise ConnectionError("redis down")

    async def get(self, key: str) -> Optional[str]:
        self._check()
        return self.kv.get(key)

    async def set(self, key: str, value: str, ex: Optional[int] = None) -> None:
        self._check()
        self.kv[key] = value
        if ex:
            self.ttl[key] = ex

    async def incr(self, key: str) -> int:
        self._check()
        self.kv[key] = int(self.kv.get(key, 0)) + 1
        return self.kv[key]

    async def expire(self, key: str, sec: int) -> None:
        self._check()
        self.ttl[key] = sec

    async def lrange(self, key: str, start: int, end: int) -> list[str]:
        self._check()
        items = self.lists.get(key, [])
        return items[start: end + 1]

    async def lpush(self, key: str, value: str) -> None:
        self._check()
        self.lists.setdefault(key, []).insert(0, value)

    async def ltrim(self, key: str, start: int, end: int) -> None:
        self._check()
        self.lists[key] = self.lists.get(key, [])[start: end + 1]
