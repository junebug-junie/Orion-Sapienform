"""Shared Retro facts across Hub's two processes (hub-app and hub-retro-gateway), in Redis.

Browser traffic for /api/retro reaches the gateway process; the leisure runner may run in
hub-app. Neither can see the other's memory, and Retro itself does not expose who is
connected. These keys are the one place both read and write:

- ``orion:retro:presence:human_at``     epoch seconds of the last human Retro action
  (new game, controller acquire/takeover, script handoff, input socket open). 1-day TTL.
- ``orion:retro:presence:browser:<id>`` one key per open relay socket (input or media),
  refreshed every BROWSER_BEAT_SEC with a BROWSER_TTL_SEC TTL, deleted on close. A crashed
  process's keys expire on their own.
- ``orion:retro:leisure:status``        the leisure runner's live status (JSON) for the page.
- ``orion:retro:leisure:ledger``        recent leisure sessions (JSON list) for the caps.
- ``orion:retro:leisure:journal``       named refusals/ends (Redis list, newest first, 200).
- ``orion:retro:leisure:lock``          one leisure session at a time, across processes.

Readers fail closed: a read error returns ``None`` and the leisure gate refuses on it.
Writers on the human path are best-effort and bounded (1 s), so play never waits on Redis.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Optional

logger = logging.getLogger('orion-hub.retro.presence')

HUMAN_AT_KEY = 'orion:retro:presence:human_at'
BROWSER_PREFIX = 'orion:retro:presence:browser:'
STATUS_KEY = 'orion:retro:leisure:status'
LEDGER_KEY = 'orion:retro:leisure:ledger'
JOURNAL_KEY = 'orion:retro:leisure:journal'
LOCK_KEY = 'orion:retro:leisure:lock'
BROWSER_BEAT_SEC = 5.0
BROWSER_TTL_SEC = 15
HUMAN_TTL_SEC = 86400
STATUS_TTL_SEC = 3600
LEDGER_MAX = 64
JOURNAL_MAX = 200
OP_TIMEOUT_SEC = 1.0


def _text(value: Any) -> Optional[str]:
    if value is None:
        return None
    return value.decode() if isinstance(value, (bytes, bytearray)) else str(value)


class PresenceStore:
    """Thin wrapper over an async Redis client (``redis.asyncio`` API subset)."""

    def __init__(self, redis: Any, *, clock=time.time):
        self.redis = redis
        self.clock = clock

    async def _op(self, coro):
        return await asyncio.wait_for(coro, OP_TIMEOUT_SEC)

    # --- human path (best effort) ---------------------------------------------------------
    async def mark_human(self) -> None:
        try:
            await self._op(self.redis.set(HUMAN_AT_KEY, f'{self.clock():.3f}', ex=HUMAN_TTL_SEC))
        except Exception:  # noqa: BLE001 -- never block or fail human play
            logger.warning('retro_presence_write_failed key=human_at', exc_info=True)

    async def beat_browser(self, socket_id: str) -> None:
        try:
            await self._op(self.redis.set(BROWSER_PREFIX + socket_id, '1', ex=BROWSER_TTL_SEC))
        except Exception:  # noqa: BLE001
            logger.warning('retro_presence_write_failed key=browser', exc_info=True)

    async def drop_browser(self, socket_id: str) -> None:
        try:
            await self._op(self.redis.delete(BROWSER_PREFIX + socket_id))
        except Exception:  # noqa: BLE001
            logger.warning('retro_presence_write_failed key=browser_drop', exc_info=True)

    # --- leisure reads (fail closed: None = unknown) --------------------------------------
    async def seconds_since_human(self) -> tuple[bool, Optional[float]]:
        """(read_ok, seconds). read_ok with None seconds: no human action in the last day."""
        try:
            raw = _text(await self._op(self.redis.get(HUMAN_AT_KEY)))
        except Exception:  # noqa: BLE001
            return False, None
        if raw is None:
            return True, None
        try:
            return True, max(0.0, self.clock() - float(raw))
        except ValueError:
            return False, None

    async def browsers_connected(self) -> Optional[int]:
        try:
            async def count():
                n = 0
                async for _ in self.redis.scan_iter(match=BROWSER_PREFIX + '*', count=100):
                    n += 1
                return n
            return await self._op(count())
        except Exception:  # noqa: BLE001
            return None

    async def ledger(self) -> Optional[list]:
        try:
            raw = _text(await self._op(self.redis.get(LEDGER_KEY)))
            rows = json.loads(raw) if raw else []
            return rows if isinstance(rows, list) else None
        except Exception:  # noqa: BLE001
            return None

    async def status(self) -> Optional[dict]:
        try:
            raw = _text(await self._op(self.redis.get(STATUS_KEY)))
            return json.loads(raw) if raw else {}
        except Exception:  # noqa: BLE001
            return None

    async def journal(self, limit: int = 20) -> Optional[list]:
        try:
            rows = await self._op(self.redis.lrange(JOURNAL_KEY, 0, max(0, limit - 1)))
            return [json.loads(_text(r)) for r in rows]
        except Exception:  # noqa: BLE001
            return None

    # --- leisure writes (raise: the runner decides) ---------------------------------------
    async def write_status(self, status: dict) -> None:
        await self._op(self.redis.set(STATUS_KEY, json.dumps(status, separators=(',', ':')), ex=STATUS_TTL_SEC))

    async def write_ledger(self, rows: list) -> None:
        await self._op(self.redis.set(LEDGER_KEY, json.dumps(rows[-LEDGER_MAX:], separators=(',', ':'))))

    async def append_journal(self, row: dict) -> None:
        await self._op(self.redis.lpush(JOURNAL_KEY, json.dumps(row, separators=(',', ':'))))
        await self._op(self.redis.ltrim(JOURNAL_KEY, 0, JOURNAL_MAX - 1))

    async def acquire_lock(self, owner: str, ttl_sec: int) -> bool:
        return bool(await self._op(self.redis.set(LOCK_KEY, owner, nx=True, ex=ttl_sec)))

    async def refresh_lock(self, owner: str, ttl_sec: int) -> bool:
        # Compare-and-expire in one step: never extend another runner's lock.
        return bool(await self._op(self.redis.eval(_REFRESH_LUA, 1, LOCK_KEY, owner, int(ttl_sec))))

    async def release_lock(self, owner: str) -> None:
        # Compare-and-delete in one step: never delete another runner's lock.
        await self._op(self.redis.eval(_RELEASE_LUA, 1, LOCK_KEY, owner))


_REFRESH_LUA = ("if redis.call('get', KEYS[1]) == ARGV[1] then "
                "return redis.call('expire', KEYS[1], ARGV[2]) else return 0 end")
_RELEASE_LUA = ("if redis.call('get', KEYS[1]) == ARGV[1] then "
                "return redis.call('del', KEYS[1]) else return 0 end")


_stores: dict[tuple[int, str], PresenceStore] = {}


def store_for(config) -> Optional[PresenceStore]:
    """One client per event loop and bus URL; None when the process has no bus URL."""
    url = getattr(config, 'ORION_BUS_URL', None)
    if not url:
        return None
    try:
        loop = id(asyncio.get_running_loop())
    except RuntimeError:
        return None
    key = (loop, str(url))
    if key not in _stores:
        import redis.asyncio as aioredis
        _stores[key] = PresenceStore(aioredis.Redis.from_url(
            str(url), socket_timeout=OP_TIMEOUT_SEC, socket_connect_timeout=OP_TIMEOUT_SEC))
    return _stores[key]
