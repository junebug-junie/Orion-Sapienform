"""Background-priority admission gate for llm-gateway routes.

See docs/superpowers/specs/2026-07-30-llm-gateway-background-priority-design.md
and README.md's "Background-priority routes" section: AI Town's NPC dialogue
shares GPU1's `quick` llama.cpp process (`atlas-worker-fast-1`) with several
other, snappier consumers (orion-mind, orion-hub, orion-embodiment hub-speech),
and no second GPU is available to give it its own dedicated model. Rather than
compete evenly for atlas-worker-fast-1's 4 continuous-batching slots, a
background-tagged route (RouteTarget.priority == "background") waits for slot
slack before dispatching -- llama.cpp's own `/slots` endpoint is the live
source of truth for "is there room right now," and every existing
(non-background) route is completely unaffected by any of this.

This is deliberately a fail-open gate, not a hard block: an unreachable
`/slots` endpoint or a permanently busy upstream never silently drops a
caller's request -- it just proceeds without the wait, same as
orion-embodiment's existing fail-open speech path.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, Optional

import httpx

from .llm_backend import RouteTarget

logger = logging.getLogger("orion-llm-gateway.priority_admission")

_DEFAULT_RESERVED_FREE_SLOTS = 1
_MIN_POLL_INTERVAL_SEC = 0.05  # floor so a misconfigured 0/negative interval can't hot-loop /slots

_semaphores: Dict[str, asyncio.Semaphore] = {}


def _semaphore_for(route_key: str, concurrency: int) -> asyncio.Semaphore:
    """One semaphore per route key, created lazily and reused for the process lifetime."""
    sem = _semaphores.get(route_key)
    if sem is None:
        sem = asyncio.Semaphore(max(1, concurrency))
        _semaphores[route_key] = sem
    return sem


async def _free_slot_count(base_url: str, *, timeout_sec: float = 5.0) -> Optional[int]:
    """Return the number of currently-idle slots, or None if /slots is unreachable/disabled."""
    try:
        async with httpx.AsyncClient(timeout=timeout_sec) as client:
            resp = await client.get(f"{base_url.rstrip('/')}/slots")
            resp.raise_for_status()
            slots = resp.json()
    except Exception as exc:  # noqa: BLE001 -- any failure here just means "can't check", not a hard error
        logger.warning("[LLM-GW background] could not read /slots at %s: %s", base_url, exc)
        return None
    if not isinstance(slots, list):
        return None
    return sum(1 for s in slots if isinstance(s, dict) and not s.get("is_processing"))


async def wait_for_slack(
    target: RouteTarget,
    *,
    poll_interval_sec: float,
    max_wait_sec: float,
) -> bool:
    """Block until the upstream has enough free slots for background dispatch.

    Returns True if slack was confirmed, False if we're proceeding without
    confirmation (either /slots was unreachable, or we timed out waiting).
    Either way the caller should forward the request regardless -- this
    function only ever delays, never blocks a request permanently.
    """
    reserved = target.reserved_free_slots or _DEFAULT_RESERVED_FREE_SLOTS
    interval = max(poll_interval_sec, _MIN_POLL_INTERVAL_SEC)
    deadline = time.monotonic() + max(0.0, max_wait_sec)
    while True:
        free = await _free_slot_count(target.url)
        if free is None:
            return False
        if free >= reserved:
            return True
        if time.monotonic() >= deadline:
            logger.warning(
                "[LLM-GW background] timed out after %.1fs waiting for %d free slot(s) on %s -- forwarding anyway",
                max_wait_sec, reserved, target.url,
            )
            return False
        await asyncio.sleep(interval)


class background_admission:
    """Async context manager: caps concurrent background requests to one
    upstream (so a burst of, say, several NPCs speaking at once can't claim
    more than `concurrency` slots) and waits for slot slack before letting
    the caller proceed.

    Usage:
        async with background_admission(route_key, target, concurrency=1,
                                         poll_interval_sec=0.5, max_wait_sec=30):
            ...forward the request...
    """

    def __init__(
        self,
        route_key: str,
        target: RouteTarget,
        *,
        concurrency: int,
        poll_interval_sec: float,
        max_wait_sec: float,
    ) -> None:
        self._target = target
        self._poll_interval_sec = poll_interval_sec
        self._max_wait_sec = max_wait_sec
        self._sem = _semaphore_for(route_key, concurrency)

    async def __aenter__(self) -> "background_admission":
        await self._sem.acquire()
        # If the wait itself is cancelled (e.g. client disconnect mid-poll),
        # __aexit__ never runs since __aenter__ never returned -- release
        # here or the acquired permit leaks forever, wedging this route's
        # background lane at its concurrency cap until process restart.
        try:
            await wait_for_slack(
                self._target,
                poll_interval_sec=self._poll_interval_sec,
                max_wait_sec=self._max_wait_sec,
            )
        except BaseException:
            self._sem.release()
            raise
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._sem.release()
