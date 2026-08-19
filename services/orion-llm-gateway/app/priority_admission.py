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

Two implementations live here, not one: `run_llm_chat` (llm_backend.py) is
plain `def`, not `async def` -- it's dispatched via `asyncio.to_thread` from
main.py, so it and everything it calls (including its LLM backend execution
helpers) is blocking/sync code, not compatible with the asyncio-based
`wait_for_slack`/`background_admission` above. `wait_for_slack_sync` is the
same polling logic against a blocking `httpx` call and `time.sleep`, for that
call site (used by orion-embodiment's ai-town speech, which reaches the
gateway through this bus-native path rather than the OpenAI-compatible HTTP
passthrough). It deliberately skips the per-route-key concurrency cap the
async version has: `run_llm_chat`'s only caller in that context is
orion-embodiment's speech path, which fires speech generation via
`asyncio.create_task` (not strictly serialized) but is bounded to ~1, rarely
2, concurrent calls in practice (one `active_conversation` at a time, guarded
by the `_speaking_conversations` per-conversation set) -- not a hard
guarantee, but nowhere near the kind of burst the async path's cap exists
for. Wrapping this large, heavily-branched function's full dispatch in a
semaphore for that marginal case would add real risk for little benefit.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, Optional

import httpx

from .admission_ledger import get_ledger
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


# ROADMAP A4. A deferral is the arc's whole destination -- "I wanted to think and had to wait,
# this long, while that ran instead" -- and until now it left NO TRACE. `wait_for_slack`
# returned True the instant slack appeared, however long it had blocked, and logged only on
# TIMEOUT. So the one event Track A exists to observe was computed and discarded, and A4's gate
# (">=1 admission-wait event ... with its duration") could never pass regardless of traffic.
#
# Every wait is now recorded with its duration and how many polls it took. `waited=0.0s` on the
# fast path is deliberate: it distinguishes "admitted immediately" from "never checked", which
# are different facts and were previously the same silence.
_DEFERRAL_LOG = "[LLM-GW background] admission waited=%.3fs polls=%d reserved=%d url=%s outcome=%s"


# ROADMAP A5. The log line above is a durable record for an operator and is invisible to Orion.
# Every decision is now also written to the process ledger (admission_ledger.py), which is what
# `GET /admission` reads and what puts the wait into Orion's own context. Logging and recording
# go through this one helper on purpose: six call sites emitted the log line, and any of them
# forgetting to also record would make the ledger quietly under-count the very event it exists
# to hold.
def _log_and_record(
    *,
    route_key: str,
    target: "RouteTarget",
    waited_s: float,
    polls: int,
    reserved: int,
    outcome: str,
) -> None:
    log = logger.warning if outcome == "timeout_forwarded" else logger.info
    log(_DEFERRAL_LOG, waited_s, polls, reserved, target.url, outcome)
    try:
        get_ledger().record(
            route_key=route_key,
            url=target.url,
            waited_s=waited_s,
            polls=polls,
            reserved=reserved,
            outcome=outcome,
        )
    except Exception:  # noqa: BLE001 -- bookkeeping must never break a dispatch
        logger.debug("[LLM-GW background] admission ledger record failed", exc_info=True)


async def wait_for_slack(
    target: RouteTarget,
    *,
    poll_interval_sec: float,
    max_wait_sec: float,
    route_key: str = "",
) -> bool:
    """Block until the upstream has enough free slots for background dispatch.

    Returns True if slack was confirmed, False if we're proceeding without
    confirmation (either /slots was unreachable, or we timed out waiting).
    Either way the caller should forward the request regardless -- this
    function only ever delays, never blocks a request permanently.
    """
    reserved = target.reserved_free_slots or _DEFAULT_RESERVED_FREE_SLOTS
    interval = max(poll_interval_sec, _MIN_POLL_INTERVAL_SEC)
    started = time.monotonic()
    deadline = started + max(0.0, max_wait_sec)
    polls = 0
    while True:
        free = await _free_slot_count(target.url)
        polls += 1
        if free is None:
            _log_and_record(route_key=route_key, target=target,
                            waited_s=time.monotonic() - started, polls=polls,
                            reserved=reserved, outcome="unchecked")
            return False
        if free >= reserved:
            _log_and_record(route_key=route_key, target=target,
                            waited_s=time.monotonic() - started, polls=polls,
                            reserved=reserved, outcome="admitted")
            return True
        if time.monotonic() >= deadline:
            _log_and_record(route_key=route_key, target=target,
                            waited_s=time.monotonic() - started, polls=polls,
                            reserved=reserved, outcome="timeout_forwarded")
            return False
        await asyncio.sleep(interval)


def _free_slot_count_sync(base_url: str, *, timeout_sec: float = 5.0) -> Optional[int]:
    """Blocking counterpart to _free_slot_count, for run_llm_chat's sync call path."""
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/slots", timeout=timeout_sec)
        resp.raise_for_status()
        slots = resp.json()
    except Exception as exc:  # noqa: BLE001 -- any failure here just means "can't check", not a hard error
        logger.warning("[LLM-GW background] could not read /slots at %s: %s", base_url, exc)
        return None
    if not isinstance(slots, list):
        return None
    return sum(1 for s in slots if isinstance(s, dict) and not s.get("is_processing"))


def wait_for_slack_sync(
    target: RouteTarget,
    *,
    poll_interval_sec: float,
    max_wait_sec: float,
    route_key: str = "",
) -> bool:
    """Blocking counterpart to wait_for_slack, for run_llm_chat's sync call path.

    Same contract: returns True if slack was confirmed, False if proceeding
    without confirmation (unreachable /slots or timed out) -- the caller
    forwards the request regardless either way.
    """
    reserved = target.reserved_free_slots or _DEFAULT_RESERVED_FREE_SLOTS
    interval = max(poll_interval_sec, _MIN_POLL_INTERVAL_SEC)
    started = time.monotonic()
    deadline = started + max(0.0, max_wait_sec)
    polls = 0
    while True:
        free = _free_slot_count_sync(target.url)
        polls += 1
        if free is None:
            _log_and_record(route_key=route_key, target=target,
                            waited_s=time.monotonic() - started, polls=polls,
                            reserved=reserved, outcome="unchecked")
            return False
        if free >= reserved:
            _log_and_record(route_key=route_key, target=target,
                            waited_s=time.monotonic() - started, polls=polls,
                            reserved=reserved, outcome="admitted")
            return True
        if time.monotonic() >= deadline:
            _log_and_record(route_key=route_key, target=target,
                            waited_s=time.monotonic() - started, polls=polls,
                            reserved=reserved, outcome="timeout_forwarded")
            return False
        time.sleep(interval)


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
        self._route_key = route_key
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
                route_key=self._route_key,
            )
        except BaseException:
            self._sem.release()
            raise
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._sem.release()
