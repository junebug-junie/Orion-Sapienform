"""Relay harness governor FCC steps to per-correlation WebSocket queues."""
from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import Any, Dict, Optional, Set

from orion.core.bus.async_service import OrionBusAsync
from orion.schemas.harness_finalize import HarnessRunStepV1

logger = logging.getLogger("orion-hub.harness_step_relay")

HARNESS_RUN_STEP_KIND = "harness.run.step.v1"


class HarnessStepRelay:
    # Sweep at most this often (cheap amortized cost even under high step volume).
    _SWEEP_INTERVAL_SEC = 300.0

    def __init__(self, *, channel: str, last_seen_ttl_sec: float = 7200.0) -> None:
        self.channel = channel
        self._bus: Optional[OrionBusAsync] = None
        self._task: Optional[asyncio.Task] = None
        self._queues: Dict[str, Set[asyncio.Queue]] = defaultdict(set)
        self._last_seen: Dict[str, float] = {}
        # Safety-net TTL: _dispatch_step writes an entry for EVERY correlation_id this
        # process observes on the shared step channel, whether or not this process is the
        # one running execute_unified_turn (and therefore calling forget()) for that turn
        # (e.g. a sibling hub replica handling the same broadcast). Without this, entries
        # for turns this instance never explicitly forgets would grow unbounded.
        self._last_seen_ttl_sec = max(60.0, float(last_seen_ttl_sec))
        self._last_sweep_monotonic = 0.0

    async def start(self, bus: OrionBusAsync) -> None:
        if self._task and not self._task.done():
            return
        self._bus = bus
        self._task = asyncio.create_task(self._run(), name="hub-harness-step-relay")

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    def register_queue(self, correlation_id: str, queue: asyncio.Queue) -> None:
        self._queues[str(correlation_id)].add(queue)

    def unregister_queue(self, correlation_id: str, queue: asyncio.Queue) -> None:
        cid = str(correlation_id)
        self._queues.get(cid, set()).discard(queue)
        if cid in self._queues and not self._queues[cid]:
            self._queues.pop(cid, None)

    def forget(self, correlation_id: str) -> None:
        """Drop liveness bookkeeping for a finished correlation_id."""
        self._last_seen.pop(str(correlation_id), None)

    def seen_recently(self, correlation_id: str, *, within_sec: float) -> bool:
        """True if a harness step for this correlation_id was observed within `within_sec`."""
        ts = self._last_seen.get(str(correlation_id))
        if ts is None:
            return False
        return (time.monotonic() - ts) <= within_sec

    def _sweep_last_seen(self, *, now: float) -> None:
        """Opportunistically evict liveness entries older than the TTL. Cheap: only scans
        the dict at most once per _SWEEP_INTERVAL_SEC, not on every dispatch."""
        if now - self._last_sweep_monotonic < self._SWEEP_INTERVAL_SEC:
            return
        self._last_sweep_monotonic = now
        stale = [cid for cid, ts in self._last_seen.items() if now - ts > self._last_seen_ttl_sec]
        for cid in stale:
            self._last_seen.pop(cid, None)

    async def _run(self) -> None:
        if not self._bus:
            return
        logger.info("Subscribing to harness FCC steps: %s", self.channel)
        try:
            async with self._bus.subscribe(self.channel) as pubsub:
                async for msg in self._bus.iter_messages(pubsub):
                    decoded = self._bus.codec.decode(msg.get("data"))
                    if not decoded.ok:
                        continue
                    env = decoded.envelope
                    if str(env.kind) != HARNESS_RUN_STEP_KIND:
                        continue
                    payload = env.payload
                    if not isinstance(payload, dict):
                        continue
                    try:
                        step_event = HarnessRunStepV1.model_validate(payload)
                    except Exception:
                        logger.debug(
                            "harness step relay skipped invalid payload corr=%s",
                            payload.get("correlation_id"),
                            exc_info=True,
                        )
                        continue
                    await self._dispatch_step(step_event)
        except asyncio.CancelledError:
            logger.info("Harness step relay cancelled.")
        except Exception as exc:
            logger.error("Harness step relay loop failed: %s", exc, exc_info=True)

    async def _dispatch_step(self, step_event: HarnessRunStepV1) -> None:
        cid = str(step_event.correlation_id)
        now = time.monotonic()
        self._last_seen[cid] = now
        self._sweep_last_seen(now=now)
        queues = self._queues.get(cid)
        if not queues:
            return
        item = {
            "kind": "claude_step",
            "mode": "orion",
            "correlation_id": cid,
            "step": step_event.step,
            "step_index": step_event.step_index,
        }
        for queue in list(queues):
            try:
                queue.put_nowait(item)
            except asyncio.QueueFull:
                logger.debug(
                    "harness_step relay queue full corr=%s; dropping frame",
                    cid,
                )
