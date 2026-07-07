"""Minimal Hub trace consumer for ``orion:embodiment:outcome``.

The embodiment outcome channel declares ``orion-hub`` as a consumer, but the Hub
has no catch-all subscriber, so a lightweight, gated trace cache is wired here.
It keeps a bounded ring of the most recent ``EmbodimentOutcomeV1`` payloads for
debug/observability. It is NOT a UI panel and does not expose private material.

Default-off (``EMBODIMENT_OUTCOME_TRACE_ENABLED``) and fail-open: a decode or
loop error must never take down the Hub.
"""
from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Any, Deque, Dict, List, Optional

from orion.core.bus.async_service import OrionBusAsync

logger = logging.getLogger("orion-hub.embodiment_outcome_cache")

EMBODIMENT_OUTCOME_KIND = "embodiment.outcome.v1"


class EmbodimentOutcomeCache:
    def __init__(self, *, enabled: bool, channel: str, max_entries: int = 200) -> None:
        self.enabled = bool(enabled)
        self.channel = channel
        self.max_entries = int(max_entries)
        self._bus: Optional[OrionBusAsync] = None
        self._task: Optional[asyncio.Task] = None
        self._entries: Deque[Dict[str, Any]] = deque(maxlen=self.max_entries)

    async def start(self, bus: OrionBusAsync) -> None:
        if not self.enabled:
            logger.info("embodiment_outcome_cache_disabled channel=%s", self.channel)
            return
        if self._task and not self._task.done():
            return
        self._bus = bus
        self._task = asyncio.create_task(self._run(), name="hub-embodiment-outcome-cache")

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    def recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        items = list(self._entries)
        return items[-int(limit):]

    async def _run(self) -> None:
        if not self._bus:
            return
        logger.info("Subscribing to embodiment outcomes: %s", self.channel)
        try:
            async with self._bus.subscribe(self.channel) as pubsub:
                async for msg in self._bus.iter_messages(pubsub):
                    try:
                        decoded = self._bus.codec.decode(msg.get("data"))
                        if not decoded.ok:
                            continue
                        env = decoded.envelope
                        payload = env.payload if isinstance(env.payload, dict) else {}
                        self._entries.append(
                            {
                                "intent_correlation_id": payload.get("intent_correlation_id"),
                                "source": payload.get("source"),
                                "status": payload.get("status"),
                                "reason": payload.get("reason"),
                                "player_id": payload.get("player_id"),
                            }
                        )
                        logger.info(
                            "embodiment_outcome corr=%s source=%s status=%s reason=%s",
                            payload.get("intent_correlation_id"),
                            payload.get("source"),
                            payload.get("status"),
                            payload.get("reason"),
                        )
                    except Exception:
                        logger.warning("embodiment_outcome_decode_failed", exc_info=True)
                        continue
        except asyncio.CancelledError:
            logger.info("Embodiment outcome cache cancelled.")
        except Exception as exc:
            logger.error("Embodiment outcome cache loop failed: %s", exc, exc_info=True)
