"""Cortex-exec (D) background embodiment emit + perception read-model.

Thin seam: a latest-wins TTL cache of ``WorldPerceptionV1`` (fed by a gated
consumer on ``orion:embodiment:perception``) plus a pure helper that maps the
cached perception into a ``source=deliberate`` embodiment intent. A gated
background loop publishes that intent to ``orion:embodiment:intent``.

Everything is default-off and fail-open: a broken perception payload or a
publish failure must never crash the cortex-exec runtime.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Callable, Optional
from uuid import uuid4

from orion.embodiment.intents import build_intent
from orion.schemas.embodiment import (
    EMBODIMENT_INTENT_KIND,
    EmbodimentIntentV1,
    WorldPerceptionV1,
)

logger = logging.getLogger("orion.cortex.exec.embodiment_background")

EMBODIMENT_INTENT_CHANNEL = "orion:embodiment:intent"
DEFAULT_PERCEPTION_TTL_SEC = 30.0


@dataclass
class _Entry:
    perception: WorldPerceptionV1
    stored_at: float


class LatestPerceptionCache:
    """Latest-wins perception cache keyed by ``player_id`` with TTL eviction."""

    def __init__(
        self,
        *,
        ttl_sec: float = DEFAULT_PERCEPTION_TTL_SEC,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = float(ttl_sec)
        self._clock = clock
        self._entries: dict[str, _Entry] = {}

    def put(self, perception: WorldPerceptionV1) -> None:
        self._entries[perception.player_id] = _Entry(perception, self._clock())

    def get(self, player_id: str) -> Optional[WorldPerceptionV1]:
        entry = self._entries.get(player_id)
        if entry is None:
            return None
        if self._clock() - entry.stored_at > self._ttl:
            self._entries.pop(player_id, None)
            return None
        return entry.perception

    def latest(self) -> Optional[WorldPerceptionV1]:
        """Most-recently stored non-expired perception across all players."""
        now = self._clock()
        best: Optional[_Entry] = None
        for pid, entry in list(self._entries.items()):
            if now - entry.stored_at > self._ttl:
                self._entries.pop(pid, None)
                continue
            if best is None or entry.stored_at > best.stored_at:
                best = entry
        return best.perception if best is not None else None


def build_background_embodiment_intent(
    perception: WorldPerceptionV1, correlation_id: str
) -> EmbodimentIntentV1:
    """Pure map: cached perception -> deliberate (D) intent.

    Approaches the nearest nearby player when present, otherwise wanders. The
    ``reason`` is always non-empty (anti empty-shell contract).
    """
    nearby = perception.nearby_players or []
    if nearby:
        nearest = min(nearby, key=lambda n: float(n.get("distance", math.inf)))
        ref = nearest.get("name") or nearest.get("player_id")
        return build_intent(
            kind="approach_player",
            source="deliberate",
            reason=f"background drive: approach nearest player {ref}",
            correlation_id=correlation_id,
            ref=str(ref) if ref is not None else None,
            player_id=perception.player_id,
        )
    return build_intent(
        kind="wander",
        source="deliberate",
        reason="background drive: no players nearby, wander",
        correlation_id=correlation_id,
        player_id=perception.player_id,
    )


_cache: LatestPerceptionCache | None = None


def get_perception_cache() -> LatestPerceptionCache:
    global _cache
    if _cache is None:
        _cache = LatestPerceptionCache()
    return _cache


def ingest_perception_payload(payload: object) -> Optional[WorldPerceptionV1]:
    """Validate + store a perception payload; fail-open (returns None on error)."""
    try:
        perception = WorldPerceptionV1.model_validate(payload or {})
    except Exception:
        logger.warning("embodiment_perception_ingest_invalid", exc_info=True)
        return None
    get_perception_cache().put(perception)
    return perception


async def maybe_emit_background_intent(
    *,
    bus,
    source,
    player_id: str,
    enabled: bool,
    channel: str = EMBODIMENT_INTENT_CHANNEL,
) -> Optional[EmbodimentIntentV1]:
    """Gated + fail-open background (D) emit from the cached perception."""
    if not enabled:
        return None
    try:
        perception = get_perception_cache().get(player_id) or get_perception_cache().latest()
        if perception is None:
            return None
        intent = build_background_embodiment_intent(perception, correlation_id=str(uuid4()))
        from orion.core.bus.bus_schemas import BaseEnvelope

        envelope = BaseEnvelope(
            kind=EMBODIMENT_INTENT_KIND,
            source=source,
            correlation_id=uuid4(),
            payload=intent.model_dump(mode="json"),
        )
        await bus.publish(channel, envelope)
        logger.info(
            "embodiment_background_intent_published kind=%s ref=%s player=%s",
            intent.kind,
            intent.ref,
            player_id,
        )
        return intent
    except Exception:
        logger.warning("embodiment_background_intent_emit_failed", exc_info=True)
        return None
