from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from orion.core.bus.async_service import OrionBusAsync
    from app.store import CompressionStore

logger = logging.getLogger("orion.graph-compression.stale_listener")

# orion:rdf:enqueue subscription and its _GRAPH_TO_SCOPE mapping removed
# 2026-07-28: orion-rdf-writer (the only other consumer of that channel) was
# retired, and live verification of this listener's own consumption found it
# was already dead weight -- the Redis stream key didn't exist (zero messages
# ever), and the one real producer (orion-cortex-exec's self_concept_induce/
# reflect verbs) wrote graph names this mapping didn't even recognize, so it
# only ever hit the "mark everything stale" fallback branch, and never
# observably did so (stale_queue was empty). See
# orion/bus/channels.yaml's orion:rdf:enqueue entry for the full writeup.


async def run_stale_listener(
    *,
    bus: "OrionBusAsync",
    store: "CompressionStore",
    channel_stale: str,
) -> None:
    """
    Subscribes to orion:graph:compression:stale — explicit staleness marks
    from other services.
    """
    async def _handle_explicit(envelope: Any) -> None:
        try:
            payload = envelope.payload or {}
            scope = payload.get("scope")
            region_id = payload.get("region_id")
            reason = payload.get("reason", "explicit_stale_mark")
            store.enqueue_stale(scope=scope, region_id=region_id, reason=reason)
        except Exception as exc:
            logger.warning("stale_listener_explicit_handle_error reason=%s", exc)

    await bus.subscribe(channel_stale, _handle_explicit)
    logger.info("stale_listener_started channels=[%s]", channel_stale)
