from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from orion.core.bus.async_service import OrionBusAsync
    from app.store import CompressionStore

logger = logging.getLogger("orion.graph-compression.stale_listener")

# Map RDF enqueue graph names to compression scopes
_GRAPH_TO_SCOPE = {
    "orion:chat": "episodic",
    "orion:enrichment": "episodic",
    "orion:collapse": "episodic",
    "orion:cognition": "episodic",
    "orion:metacog": "episodic",
    "orion:chat:social": "episodic",
    "orion:autonomy:identity": "episodic",
    "orion:autonomy:drives": "episodic",
    "orion:autonomy:goals": "episodic",
    "orion:self": "self_study",
    "orion:self:induced": "self_study",
    "orion:self:reflective": "self_study",
    "orion:substrate": "substrate",
}


async def run_stale_listener(
    *,
    bus: "OrionBusAsync",
    store: "CompressionStore",
    channel_rdf_enqueue: str,
    channel_stale: str,
) -> None:
    """
    Subscribes to two channels:
    - orion:rdf:enqueue — mark affected scope stale when a graph is written
    - orion:graph:compression:stale — explicit staleness marks from other services
    """
    async def _handle(envelope: Any) -> None:
        try:
            payload = envelope.payload or {}
            # From orion:rdf:enqueue: look for graph_name field
            graph_name = (
                payload.get("graph_name")
                or payload.get("named_graph")
                or payload.get("graph")
                or ""
            )
            scope = _GRAPH_TO_SCOPE.get(graph_name)
            if scope:
                store.enqueue_stale(scope=scope, reason=f"rdf_enqueue:{graph_name}")
                logger.debug("stale_marked scope=%s graph=%s", scope, graph_name)
            else:
                # Mark all scopes stale on unknown graph writes
                for s in ("episodic", "substrate", "self_study"):
                    store.enqueue_stale(scope=s, reason="rdf_enqueue:unknown_graph")
        except Exception as exc:
            logger.warning("stale_listener_handle_error reason=%s", exc)

    async def _handle_explicit(envelope: Any) -> None:
        try:
            payload = envelope.payload or {}
            scope = payload.get("scope")
            region_id = payload.get("region_id")
            reason = payload.get("reason", "explicit_stale_mark")
            store.enqueue_stale(scope=scope, region_id=region_id, reason=reason)
        except Exception as exc:
            logger.warning("stale_listener_explicit_handle_error reason=%s", exc)

    await bus.subscribe(channel_rdf_enqueue, _handle)
    await bus.subscribe(channel_stale, _handle_explicit)
    logger.info("stale_listener_started channels=[%s, %s]", channel_rdf_enqueue, channel_stale)
