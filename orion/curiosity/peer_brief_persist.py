from __future__ import annotations

from typing import Any, Callable, Dict

from orion.curiosity.peer_briefs import peer_brief_merge_cypher
from orion.schemas.curiosity_peer import PEER_BRIEF_CHANNEL, PeerBriefV1

GraphExecute = Callable[[str], Any]
BusPublish = Callable[[str, Dict[str, Any]], Any]


def persist_peer_brief(
    *,
    brief: PeerBriefV1,
    graph_execute: GraphExecute,
    bus_publish: BusPublish,
) -> dict[str, bool]:
    """Dual-write PeerBrief to worldview (MERGE) and bus (sql-writer / consumers)."""
    graph_ok = False
    bus_ok = False
    graph_execute(peer_brief_merge_cypher(brief))
    graph_ok = True
    payload = brief.model_dump(mode="json")
    bus_publish(PEER_BRIEF_CHANNEL, payload)
    bus_ok = True
    return {"graph_ok": graph_ok, "bus_ok": bus_ok}
