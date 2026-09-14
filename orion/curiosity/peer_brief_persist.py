from __future__ import annotations

from typing import Any, Callable, Dict, Union

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.curiosity.peer_briefs import peer_brief_merge_cypher
from orion.schemas.curiosity_peer import PEER_BRIEF_CHANNEL, PEER_BRIEF_KIND, PeerBriefV1

GraphExecute = Callable[..., Any]
BusPublish = Callable[[str, Union[BaseEnvelope, Dict[str, Any]]], Any]


def persist_peer_brief(
    *,
    brief: PeerBriefV1,
    graph_execute: GraphExecute,
    bus_publish: BusPublish,
    source: dict[str, Any] | None = None,
) -> dict[str, bool]:
    """Dual-write PeerBrief to worldview (MERGE) and bus (sql-writer / consumers).

    Bus publish is always a Titanium ``BaseEnvelope`` with
    ``kind=PEER_BRIEF_KIND``. A bare dict would decode as ``legacy.message``
    and never route to ``CuriosityPeerBriefSQL``.
    """
    graph_ok = False
    bus_ok = False
    cypher, params = peer_brief_merge_cypher(brief)
    graph_execute(cypher, params)
    graph_ok = True
    envelope = BaseEnvelope(
        kind=PEER_BRIEF_KIND,
        source=source or {"name": "orion-curiosity-peer"},
        payload=brief.model_dump(mode="json"),
    )
    bus_publish(PEER_BRIEF_CHANNEL, envelope)
    bus_ok = True
    return {"graph_ok": graph_ok, "bus_ok": bus_ok}
