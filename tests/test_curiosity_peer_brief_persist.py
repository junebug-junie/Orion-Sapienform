from __future__ import annotations

from orion.curiosity.peer_brief_persist import persist_peer_brief
from orion.schemas.curiosity_peer import PEER_BRIEF_CHANNEL, PeerBriefV1


def test_persist_dual_writes_graph_and_bus() -> None:
    brief = PeerBriefV1(
        brief_id="brief-persist-1",
        help_id="help-1",
        run_id="abcd1234abcd",
        peer="cursor_auto",
        status="ok",
        summary="RO_QUERY only on Hub.",
        evidence_pointers=["orion/curiosity/worldview.py"],
    )
    graphs: list[str] = []
    buses: list[tuple[str, dict]] = []

    def graph_execute(cypher: str) -> None:
        graphs.append(cypher)

    def bus_publish(channel: str, payload: dict) -> None:
        buses.append((channel, payload))

    result = persist_peer_brief(
        brief=brief, graph_execute=graph_execute, bus_publish=bus_publish
    )
    assert result["graph_ok"] is True
    assert result["bus_ok"] is True
    assert graphs and "PeerBrief" in graphs[0]
    assert ":Prior" not in graphs[0]
    assert buses[0][0] == PEER_BRIEF_CHANNEL
    assert buses[0][1]["brief_id"] == "brief-persist-1"
    assert buses[0][1]["schema_version"] == "curiosity.peer.brief.v1"
