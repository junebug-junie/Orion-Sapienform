from __future__ import annotations

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.core.bus.codec import OrionCodec
from orion.curiosity.peer_brief_persist import persist_peer_brief
from orion.schemas.curiosity_peer import PEER_BRIEF_CHANNEL, PEER_BRIEF_KIND, PeerBriefV1


def test_persist_dual_writes_graph_and_bus_envelope() -> None:
    brief = PeerBriefV1(
        brief_id="brief-persist-1",
        help_id="help-1",
        run_id="abcd1234abcd",
        peer="cursor_auto",
        status="ok",
        summary="RO_QUERY only on Hub.",
        evidence_pointers=["orion/curiosity/worldview.py"],
    )
    graphs: list[tuple[str, dict | None]] = []
    buses: list[tuple[str, BaseEnvelope]] = []

    def graph_execute(cypher: str, params: dict | None = None) -> None:
        graphs.append((cypher, params))

    def bus_publish(channel: str, payload: BaseEnvelope) -> None:
        buses.append((channel, payload))

    result = persist_peer_brief(
        brief=brief, graph_execute=graph_execute, bus_publish=bus_publish
    )
    assert result["graph_ok"] is True
    assert result["bus_ok"] is True
    assert graphs and "PeerBrief" in graphs[0][0]
    assert graphs[0][1] is not None and graphs[0][1]["brief_id"] == "brief-persist-1"
    assert ":Prior" not in graphs[0][0]
    assert buses[0][0] == PEER_BRIEF_CHANNEL
    env = buses[0][1]
    assert isinstance(env, BaseEnvelope)
    assert env.kind == PEER_BRIEF_KIND
    assert env.payload["brief_id"] == "brief-persist-1"
    assert env.payload["schema_version"] == "curiosity.peer.brief.v1"

    # Real codec round-trip — not FakeBus dict capture alone.
    codec = OrionCodec()
    decoded = codec.decode(codec.encode(env))
    assert decoded.ok
    assert decoded.envelope.kind == PEER_BRIEF_KIND
    assert decoded.envelope.payload["brief_id"] == "brief-persist-1"
