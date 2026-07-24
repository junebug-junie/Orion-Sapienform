"""Read-only debug API for the bus synaptic graph (services/orion-bus-mirror).

Surfaces structure that already exists in the live FalkorDB graph but had no
human-visible view -- the "Idea 5" starting point from
docs/superpowers/specs/2026-07-24-bus-vitality-field-signal-brainstorm.md's
Phase 3+ brainstorm: before building any new signal, show what's already
sitting in the data. No new edge/node type, no write path -- pure read.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from orion.graph.falkor_client import RedisGraphQueryClient

from app.settings import settings

router = APIRouter(prefix="/api/bus-synaptic-graph", tags=["bus-synaptic-graph"])


def _client() -> RedisGraphQueryClient:
    uri = (settings.FALKORDB_URI or "").strip()
    if not uri:
        raise HTTPException(status_code=503, detail="falkordb_uri_not_configured")
    graph_name = getattr(settings, "FALKORDB_BUS_GRAPH", "") or "orion_bus_synapse"
    return RedisGraphQueryClient(uri=uri, graph_name=graph_name)


@router.get("/summary")
async def summary() -> dict[str, Any]:
    client = _client()
    organs = client.graph_query("MATCH (o:Organ) RETURN count(o) AS c")
    channels = client.graph_query("MATCH (c:Channel) RETURN count(c) AS c")
    verbs = client.graph_query("MATCH (v:Verb) RETURN count(v) AS c")
    publishes = client.graph_query("MATCH ()-[e:PUBLISHES]->() RETURN count(e) AS c")
    causal = client.graph_query("MATCH ()-[e:CAUSALLY_FOLLOWED_BY]->() RETURN count(e) AS c")
    verb_edges = client.graph_query("MATCH ()-[e:EXECUTES_VERB]->() RETURN count(e) AS c")

    def _count(rows: list[dict[str, Any]]) -> int:
        return int(rows[0]["c"]) if rows else 0

    return {
        "organ_count": _count(organs),
        "channel_count": _count(channels),
        "verb_count": _count(verbs),
        "publishes_edge_count": _count(publishes),
        "causally_followed_by_edge_count": _count(causal),
        "executes_verb_edge_count": _count(verb_edges),
    }


@router.get("/hot-organs")
async def hot_organs(limit: int = Query(default=10, ge=1, le=100)) -> dict[str, Any]:
    """Organs ranked by PUBLISHES out-degree (how many distinct channels they
    fan out to) -- a real, already-visible centrality signal, not a new one.
    """
    client = _client()
    rows = client.graph_query(
        """
        MATCH (o:Organ)-[e:PUBLISHES]->(:Channel)
        RETURN o.organ_id AS organ_id, count(e) AS channel_out_degree
        ORDER BY channel_out_degree DESC
        LIMIT $limit
        """,
        {"limit": limit},
    )
    return {"organs": rows}


@router.get("/hot-edges")
async def hot_edges(limit: int = Query(default=10, ge=1, le=100)) -> dict[str, Any]:
    """Real cross-organ hop pairs ranked by observed count -- the structurally
    dominant flows in the mesh right now, e.g. a hot vision pipeline or a
    high-volume cognition loop, visible with no new instrumentation.
    """
    client = _client()
    rows = client.graph_query(
        """
        MATCH (a:Organ)-[e:CAUSALLY_FOLLOWED_BY]->(b:Organ)
        RETURN a.organ_id AS source_organ, b.organ_id AS target_organ,
               e.count AS count, e.latency_ewma_sec AS latency_ewma_sec
        ORDER BY e.count DESC
        LIMIT $limit
        """,
        {"limit": limit},
    )
    return {"edges": rows}


@router.get("/anomalies")
async def anomalies(zscore_threshold: float = 3.0, min_count: int = 5) -> dict[str, Any]:
    """Edges whose most recent observation deviated sharply from that edge's
    own rolling baseline. min_count guards against the cold-start z-score
    instability documented in orion-bus-mirror's README (an edge's second
    observation can read as an extreme z-score before a real baseline
    exists) -- only edges with enough history to trust are surfaced here.
    """
    client = _client()
    publish_rows = client.graph_query(
        """
        MATCH (o:Organ)-[e:PUBLISHES]->(c:Channel)
        WHERE e.gap_zscore IS NOT NULL AND abs(e.gap_zscore) > $threshold AND e.count > $min_count
        RETURN o.organ_id AS organ_id, c.channel AS channel,
               e.gap_zscore AS zscore, e.count AS count
        ORDER BY abs(e.gap_zscore) DESC
        LIMIT 20
        """,
        {"threshold": zscore_threshold, "min_count": min_count},
    )
    causal_rows = client.graph_query(
        """
        MATCH (a:Organ)-[e:CAUSALLY_FOLLOWED_BY]->(b:Organ)
        WHERE e.latency_zscore IS NOT NULL AND abs(e.latency_zscore) > $threshold AND e.count > $min_count
        RETURN a.organ_id AS source_organ, b.organ_id AS target_organ,
               e.latency_zscore AS zscore, e.count AS count
        ORDER BY abs(e.latency_zscore) DESC
        LIMIT 20
        """,
        {"threshold": zscore_threshold, "min_count": min_count},
    )
    return {"publish_gap_anomalies": publish_rows, "causal_latency_anomalies": causal_rows}
