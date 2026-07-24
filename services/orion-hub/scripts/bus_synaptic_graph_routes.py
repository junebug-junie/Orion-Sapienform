"""Read-only debug API for the bus synaptic graph (services/orion-bus-mirror).

Surfaces structure that already exists in the live FalkorDB graph but had no
human-visible view -- the "Idea 5" starting point from
docs/superpowers/specs/2026-07-24-bus-vitality-field-signal-brainstorm.md's
Phase 3+ brainstorm: before building any new signal, show what's already
sitting in the data. No new edge/node type, no write path -- pure read.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from orion.graph.falkor_client import RedisGraphQueryClient

from app.settings import settings

router = APIRouter(prefix="/api/bus-synaptic-graph", tags=["bus-synaptic-graph"])


def propagate_from_organ(
    edges: list[dict[str, Any]],
    *,
    seed_organ_id: str,
    max_hops: int,
    zscore_threshold: float,
    min_count: int,
) -> dict[str, Any]:
    """Pure bounded-BFS over an already-fetched CAUSALLY_FOLLOWED_BY edge list
    (small -- 150 live edges as of this patch, cheap to load whole and walk
    in Python rather than express as a Cypher variable-length path, which
    FalkorDB requires a literal hop bound for, not a runtime parameter).

    Idea A of docs/superpowers/specs/2026-07-24-bus-synaptic-graph-wider-net-
    brainstorm.md: trace a spike outward from ``seed_organ_id`` via real
    observed cross-organ hops, to see which *downstream* organs are
    currently absorbing that stress -- a propagation trace, not an isolated
    per-edge alarm. "Elevated" uses the same threshold/min_count semantics
    as the existing /anomalies route (min_count guards the same cold-start
    z-score instability documented in orion-bus-mirror's README).

    Separated from DB I/O so the traversal logic is testable against a
    fixed edge list, matching this arc's established pattern (e.g.
    stale_channel_node_cleanup.py's compute_stale_channel_candidates).
    """
    adjacency: dict[str, list[dict[str, Any]]] = {}
    for edge in edges:
        adjacency.setdefault(edge["source_organ"], []).append(edge)

    edges_traversed: list[dict[str, Any]] = []
    elevated_edges: list[dict[str, Any]] = []
    organs_reached: set[str] = set()
    visited_organs = {seed_organ_id}

    frontier: deque[tuple[str, int]] = deque([(seed_organ_id, 0)])
    while frontier:
        organ_id, hop = frontier.popleft()
        if hop >= max_hops:
            continue
        for edge in adjacency.get(organ_id, []):
            target = edge["target_organ"]
            zscore = edge.get("zscore")
            is_elevated = (
                zscore is not None
                and abs(zscore) > zscore_threshold
                and (edge.get("count") or 0) > min_count
            )
            entry = {**edge, "hop": hop + 1, "elevated": is_elevated}
            edges_traversed.append(entry)
            # A cycle back to the seed is real structure (kept in
            # edges_traversed) but the seed is where the stress originated,
            # not a downstream organ absorbing it -- excluded here so a
            # cyclic CAUSALLY_FOLLOWED_BY chain doesn't report the seed as
            # part of its own blast radius.
            if target != seed_organ_id:
                organs_reached.add(target)
            if is_elevated:
                elevated_edges.append(entry)
            if target not in visited_organs:
                visited_organs.add(target)
                frontier.append((target, hop + 1))

    return {
        "seed_organ_id": seed_organ_id,
        "max_hops": max_hops,
        "edges_traversed": edges_traversed,
        "elevated_edges": elevated_edges,
        "organs_reached": sorted(organs_reached),
    }


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


@router.get("/node-centrality")
async def node_centrality(limit: int = Query(default=15, ge=1, le=100)) -> dict[str, Any]:
    """Real in+out degree over the CAUSALLY_FOLLOWED_BY organ-to-organ
    subgraph -- Idea B of docs/superpowers/specs/2026-07-24-bus-synaptic-graph-
    wider-net-brainstorm.md's "node centrality" wider-net idea.

    Deliberately scoped to CAUSALLY_FOLLOWED_BY (organ-to-organ), not
    PUBLISHES (organ-to-channel): the latter's Channel-node side was found
    live to hold ~9K stale pre-normalization duplicates (fixed via
    services/orion-bus-mirror/scripts/stale_channel_node_cleanup.py), and
    even post-cleanup, PUBLISHES out-degree already has its own partial view
    via /hot-organs. This route answers a genuinely different question: which
    organs are structurally load-bearing in the *causal* flow of real
    traffic, based on live observed cross-organ hops -- distinct from
    ORGAN_REGISTRY's hand-authored, admittedly-approximate causal_parent_organs
    edges. Degree only (not betweenness) -- FalkorDB has no native
    betweenness-centrality query, and loading the whole graph into an
    external library for it is an unmeasured cost, deferred to a later pass.
    """
    client = _client()
    rows = client.graph_query(
        """
        MATCH (o:Organ)
        OPTIONAL MATCH (o)-[out:CAUSALLY_FOLLOWED_BY]->()
        OPTIONAL MATCH ()-[inc:CAUSALLY_FOLLOWED_BY]->(o)
        WITH o, count(DISTINCT out) AS out_degree, count(DISTINCT inc) AS in_degree
        WHERE out_degree > 0 OR in_degree > 0
        RETURN o.organ_id AS organ_id, out_degree, in_degree, (out_degree + in_degree) AS total_degree
        ORDER BY total_degree DESC
        LIMIT $limit
        """,
        {"limit": limit},
    )
    return {"organs": rows}


@router.get("/cross-node-imbalance")
async def cross_node_imbalance(limit: int = Query(default=20, ge=1, le=100)) -> dict[str, Any]:
    """Real cross-organ hops that cross a physical host boundary (e.g.
    athena -> atlas), ranked by how often that's happened -- Idea D of
    docs/superpowers/specs/2026-07-24-bus-synaptic-graph-wider-net-brainstorm.md.

    Live-verified before building this: a 2000-envelope sample of real
    traffic found envelope.source.node values of athena/atlas/circe all
    present today -- the mesh genuinely spans multiple hosts, this isn't
    speculative plumbing for a topology that doesn't exist.

    ``cross_node_count`` / ``count`` on each edge is the real
    infrastructure-load-balance signal: an edge that's *always* cross-node
    is architecturally expected (e.g. a producer pinned to one host calling
    a service pinned to another); a low but nonzero ratio suggests a
    genuinely floating/unpinned deployment. Only edges that have observed at
    least one hop with both sides' node populated are returned -- edges with
    no node data yet (e.g. not touched since this feature deployed) are
    correctly absent, not shown as zero.
    """
    client = _client()
    rows = client.graph_query(
        """
        MATCH (a:Organ)-[e:CAUSALLY_FOLLOWED_BY]->(b:Organ)
        WHERE e.cross_node_count IS NOT NULL
        RETURN a.organ_id AS source_organ, b.organ_id AS target_organ,
               e.last_source_node AS last_source_node, e.last_target_node AS last_target_node,
               e.cross_node_count AS cross_node_count, e.count AS count
        ORDER BY cross_node_count DESC
        LIMIT $limit
        """,
        {"limit": limit},
    )
    return {"edges": rows}


@router.get("/anomaly-propagation")
async def anomaly_propagation(
    organ_id: str = Query(...),
    max_hops: int = Query(default=2, ge=1, le=5),
    zscore_threshold: float = 3.0,
    min_count: int = 5,
) -> dict[str, Any]:
    """Trace downstream from ``organ_id`` via real CAUSALLY_FOLLOWED_BY hops,
    flagging which edges along the way are themselves currently elevated --
    a blast-radius view, not just an isolated per-edge alarm. Idea A of
    docs/superpowers/specs/2026-07-24-bus-synaptic-graph-wider-net-brainstorm.md.

    Loads the whole (small, ~150-edge) CAUSALLY_FOLLOWED_BY subgraph in one
    query and walks it in Python via ``propagate_from_organ`` -- see that
    function's docstring for why (FalkorDB variable-length paths need a
    literal hop bound, not a runtime parameter).
    """
    client = _client()
    rows = client.graph_query(
        """
        MATCH (a:Organ)-[e:CAUSALLY_FOLLOWED_BY]->(b:Organ)
        RETURN a.organ_id AS source_organ, b.organ_id AS target_organ,
               e.latency_zscore AS zscore, e.count AS count
        """
    )
    return propagate_from_organ(
        rows,
        seed_organ_id=organ_id,
        max_hops=max_hops,
        zscore_threshold=zscore_threshold,
        min_count=min_count,
    )


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
