"""Entity-graph reasoning primitives over orion_recall's MENTIONS_ENTITY
co-occurrence graph. Phase 1 of the entity-graph-reasoning work (Phase 0:
data-quality cleanup, PR #1203).

This bipartite ChatTurn<->Entity graph supports exactly one class of
reasoning: co-occurrence-based association ("these things travel together"),
via projecting turn<->entity edges onto an entity<->entity relatedness
signal. It does NOT support typed/predicate reasoning ("X supports Y") --
there is one edge type, MENTIONS_ENTITY. See
services/orion-recall/README.md's "Graphtri" section for the fuller
discussion of what this graph shape can and cannot do.

Live-verified against the real orion_recall graph (post Phase 0 cleanup,
737 Entity nodes, 1,901 edges) throughout development -- see each function's
docstring for the specific evidence.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from ..recall_falkor_store import get_recall_falkor_client

logger = logging.getLogger(__name__)


async def fetch_related_entities(
    *,
    name: str,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """Co-occurrence relatedness, ranked by Jaccard similarity (shared turns
    / union of turns each entity appears in) rather than raw shared-turn
    count.

    Raw count alone is a bad ranking signal for this graph: live-verified
    that "nvidia"'s top co-occurring entity by raw count included "athena"
    (shared=4, but athena's own overall degree is 32 -- it co-occurs with
    almost everything) tied with "tesla" (shared=4, degree=7 -- genuinely
    nvidia-specific). Jaccard correctly separates them: tesla scores 0.154,
    athena drops to 0.078, six places lower. This is the standard IR fix for
    exactly this failure mode (a document-frequency-blind co-occurrence
    count conflating "common" with "related") -- the same theory-anchored
    critique this whole entity-graph-reasoning effort started from.

    Returns [] (not an error) if the entity doesn't exist or the client is
    unavailable -- this is a reasoning/debug primitive, not a hard recall
    dependency.
    """
    client = get_recall_falkor_client()
    if client is None or not name:
        return []

    try:
        rows = await asyncio.to_thread(
            client.graph_query,
            "MATCH (e1:Entity {name: $name})<-[:MENTIONS_ENTITY]-(t:ChatTurn) "
            "WITH e1, count(t) AS degree1 "
            "MATCH (e1)<-[:MENTIONS_ENTITY]-(t)-[:MENTIONS_ENTITY]->(e2:Entity) "
            "WHERE e2 <> e1 "
            "WITH e1, degree1, e2, count(t) AS shared "
            "MATCH (e2)<-[:MENTIONS_ENTITY]-(t2:ChatTurn) "
            "WITH e2.name AS name, shared, degree1, count(t2) AS degree2 "
            "RETURN name, shared, degree1, degree2, "
            "toFloat(shared) / (degree1 + degree2 - shared) AS jaccard "
            "ORDER BY jaccard DESC, shared DESC "
            "LIMIT $max_results",
            {"name": name, "max_results": int(max(1, min(max_results, 50)))},
        )
    except Exception as exc:
        logger.debug("fetch_related_entities skipped: %s", exc)
        return []

    out: List[Dict[str, Any]] = []
    for row in rows or []:
        related_name = str(row.get("name") or "").strip()
        if not related_name:
            continue
        out.append(
            {
                "name": related_name,
                "shared_turns": int(row.get("shared") or 0),
                "jaccard": float(row.get("jaccard") or 0.0),
            }
        )
    return out


async def fetch_bridging_turns(
    *,
    entity_a: str,
    entity_b: str,
    max_results: int = 5,
) -> Dict[str, Any]:
    """How are entity_a and entity_b connected? Checks direct co-mention
    first (both entities in the same turn); if none exists, falls back to a
    2-hop bridge (a's turn mentions some intermediate entity, which a
    different turn also mentions alongside b).

    Live-verified against the real graph: "nvidia"/"atlas" have both a
    direct co-mention (3 turns) AND multiple 2-hop bridges (e.g. via
    "athena") -- direct co-mention is checked first and, when present, is
    always the more informative answer, so the 2-hop query only runs when
    direct comes back empty.

    Returns {"mode": "direct"|"bridge"|"none", "results": [...]}.
    """
    client = get_recall_falkor_client()
    if client is None or not entity_a or not entity_b:
        return {"mode": "none", "results": []}

    limit = int(max(1, min(max_results, 25)))

    try:
        direct_rows = await asyncio.to_thread(
            client.graph_query,
            "MATCH (a:Entity {name: $a})<-[:MENTIONS_ENTITY]-(t:ChatTurn)-[:MENTIONS_ENTITY]->(b:Entity {name: $b}) "
            "RETURN t.turn_id AS turn_id, t.ts AS ts "
            "ORDER BY t.ts DESC LIMIT $max_results",
            {"a": entity_a, "b": entity_b, "max_results": limit},
        )
    except Exception as exc:
        logger.debug("fetch_bridging_turns direct-check skipped: %s", exc)
        direct_rows = []

    if direct_rows:
        return {
            "mode": "direct",
            "results": [
                {"turn_id": str(r.get("turn_id")), "ts": r.get("ts")}
                for r in direct_rows
                if r.get("turn_id")
            ],
        }

    try:
        bridge_rows = await asyncio.to_thread(
            client.graph_query,
            "MATCH (a:Entity {name: $a})<-[:MENTIONS_ENTITY]-(t1:ChatTurn)-[:MENTIONS_ENTITY]->(mid:Entity)"
            "<-[:MENTIONS_ENTITY]-(t2:ChatTurn)-[:MENTIONS_ENTITY]->(b:Entity {name: $b}) "
            "WHERE mid <> a AND mid <> b AND a <> b "
            "RETURN DISTINCT mid.name AS bridge, t1.turn_id AS turn1, t2.turn_id AS turn2 "
            "LIMIT $max_results",
            {"a": entity_a, "b": entity_b, "max_results": limit},
        )
    except Exception as exc:
        logger.debug("fetch_bridging_turns 2-hop skipped: %s", exc)
        return {"mode": "none", "results": []}

    results = [
        {
            "bridge_entity": str(r.get("bridge")),
            "turn1": str(r.get("turn1")),
            "turn2": str(r.get("turn2")),
        }
        for r in (bridge_rows or [])
        if r.get("bridge")
    ]
    return {"mode": "bridge" if results else "none", "results": results}


async def fetch_entity_mention_timeline(
    *,
    name: str,
    max_results: int = 100,
) -> List[Dict[str, Any]]:
    """Raw mention timestamps for an entity, most recent first -- the "when"
    dimension relatedness alone doesn't have. Deliberately returns raw
    timestamps rather than pre-bucketed counts: bucketing (by day/week/month)
    is a presentation choice for whatever calls this, not something to bake
    into the query at this data volume (~1,900 edges total)."""
    client = get_recall_falkor_client()
    if client is None or not name:
        return []

    try:
        rows = await asyncio.to_thread(
            client.graph_query,
            "MATCH (e:Entity {name: $name})<-[r:MENTIONS_ENTITY]-(t:ChatTurn) "
            "RETURN t.turn_id AS turn_id, r.ts AS ts "
            "ORDER BY r.ts DESC LIMIT $max_results",
            {"name": name, "max_results": int(max(1, min(max_results, 500)))},
        )
    except Exception as exc:
        logger.debug("fetch_entity_mention_timeline skipped: %s", exc)
        return []

    return [
        {"turn_id": str(r.get("turn_id")), "ts": r.get("ts")}
        for r in (rows or [])
        if r.get("turn_id")
    ]


__all__ = ["fetch_related_entities", "fetch_bridging_turns", "fetch_entity_mention_timeline"]
