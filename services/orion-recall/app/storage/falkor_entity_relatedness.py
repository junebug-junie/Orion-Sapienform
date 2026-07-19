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
737 Entity nodes, 1,901 edges) during development -- see
scripts/live_verify_entity_relatedness.py for a re-runnable version of that
verification (not hand-typed test fixture numbers only).

Phase 1 status, stated plainly: these three functions are reachable today
only via the /debug/entity-graph/* routes in app/main.py -- a human-curlable
surface for verifying the primitives are correct, not (yet) an input to
worker.py's process_recall/_query_backends fusion pipeline the way
RECALL_FALKOR_IN_CHAT/RECALL_FALKOR_GRAPHTRI_IN_CHAT are. Wiring one of
these into live recall ranking is Phase 2, not done here.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

from ..recall_falkor_store import get_recall_falkor_client

logger = logging.getLogger(__name__)


async def _safe_graph_query(client: Any, cypher: str, params: Dict[str, Any], *, log_ctx: str) -> List[Dict[str, Any]]:
    """Single source of truth for the run-Cypher-off-the-event-loop,
    degrade-to-empty-on-any-failure contract every function in this module
    (and its Falkor-adapter siblings) follows. Centralizing this closes a
    real bug found in review: row-shaping code that runs AFTER a per-call
    try/except (e.g. int()/float() coercion) can still raise uncaught -- by
    keeping callers thin (row-shaping happens on the return value, outside
    this helper, but the raising-prone Falkor round-trip itself is fully
    contained here), the failure mode this fixes can't silently reappear at
    a 4th call site the way it could with copy-pasted try/except blocks."""
    try:
        rows = await asyncio.to_thread(client.graph_query, cypher, params)
        return list(rows or [])
    except Exception as exc:
        logger.debug("%s skipped: %s", log_ctx, exc)
        return []


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

    Aggregates by e2 node identity, not e2.name -- if the graph ever
    regressed to having two distinct Entity nodes sharing a name (the exact
    class of bug Phase 0 fixed), this would still produce one row per node,
    not silently merge or duplicate. Node-identity uniqueness-by-name is
    itself an invariant of the write path's `MERGE (g:Entity {name: name})`
    pattern (services/orion-meta-tags/app/falkor_recall_writer.py), not
    re-verified here.

    Returns [] (not an error) if the entity doesn't exist or the client is
    unavailable -- this is a reasoning/debug primitive, not a hard recall
    dependency.
    """
    client = get_recall_falkor_client()
    if client is None or not name:
        return []

    rows = await _safe_graph_query(
        client,
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
        log_ctx="fetch_related_entities",
    )

    out: List[Dict[str, Any]] = []
    for row in rows:
        related_name = str(row.get("name") or "").strip()
        if not related_name:
            continue
        try:
            shared_turns = int(row.get("shared") or 0)
            jaccard = float(row.get("jaccard") or 0.0)
        except (TypeError, ValueError):
            # A malformed numeric value from Falkor must not 500 the debug
            # endpoint -- skip just this row rather than the whole request.
            logger.debug("fetch_related_entities: unparseable row %r", row)
            continue
        out.append({"name": related_name, "shared_turns": shared_turns, "jaccard": jaccard})
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

    Returns {"mode": "direct"|"bridge"|"none", "entity_a", "entity_b", "results": [...]}.
    """
    client = get_recall_falkor_client()
    if client is None or not entity_a or not entity_b:
        return {"mode": "none", "entity_a": entity_a, "entity_b": entity_b, "results": []}

    limit = int(max(1, min(max_results, 25)))

    direct_rows = await _safe_graph_query(
        client,
        "MATCH (a:Entity {name: $a})<-[:MENTIONS_ENTITY]-(t:ChatTurn)-[:MENTIONS_ENTITY]->(b:Entity {name: $b}) "
        "RETURN t.turn_id AS turn_id, t.ts AS ts "
        "ORDER BY t.ts DESC LIMIT $max_results",
        {"a": entity_a, "b": entity_b, "max_results": limit},
        log_ctx="fetch_bridging_turns direct-check",
    )

    if direct_rows:
        return {
            "mode": "direct",
            "entity_a": entity_a,
            "entity_b": entity_b,
            "results": [
                {"turn_id": str(r.get("turn_id")), "ts": r.get("ts")}
                for r in direct_rows
                if r.get("turn_id")
            ],
        }

    bridge_rows = await _safe_graph_query(
        client,
        "MATCH (a:Entity {name: $a})<-[:MENTIONS_ENTITY]-(t1:ChatTurn)-[:MENTIONS_ENTITY]->(mid:Entity)"
        "<-[:MENTIONS_ENTITY]-(t2:ChatTurn)-[:MENTIONS_ENTITY]->(b:Entity {name: $b}) "
        "WHERE mid <> a AND mid <> b AND a <> b AND t1 <> t2 "
        "RETURN DISTINCT mid.name AS bridge, t1.turn_id AS turn1, t2.turn_id AS turn2 "
        "LIMIT $max_results",
        {"a": entity_a, "b": entity_b, "max_results": limit},
        log_ctx="fetch_bridging_turns 2-hop",
    )

    results = [
        {
            "bridge_entity": str(r.get("bridge")),
            "turn1": str(r.get("turn1")),
            "turn2": str(r.get("turn2")),
        }
        for r in bridge_rows
        if r.get("bridge")
    ]
    return {
        "mode": "bridge" if results else "none",
        "entity_a": entity_a,
        "entity_b": entity_b,
        "results": results,
    }


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

    rows = await _safe_graph_query(
        client,
        "MATCH (e:Entity {name: $name})<-[r:MENTIONS_ENTITY]-(t:ChatTurn) "
        "RETURN t.turn_id AS turn_id, r.ts AS ts "
        "ORDER BY r.ts DESC LIMIT $max_results",
        {"name": name, "max_results": int(max(1, min(max_results, 500)))},
        log_ctx="fetch_entity_mention_timeline",
    )

    return [
        {"turn_id": str(r.get("turn_id")), "ts": r.get("ts")}
        for r in rows
        if r.get("turn_id")
    ]


async def fetch_entity_matches_for_turns(
    *,
    turn_ids: List[str],
    target_names: List[str],
) -> Dict[str, List[str]]:
    """Phase 2 (fusion-weight boost): for each of turn_ids, which of
    target_names does that turn's MENTIONS_ENTITY set include? One batched
    UNWIND query, not one round trip per turn -- this is called from the
    recall hot path (fuse_candidates), unlike the Phase 1 debug-only
    primitives above, so round-trip count actually matters here.

    Returns only turn_ids with at least one match (empty/no-match turns are
    simply absent from the returned dict, not present with an empty list) --
    callers should treat a missing key as "no boost," not distinguish it
    from an explicit empty match set.
    """
    client = get_recall_falkor_client()
    if client is None or not turn_ids or not target_names:
        return {}

    rows = await _safe_graph_query(
        client,
        "UNWIND $turn_ids AS tid "
        "MATCH (t:ChatTurn {turn_id: tid})-[:MENTIONS_ENTITY]->(e:Entity) "
        "WHERE e.name IN $target_names "
        "RETURN tid, collect(DISTINCT e.name) AS matched",
        {"turn_ids": list(turn_ids), "target_names": list(target_names)},
        log_ctx="fetch_entity_matches_for_turns",
    )

    out: Dict[str, List[str]] = {}
    for row in rows:
        tid = str(row.get("tid") or "").strip()
        matched = [str(m) for m in (row.get("matched") or []) if m]
        if tid and matched:
            out[tid] = matched
    return out


async def fetch_turns_mentioning_entities(
    *,
    target_names: List[str],
    max_results: int = 8,
) -> List[Dict[str, Any]]:
    """Phase 2 (fusion-weight boost), added after live evidence showed the
    boost alone never fires: fetch_entity_matches_for_turns can only re-rank
    turn_ids ALREADY in the candidate pool, and falkor_chat's own fetch
    (falkor_chat_adapter.py::fetch_falkor_chatturn_fragments) is
    deliberately unfiltered by query -- it returns the most-recent-N turns
    regardless of content (that's Phase 4's own documented design: chatturn
    recall is recency-based, not relevance-filtered). Live-verified across
    3 profiles and 6 real queries: an entity from months-old turns never
    appears in a recency-windowed pool of 4-12 items, so the boost had
    nothing to act on and never changed a single ranking.

    This closes that gap the other direction: given the target entity set
    already computed for the query, fetch actual ChatTurn ids that mention
    them directly (independent of recency), for injection as NEW candidates
    -- not just re-ranking of whatever recency already happened to fetch.
    Text hydration (Postgres join) and fragment-shape construction stay in
    worker.py, matching where falkor_chat_adapter.py already does the same
    join for the recency-fetched candidates -- this function only resolves
    which turn_ids are relevant, same division of labor as the rest of this
    module.
    """
    client = get_recall_falkor_client()
    if client is None or not target_names:
        return []

    rows = await _safe_graph_query(
        client,
        "MATCH (t:ChatTurn {source_kind: 'chat.history'})-[r:MENTIONS_ENTITY]->(e:Entity) "
        "WHERE e.name IN $target_names "
        "RETURN DISTINCT t.turn_id AS turn_id, t.ts AS ts "
        "ORDER BY t.ts DESC LIMIT $max_results",
        {"target_names": list(target_names), "max_results": int(max(1, min(max_results, 50)))},
        log_ctx="fetch_turns_mentioning_entities",
    )

    return [
        {"turn_id": str(r.get("turn_id")), "ts": r.get("ts")}
        for r in rows
        if r.get("turn_id")
    ]


async def fetch_entity_degrees(*, names: List[str]) -> Dict[str, int]:
    """Phase 2 fix: live-confirmed the entity-relatedness boost's direct-
    query-entity-match path scored EVERY extracted query entity at a flat
    1.0, with no document-frequency awareness -- unlike the Jaccard-ranked
    "related entities" expansion, which was specifically designed to avoid
    exactly this failure mode (see fetch_related_entities' own docstring).
    A mundane message that simply addresses the assistant by name ("Orion,
    what do you think...") extracted "Orion" as a query entity and, because
    "orion" is one of the two most frequent nodes in the whole graph (282
    mentions, confirmed live -- nearly every turn), injected 6 generic
    filler turns ("thanks, appreciated.", "I'm all good.") at full boost
    strength, purely because they happened to mention Orion by name.

    This returns each name's live mention-count (same shape as
    fetch_related_entities' internal degree1/degree2, just not currently
    exposed there), letting the caller apply the identical document-
    frequency principle to DIRECT matches that Jaccard already applies to
    RELATED ones -- one query, not one round trip per entity.
    """
    client = get_recall_falkor_client()
    if client is None or not names:
        return {}

    rows = await _safe_graph_query(
        client,
        "UNWIND $names AS name "
        "MATCH (e:Entity {name: name})<-[r:MENTIONS_ENTITY]-() "
        "RETURN name, count(r) AS degree",
        {"names": list(names)},
        log_ctx="fetch_entity_degrees",
    )

    return {str(r.get("name")): int(r.get("degree") or 0) for r in rows if r.get("name")}


__all__ = [
    "fetch_related_entities",
    "fetch_bridging_turns",
    "fetch_entity_mention_timeline",
    "fetch_entity_matches_for_turns",
    "fetch_turns_mentioning_entities",
    "fetch_entity_degrees",
]
