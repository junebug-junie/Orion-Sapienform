"""Cypher-native graphtri (entity-relevance) recall, standing in for
storage/rdf_adapter.py::fetch_rdf_graphtri_fragments and
::fetch_graphtri_anchors when settings.RECALL_FALKOR_GRAPHTRI_IN_CHAT is
true.

Second, independent piece of the recall RDF->Falkor cutover
(docs/superpowers/specs/2026-07-17-recall-rdf-writer-falkor-cutover-phase2-spec.md).
Deliberately separate from RECALL_FALKOR_IN_CHAT (the chatturn-text swap,
storage/falkor_chat_adapter.py) -- see settings.py's
RECALL_FALKOR_GRAPHTRI_IN_CHAT comment for why.

Old RDF "Claim" shape supported arbitrary (subject, predicate, object)
statements; Falkor's MENTIONS_ENTITY edges only support "this turn mentions
this entity". A live audit (Phase 0 spec's "Ground truth" section) found
this is not a real functional loss: Claim.predicate only ever took 2 fixed
values in production (hasTag, mentionsEntity -- never open-vocabulary),
confidence/salience were always 0.0/0.0 (dead constants), and no downstream
code (fusion.py, render.py) ever parsed the predicate/object structure --
the whole "Claim: ..." string was always carried as opaque text, bucketed
only by a literal "claim" tag. :Tag/HAS_TAG is also empty by design (see
falkor_chat_adapter.py's sibling comment), so MENTIONS_ENTITY already
covers the only part of the old Claim shape that was ever real.

Filtering divergence from the RDF version, named not silent: the old
SPARQL filtered at the TURN level (keyword CONTAINS on raw prompt/response
text, which requires the text to exist somewhere queryable -- Falkor's thin
ChatTurn node doesn't have it). This filters at the ENTITY level instead
(keyword match against Entity.name directly in Cypher) -- no Postgres join
needed, and arguably more semantically correct for a graph meant to
represent entity relevance in the first place.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

from ..recall_falkor_store import get_recall_falkor_client
from ..sql_chat import _to_epoch

logger = logging.getLogger(__name__)

_EMPTY_ANCHORS: Dict[str, List[str]] = {
    "entities_terms": [],
    "tags_terms": [],
    "claim_objs": [],
    "related_terms": [],
}

# Matches fetch_rdf_graphtri_fragments's shape exactly, per its own comment
# and the fusion.py falkor_chat precedent: don't reproduce the fixed 0.0
# confidence/salience (confirmed dead constants), keep the fixed score.
_FIXED_SCORE = 0.6


def _entity_match_clause(alias: str) -> str:
    """No index backs this today (confirmed via GRAPH.EXPLAIN + db.indexes()
    against the live orion_recall graph -- a plain label scan, not an issue
    at today's scale of ~1,500 Entity nodes). CONTAINS can't use a standard
    property index even if one is added later (would need a full-text
    index) -- worth revisiting if Entity count grows enough for this to
    show up in latency, not something to build preemptively here."""
    return f"ANY(k IN $keywords WHERE {alias}.name CONTAINS k OR k CONTAINS {alias}.name)"


async def fetch_falkor_graphtri_fragments(
    *,
    query_text: str,
    session_id: str | None,
    keywords: List[str],
    max_items: int = 8,
) -> List[Dict[str, Any]]:
    """Entity-mention fragments, standing in for the old Claim-based ones.
    Same return-fragment shape as fetch_rdf_graphtri_fragments
    (id/source/source_ref/uri/text/ts/tags/score/meta), including the
    literal "claim" tag render.py's is_graphtri bucketing keys off of --
    dropping that tag would silently break the "High-salience claims"
    render group for graphtri-mode profiles.

    Requires real keywords: with none, there's no entity-relevance signal
    to filter on (unlike the chatturn fetch, which intentionally has no
    query-time filter and lets fusion rank recent turns generically --
    graphtri fragments only exist to represent query-anchored relevance).

    session_id is accepted (matches fetch_rdf_graphtri_fragments's
    signature so this drops in cleanly at both worker.py call sites) but
    not used for filtering -- deliberate parity with the RDF version, which
    also never filtered by session_id in practice, and this service's own
    documented system-wide behavior (README.md: "recall ignores session_id
    for retrieval and ranking"). Named explicitly here, not left implicit,
    per code review.

    Never raises: any Falkor failure degrades to [].
    """
    if not query_text or not keywords:
        return []
    client = get_recall_falkor_client()
    if client is None:
        return []

    try:
        rows = await asyncio.to_thread(
            client.graph_query,
            "MATCH (t:ChatTurn {source_kind: 'chat.history'})-[r:MENTIONS_ENTITY]->(e:Entity) "
            f"WHERE {_entity_match_clause('e')} "
            "RETURN t.turn_id AS turn_id, e.name AS entity, r.ts AS ts "
            "ORDER BY r.ts DESC "
            "LIMIT $max_items",
            {"keywords": keywords, "max_items": int(max(1, min(max_items, 50)))},
        )
    except Exception as exc:
        logger.debug("falkor graphtri fetch skipped: %s", exc)
        return []

    out: List[Dict[str, Any]] = []
    for row in rows or []:
        turn_id = str(row.get("turn_id") or "").strip()
        entity = str(row.get("entity") or "").strip()
        if not turn_id or not entity:
            continue
        text = f"Claim: mentions {entity} | evidence={turn_id}"
        out.append(
            {
                "id": f"{turn_id}:{entity}",
                "source": "falkor_graphtri",
                "source_ref": "falkordb",
                "uri": turn_id,
                "text": text,
                "ts": _to_epoch(row.get("ts")),
                "tags": ["falkor", "graphtri", "claim"],
                "score": _FIXED_SCORE,
                "meta": {"subject": turn_id, "entity": entity},
            }
        )
    return out


async def fetch_falkor_graphtri_anchors(
    *,
    session_id: str | None,
    query_terms: List[str],
    max_terms: int = 8,
) -> Dict[str, Any]:
    """Entity-name expansion terms, standing in for fetch_graphtri_anchors.

    tags_terms is always [] -- Falkor has no :Tag/HAS_TAG data (see module
    docstring), and this is the honest answer, not a bug to paper over.
    claim_objs mirrors entities_terms: in the old RDF shape, Claim.obj was
    also, in practice, always an entity mention (per the ground-truth audit
    -- there was no other real predicate), so there's no separate data
    source to draw a distinct claim_objs list from.

    Same filtered-first, fall-back-to-unfiltered-if-thin retry pattern as
    fetch_graphtri_anchors: a real query gets query-anchored terms; a query
    with no extractable keywords (or too few real matches) still gets
    *some* expansion signal from globally-recent entities, rather than
    nothing.

    session_id is accepted (matches fetch_graphtri_anchors's signature) but
    not used for filtering -- same deliberate parity noted in
    fetch_falkor_graphtri_fragments's docstring.

    `RETURN DISTINCT e.name ... ORDER BY r.ts` orders by a relationship
    property not in the DISTINCT projection -- undefined in strict
    openCypher, but live-verified against this real FalkorDB instance
    (including a 254-mention entity and the fully unfiltered query shape)
    to neither error nor return garbage; this codebase targets FalkorDB
    specifically, not generic Cypher portability.
    """
    client = get_recall_falkor_client()
    if client is None:
        return dict(_EMPTY_ANCHORS)

    async def _run(filtered: bool) -> List[str]:
        if filtered and query_terms:
            cypher = (
                "MATCH (t:ChatTurn {source_kind: 'chat.history'})-[r:MENTIONS_ENTITY]->(e:Entity) "
                f"WHERE {_entity_match_clause('e')} "
                "RETURN DISTINCT e.name AS name "
                "ORDER BY r.ts DESC "
                "LIMIT $max_terms"
            )
            params = {"keywords": query_terms, "max_terms": int(max(1, min(max_terms, 50)))}
        else:
            cypher = (
                "MATCH (t:ChatTurn {source_kind: 'chat.history'})-[r:MENTIONS_ENTITY]->(e:Entity) "
                "RETURN DISTINCT e.name AS name "
                "ORDER BY r.ts DESC "
                "LIMIT $max_terms"
            )
            params = {"max_terms": int(max(1, min(max_terms, 50)))}
        try:
            rows = await asyncio.to_thread(client.graph_query, cypher, params)
        except Exception as exc:
            logger.debug("falkor graphtri anchors fetch skipped: %s", exc)
            return []
        return [str(r.get("name") or "").strip() for r in (rows or []) if r.get("name")]

    names = await _run(filtered=True)
    if len(names) < 3:
        names = await _run(filtered=False)

    deduped: List[str] = []
    seen = set()
    for name in names:
        if name and name not in seen:
            seen.add(name)
            deduped.append(name)
    deduped = deduped[:max_terms]

    return {
        "entities_terms": list(deduped),
        "tags_terms": [],
        "claim_objs": list(deduped),
        "related_terms": list(deduped),
    }


__all__ = ["fetch_falkor_graphtri_fragments", "fetch_falkor_graphtri_anchors"]
