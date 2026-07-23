"""Cypher-native keyword-to-entity-neighborhood recall fragments, standing in
for storage/rdf_adapter.py::fetch_rdf_fragments
(_fetch_rdf_neighborhood_fragments) when settings.RECALL_FALKOR_NEIGHBORHOOD_IN_CHAT
is true.

This is the last live Fuseki read path in orion-recall (the chatturn and
graphtri/Claim paths were already migrated/retired -- see
docs/superpowers/specs/2026-07-17-recall-rdf-writer-falkor-cutover-phase2-spec.md
and de1336455d's graphtri retirement). Confirmed live (2026-07-22, direct
Fuseki container log read) that the RDF version is not dead code -- it fires
on real recall calls with real keywords (e.g. "nico", "sable",
"reverie_narrate").

Deliberate scope narrowing from the RDF version, not a silent behavior
change: `_fetch_rdf_neighborhood_fragments` ran a blind
`?s ?p ?o FILTER(CONTAINS(...))` scan across every triple in the store --
any predicate, any object literal, matching generic stopwords ("the",
"with") as readily as real content. This function instead matches only
real, canonical `:Entity` nodes (deduplicated by
services/orion-meta-tags/app/falkor_recall_writer.py's `MERGE` write) and
walks their `MENTIONS_ENTITY` edges back to `:ChatTurn`s -- the same
graph-native traversal the Phase 2 spec's schema section prescribed
("real neighbor traversal... becomes a 1-hop query against a stable node
instead of a CONTAINS(LCASE(...)) string scan"). A keyword that matches no
real entity name (e.g. a generic word) now returns [] instead of a noisy
partial match -- narrower, not broader, and the narrowing is the point:
the doctrine's own "no property without a consumer" rule already flagged
loose keyword-string matching as the wrong tool.

A second, real (not just cosmetic) scope narrowing, disclosed rather than
silently inherited: this reuses `falkor_entity_relatedness.py`'s
`fetch_turns_mentioning_entities`, which filters `(:ChatTurn
{source_kind: 'chat.history'})` -- by design, since its Postgres join
(`fetch_chat_turns_by_id`) only resolves rows from `chat_history_log`.
The old SPARQL scan had no such filter and could surface matches from
any Fuseki graph, including `orion:chat:social` (SocialRoomTurn) and
`orion:enrichment` (Entity/Mention/Claim). This function cannot reach
those -- it is chat.history-only. If social-room or enrichment content
turns out to matter for this fetch specifically, that's separate,
not-yet-built work (a SocialRoomTurn-aware variant), not something this
swap silently covers already.

A third limitation, live-confirmed and not fully closed: entity matching is
`CONTAINS`, not word-bounded, so a short keyword can still false-positive
inside an unrelated entity name -- live-verified the keyword "nico" matching
only the entity "unicode" (a real node, wrong reason). A stopword filter
(`_STOPWORDS` below) closes the much larger, live-confirmed instance of this
same failure mode (generic words like "and"/"about" matching dozens of
unrelated entities), but does not fully eliminate short-substring
collisions. Same class of risk the RDF version had (its own CONTAINS filter
was equally unbounded); not solved here, disclosed instead.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, List

from ..recall_falkor_store import get_recall_falkor_client
from ..sql_chat import _to_epoch, fetch_chat_turns_by_id
from .falkor_entity_relatedness import fetch_turns_mentioning_entities

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]{3,}")

# Live-verified necessary (2026-07-22, real orion_recall query): without this,
# common function words survive the 3+ char token filter and false-positive
# match as CONTAINS substrings inside unrelated real entity names -- "and"
# alone matched "sandra bullock", "nelson mandela", "england", "landing pad",
# "amanda", "grand canyon skywalk" in one live test query. The old SPARQL
# version had the same unfiltered-keyword weakness (rdf_adapter.py's
# _extract_keywords is identical), just diluted across a much larger match
# space (any triple, not a curated ~850-node Entity set) where the noise was
# less visible per-query. Deliberately short and boring (not a general NLP
# stopword list) -- only words actually observed causing false-positive
# entity matches live, per doctrine's "reject stopword values" precedent
# (docs/superpowers/specs/2026-07-17-recall-rdf-writer-falkor-cutover-phase2-spec.md,
# point 4), extend if a live-verify run surfaces more.
_STOPWORDS = frozenset({
    "and", "the", "for", "are", "was", "did", "tell", "what", "about",
    "with", "that", "this", "from", "have", "has", "not", "you", "your",
    "she", "him", "her", "his", "its", "who", "how", "why", "can", "will",
    "just", "like", "when", "where", "were", "been", "would", "could",
    "should", "there", "their", "than", "then", "some", "such", "into",
})


def _extract_keywords(query_text: str, *, max_keywords: int = 6) -> List[str]:
    """Same tokenization as rdf_adapter.py's own helper (3+ char alnum/underscore
    tokens, order-preserving, deduped), duplicated rather than imported so this
    module has no dependency on the SPARQL module it's replacing -- plus a
    stopword filter the RDF version never had (see _STOPWORDS above)."""
    tokens = _TOKEN_RE.findall(query_text.lower())
    seen = set()
    keywords: List[str] = []
    for token in tokens:
        if token in seen or token in _STOPWORDS:
            continue
        seen.add(token)
        keywords.append(token)
        if len(keywords) >= max_keywords:
            break
    return keywords


async def _match_entities_by_keyword(
    client: Any, keywords: List[str], *, max_entities: int
) -> List[str]:
    if not keywords:
        return []
    try:
        rows = await asyncio.to_thread(
            client.graph_query,
            "UNWIND $keywords AS kw "
            "MATCH (e:Entity) "
            "WHERE toLower(e.name) CONTAINS kw "
            "RETURN DISTINCT e.name AS name "
            "LIMIT $max_entities",
            {"keywords": list(keywords), "max_entities": int(max(1, min(max_entities, 50)))},
        )
    except Exception as exc:
        logger.debug("falkor neighborhood entity match skipped: %s", exc)
        return []
    return [str(r.get("name")) for r in (rows or []) if r.get("name")]


async def fetch_falkor_neighborhood_fragments(
    *,
    query_text: str,
    max_items: int = 8,
) -> List[Dict[str, Any]]:
    """Keyword-match real :Entity nodes in orion_recall, walk MENTIONS_ENTITY
    back to the ChatTurns that mention them, join Postgres for quoted text.

    Same return-fragment shape as rdf_adapter.py's neighborhood fetch
    (id/source/source_ref/uri/text/ts/tags/score/meta) so fusion.py treats it
    identically. Never raises: any Falkor/Postgres failure degrades to [],
    same fail-open contract as every other Falkor adapter in this service.
    """
    if not query_text:
        return []
    keywords = _extract_keywords(query_text, max_keywords=6)
    if not keywords:
        return []

    client = get_recall_falkor_client()
    if client is None:
        return []

    matched_names = await _match_entities_by_keyword(client, keywords, max_entities=12)
    if not matched_names:
        return []

    try:
        turns = await fetch_turns_mentioning_entities(
            target_names=matched_names,
            max_results=max_items,
        )
    except Exception as exc:
        logger.debug("falkor neighborhood turn fetch skipped: %s", exc)
        return []
    if not turns:
        return []

    turn_ids = [str(t.get("turn_id") or "").strip() for t in turns]
    turn_ids = [t for t in turn_ids if t]
    if not turn_ids:
        return []

    try:
        text_map = await fetch_chat_turns_by_id(turn_ids)
    except Exception as exc:
        logger.debug("falkor neighborhood postgres text join skipped: %s", exc)
        return []

    out: List[Dict[str, Any]] = []
    for row in turns:
        turn_id = str(row.get("turn_id") or "").strip()
        if not turn_id or turn_id not in text_map:
            # No matching Postgres row -- nothing to quote, same drop rule
            # falkor_chat_adapter.py uses for the same reason.
            continue
        prompt, response = text_map[turn_id]
        text = f'ExactUserText: "{prompt}"\nOrionResponse: "{response}"'.strip()
        out.append(
            {
                "id": turn_id,
                "source": "falkor_neighborhood",
                "source_ref": "falkordb",
                "uri": turn_id,
                "text": text[:1500],
                "ts": _to_epoch(row.get("ts")),
                "tags": ["falkor", "neighborhood"],
                "score": 0.5,
                "meta": {"matched_entities": matched_names},
            }
        )
        if len(out) >= max_items:
            break
    return out


__all__ = ["fetch_falkor_neighborhood_fragments"]
