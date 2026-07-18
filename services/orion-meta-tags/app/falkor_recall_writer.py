"""Cypher-native Falkor write of chat turn tag/entity enrichment.

Phase 2 of the recall/Falkor cutover
(docs/superpowers/plans/2026-07-18-recall-tag-entity-falkor-writer-plan.md).
Dark-shipped, additive: this runs alongside orion-rdf-writer's existing
Fuseki write of the same `tags.enriched` event, does not replace it.

No RDF/SPARQL anywhere in this module -- pure Cypher MERGE against
FalkorDB via orion.graph.falkor_client.RedisGraphQueryClient.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from orion.graph.falkor_client import FalkorGraphClient

logger = logging.getLogger("orion-meta-tags.falkor_recall_writer")

_SENTIMENT_PREFIX = "sentiment:"

# Noise categories confirmed against the live Fuseki store (2026-07-17 audit,
# see the Phase 0 spec): bare numbers, generic stopwords, and relative-time
# expressions are not real tags/entities -- nothing downstream ever queries
# on them, and graphing them adds no traversal value.
_STOPWORDS = {
    "today", "yesterday", "tomorrow", "tonight", "now", "then", "later",
    "earlier", "recently", "currently", "still", "already", "soon",
    "one", "two", "three", "dozen", "few", "several", "many",
}

_RELATIVE_TIME_RE = re.compile(
    r"^(about\s+)?(a|an|\d+)\s+(year|month|week|day|hour|minute)s?\s+ago$"
    r"|^(that\s+day\s+on|last\s+(week|month|year|night))$",
    re.IGNORECASE,
)


def _normalize_identity_key(value: str) -> str:
    return str(value or "").strip().lower()


def _is_noise(value: str) -> bool:
    if not value:
        return True
    if value.isdigit():
        return True
    if value in _STOPWORDS:
        return True
    if _RELATIVE_TIME_RE.match(value):
        return True
    return False


def extract_sentiment(tags: list[str]) -> tuple[str | None, list[str]]:
    """Split the `sentiment:*` string-tag convention out into a real value.

    Returns (sentiment_or_none, remaining_tags). The sentiment heuristic
    itself is computed upstream in main.py; this just stops it from being
    smuggled through the tags list into a Tag node.
    """
    sentiment: str | None = None
    remaining: list[str] = []
    for tag in tags:
        if isinstance(tag, str) and tag.startswith(_SENTIMENT_PREFIX):
            value = tag[len(_SENTIMENT_PREFIX):].strip()
            if value:
                sentiment = value
        else:
            remaining.append(tag)
    return sentiment, remaining


def filter_noise(values: list[str]) -> tuple[list[str], list[str]]:
    """Normalize + dedupe + reject noise. Returns (kept_normalized, rejected_raw)."""
    kept: list[str] = []
    rejected: list[str] = []
    seen: set[str] = set()
    for raw in values:
        norm = _normalize_identity_key(raw)
        if not norm or _is_noise(norm):
            rejected.append(raw)
            continue
        if norm in seen:
            continue
        seen.add(norm)
        kept.append(norm)
    return kept, rejected


def _set_clause(alias: str, params: dict[str, Any], *, skip: set[str]) -> str:
    keys = sorted(k for k in params if k not in skip)
    return ", ".join(f"{alias}.{key} = ${key}" for key in keys)


def write_chat_turn_tags_to_falkor(
    client: FalkorGraphClient,
    *,
    turn_id: str,
    session_id: str | None,
    ts: str,
    correlation_id: str | None,
    tags: list[str],
    entities: list[str],
) -> dict[str, Any]:
    """Synchronous Cypher write. Caller must keep this off the event loop
    (e.g. `asyncio.to_thread`) -- the underlying redis client is sync.
    """
    sentiment, clean_tags = extract_sentiment(tags or [])
    kept_tags, rejected_tags = filter_noise(clean_tags)
    kept_entities, rejected_entities = filter_noise(entities or [])

    if rejected_tags or rejected_entities:
        logger.info(
            "property_cathedral_rejected workload=recall.tag_entity turn_id=%s "
            "rejected_tags=%s rejected_entities=%s",
            turn_id,
            rejected_tags,
            rejected_entities,
        )

    turn_params: dict[str, Any] = {"turn_id": turn_id, "ts": ts}
    if correlation_id:
        turn_params["correlation_id"] = correlation_id
    if sentiment:
        turn_params["sentiment"] = sentiment
    set_clause = _set_clause("t", turn_params, skip={"turn_id"})
    if set_clause:
        client.graph_query(
            f"MERGE (t:ChatTurn {{turn_id: $turn_id}}) SET {set_clause}",
            turn_params,
        )
    else:
        client.graph_query("MERGE (t:ChatTurn {turn_id: $turn_id})", {"turn_id": turn_id})

    if session_id:
        client.graph_query(
            "MATCH (t:ChatTurn {turn_id: $turn_id}) "
            "MERGE (s:ChatSession {session_id: $session_id}) "
            "MERGE (s)-[:HAS_TURN]->(t)",
            {"turn_id": turn_id, "session_id": session_id},
        )

    if kept_tags:
        client.graph_query(
            "MATCH (t:ChatTurn {turn_id: $turn_id}) "
            "UNWIND $names AS name "
            "MERGE (g:Tag {name: name}) "
            "MERGE (t)-[r:HAS_TAG]->(g) "
            "SET r.ts = $ts",
            {"turn_id": turn_id, "names": kept_tags, "ts": ts},
        )

    if kept_entities:
        client.graph_query(
            "MATCH (t:ChatTurn {turn_id: $turn_id}) "
            "UNWIND $names AS name "
            "MERGE (g:Entity {name: name}) "
            "MERGE (t)-[r:MENTIONS_ENTITY]->(g) "
            "SET r.ts = $ts",
            {"turn_id": turn_id, "names": kept_entities, "ts": ts},
        )

    return {
        "tags_written": len(kept_tags),
        "tags_rejected": rejected_tags,
        "entities_written": len(kept_entities),
        "entities_rejected": rejected_entities,
        "sentiment": sentiment,
        "session_linked": bool(session_id),
    }
