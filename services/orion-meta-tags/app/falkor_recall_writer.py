"""Cypher-native Falkor write of chat/social-turn and collapse-triage
tag/entity enrichment.

Phase 2 of the recall/Falkor cutover
(docs/superpowers/plans/2026-07-18-recall-tag-entity-falkor-writer-plan.md).
Originally shipped dark/additive alongside orion-rdf-writer's Fuseki write
of the same `tags.enriched` event. As of 2026-07-18 (same-day follow-up,
fix/meta-tags-kill-rdf-chat-dual-write) that Fuseki write was killed as
redundant -- this is now the SOLE persistence path for chat/social-turn
enrichment, covering both `chat.history` and `social.turn.stored.v1`
(previously chat.history only). See services/orion-meta-tags/README.md's
"Historical note" section.

2026-07-22: added write_collapse_triage_tags_to_falkor for the remaining
`tags.enriched` producer -- app/main.py's generic handle_triage_event
branch, fed by `orion:collapse:triage` (Juniper-observer-gated collapse-
mirror events, CollapseMirrorEntryV2, `should_route_to_triage`/`_is_juniper`
strict gate). Unlike chat/social turns, this content has no Falkor
equivalent yet -- Fuseki (via orion-rdf-writer) is still its only
persistence, and live volume is low (2 real Enrichment records over three
weeks as of this writing, confirmed via direct Fuseki query) but genuinely
unique: it is not redundant with any Postgres store the way chat.history's
RDF copy was. Ships dark/additive, same ladder as the chat/social writer --
Fuseki keeps writing until this is verified live and explicitly cut over.

No RDF/SPARQL anywhere in this module -- pure Cypher MERGE against
FalkorDB via orion.graph.falkor_client.RedisGraphQueryClient.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any

from orion.graph.falkor_client import FalkorGraphClient, set_assignments

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
    r"|^(that\s+day(\s+on)?|last\s+(week|month|year|night))$",
    re.IGNORECASE,
)


def _normalize_identity_key(value: str) -> str:
    """Strip diacritics before lowercasing -- without this, an accent variant
    like "oríon" normalizes to a distinct key from "orion" and never merges
    with it (confirmed live: both existed as separate Entity nodes).

    Deliberately NOT app.main._normalize_observer's NFKD+encode('ascii',
    'ignore') idiom, despite the visual similarity -- that idiom drops any
    character outside ASCII, which is fine for _normalize_observer's closed,
    always-Latin-script value set (Juniper/Orion) but is wrong here: an
    entity name written entirely in a non-Latin script (Cyrillic/CJK/Arabic/
    Hebrew/Greek) doesn't decompose into base+combining-mark form under NFKD,
    so ascii-encoding it drops the whole string to "" and filter_noise()
    rejects it as noise -- confirmed live:
    unicodedata.normalize("NFKD","北京").encode("ascii","ignore") == b"".
    Stripping only Unicode category Mn (combining marks) instead folds
    accented Latin text ("oríon" -> "orion") while leaving non-Latin scripts
    untouched. This still merges genuinely distinct entities that differ only
    by accent (e.g. "Peña" and "Pena" both normalize to "pena") -- a real,
    known, accepted limitation of accent-insensitive matching, not something
    this fix attempts to solve; true entity disambiguation is out of scope
    here."""
    decomposed = unicodedata.normalize("NFKD", str(value or ""))
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return stripped.strip().lower()


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


def write_entity_edges(client: FalkorGraphClient, *, turn_id: str, ts: str, names: list[str]) -> None:
    """The single source of truth for MERGE(:Entity)+MERGE(:MENTIONS_ENTITY)
    -- also called directly by scripts/backfill_recall_entity_graph_cleanup_
    reconcile.py so a future change to this write shape (e.g. adding an edge
    property) can't silently diverge between the live writer and any
    backfill/reconcile job that adds missing edges."""
    if not names:
        return
    client.graph_query(
        "MATCH (t:ChatTurn {turn_id: $turn_id}) "
        "UNWIND $names AS name "
        "MERGE (g:Entity {name: name}) "
        "MERGE (t)-[r:MENTIONS_ENTITY]->(g) "
        "SET r.ts = $ts",
        {"turn_id": turn_id, "names": names, "ts": ts},
    )


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


def write_chat_turn_tags_to_falkor(
    client: FalkorGraphClient,
    *,
    turn_id: str,
    source_kind: str,
    session_id: str | None,
    ts: str,
    correlation_id: str | None,
    tags: list[str],
    entities: list[str],
) -> dict[str, Any]:
    """Synchronous Cypher write. Caller must keep this off the event loop
    (e.g. `asyncio.to_thread`) -- the underlying redis client is sync.

    `source_kind` (the producing envelope's kind, e.g. "chat.history" or
    "social.turn.stored.v1") is stored on the `:ChatTurn` node. Both kinds
    share this same node label/shape -- without a discriminator property, a
    `turn_id` collision between a Hub chat turn and a social-room turn (both
    producer-supplied strings, not namespaced against each other) would
    silently fuse two unrelated conversations into one node with no way to
    tell them apart afterward.
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

    # turn_params always has at least "ts" and "source_kind" beyond "turn_id"
    # (both required, non-optional parameters), so set_clause is never empty
    # -- no bare-anchor branch needed.
    turn_params: dict[str, Any] = {"turn_id": turn_id, "ts": ts, "source_kind": source_kind}
    if correlation_id:
        turn_params["correlation_id"] = correlation_id
    if sentiment:
        turn_params["sentiment"] = sentiment
    set_clause = set_assignments("t", turn_params, skip={"turn_id"})
    client.graph_query(
        f"MERGE (t:ChatTurn {{turn_id: $turn_id}}) SET {set_clause}",
        turn_params,
    )

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
        write_entity_edges(client, turn_id=turn_id, ts=ts, names=kept_entities)

    return {
        "tags_written": len(kept_tags),
        "tags_rejected": rejected_tags,
        "entities_written": len(kept_entities),
        "entities_rejected": rejected_entities,
        "sentiment": sentiment,
        "session_linked": bool(session_id),
    }


def write_collapse_triage_tags_to_falkor(
    client: FalkorGraphClient,
    *,
    collapse_id: str,
    correlation_id: str | None,
    ts: str,
    tags: list[str],
    entities: list[str],
) -> dict[str, Any]:
    """Synchronous Cypher write for Juniper-observed collapse-triage events.

    Same shape as write_chat_turn_tags_to_falkor (:X {id})-[:HAS_TAG]->(:Tag),
    -[:MENTIONS_ENTITY]->(:Entity), deliberately reusing the *same* Tag/Entity
    node universe as chat turns (not a separate namespace) -- an entity
    mentioned in both a real conversation and a collapse-mirror moment should
    resolve to one node, not two, so entity-relatedness queries see both.
    No ChatSession-equivalent grouping exists for collapse events (they are
    not part of a session the way chat turns are), so there is no analogue
    to write_chat_turn_tags_to_falkor's session-linking step.

    Caller must keep this off the event loop (e.g. `asyncio.to_thread`) --
    the underlying redis client is sync.
    """
    sentiment, clean_tags = extract_sentiment(tags or [])
    kept_tags, rejected_tags = filter_noise(clean_tags)
    kept_entities, rejected_entities = filter_noise(entities or [])

    if rejected_tags or rejected_entities:
        logger.info(
            "property_cathedral_rejected workload=recall.collapse_triage collapse_id=%s "
            "rejected_tags=%s rejected_entities=%s",
            collapse_id,
            rejected_tags,
            rejected_entities,
        )

    event_params: dict[str, Any] = {"collapse_id": collapse_id, "ts": ts}
    if correlation_id:
        event_params["correlation_id"] = correlation_id
    if sentiment:
        event_params["sentiment"] = sentiment
    set_clause = set_assignments("c", event_params, skip={"collapse_id"})
    client.graph_query(
        f"MERGE (c:CollapseEvent {{collapse_id: $collapse_id}}) SET {set_clause}",
        event_params,
    )

    if kept_tags:
        client.graph_query(
            "MATCH (c:CollapseEvent {collapse_id: $collapse_id}) "
            "UNWIND $names AS name "
            "MERGE (g:Tag {name: name}) "
            "MERGE (c)-[r:HAS_TAG]->(g) "
            "SET r.ts = $ts",
            {"collapse_id": collapse_id, "names": kept_tags, "ts": ts},
        )

    if kept_entities:
        client.graph_query(
            "MATCH (c:CollapseEvent {collapse_id: $collapse_id}) "
            "UNWIND $names AS name "
            "MERGE (g:Entity {name: name}) "
            "MERGE (c)-[r:MENTIONS_ENTITY]->(g) "
            "SET r.ts = $ts",
            {"collapse_id": collapse_id, "names": kept_entities, "ts": ts},
        )

    return {
        "tags_written": len(kept_tags),
        "tags_rejected": rejected_tags,
        "entities_written": len(kept_entities),
        "entities_rejected": rejected_entities,
        "sentiment": sentiment,
    }
