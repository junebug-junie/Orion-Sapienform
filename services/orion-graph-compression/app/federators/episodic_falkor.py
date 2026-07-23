from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple
from urllib.parse import quote

logger = logging.getLogger("orion.graph-compression.federator.episodic_falkor")

Triple = Tuple[str, str, str]

# Downstream (region_builder.py -> writer.py::_build_sparql_update) writes
# node identity strings straight into SPARQL IRIREF position (`<{value}>`)
# with zero escaping -- an invariant every SPARQL federator satisfies for
# free (bindings are already well-formed IRIs). ChatTurn/ChatSession
# node_id-shaped values are safe as-is, but Entity/Tag `.name` values are
# raw free chat text -- confirmed live against orion_recall that real values
# include spaces and apostrophes (e.g. "solar system",
# "the 'sentience striving program'"), which are illegal inside a SPARQL
# IRIREF and would make CompressionWriter's SPARQL UPDATE invalid, silently
# dropping exactly the regions this federator was added to surface. Wrap
# every node in a synthetic namespace + percent-encoding so the identity
# string is always IRIREF-safe regardless of which label it came from.
_NODE_NS = "http://conjourney.net/orion/recall/falkor/"


def _to_iri(value: str) -> str:
    return _NODE_NS + quote(str(value), safe="")

# orion_recall shape per services/orion-meta-tags/app/falkor_recall_writer.py
# (the real, live producer): (:ChatSession {session_id})-[:HAS_TURN]->
# (:ChatTurn {turn_id})-[:HAS_TAG]->(:Tag {name}),
# (:ChatTurn)-[:MENTIONS_ENTITY]->(:Entity {name}). As of 2026-07-22,
# (:CollapseEvent {collapse_id}) writes the same HAS_TAG/MENTIONS_ENTITY
# shape too (dark by default, RECALL_FALKOR_COLLAPSE_TRIAGE_ENABLED) --
# included in the coalesce below so a CollapseEvent-sourced edge doesn't
# silently resolve to a NULL node id and get dropped once that flag flips on.
# Node identity property differs per label (turn_id/session_id/collapse_id/
# name) -- coalesce recovers a single stable node id per row without needing
# a label-specific query per pair.
_QUERY = """
MATCH (s)-[r:HAS_TURN|HAS_TAG|MENTIONS_ENTITY]->(o)
RETURN coalesce(s.turn_id, s.session_id, s.collapse_id, s.name) AS s,
       type(r) AS p,
       coalesce(o.turn_id, o.session_id, o.collapse_id, o.name) AS o
LIMIT $max_edges
"""


class FalkorEpisodicFederator:
    """Cypher-native replacement for the ``orion:enrichment``/``orion:chat``
    slice of the SPARQL-based ``EpisodicFederator``.

    Covers what has a live Falkor writer today (orion-meta-tags' Entity/
    Tag/ChatTurn/CollapseEvent graph, ``orion_recall``). ``orion:cognition``/
    ``orion:metacog`` (Postgres-owned, no clustering value -- see the
    rdf-writer kill this ships alongside) have no Falkor equivalent and
    never had one; the SPARQL ``EpisodicFederator`` remains the source for
    that content, such as it is (it's dead going forward, frozen history).

    ``orion:chat:social`` was removed from the SPARQL federator's graph list
    entirely 2026-07-23 (live-verified pure redundancy with Postgres
    ``social_room_turns``, no other reader) -- it is not "covered" by this
    Falkor federator so much as the whole scope's dependency on it was
    retired. orion-meta-tags does write social-turn tags/entities into this
    same query's underlying data (unconditional for both ``chat.history``
    and ``social.turn.stored.v1`` since 2026-07-18), but live-verified thin
    as of this writing (2 ``social.turn.stored.v1`` ChatTurn nodes, zero
    HAS_TAG/MENTIONS_ENTITY edges between them, vs 1,772 for chat.history) --
    disclosed here rather than overclaimed as equivalent coverage. Degrades
    to an empty list on any error, matching every other federator's
    fail-open contract.
    """

    def __init__(self, *, client: Optional[Any] = None) -> None:
        self._client = client

    def fetch(self, *, max_edges: int = 4000) -> List[Triple]:
        client = self._client
        if client is None:
            from app.falkor_store import get_recall_falkor_client

            client = get_recall_falkor_client()
        if client is None:
            return []
        try:
            rows = client.graph_query(_QUERY, {"max_edges": max_edges})
        except Exception as exc:
            logger.warning("episodic_falkor_federator_fetch_failed reason=%s", exc)
            return []
        triples: List[Triple] = []
        for row in rows or []:
            s = row.get("s")
            p = row.get("p")
            o = row.get("o")
            if s and p and o:
                triples.append((_to_iri(s), str(p), _to_iri(o)))
        return triples
