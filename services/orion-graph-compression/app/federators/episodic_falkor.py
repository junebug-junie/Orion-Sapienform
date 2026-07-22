from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

logger = logging.getLogger("orion.graph-compression.federator.episodic_falkor")

Triple = Tuple[str, str, str]

# orion_recall shape per services/orion-meta-tags/app/falkor_recall_writer.py
# (the real, live producer): (:ChatSession {session_id})-[:HAS_TURN]->
# (:ChatTurn {turn_id})-[:HAS_TAG]->(:Tag {name}),
# (:ChatTurn)-[:MENTIONS_ENTITY]->(:Entity {name}). Node identity property
# differs per label (turn_id/session_id/name) -- coalesce recovers a single
# stable node id per row without needing a label-specific query per pair.
_QUERY = """
MATCH (s)-[r:HAS_TURN|HAS_TAG|MENTIONS_ENTITY]->(o)
RETURN coalesce(s.turn_id, s.session_id, s.name) AS s,
       type(r) AS p,
       coalesce(o.turn_id, o.session_id, o.name) AS o
LIMIT $max_edges
"""


class FalkorEpisodicFederator:
    """Cypher-native replacement for the ``orion:enrichment``/``orion:chat``
    slice of the SPARQL-based ``EpisodicFederator``.

    Covers only what has a live Falkor writer today (orion-meta-tags' Entity/
    Tag/ChatTurn graph, ``orion_recall``). ``orion:collapse``,
    ``orion:cognition``/``orion:metacog`` (Postgres-owned, no clustering
    value -- see the rdf-writer kill this ships alongside), and
    ``orion:chat:social`` have no Falkor equivalent yet and are not covered
    here -- the SPARQL ``EpisodicFederator`` remains the source for those
    scopes until they get their own Falkor writer. Degrades to an empty list
    on any error, matching every other federator's fail-open contract.
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
                triples.append((str(s), str(p), str(o)))
        return triples
