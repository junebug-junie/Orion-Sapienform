from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

logger = logging.getLogger("orion.graph-compression.federator.substrate_falkor")

Triple = Tuple[str, str, str]

# SubstrateNode/edge shape per orion/substrate/falkor_store.py::upsert_node/
# upsert_edge: nodes are (:SubstrateNode:<label> {node_id: ...}), edges are
# dynamically-typed relationships keyed by edge.predicate (CONTRADICTS/
# SUPPORTS/REFINES/CO_OCCURS_WITH/...). type(r) recovers the predicate
# without needing to enumerate the closed predicate set here.
_QUERY = """
MATCH (s:SubstrateNode)-[r]->(o:SubstrateNode)
RETURN s.node_id AS s, type(r) AS p, o.node_id AS o
LIMIT $max_edges
"""


class FalkorSubstrateFederator:
    """Cypher-native replacement for the SPARQL-based ``SubstrateFederator``.

    Substrate-runtime has been Falkor-primary (``SUBSTRATE_STORE_BACKEND=falkor``)
    since PR #1153 -- the SPARQL federator reads a graph nothing has written
    to since that cutover. This reads the real live substrate graph directly.
    Degrades to an empty list on any error (no client configured, FalkorDB
    unreachable, malformed rows) -- matches every other federator's
    fail-open contract; a quiet scope is skipped by the caller, not a crash.
    """

    def __init__(self, *, client: Optional[Any] = None) -> None:
        self._client = client

    def fetch(self, *, max_edges: int = 4000) -> List[Triple]:
        client = self._client
        if client is None:
            from app.falkor_store import get_substrate_falkor_client

            client = get_substrate_falkor_client()
        if client is None:
            return []
        try:
            rows = client.graph_query(_QUERY, {"max_edges": max_edges})
        except Exception as exc:
            logger.warning("substrate_falkor_federator_fetch_failed reason=%s", exc)
            return []
        triples: List[Triple] = []
        for row in rows or []:
            s = row.get("s")
            p = row.get("p")
            o = row.get("o")
            if s and p and o:
                triples.append((str(s), str(p), str(o)))
        return triples
