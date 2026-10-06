from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple
from urllib.parse import quote

logger = logging.getLogger("orion.graph-compression.federator.substrate_falkor")

Triple = Tuple[str, str, str]

# Downstream (region_builder.py -> writer.py::_build_sparql_update) writes
# node identity strings straight into SPARQL IRIREF position with zero
# escaping -- every SPARQL federator satisfies this for free (bindings are
# already well-formed IRIs). Live substrate node_ids are alnum/dash/
# underscore slugs today (verified against the real orion_substrate graph),
# but wrapping unconditionally costs nothing and keeps the contract
# identical to episodic_falkor.py -- a future node_id shape change can't
# silently reintroduce an invalid-SPARQL-IRIREF failure mode here.
_NODE_NS = "http://conjourney.net/orion/substrate/falkor/"


def _to_iri(value: str) -> str:
    return _NODE_NS + quote(str(value), safe="")

# SubstrateNode/edge shape per orion/substrate/falkor_store.py::upsert_node/
# upsert_edge: nodes are (:SubstrateNode:<label> {node_id: ...}), edges are
# dynamically-typed relationships keyed by edge.predicate (CONTRADICTS/
# SUPPORTS/REFINES/CO_OCCURS_WITH/...). type(r) recovers the predicate
# without needing to enumerate the closed predicate set here.
# Only walkable relationships (orion.substrate.neighborhood): legacy edges as before and
# accepted projections; never assertion structure, provenance, or a projection whose
# assertion is not accepted at its revision (checked in one batched lookup after the scan).
from orion.substrate.neighborhood import (  # noqa: E402
    ASSERTION_STATE_CYPHER, accepted_revisions, role_prefilter, walkable_given,
)

_QUERY = (
    "MATCH (s:SubstrateNode)-[r]->(o:SubstrateNode) WHERE " + role_prefilter("r")
    + " RETURN s.node_id AS s, type(r) AS p, o.node_id AS o, r.edge_role AS edge_role,"
    " r.assertion_id AS assertion_id, r.assertion_revision AS assertion_revision LIMIT $max_edges"
)


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
        rows = list(rows or [])
        ids = sorted({str(r.get("assertion_id")) for r in rows if r.get("edge_role") == "semantic_projection"})
        accepted: dict = {}
        if ids:
            try:
                accepted = accepted_revisions(client.graph_query(ASSERTION_STATE_CYPHER, {"ids": ids}) or [])
            except Exception as exc:
                logger.warning("substrate_falkor_federator_assertion_lookup_failed reason=%s", exc)
        triples: List[Triple] = []
        for row in rows:
            if not walkable_given(row.get("edge_role"), row.get("assertion_id"), row.get("assertion_revision"),
                                  accepted):
                continue
            s = row.get("s")
            p = row.get("p")
            o = row.get("o")
            if s and p and o:
                triples.append((_to_iri(s), str(p), _to_iri(o)))
        return triples
