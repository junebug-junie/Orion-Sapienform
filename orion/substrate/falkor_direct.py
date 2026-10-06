"""Hydration-free FalkorDB reads for per-turn callers (orion-recall's concept_region).

``FalkorSubstrateStore`` serves reads from a complete in-process copy of the
graph that it hydrates on first use. Since PR #2500 made that copy complete
(37k+ edges), building it takes 17-25s -- fine for a background writer, not
for a recall turn. This module answers the one question recall asks without
building that copy: bounded Cypher reads straight against FalkorDB, decoded by
the same codec the hydration path uses.

Semantics contract (proved against the cache by
``orion/substrate/tests/test_falkor_direct.py`` on a live throwaway FalkorDB,
and against the production graph by
``services/orion-recall/scripts/compare_concept_region_direct_vs_cache.py``):

``read_concept_region(limit_nodes=N, limit_edges=M)`` returns exactly what
``InMemorySubstrateGraphStore.read_concept_region`` returns after a complete
hydrate of the same graph:

* nodes: every native ``concept`` node, ranked by decoded
  ``(signals.salience, signals.confidence)`` descending, ties in hydration
  order (ascending Falkor object id), top N;
* edges: every native substrate edge touching one of those N nodes, ranked by
  decoded ``(salience, confidence)`` descending, ties in hydration order
  (ascending object id), top M.

The ranking expressions mirror the codec's decode defaults
(``float(salience or 0.0)``, ``float(confidence or 0.5)``) so a NULL or 0
stored value ranks the way the decoded model would.

``read_concept_region_matching(keep_label=p, ...)`` returns the same slice
restricted to the ranked nodes whose label satisfies ``p`` and the ranked
edges touching them -- precisely the filter concept_region applies to the
full slice. It fetches full rows only for what survives, so a turn with no
label match costs one light ranking query.

Not served (by design, documented divergence): legacy ``payload_json`` rows.
Hydration decodes them and a writer then rewrites them to native properties;
this reader does not write, so it only sees native rows. Live count of legacy
rows on orion_substrate: 0 nodes, 0 edges (2026-10-06).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable
from urllib.parse import urlparse

from orion.core.schemas.cognitive_substrate import BaseSubstrateNodeV1, SubstrateEdgeV1
from orion.graph.falkor_client import FalkorGraphClient, RedisGraphQueryClient
from orion.substrate.falkor_codec import decode_edge, decode_node
from orion.substrate.falkor_store import (
    NATIVE_EDGE_RETURN_FIELDS,
    NATIVE_NODE_RETURN_FIELDS,
    FalkorSubstrateStore,
    FalkorSubstrateStoreConfig,
    _edge_hydrate_return_clause,
    _normalize_rows,
    _return_clause,
    ensure_substrate_indexes,
)
from orion.substrate.store import SubstrateNeighborhoodSliceV1

logger = logging.getLogger("orion.substrate.falkor_direct")

__all__ = [
    "FalkorDirectConceptStore",
    "build_falkor_direct_concept_store_from_env",
    "CONCEPT_RANK_CYPHER",
    "CONCEPT_ROWS_CYPHER",
    "CONCEPT_EDGE_CUT_CYPHER",
    "NODE_BY_ID_CYPHER",
]


def _salience(alias: str) -> str:
    # falkor_codec: float(row.get("salience") or 0.0)
    return f"coalesce({alias}.salience, 0.0)"


def _confidence(alias: str) -> str:
    # falkor_codec: float(row.get("confidence") or 0.5) -- NULL and 0 both decode to 0.5.
    return f"CASE WHEN coalesce({alias}.confidence, 0.0) = 0.0 THEN 0.5 ELSE {alias}.confidence END"


_CONCEPT_MATCH = "MATCH (n:SubstrateNode) WHERE n.node_kind = 'concept' AND n.payload_json IS NULL "
_CONCEPT_RANKED = (
    f"WITH n, {_salience('n')} AS rank_salience, {_confidence('n')} AS rank_confidence, id(n) AS object_id "
    "ORDER BY rank_salience DESC, rank_confidence DESC, object_id ASC LIMIT $limit_nodes "
)

# Light ranking pass: object id + label + rank keys only, never the full row.
CONCEPT_RANK_CYPHER = (
    _CONCEPT_MATCH + _CONCEPT_RANKED + "RETURN object_id, n.label AS label"
)

# Full rows for a handful of already-ranked nodes, by object id.
CONCEPT_ROWS_CYPHER = (
    "UNWIND $object_ids AS object_id "
    "MATCH (n:SubstrateNode) WHERE id(n) = object_id "
    f"RETURN {_return_clause('n', NATIVE_NODE_RETURN_FIELDS)}, id(n) AS object_id"
)

# The edge cut is computed over every edge touching the ranked concepts (the
# cache's semantics); only survivors touching $selected_ids come back, with
# full rows. $selected_all=true returns the whole cut.
#
# The ranking is recomputed here rather than passed in as object ids on
# purpose. FalkorDB 6.0 (graph module 60001, falkordb/falkordb:latest as of
# 2026-10-06) drops the id filter in
# "MATCH (n) WHERE id(n) = $x MATCH (n)-[e]-()" and walks every edge in the
# graph; the single-clause form keeps the filter but traverses from every node
# before applying it. Projecting n through WITH ... LIMIT gives the correct,
# bounded plan on both 4.x (production) and 6.0. Caught by the CI equivalence
# lane, which pulls the newer image.
CONCEPT_EDGE_CUT_CYPHER = (
    _CONCEPT_MATCH
    + _CONCEPT_RANKED
    + "MATCH (n)-[e]-(:SubstrateNode) WHERE e.substrate_edge = true AND e.payload_json IS NULL "
    + "WITH DISTINCT e "
    + f"WITH e, {_salience('e')} AS rank_salience, {_confidence('e')} AS rank_confidence "
    + "ORDER BY rank_salience DESC, rank_confidence DESC, id(e) ASC LIMIT $limit_edges "
    + "WITH e, startNode(e) AS source, endNode(e) AS target "
    + "WHERE $selected_all OR id(source) IN $selected_ids OR id(target) IN $selected_ids "
    + f"RETURN {_edge_hydrate_return_clause(NATIVE_EDGE_RETURN_FIELDS)}, id(e) AS object_id"
)

NODE_BY_ID_CYPHER = (
    "MATCH (n:SubstrateNode) WHERE n.node_id = $node_id AND n.payload_json IS NULL "
    f"RETURN {_return_clause('n', NATIVE_NODE_RETURN_FIELDS)} LIMIT 2"
)


def _node_rank_key(node: BaseSubstrateNodeV1, object_id: int) -> tuple[float, float, int]:
    return (-float(node.signals.salience), -float(node.signals.confidence), object_id)


def _edge_rank_key(edge: SubstrateEdgeV1, object_id: int) -> tuple[float, float, int]:
    return (-float(edge.salience), -float(edge.confidence), object_id)


class FalkorDirectConceptStore:
    """The slice of ``SubstrateGraphStore`` concept_region uses, with no cache.

    Reads go through ``read_client`` (meant to be a ``GRAPH.RO_QUERY`` client)
    on every call. The one write -- activation reinforcement -- goes through a
    ``FalkorSubstrateStore`` built with ``hydrate=False``; its upsert is a
    single ``MERGE ... SET`` and never reads the graph.

    Deliberately has no ``snapshot()``: nothing holding this handle can ask
    for the whole graph.
    """

    def __init__(self, *, read_client: FalkorGraphClient, writer: FalkorSubstrateStore) -> None:
        self._read = read_client
        self._writer = writer

    # -- reads ---------------------------------------------------------------

    def _query(self, cypher: str, params: dict[str, Any], fields: tuple[str, ...]) -> list[dict[str, Any]]:
        return _normalize_rows(self._read.graph_query(cypher, params), fields=fields, strict=True)

    def read_concept_region(self, *, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateNeighborhoodSliceV1:
        return self.read_concept_region_matching(keep_label=None, limit_nodes=limit_nodes, limit_edges=limit_edges)

    def read_concept_region_matching(
        self,
        *,
        keep_label: Callable[[str], bool] | None,
        limit_nodes: int = 32,
        limit_edges: int = 64,
    ) -> SubstrateNeighborhoodSliceV1:
        bounded_nodes = max(1, int(limit_nodes))
        bounded_edges = max(1, int(limit_edges))
        ranked = self._query(
            CONCEPT_RANK_CYPHER, {"limit_nodes": bounded_nodes}, ("object_id", "label")
        )
        ranked_ids = [int(row["object_id"]) for row in ranked]
        if keep_label is None:
            selected_ids = ranked_ids
        else:
            # str(None) == "None" mirrors decode_concept_node's str(row["label"]).
            selected_ids = [int(row["object_id"]) for row in ranked if keep_label(str(row.get("label")))]
        if not selected_ids:
            return SubstrateNeighborhoodSliceV1(nodes=[], edges=[])

        node_rows = self._query(
            CONCEPT_ROWS_CYPHER, {"object_ids": selected_ids}, (*NATIVE_NODE_RETURN_FIELDS, "object_id")
        )
        decoded_nodes: list[tuple[tuple[float, float, int], BaseSubstrateNodeV1]] = []
        for row in node_rows:
            node = decode_node(row)
            if node is None or node.node_kind != "concept":
                logger.warning("falkor_direct_concept_row_undecodable object_id=%s", row.get("object_id"))
                continue
            decoded_nodes.append((_node_rank_key(node, int(row["object_id"])), node))
        decoded_nodes.sort(key=lambda item: item[0])

        edge_rows = self._query(
            CONCEPT_EDGE_CUT_CYPHER,
            {
                "limit_nodes": bounded_nodes,
                "selected_ids": selected_ids,
                "selected_all": keep_label is None,
                "limit_edges": bounded_edges,
            },
            (*NATIVE_EDGE_RETURN_FIELDS, "object_id"),
        )
        decoded_edges: list[tuple[tuple[float, float, int], SubstrateEdgeV1]] = []
        for row in edge_rows:
            edge = decode_edge(row)
            if edge is None:
                logger.warning("falkor_direct_edge_row_undecodable object_id=%s", row.get("object_id"))
                continue
            decoded_edges.append((_edge_rank_key(edge, int(row["object_id"])), edge))
        decoded_edges.sort(key=lambda item: item[0])

        return SubstrateNeighborhoodSliceV1(
            nodes=[node for _key, node in decoded_nodes],
            edges=[edge for _key, edge in decoded_edges],
        )

    def _node_row(self, node_id: str) -> dict[str, Any] | None:
        rows = self._query(NODE_BY_ID_CYPHER, {"node_id": str(node_id)}, NATIVE_NODE_RETURN_FIELDS)
        if len(rows) != 1:
            # 0 = absent; 2 = duplicate node_id, which hydration also rejects.
            return None
        return rows[0]

    def get_node_and_identity_key(self, node_id: str) -> tuple[BaseSubstrateNodeV1 | None, str | None]:
        """One query for both halves of a reinforcement read (the separate
        getters below each issue NODE_BY_ID_CYPHER for the same row)."""
        row = self._node_row(node_id)
        if row is None:
            return None, None
        identity = row.get("identity_key")
        return decode_node(row), (str(identity) if identity else None)

    def get_node_by_id(self, node_id: str) -> BaseSubstrateNodeV1 | None:
        row = self._node_row(node_id)
        return decode_node(row) if row is not None else None

    def get_identity_key_by_node_id(self, node_id: str) -> str | None:
        row = self._node_row(node_id)
        if row is None:
            return None
        identity = row.get("identity_key")
        return str(identity) if identity else None

    # -- write ---------------------------------------------------------------

    def upsert_node(
        self,
        *,
        identity_key: str | None,
        node: BaseSubstrateNodeV1,
        skip_metadata_keys: frozenset[str] | None = None,
    ) -> None:
        self._writer.upsert_node(identity_key=identity_key, node=node, skip_metadata_keys=skip_metadata_keys)


def build_falkor_direct_concept_store_from_env(
    *,
    socket_timeout_s: float | None = None,
    socket_connect_timeout_s: float | None = None,
) -> FalkorDirectConceptStore | None:
    """Build from ``FALKORDB_URI``/``FALKORDB_SUBSTRATE_GRAPH``. No network I/O
    happens here (redis-py connects lazily), so construction cannot block.
    Returns None when ``FALKORDB_URI`` is unset."""
    uri = str(os.getenv("FALKORDB_URI", "")).strip()
    if not uri:
        logger.warning("falkor_direct_concept_store_unconfigured reason=FALKORDB_URI_missing")
        return None
    graph_name = str(os.getenv("FALKORDB_SUBSTRATE_GRAPH", "orion_substrate")).strip() or "orion_substrate"
    timeouts: dict[str, float] = {}
    if socket_timeout_s is not None:
        timeouts["socket_timeout"] = float(socket_timeout_s)
    if socket_connect_timeout_s is not None:
        timeouts["socket_connect_timeout"] = float(socket_connect_timeout_s)
    read_client = RedisGraphQueryClient(uri=uri, graph_name=graph_name, read_only=True, **timeouts)
    write_client = RedisGraphQueryClient(uri=uri, graph_name=graph_name, **timeouts)
    writer = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri=uri, graph_name=graph_name), client=write_client, hydrate=False
    )
    # The writer gets an injected client, so its own constructor skips the
    # index bootstrap; run it here on that client (same socket timeouts).
    ensure_substrate_indexes(uri, graph_name, client=write_client)
    logger.info(
        "substrate_store_backend_selected backend=falkor_direct uri_host=%s graph=%s",
        urlparse(uri).hostname or "",
        graph_name,
    )
    return FalkorDirectConceptStore(read_client=read_client, writer=writer)
