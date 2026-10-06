"""Hydration-free store for the cognitive unification layer (stance builds).

``CognitiveUnificationLayer.beliefs_for_stance`` asks its store for three
things: ``snapshot().nodes`` filtered by anchor scope, the concept region
(concept_induction's adapter), and -- only for a write-through producer that
returns a record -- the materializer's lookups and upserts.

On ``FalkorSubstrateStore`` the first of those is a complete hydrate of the
graph, repeated whenever the 30 s refresh ceiling has elapsed, which on a
human turn it always has. Measured 2026-10-06 against the production graph
(5,051 nodes, 38,393 edges, read-only): 14.0 s per hydrate, 10.6 s of it in
49 Falkor page queries. Live stance builds paid it on every Hub turn
(build_ms 28632 cold after restart, 17316 warm). The layer used about 215 of
those nodes and none of the edges.

This store answers the same three questions with bounded Cypher reads:

* ``snapshot()``: every native node whose ``anchor_scope`` is one of
  ``anchor_scopes`` (default: every scope except ``world``, which holds the
  evidence/entity/world-concept bulk), decoded by the same codec as hydration,
  so ``snapshot().nodes`` filtered by any of those anchors equals the hydrated
  snapshot filtered the same way. No edges -- the layer reads none.
* ``query_concept_region``: ``FalkorDirectConceptStore.read_concept_region``,
  whose equivalence with the hydrated cache is proved in
  ``orion/substrate/tests/test_falkor_direct.py``.
* lookups by id/identity: single-row Cypher reads; upserts: a
  ``FalkorSubstrateStore(hydrate=False)`` writer (one ``MERGE ... SET``).

Not served: legacy ``payload_json`` rows (0 live, see falkor_direct.py), and
anchor scopes outside ``anchor_scopes`` -- the layer refuses to build beliefs
for those instead of returning an empty slice.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Iterable
from urllib.parse import urlparse

from orion.core.schemas.cognitive_substrate import BaseSubstrateNodeV1, SubstrateEdgeV1
from orion.graph.falkor_client import FalkorGraphClient, RedisGraphQueryClient
from orion.substrate.falkor_codec import decode_edge, decode_node
from orion.substrate.falkor_direct import FalkorDirectConceptStore
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
from orion.substrate.store import MaterializedSubstrateGraphState, SubstrateQueryResultV1

logger = logging.getLogger("orion.substrate.falkor_anchor_store")

# Every SubstrateAnchorScopeV1 except "world".
DEFAULT_STANCE_ANCHOR_SCOPES: tuple[str, ...] = ("orion", "juniper", "claude", "relationship", "session")

ANCHOR_NODES_CYPHER = (
    "MATCH (n:SubstrateNode) WHERE n.anchor_scope IN $anchors AND n.payload_json IS NULL "
    f"RETURN {_return_clause('n', NATIVE_NODE_RETURN_FIELDS)}, id(n) AS object_id "
    "ORDER BY object_id"
)
NODE_ID_BY_IDENTITY_CYPHER = (
    "MATCH (n:SubstrateNode) WHERE n.identity_key = $identity_key AND n.payload_json IS NULL "
    "RETURN n.node_id AS node_id LIMIT 2"
)
_EDGE_MATCH = (
    "MATCH (source:SubstrateNode)-[e]->(target:SubstrateNode) "
    "WHERE e.substrate_edge = true AND e.payload_json IS NULL AND "
)
EDGE_BY_ID_CYPHER = (
    _EDGE_MATCH + "e.edge_id = $edge_id "
    f"RETURN {_edge_hydrate_return_clause(NATIVE_EDGE_RETURN_FIELDS)} LIMIT 2"
)
EDGE_ID_BY_IDENTITY_CYPHER = (
    _EDGE_MATCH + "e.identity_key = $identity_key RETURN e.edge_id AS edge_id LIMIT 2"
)


class FalkorAnchorStanceStore:
    """The slice of ``SubstrateGraphStore`` the unification layer uses, with no cache."""

    def __init__(
        self,
        *,
        read_client: FalkorGraphClient,
        writer: FalkorSubstrateStore,
        anchor_scopes: Iterable[str] = DEFAULT_STANCE_ANCHOR_SCOPES,
    ) -> None:
        self._read = read_client
        self._writer = writer
        self._concepts = FalkorDirectConceptStore(read_client=read_client, writer=writer)
        self.snapshot_anchor_scopes: frozenset[str] = frozenset(anchor_scopes)
        if "world" in self.snapshot_anchor_scopes:
            raise ValueError("the world scope is the bulk of the graph; it is not a stance read")
        # Telemetry for the stance build's phase timing line.
        self._stats_lock = threading.Lock()
        self.snapshot_calls = 0
        self.snapshot_ms_total = 0.0

    def _query(self, cypher: str, params: dict[str, Any], fields: tuple[str, ...]) -> list[dict[str, Any]]:
        return _normalize_rows(self._read.graph_query(cypher, params), fields=fields, strict=True)

    # -- reads ---------------------------------------------------------------

    def snapshot(self) -> MaterializedSubstrateGraphState:
        started = time.perf_counter()
        rows = self._query(
            ANCHOR_NODES_CYPHER,
            {"anchors": sorted(self.snapshot_anchor_scopes)},
            (*NATIVE_NODE_RETURN_FIELDS, "object_id"),
        )
        nodes: dict[str, BaseSubstrateNodeV1] = {}
        identity_index: dict[str, str] = {}
        for row in rows:
            node = decode_node(row)
            if node is None or not node.node_id or node.node_id in nodes:
                # Hydration rejects the whole scan on these; one bad row must
                # not blank the stance, so it is skipped and named instead.
                logger.warning("falkor_anchor_snapshot_row_skipped object_id=%s", row.get("object_id"))
                continue
            nodes[node.node_id] = node
            identity = row.get("identity_key")
            if identity:
                identity_index[str(identity)] = node.node_id
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        with self._stats_lock:
            self.snapshot_calls += 1
            self.snapshot_ms_total += elapsed_ms
        return MaterializedSubstrateGraphState(
            nodes=nodes, edges={}, node_identity_index=identity_index, edge_identity_index={},
        )

    def query_concept_region(self, *, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateQueryResultV1:
        nodes_limit = max(1, int(limit_nodes))
        edges_limit = max(1, int(limit_edges))
        try:
            region = self._concepts.read_concept_region(limit_nodes=nodes_limit, limit_edges=edges_limit)
        except Exception as exc:  # noqa: BLE001 -- the adapter reads degraded as unavailable
            from orion.substrate.store import SubstrateNeighborhoodSliceV1

            return SubstrateQueryResultV1(
                query_kind="concept_region", slice=SubstrateNeighborhoodSliceV1(nodes=[], edges=[]),
                source_kind="falkor_direct", degraded=True, error=f"{type(exc).__name__}: {exc}",
                limits={"limit_nodes": nodes_limit, "limit_edges": edges_limit},
            )
        return SubstrateQueryResultV1(
            query_kind="concept_region",
            slice=region,
            source_kind="falkor_direct",
            limits={"limit_nodes": nodes_limit, "limit_edges": edges_limit},
            truncated=len(region.nodes) >= nodes_limit or len(region.edges) >= edges_limit,
        )

    def get_node_by_id(self, node_id: str) -> BaseSubstrateNodeV1 | None:
        return self._concepts.get_node_by_id(node_id)

    def get_identity_key_by_node_id(self, node_id: str) -> str | None:
        return self._concepts.get_identity_key_by_node_id(node_id)

    def get_node_id_by_identity(self, identity_key: str) -> str | None:
        rows = self._query(NODE_ID_BY_IDENTITY_CYPHER, {"identity_key": str(identity_key)}, ("node_id",))
        return str(rows[0]["node_id"]) if len(rows) == 1 and rows[0].get("node_id") else None

    def get_edge_by_id(self, edge_id: str) -> SubstrateEdgeV1 | None:
        rows = self._query(EDGE_BY_ID_CYPHER, {"edge_id": str(edge_id)}, NATIVE_EDGE_RETURN_FIELDS)
        return decode_edge(rows[0]) if len(rows) == 1 else None

    def get_edge_id_by_identity(self, identity_key: str) -> str | None:
        rows = self._query(EDGE_ID_BY_IDENTITY_CYPHER, {"identity_key": str(identity_key)}, ("edge_id",))
        return str(rows[0]["edge_id"]) if len(rows) == 1 and rows[0].get("edge_id") else None

    # -- writes --------------------------------------------------------------

    def upsert_node(
        self,
        *,
        identity_key: str | None,
        node: BaseSubstrateNodeV1,
        skip_metadata_keys: frozenset[str] | None = None,
    ) -> None:
        self._writer.upsert_node(identity_key=identity_key, node=node, skip_metadata_keys=skip_metadata_keys)

    def upsert_edge(self, *, identity_key: str | None, edge: SubstrateEdgeV1) -> None:
        self._writer.upsert_edge(identity_key=identity_key, edge=edge)


def build_unification_store_from_env() -> Any:
    """The store a ``CognitiveUnificationLayer`` should hold.

    Falkor backend (``SUBSTRATE_STORE_BACKEND=falkor`` and ``FALKORDB_URI``):
    ``FalkorAnchorStanceStore``, which never hydrates. Any other backend:
    ``build_substrate_store_from_env()``, unchanged. Construction does no
    network I/O apart from the node_id index bootstrap (idempotent).
    """
    from orion.substrate.graphdb_store import build_substrate_store_from_env

    backend = str(os.getenv("SUBSTRATE_STORE_BACKEND", "")).strip().lower()
    uri = str(os.getenv("FALKORDB_URI", "")).strip()
    if backend not in {"falkor", "falkordb"} or not uri:
        return build_substrate_store_from_env()
    graph_name = str(os.getenv("FALKORDB_SUBSTRATE_GRAPH", "orion_substrate")).strip() or "orion_substrate"
    read_client = RedisGraphQueryClient(uri=uri, graph_name=graph_name, read_only=True)
    write_client = RedisGraphQueryClient(uri=uri, graph_name=graph_name)
    writer = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri=uri, graph_name=graph_name), client=write_client, hydrate=False
    )
    ensure_substrate_indexes(uri, graph_name, client=write_client)
    logger.info(
        "substrate_store_backend_selected backend=falkor_anchor_stance uri_host=%s graph=%s",
        urlparse(uri).hostname or "",
        graph_name,
    )
    return FalkorAnchorStanceStore(read_client=read_client, writer=writer)
