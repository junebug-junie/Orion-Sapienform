"""Primary/shadow SubstrateGraphStore wrapper for dual-run migration."""

from __future__ import annotations

import logging
import os
from typing import Any

from orion.core.schemas.cognitive_substrate import BaseSubstrateNodeV1, SubstrateEdgeV1
from orion.substrate.store import (
    InMemorySubstrateGraphStore,
    MaterializedSubstrateGraphState,
    SubstrateGraphStore,
    SubstrateNeighborhoodSliceV1,
    SubstrateQueryResultV1,
)

logger = logging.getLogger("orion.substrate.routed_store")


class RoutedSubstrateGraphStore:
    """Writes to primary then best-effort shadow; reads primary only."""

    def __init__(
        self,
        *,
        primary: SubstrateGraphStore,
        shadow: SubstrateGraphStore | None = None,
        workload: str = "substrate.runtime",
    ) -> None:
        self._primary = primary
        self._shadow = shadow
        self._workload = workload

    def get_node_by_id(self, node_id: str) -> BaseSubstrateNodeV1 | None:
        return self._primary.get_node_by_id(node_id)

    def get_edge_by_id(self, edge_id: str) -> SubstrateEdgeV1 | None:
        return self._primary.get_edge_by_id(edge_id)

    def get_node_id_by_identity(self, identity_key: str) -> str | None:
        return self._primary.get_node_id_by_identity(identity_key)

    def get_edge_id_by_identity(self, identity_key: str) -> str | None:
        return self._primary.get_edge_id_by_identity(identity_key)

    def upsert_node(self, *, identity_key: str | None, node: BaseSubstrateNodeV1) -> None:
        self._primary.upsert_node(identity_key=identity_key, node=node)
        self._shadow_write("upsert_node", identity_key=identity_key, node=node)

    def upsert_edge(self, *, identity_key: str, edge: SubstrateEdgeV1) -> None:
        self._primary.upsert_edge(identity_key=identity_key, edge=edge)
        self._shadow_write("upsert_edge", identity_key=identity_key, edge=edge)

    def _shadow_write(self, op: str, **kwargs: Any) -> None:
        if self._shadow is None:
            return
        try:
            getattr(self._shadow, op)(**kwargs)
        except Exception as exc:
            logger.warning(
                "substrate_route_shadow_write_failed workload=%s op=%s error=%s",
                self._workload,
                op,
                exc,
            )

    def snapshot(self) -> MaterializedSubstrateGraphState:
        return self._primary.snapshot()

    def query_focal_slice(self, *, node_ids: list[str], max_edges: int = 64) -> SubstrateQueryResultV1:
        return self._primary.query_focal_slice(node_ids=node_ids, max_edges=max_edges)

    def query_hotspot_region(
        self, *, min_salience: float = 0.6, limit_nodes: int = 32, limit_edges: int = 64
    ) -> SubstrateQueryResultV1:
        return self._primary.query_hotspot_region(
            min_salience=min_salience, limit_nodes=limit_nodes, limit_edges=limit_edges
        )

    def query_contradiction_region(
        self, *, limit_nodes: int = 32, limit_edges: int = 64
    ) -> SubstrateQueryResultV1:
        return self._primary.query_contradiction_region(limit_nodes=limit_nodes, limit_edges=limit_edges)

    def query_concept_region(
        self, *, limit_nodes: int = 32, limit_edges: int = 64
    ) -> SubstrateQueryResultV1:
        return self._primary.query_concept_region(limit_nodes=limit_nodes, limit_edges=limit_edges)

    def query_provenance_neighborhood(
        self, *, evidence_ref: str, limit_nodes: int = 32, limit_edges: int = 64
    ) -> SubstrateQueryResultV1:
        return self._primary.query_provenance_neighborhood(
            evidence_ref=evidence_ref, limit_nodes=limit_nodes, limit_edges=limit_edges
        )

    def read_focal_slice(self, *, node_ids: list[str], max_edges: int = 64) -> SubstrateNeighborhoodSliceV1:
        return self._primary.read_focal_slice(node_ids=node_ids, max_edges=max_edges)

    def read_hotspot_region(
        self, *, min_salience: float = 0.6, limit_nodes: int = 32, limit_edges: int = 64
    ) -> SubstrateNeighborhoodSliceV1:
        return self._primary.read_hotspot_region(
            min_salience=min_salience, limit_nodes=limit_nodes, limit_edges=limit_edges
        )

    def read_contradiction_region(
        self, *, limit_nodes: int = 32, limit_edges: int = 64
    ) -> SubstrateNeighborhoodSliceV1:
        return self._primary.read_contradiction_region(limit_nodes=limit_nodes, limit_edges=limit_edges)

    def read_concept_region(
        self, *, limit_nodes: int = 32, limit_edges: int = 64
    ) -> SubstrateNeighborhoodSliceV1:
        return self._primary.read_concept_region(limit_nodes=limit_nodes, limit_edges=limit_edges)

    def read_provenance_neighborhood(
        self, *, evidence_ref: str, limit_nodes: int = 32, limit_edges: int = 64
    ) -> SubstrateNeighborhoodSliceV1:
        return self._primary.read_provenance_neighborhood(
            evidence_ref=evidence_ref, limit_nodes=limit_nodes, limit_edges=limit_edges
        )


def _build_named_backend(name: str) -> SubstrateGraphStore:
    """Build a leaf backend without recursing into routed."""
    key = str(name or "").strip().lower()
    if key in {"", "in_memory", "memory", "mem", "local"}:
        return InMemorySubstrateGraphStore()
    if key in {"falkor", "falkordb"}:
        from orion.substrate.falkor_store import build_falkor_substrate_store_from_env

        return build_falkor_substrate_store_from_env()
    if key in {"sparql", "sparql_http", "graphdb", "graph_db", "rdf"}:
        # Re-enter builder with temporary env mutation avoided: call sparql/graphdb paths
        # by setting backend via a nested helper that only accepts leaf names.
        from orion.substrate import graphdb_store as gdb

        prev = os.environ.get("SUBSTRATE_STORE_BACKEND")
        try:
            os.environ["SUBSTRATE_STORE_BACKEND"] = key
            return gdb.build_substrate_store_from_env()
        finally:
            if prev is None:
                os.environ.pop("SUBSTRATE_STORE_BACKEND", None)
            else:
                os.environ["SUBSTRATE_STORE_BACKEND"] = prev
    logger.warning("routed_substrate_unknown_leaf backend=%r; using in-memory", name)
    return InMemorySubstrateGraphStore()


def build_routed_substrate_store_from_env() -> SubstrateGraphStore:
    primary_name = str(os.getenv("SUBSTRATE_STORE_PRIMARY", "in_memory")).strip().lower() or "in_memory"
    shadow_raw = str(os.getenv("SUBSTRATE_STORE_SHADOW", "")).strip().lower()
    workload = str(os.getenv("SUBSTRATE_ROUTE_WORKLOAD", "substrate.runtime")).strip() or "substrate.runtime"
    if primary_name == "routed":
        logger.error("SUBSTRATE_STORE_PRIMARY cannot be routed; falling back to in_memory")
        primary_name = "in_memory"
    primary = _build_named_backend(primary_name)
    shadow = None
    if shadow_raw and shadow_raw not in {"none", "off", "disabled"}:
        if shadow_raw == "routed":
            logger.warning("SUBSTRATE_STORE_SHADOW=routed ignored")
        else:
            shadow = _build_named_backend(shadow_raw)
    logger.info(
        "substrate_store_backend_selected backend=routed primary=%s shadow=%s workload=%s",
        primary_name,
        shadow_raw or "none",
        workload,
    )
    return RoutedSubstrateGraphStore(primary=primary, shadow=shadow, workload=workload)
