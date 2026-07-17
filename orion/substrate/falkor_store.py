"""FalkorDB-backed SubstrateGraphStore (write-through cache + payload_json).

Queries and reads are served from the in-process cache (same shape as a warm
GraphDBSubstrateStore). Durable writes go through an injectable sync client so
unit tests never need a live FalkorDB. Cold-start hydration is best-effort via
MATCH … RETURN payload_json when the client can answer.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, replace
from typing import Any, Protocol
from urllib.parse import urlparse

from pydantic import TypeAdapter

from orion.core.schemas.cognitive_substrate import BaseSubstrateNodeV1, SubstrateEdgeV1, SubstrateNodeV1
from orion.graph.property_guard import sanitize_metadata
from orion.substrate.store import (
    InMemorySubstrateGraphStore,
    MaterializedSubstrateGraphState,
    SubstrateNeighborhoodSliceV1,
    SubstrateQueryResultV1,
)

logger = logging.getLogger("orion.substrate.falkor_store")

NODE_ADAPTER = TypeAdapter(SubstrateNodeV1)


class FalkorGraphClient(Protocol):
    def graph_query(self, cypher: str, params: dict[str, Any] | None = None) -> Any: ...


@dataclass(frozen=True)
class FalkorSubstrateStoreConfig:
    uri: str
    graph_name: str = "orion_substrate"


class RecordingFalkorClient:
    """Test double that records Cypher and optionally returns scripted rows."""

    def __init__(self, *, hydrate_rows: list[dict[str, Any]] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any] | None]] = []
        self._hydrate_rows = list(hydrate_rows or [])

    def graph_query(self, cypher: str, params: dict[str, Any] | None = None) -> Any:
        self.calls.append((cypher, params))
        if "RETURN n.payload_json" in cypher or "RETURN e.payload_json" in cypher:
            return self._hydrate_rows
        return []


class RedisGraphQueryClient:
    """Minimal sync Redis GRAPH.QUERY client for FalkorDB."""

    def __init__(self, *, uri: str, graph_name: str) -> None:
        import redis

        parsed = urlparse(uri or "redis://localhost:6379")
        self._graph_name = graph_name
        self._r = redis.Redis(
            host=parsed.hostname or "localhost",
            port=int(parsed.port or 6379),
            db=int((parsed.path or "/0").lstrip("/") or 0),
            decode_responses=True,
        )

    def graph_query(self, cypher: str, params: dict[str, Any] | None = None) -> Any:
        # Falkor/RedisGraph: GRAPH.QUERY key cypher [params...]
        # Keep params embedded via $name and Redis graph parameter map when available.
        if params:
            # redis-py graph module is optional; fall back to string replace for simple scalars
            # by using the official GRAPH.QUERY with --params JSON when supported.
            args: list[Any] = [self._graph_name, cypher, "--params", json.dumps(params)]
            return self._r.execute_command("GRAPH.QUERY", *args)
        return self._r.execute_command("GRAPH.QUERY", self._graph_name, cypher)


def _with_sanitized_metadata(model: Any) -> Any:
    cleaned, _rejected = sanitize_metadata(getattr(model, "metadata", None) or {})
    if cleaned == (getattr(model, "metadata", None) or {}):
        return model
    return model.model_copy(update={"metadata": cleaned})


class FalkorSubstrateStore:
    """SubstrateGraphStore with Falkor write-through and in-memory read cache."""

    def __init__(
        self,
        cfg: FalkorSubstrateStoreConfig,
        *,
        client: FalkorGraphClient | None = None,
        hydrate: bool = True,
    ) -> None:
        self._cfg = cfg
        self._client: FalkorGraphClient = client or RedisGraphQueryClient(
            uri=cfg.uri, graph_name=cfg.graph_name
        )
        self._cache = InMemorySubstrateGraphStore()
        self._result_source_kind = "falkor"
        if hydrate:
            self._hydrate_from_durable()

    def _hydrate_from_durable(self) -> None:
        try:
            node_rows = self._client.graph_query(
                "MATCH (n:SubstrateNode) RETURN n.payload_json AS payload_json, n.identity_key AS identity_key"
            )
            edge_rows = self._client.graph_query(
                "MATCH (e:SubstrateEdge) RETURN e.payload_json AS payload_json, e.identity_key AS identity_key"
            )
        except Exception as exc:
            logger.warning("falkor_substrate_hydrate_failed error=%s", exc)
            return
        for row in _normalize_rows(node_rows):
            payload = row.get("payload_json")
            if not payload:
                continue
            try:
                node = NODE_ADAPTER.validate_json(payload)
            except Exception:
                logger.warning("falkor_substrate_hydrate_node_invalid")
                continue
            identity = row.get("identity_key")
            self._cache.upsert_node(identity_key=str(identity) if identity else None, node=node)
        for row in _normalize_rows(edge_rows):
            payload = row.get("payload_json")
            if not payload:
                continue
            try:
                edge = SubstrateEdgeV1.model_validate_json(payload)
            except Exception:
                logger.warning("falkor_substrate_hydrate_edge_invalid")
                continue
            identity = row.get("identity_key") or self._edge_identity(edge)
            self._cache.upsert_edge(identity_key=str(identity), edge=edge)

    @staticmethod
    def _edge_identity(edge: SubstrateEdgeV1) -> str:
        return f"{edge.source.node_id}|{edge.predicate}|{edge.target.node_id}"

    def get_node_by_id(self, node_id: str) -> BaseSubstrateNodeV1 | None:
        return self._cache.get_node_by_id(node_id)

    def get_edge_by_id(self, edge_id: str) -> SubstrateEdgeV1 | None:
        return self._cache.get_edge_by_id(edge_id)

    def get_node_id_by_identity(self, identity_key: str) -> str | None:
        return self._cache.get_node_id_by_identity(identity_key)

    def get_edge_id_by_identity(self, identity_key: str) -> str | None:
        return self._cache.get_edge_id_by_identity(identity_key)

    def upsert_node(self, *, identity_key: str | None, node: BaseSubstrateNodeV1) -> None:
        node = _with_sanitized_metadata(node)
        payload_json = json.dumps(node.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        params = {
            "node_id": node.node_id,
            "node_kind": node.node_kind,
            "payload_json": payload_json,
            "identity_key": identity_key or "",
            "salience": float(node.signals.salience),
        }
        cypher = (
            "MERGE (n:SubstrateNode {node_id: $node_id}) "
            "SET n.node_kind = $node_kind, n.payload_json = $payload_json, "
            "n.identity_key = $identity_key, n.salience = $salience"
        )
        try:
            self._client.graph_query(cypher, params)
        except Exception as exc:
            logger.error("falkor_substrate_upsert_node_failed node_id=%s error=%s", node.node_id, exc)
            raise
        self._cache.upsert_node(identity_key=identity_key, node=node)

    def upsert_edge(self, *, identity_key: str, edge: SubstrateEdgeV1) -> None:
        edge = _with_sanitized_metadata(edge)
        payload_json = json.dumps(edge.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        params = {
            "edge_id": edge.edge_id,
            "payload_json": payload_json,
            "identity_key": identity_key,
            "source_id": edge.source.node_id,
            "target_id": edge.target.node_id,
            "predicate": edge.predicate,
            "salience": float(edge.salience),
        }
        cypher = (
            "MERGE (e:SubstrateEdge {edge_id: $edge_id}) "
            "SET e.payload_json = $payload_json, e.identity_key = $identity_key, "
            "e.source_id = $source_id, e.target_id = $target_id, "
            "e.predicate = $predicate, e.salience = $salience"
        )
        try:
            self._client.graph_query(cypher, params)
        except Exception as exc:
            logger.error("falkor_substrate_upsert_edge_failed edge_id=%s error=%s", edge.edge_id, exc)
            raise
        self._cache.upsert_edge(identity_key=identity_key, edge=edge)

    def snapshot(self) -> MaterializedSubstrateGraphState:
        return self._cache.snapshot()

    def query_focal_slice(self, *, node_ids: list[str], max_edges: int = 64) -> SubstrateQueryResultV1:
        return _retag_source(
            self._cache.query_focal_slice(node_ids=node_ids, max_edges=max_edges),
            self._result_source_kind,
        )

    def query_hotspot_region(self, *, min_salience: float = 0.6, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateQueryResultV1:
        result = self._cache.query_hotspot_region(
            min_salience=min_salience, limit_nodes=limit_nodes, limit_edges=limit_edges
        )
        return _retag_source(result, self._result_source_kind)

    def query_contradiction_region(self, *, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateQueryResultV1:
        return _retag_source(
            self._cache.query_contradiction_region(limit_nodes=limit_nodes, limit_edges=limit_edges),
            self._result_source_kind,
        )

    def query_concept_region(self, *, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateQueryResultV1:
        return _retag_source(
            self._cache.query_concept_region(limit_nodes=limit_nodes, limit_edges=limit_edges),
            self._result_source_kind,
        )

    def query_provenance_neighborhood(
        self, *, evidence_ref: str, limit_nodes: int = 32, limit_edges: int = 64
    ) -> SubstrateQueryResultV1:
        return _retag_source(
            self._cache.query_provenance_neighborhood(
                evidence_ref=evidence_ref, limit_nodes=limit_nodes, limit_edges=limit_edges
            ),
            self._result_source_kind,
        )

    def read_focal_slice(self, *, node_ids: list[str], max_edges: int = 64) -> SubstrateNeighborhoodSliceV1:
        return self._cache.read_focal_slice(node_ids=node_ids, max_edges=max_edges)

    def read_hotspot_region(
        self, *, min_salience: float = 0.6, limit_nodes: int = 32, limit_edges: int = 64
    ) -> SubstrateNeighborhoodSliceV1:
        return self._cache.read_hotspot_region(
            min_salience=min_salience, limit_nodes=limit_nodes, limit_edges=limit_edges
        )

    def read_contradiction_region(
        self, *, limit_nodes: int = 32, limit_edges: int = 64
    ) -> SubstrateNeighborhoodSliceV1:
        return self._cache.read_contradiction_region(limit_nodes=limit_nodes, limit_edges=limit_edges)

    def read_concept_region(
        self, *, limit_nodes: int = 32, limit_edges: int = 64
    ) -> SubstrateNeighborhoodSliceV1:
        return self._cache.read_concept_region(limit_nodes=limit_nodes, limit_edges=limit_edges)

    def read_provenance_neighborhood(
        self, *, evidence_ref: str, limit_nodes: int = 32, limit_edges: int = 64
    ) -> SubstrateNeighborhoodSliceV1:
        return self._cache.read_provenance_neighborhood(
            evidence_ref=evidence_ref, limit_nodes=limit_nodes, limit_edges=limit_edges
        )


def _retag_source(result: SubstrateQueryResultV1, source_kind: str) -> SubstrateQueryResultV1:
    return replace(result, source_kind=source_kind)


def _normalize_rows(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, list):
        out: list[dict[str, Any]] = []
        for item in raw:
            if isinstance(item, dict):
                out.append(item)
            elif isinstance(item, (list, tuple)) and len(item) >= 1:
                # hydrate script may return [[payload, identity], ...]
                if len(item) >= 2:
                    out.append({"payload_json": item[0], "identity_key": item[1]})
                else:
                    out.append({"payload_json": item[0], "identity_key": ""})
        return out
    return []


def build_falkor_substrate_store_from_env() -> FalkorSubstrateStore | InMemorySubstrateGraphStore:
    uri = str(os.getenv("FALKORDB_URI", "")).strip()
    if not uri:
        logger.warning("SUBSTRATE_STORE_BACKEND=falkor but FALKORDB_URI missing; falling back to in-memory")
        return InMemorySubstrateGraphStore()
    graph_name = str(os.getenv("FALKORDB_SUBSTRATE_GRAPH", "orion_substrate")).strip() or "orion_substrate"
    logger.info(
        "substrate_store_backend_selected backend=falkor uri_host=%s graph=%s",
        urlparse(uri).hostname or "",
        graph_name,
    )
    return FalkorSubstrateStore(FalkorSubstrateStoreConfig(uri=uri, graph_name=graph_name))
