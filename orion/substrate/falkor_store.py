"""FalkorDB-backed SubstrateGraphStore (write-through cache + Cypher-native properties).

Queries and reads are served from the in-process cache (same shape as a warm
GraphDBSubstrateStore). Durable writes go through an injectable sync client so
unit tests never need a live FalkorDB.

Durable support is Concept + SubstrateEdge first. Cold-start hydration prefers
native scalar properties and falls back to legacy ``payload_json`` rows,
rewriting them to native properties (and removing the blob) on successful
concept/edge decode.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace
from typing import Any, Protocol
from urllib.parse import urlparse

from pydantic import TypeAdapter

from orion.core.schemas.cognitive_substrate import (
    BaseSubstrateNodeV1,
    SubstrateEdgeV1,
    SubstrateNodeV1,
)
from orion.graph.property_guard import sanitize_metadata
from orion.substrate.falkor_codec import (
    decode_concept_node,
    decode_edge,
    encode_edge_properties,
    encode_node_properties,
    node_label_for_kind,
)
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


NATIVE_NODE_RETURN_FIELDS: tuple[str, ...] = (
    "node_id",
    "node_kind",
    "identity_key",
    "label",
    "definition",
    "taxonomy_path_json",
    "anchor_scope",
    "subject_ref",
    "promotion_state",
    "risk_tier",
    "confidence",
    "salience",
    "activation",
    "recency_score",
    "decay_half_life_seconds",
    "decay_floor",
    "observed_at",
    "valid_from",
    "valid_to",
    "provenance_authority",
    "provenance_source_kind",
    "provenance_source_channel",
    "provenance_producer",
    "provenance_model_name",
    "provenance_correlation_id",
    "provenance_trace_id",
    "provenance_tier_rank",
    "evidence_refs_json",
)

NATIVE_EDGE_RETURN_FIELDS: tuple[str, ...] = (
    "edge_id",
    "identity_key",
    "source_id",
    "source_kind",
    "target_id",
    "target_kind",
    "predicate",
    "substrate_edge",
    "confidence",
    "salience",
    "observed_at",
    "valid_from",
    "valid_to",
    "provenance_authority",
    "provenance_source_kind",
    "provenance_source_channel",
    "provenance_producer",
    "provenance_model_name",
    "provenance_correlation_id",
    "provenance_trace_id",
    "provenance_tier_rank",
    "evidence_refs_json",
)


def _return_clause(alias: str, fields: tuple[str, ...]) -> str:
    return ", ".join(f"{alias}.{field} AS {field}" for field in fields)


def _set_assignments(alias: str, params: dict[str, Any], *, skip: set[str]) -> str:
    keys = sorted(k for k in params if k not in skip)
    return ", ".join(f"{alias}.{key} = ${key}" for key in keys)


class RecordingFalkorClient:
    """Test double that records Cypher and optionally returns scripted rows."""

    def __init__(
        self,
        *,
        hydrate_node_rows: list[dict[str, Any]] | None = None,
        hydrate_edge_rows: list[dict[str, Any]] | None = None,
        hydrate_legacy_node_rows: list[dict[str, Any]] | None = None,
        hydrate_legacy_edge_rows: list[dict[str, Any]] | None = None,
        hydrate_rows: list[dict[str, Any]] | None = None,
    ) -> None:
        # hydrate_rows is a compatibility alias for hydrate_node_rows.
        if hydrate_rows is not None and hydrate_node_rows is None:
            hydrate_node_rows = hydrate_rows
        self.calls: list[tuple[str, dict[str, Any] | None]] = []
        self._hydrate_node_rows = list(hydrate_node_rows or [])
        self._hydrate_edge_rows = list(hydrate_edge_rows or [])
        self._hydrate_legacy_node_rows = list(hydrate_legacy_node_rows or [])
        self._hydrate_legacy_edge_rows = list(hydrate_legacy_edge_rows or [])

    def graph_query(self, cypher: str, params: dict[str, Any] | None = None) -> Any:
        self.calls.append((cypher, params))
        if "WHERE n.payload_json IS NOT NULL" in cypher:
            return self._hydrate_legacy_node_rows
        if "WHERE e.payload_json IS NOT NULL" in cypher:
            return self._hydrate_legacy_edge_rows
        if "RETURN n.node_id AS node_id" in cypher:
            return self._hydrate_node_rows
        if "RETURN e.edge_id AS edge_id" in cypher:
            return self._hydrate_edge_rows
        return []


class RedisGraphQueryClient:
    """Minimal sync Redis GRAPH.QUERY client for FalkorDB."""

    def __init__(self, *, uri: str, graph_name: str) -> None:
        import redis
        from redis.commands.graph import Graph

        parsed = urlparse(uri or "redis://localhost:6379")
        self._r = redis.Redis(
            host=parsed.hostname or "localhost",
            port=int(parsed.port or 6379),
            db=int((parsed.path or "/0").lstrip("/") or 0),
            decode_responses=True,
        )
        self._graph = Graph(self._r, graph_name)

    def graph_query(self, cypher: str, params: dict[str, Any] | None = None) -> Any:
        result = self._graph.query(cypher, params=params)
        return result.result_set


def _with_sanitized_metadata(model: Any) -> Any:
    cleaned, _rejected = sanitize_metadata(getattr(model, "metadata", None) or {})
    if cleaned == (getattr(model, "metadata", None) or {}):
        return model
    return model.model_copy(update={"metadata": cleaned})


class FalkorSubstrateStore:
    """SubstrateGraphStore with Falkor write-through and in-memory read cache.

    Durable persistence is Concept + SubstrateEdge only. Non-concept node
    upserts raise ``ValueError`` rather than writing incomplete native rows.
    """

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
                "MATCH (n:SubstrateNode) RETURN " + _return_clause("n", NATIVE_NODE_RETURN_FIELDS)
            )
            edge_rows = self._client.graph_query(
                "MATCH (source:SubstrateNode)-[e]->(target:SubstrateNode) "
                "WHERE e.substrate_edge = true "
                "RETURN " + _return_clause("e", NATIVE_EDGE_RETURN_FIELDS)
            )
            legacy_node_rows = self._client.graph_query(
                "MATCH (n:SubstrateNode) WHERE n.payload_json IS NOT NULL "
                "RETURN n.payload_json AS payload_json, n.identity_key AS identity_key"
            )
            legacy_edge_rows = self._client.graph_query(
                "MATCH ()-[e]->() WHERE e.payload_json IS NOT NULL "
                "RETURN e.payload_json AS payload_json, e.identity_key AS identity_key"
            )
        except Exception as exc:
            logger.warning("falkor_substrate_hydrate_failed error=%s", exc)
            return

        for row in _normalize_rows(node_rows):
            try:
                node = decode_concept_node(row)
            except Exception:
                logger.warning("falkor_substrate_hydrate_node_invalid")
                continue
            if node is None:
                continue
            identity = row.get("identity_key")
            self._cache.upsert_node(identity_key=str(identity) if identity else None, node=node)

        for row in _normalize_rows(edge_rows):
            try:
                edge = decode_edge(row)
            except Exception:
                logger.warning("falkor_substrate_hydrate_edge_invalid")
                continue
            if edge is None:
                continue
            identity = row.get("identity_key") or self._edge_identity(edge)
            self._cache.upsert_edge(identity_key=str(identity), edge=edge)

        self._migrate_legacy_payload_nodes(_normalize_rows(legacy_node_rows))
        self._migrate_legacy_payload_edges(_normalize_rows(legacy_edge_rows))

    def _migrate_legacy_payload_nodes(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            payload = row.get("payload_json")
            if not payload:
                continue
            try:
                node = NODE_ADAPTER.validate_json(payload)
            except Exception:
                logger.warning("falkor_substrate_legacy_node_invalid")
                continue
            if getattr(node, "node_kind", None) != "concept":
                logger.warning(
                    "falkor_substrate_legacy_node_skipped node_kind=%s node_id=%s",
                    getattr(node, "node_kind", None),
                    getattr(node, "node_id", None),
                )
                continue
            identity = row.get("identity_key")
            identity_key = str(identity) if identity else None
            try:
                # Rewrite to native properties and REMOVE payload_json.
                self.upsert_node(identity_key=identity_key, node=node)
                logger.info(
                    "falkor_substrate_legacy_node_migrated node_id=%s identity_key=%s",
                    node.node_id,
                    identity_key or "",
                )
            except Exception as exc:
                logger.warning(
                    "falkor_substrate_legacy_node_migrate_failed node_id=%s error=%s",
                    getattr(node, "node_id", None),
                    exc,
                )

    def _migrate_legacy_payload_edges(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            payload = row.get("payload_json")
            if not payload:
                continue
            try:
                edge = SubstrateEdgeV1.model_validate_json(payload)
            except Exception:
                logger.warning("falkor_substrate_legacy_edge_invalid")
                continue
            identity = row.get("identity_key") or self._edge_identity(edge)
            try:
                self.upsert_edge(identity_key=str(identity), edge=edge)
                logger.info(
                    "falkor_substrate_legacy_edge_migrated edge_id=%s identity_key=%s",
                    edge.edge_id,
                    identity,
                )
            except Exception as exc:
                logger.warning(
                    "falkor_substrate_legacy_edge_migrate_failed edge_id=%s error=%s",
                    edge.edge_id,
                    exc,
                )

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
        if getattr(node, "node_kind", None) != "concept":
            raise ValueError(
                "FalkorSubstrateStore durable writes support concept nodes only; "
                f"got node_kind={getattr(node, 'node_kind', None)!r}"
            )
        node = _with_sanitized_metadata(node)
        params = encode_node_properties(node, identity_key)
        label = node_label_for_kind(str(node.node_kind))
        assignments = _set_assignments("n", params, skip={"node_id"})
        cypher = (
            f"MERGE (n:SubstrateNode:{label} {{node_id: $node_id}}) "
            f"SET {assignments} "
            "REMOVE n.payload_json"
        )
        try:
            self._client.graph_query(cypher, params)
        except Exception as exc:
            logger.error("falkor_substrate_upsert_node_failed node_id=%s error=%s", node.node_id, exc)
            raise
        self._cache.upsert_node(identity_key=identity_key, node=node)

    def upsert_edge(self, *, identity_key: str, edge: SubstrateEdgeV1) -> None:
        edge = _with_sanitized_metadata(edge)
        params = encode_edge_properties(edge, identity_key)
        relationship_type = edge.predicate
        assignments = _set_assignments("e", params, skip={"edge_id", "source_id", "target_id"})
        cypher = (
            "MERGE (source:SubstrateNode {node_id: $source_id}) "
            "MERGE (target:SubstrateNode {node_id: $target_id}) "
            f"MERGE (source)-[e:`{relationship_type}` {{edge_id: $edge_id}}]->(target) "
            f"SET {assignments} "
            "REMOVE e.payload_json"
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

    def query_hotspot_region(
        self, *, min_salience: float = 0.6, limit_nodes: int = 32, limit_edges: int = 64
    ) -> SubstrateQueryResultV1:
        result = self._cache.query_hotspot_region(
            min_salience=min_salience, limit_nodes=limit_nodes, limit_edges=limit_edges
        )
        return _retag_source(result, self._result_source_kind)

    def query_contradiction_region(
        self, *, limit_nodes: int = 32, limit_edges: int = 64
    ) -> SubstrateQueryResultV1:
        return _retag_source(
            self._cache.query_contradiction_region(limit_nodes=limit_nodes, limit_edges=limit_edges),
            self._result_source_kind,
        )

    def query_concept_region(
        self, *, limit_nodes: int = 32, limit_edges: int = 64
    ) -> SubstrateQueryResultV1:
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
        # Raw GRAPH.QUERY response: [header, records, statistics]. redis-py's
        # Graph.query normally strips this shape, but retain compatibility
        # with injected clients that return the wire response.
        if (
            len(raw) == 3
            and isinstance(raw[0], list)
            and isinstance(raw[1], list)
            and raw[0]
            and all(isinstance(column, (list, tuple)) for column in raw[0])
        ):
            names = [str(column[0]).split(".")[-1] for column in raw[0]]
            return [
                dict(zip(names, record))
                for record in raw[1]
                if isinstance(record, (list, tuple))
            ]
        out: list[dict[str, Any]] = []
        for item in raw:
            if isinstance(item, dict):
                out.append(item)
            elif isinstance(item, (list, tuple)) and len(item) >= 1:
                if len(item) >= 2:
                    out.append({"node_id": item[0], "identity_key": item[1]})
                else:
                    out.append({"node_id": item[0], "identity_key": ""})
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
