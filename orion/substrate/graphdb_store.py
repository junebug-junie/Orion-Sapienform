from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Optional

import requests

from pydantic import TypeAdapter
from orion.core.schemas.cognitive_substrate import BaseSubstrateNodeV1, SubstrateEdgeV1, SubstrateNodeV1

from .store import (
    InMemorySubstrateGraphStore,
    MaterializedSubstrateGraphState,
    SubstrateGraphStore,
    SubstrateNeighborhoodSliceV1,
    SubstrateQueryResultV1,
)

logger = logging.getLogger("orion.substrate.graphdb_store")

ORION_SUBSTRATE_NS = "http://conjourney.net/orion/substrate#"
DEFAULT_SUBSTRATE_GRAPH_URI = "http://conjourney.net/graph/orion/substrate"


class GraphDBSubstrateStoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class GraphDBSubstrateStoreConfig:
    endpoint: str
    graph_uri: str = DEFAULT_SUBSTRATE_GRAPH_URI
    timeout_sec: float = 5.0
    user: str | None = None
    password: str | None = None


NODE_ADAPTER = TypeAdapter(SubstrateNodeV1)


class GraphDBSubstrateStore(SubstrateGraphStore):
    """GraphDB-backed substrate store with bounded cache fallback."""

    def __init__(self, cfg: GraphDBSubstrateStoreConfig) -> None:
        self._cfg = cfg
        self._cache = InMemorySubstrateGraphStore()

    def _sparql_update_endpoint(self) -> str:
        """GraphDB accepts SPARQL UPDATE only on the RDF4J statements endpoint, not the repo root."""
        base = str(self._cfg.endpoint or "").rstrip("/")
        if not base:
            return base
        if base.endswith("/statements"):
            return base
        return f"{base}/statements"

    # ---- canonical write / lookup path ----
    def get_node_by_id(self, node_id: str) -> BaseSubstrateNodeV1 | None:
        cached = self._cache.get_node_by_id(node_id)
        if cached is not None:
            return cached
        try:
            rows = self._select(
                f"""
PREFIX orion: <{ORION_SUBSTRATE_NS}>
SELECT ?payload_json
WHERE {{
  GRAPH <{self._cfg.graph_uri}> {{
    ?node orion:nodeId {self._lit(node_id)} ;
          orion:payloadJson ?payload_json .
  }}
}}
LIMIT 1
""".strip()
            )
        except GraphDBSubstrateStoreError:
            return None
        payload = self._binding_str(rows[0], "payload_json") if rows else None
        if not payload:
            return None
        node = NODE_ADAPTER.validate_json(payload)
        self._cache.upsert_node(identity_key=None, node=node)
        return node

    def get_edge_by_id(self, edge_id: str) -> SubstrateEdgeV1 | None:
        cached = self._cache.get_edge_by_id(edge_id)
        if cached is not None:
            return cached
        try:
            rows = self._select(
                f"""
PREFIX orion: <{ORION_SUBSTRATE_NS}>
SELECT ?payload_json
WHERE {{
  GRAPH <{self._cfg.graph_uri}> {{
    ?edge orion:edgeId {self._lit(edge_id)} ;
          orion:payloadJson ?payload_json .
  }}
}}
LIMIT 1
""".strip()
            )
        except GraphDBSubstrateStoreError:
            return None
        payload = self._binding_str(rows[0], "payload_json") if rows else None
        if not payload:
            return None
        edge = SubstrateEdgeV1.model_validate_json(payload)
        self._cache.upsert_edge(identity_key=self._edge_identity(edge), edge=edge)
        return edge

    def get_node_id_by_identity(self, identity_key: str) -> str | None:
        cached = self._cache.get_node_id_by_identity(identity_key)
        if cached:
            return cached
        return self._get_canonical_id(identity_key=identity_key, identity_kind="node")

    def get_edge_id_by_identity(self, identity_key: str) -> str | None:
        cached = self._cache.get_edge_id_by_identity(identity_key)
        if cached:
            return cached
        return self._get_canonical_id(identity_key=identity_key, identity_kind="edge")

    def upsert_node(self, *, identity_key: str | None, node: BaseSubstrateNodeV1) -> None:
        node_iri = self._node_iri(node.node_id)
        payload_json = json.dumps(node.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        metadata_json = json.dumps(node.metadata or {}, ensure_ascii=False, sort_keys=True)
        provenance_json = json.dumps(node.provenance.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        evidence_values = "\n".join(
            f"    {node_iri} orion:evidenceRef {self._lit(str(ev))} ."
            for ev in sorted(set(node.provenance.evidence_refs or []))
            if str(ev).strip()
        )
        subject_clause = f"    {node_iri} orion:subjectRef {self._lit(str(node.subject_ref))} ." if node.subject_ref else ""
        update = f"""
PREFIX orion: <{ORION_SUBSTRATE_NS}>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
DELETE {{ GRAPH <{self._cfg.graph_uri}> {{ {node_iri} ?p ?o . }} }}
INSERT {{
  GRAPH <{self._cfg.graph_uri}> {{
    {node_iri} a orion:SubstrateNode ;
      orion:nodeId {self._lit(node.node_id)} ;
      orion:nodeKind {self._lit(node.node_kind)} ;
      orion:anchorScope {self._lit(node.anchor_scope)} ;
      orion:promotionState {self._lit(node.promotion_state)} ;
      orion:riskTier {self._lit(node.risk_tier)} ;
      orion:confidence {self._typed_float(node.signals.confidence)} ;
      orion:salience {self._typed_float(node.signals.salience)} ;
      orion:observedAt {self._typed_datetime(node.temporal.observed_at.isoformat())} ;
      orion:metadataJson {self._lit_long(metadata_json)} ;
      orion:provenanceJson {self._lit_long(provenance_json)} ;
      orion:payloadJson {self._lit_long(payload_json)} .
{subject_clause}
{evidence_values}
  }}
}}
WHERE {{ OPTIONAL {{ GRAPH <{self._cfg.graph_uri}> {{ {node_iri} ?p ?o . }} }} }}
""".strip()
        self._update(update)
        self._cache.upsert_node(identity_key=identity_key, node=node)
        if identity_key:
            self._upsert_identity(identity_key=identity_key, canonical_id=node.node_id, identity_kind="node")

    def upsert_edge(self, *, identity_key: str, edge: SubstrateEdgeV1) -> None:
        edge_iri = self._edge_iri(edge.edge_id)
        payload_json = json.dumps(edge.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        metadata_json = json.dumps(edge.metadata or {}, ensure_ascii=False, sort_keys=True)
        provenance_json = json.dumps(edge.provenance.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        evidence_values = "\n".join(
            f"    {edge_iri} orion:evidenceRef {self._lit(str(ev))} ."
            for ev in sorted(set(edge.provenance.evidence_refs or []))
            if str(ev).strip()
        )
        update = f"""
PREFIX orion: <{ORION_SUBSTRATE_NS}>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
DELETE {{ GRAPH <{self._cfg.graph_uri}> {{ {edge_iri} ?p ?o . }} }}
INSERT {{
  GRAPH <{self._cfg.graph_uri}> {{
    {edge_iri} a orion:SubstrateEdge ;
      orion:edgeId {self._lit(edge.edge_id)} ;
      orion:sourceNodeId {self._lit(edge.source.node_id)} ;
      orion:targetNodeId {self._lit(edge.target.node_id)} ;
      orion:predicate {self._lit(edge.predicate)} ;
      orion:confidence {self._typed_float(edge.confidence)} ;
      orion:salience {self._typed_float(edge.salience)} ;
      orion:observedAt {self._typed_datetime(edge.temporal.observed_at.isoformat())} ;
      orion:metadataJson {self._lit_long(metadata_json)} ;
      orion:provenanceJson {self._lit_long(provenance_json)} ;
      orion:payloadJson {self._lit_long(payload_json)} .
{evidence_values}
  }}
}}
WHERE {{ OPTIONAL {{ GRAPH <{self._cfg.graph_uri}> {{ {edge_iri} ?p ?o . }} }} }}
""".strip()
        self._update(update)
        self._cache.upsert_edge(identity_key=identity_key, edge=edge)
        self._upsert_identity(identity_key=identity_key, canonical_id=edge.edge_id, identity_kind="edge")

    def snapshot(self) -> MaterializedSubstrateGraphState:
        try:
            nodes, _ = self._query_nodes(limit_nodes=500)
            node_ids = [n.node_id for n in nodes]
            edges, _ = self._query_edges_for_node_ids(node_ids=node_ids, limit_edges=1000)
        except GraphDBSubstrateStoreError:
            return self._cache.snapshot()
        self._refresh_cache(nodes=nodes, edges=edges)
        return self._cache.snapshot()

    # ---- primary query layer (GraphDB-first) ----
    def query_focal_slice(self, *, node_ids: list[str], max_edges: int = 64) -> SubstrateQueryResultV1:
        node_ids = [str(node_id).strip() for node_id in node_ids if str(node_id).strip()]
        edges_limit = max(1, int(max_edges))
        if not node_ids:
            return SubstrateQueryResultV1(
                query_kind="focal_slice",
                source_kind="graphdb",
                slice=SubstrateNeighborhoodSliceV1(nodes=[], edges=[]),
                limits={"max_edges": edges_limit, "node_ids": 0},
            )
        try:
            nodes, nodes_truncated = self._query_nodes(node_ids=node_ids, limit_nodes=max(1, len(node_ids)))
            selected_ids = [node.node_id for node in nodes]
            edges, edges_truncated = self._query_edges_for_node_ids(node_ids=selected_ids, limit_edges=edges_limit)
            self._refresh_cache(nodes=nodes, edges=edges)
            return SubstrateQueryResultV1(
                query_kind="focal_slice",
                source_kind="graphdb",
                slice=SubstrateNeighborhoodSliceV1(nodes=nodes, edges=edges),
                truncated=nodes_truncated or edges_truncated,
                limits={"max_edges": edges_limit, "node_ids": len(node_ids)},
            )
        except GraphDBSubstrateStoreError as exc:
            fallback = self._cache.query_focal_slice(node_ids=node_ids, max_edges=edges_limit)
            return self._degraded(query_kind="focal_slice", fallback=fallback, error=str(exc))

    def query_hotspot_region(self, *, min_salience: float = 0.6, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateQueryResultV1:
        nodes_limit = max(1, int(limit_nodes))
        edges_limit = max(1, int(limit_edges))
        threshold = max(0.0, min(1.0, float(min_salience)))
        try:
            nodes, nodes_truncated = self._query_nodes(min_salience=threshold, limit_nodes=nodes_limit)
            node_ids = [node.node_id for node in nodes]
            edges, edges_truncated = self._query_edges_for_node_ids(node_ids=node_ids, limit_edges=edges_limit)
            self._refresh_cache(nodes=nodes, edges=edges)
            return SubstrateQueryResultV1(
                query_kind="hotspot_region",
                source_kind="graphdb",
                slice=SubstrateNeighborhoodSliceV1(nodes=nodes, edges=edges),
                truncated=nodes_truncated or edges_truncated,
                limits={"limit_nodes": nodes_limit, "limit_edges": edges_limit},
                details={"min_salience": threshold},
            )
        except GraphDBSubstrateStoreError as exc:
            fallback = self._cache.query_hotspot_region(min_salience=threshold, limit_nodes=nodes_limit, limit_edges=edges_limit)
            return self._degraded(query_kind="hotspot_region", fallback=fallback, error=str(exc))

    def query_contradiction_region(self, *, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateQueryResultV1:
        return self._query_node_kind_region(query_kind="contradiction_region", node_kind="contradiction", limit_nodes=limit_nodes, limit_edges=limit_edges)

    def query_concept_region(self, *, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateQueryResultV1:
        return self._query_node_kind_region(query_kind="concept_region", node_kind="concept", limit_nodes=limit_nodes, limit_edges=limit_edges)

    def query_provenance_neighborhood(self, *, evidence_ref: str, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateQueryResultV1:
        nodes_limit = max(1, int(limit_nodes))
        edges_limit = max(1, int(limit_edges))
        needle = str(evidence_ref or "").strip()
        if not needle:
            return SubstrateQueryResultV1(
                query_kind="provenance_neighborhood",
                source_kind="graphdb",
                slice=SubstrateNeighborhoodSliceV1(nodes=[], edges=[]),
                limits={"limit_nodes": nodes_limit, "limit_edges": edges_limit},
                details={"evidence_ref": ""},
            )
        try:
            nodes, nodes_truncated = self._query_nodes(evidence_ref=needle, limit_nodes=nodes_limit)
            node_ids = [node.node_id for node in nodes]
            edges, edges_truncated = self._query_edges_for_node_ids(node_ids=node_ids, limit_edges=edges_limit)
            self._refresh_cache(nodes=nodes, edges=edges)
            return SubstrateQueryResultV1(
                query_kind="provenance_neighborhood",
                source_kind="graphdb",
                slice=SubstrateNeighborhoodSliceV1(nodes=nodes, edges=edges),
                truncated=nodes_truncated or edges_truncated,
                limits={"limit_nodes": nodes_limit, "limit_edges": edges_limit},
                details={"evidence_ref": needle},
            )
        except GraphDBSubstrateStoreError as exc:
            fallback = self._cache.query_provenance_neighborhood(evidence_ref=needle, limit_nodes=nodes_limit, limit_edges=edges_limit)
            return self._degraded(query_kind="provenance_neighborhood", fallback=fallback, error=str(exc))

    # compatibility read helpers
    def read_focal_slice(self, *, node_ids: list[str], max_edges: int = 64) -> SubstrateNeighborhoodSliceV1:
        return self.query_focal_slice(node_ids=node_ids, max_edges=max_edges).slice

    def read_hotspot_region(self, *, min_salience: float = 0.6, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateNeighborhoodSliceV1:
        return self.query_hotspot_region(min_salience=min_salience, limit_nodes=limit_nodes, limit_edges=limit_edges).slice

    def read_contradiction_region(self, *, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateNeighborhoodSliceV1:
        return self.query_contradiction_region(limit_nodes=limit_nodes, limit_edges=limit_edges).slice

    def read_concept_region(self, *, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateNeighborhoodSliceV1:
        return self.query_concept_region(limit_nodes=limit_nodes, limit_edges=limit_edges).slice

    def read_provenance_neighborhood(self, *, evidence_ref: str, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateNeighborhoodSliceV1:
        return self.query_provenance_neighborhood(evidence_ref=evidence_ref, limit_nodes=limit_nodes, limit_edges=limit_edges).slice

    # ---- internal query helpers ----
    def _query_node_kind_region(self, *, query_kind: str, node_kind: str, limit_nodes: int, limit_edges: int) -> SubstrateQueryResultV1:
        nodes_limit = max(1, int(limit_nodes))
        edges_limit = max(1, int(limit_edges))
        try:
            nodes, nodes_truncated = self._query_nodes(node_kind=node_kind, limit_nodes=nodes_limit)
            node_ids = [node.node_id for node in nodes]
            edges, edges_truncated = self._query_edges_for_node_ids(node_ids=node_ids, limit_edges=edges_limit)
            self._refresh_cache(nodes=nodes, edges=edges)
            return SubstrateQueryResultV1(
                query_kind=query_kind,
                source_kind="graphdb",
                slice=SubstrateNeighborhoodSliceV1(nodes=nodes, edges=edges),
                truncated=nodes_truncated or edges_truncated,
                limits={"limit_nodes": nodes_limit, "limit_edges": edges_limit},
                details={"node_kind": node_kind},
            )
        except GraphDBSubstrateStoreError as exc:
            if node_kind == "concept":
                fallback = self._cache.query_concept_region(limit_nodes=nodes_limit, limit_edges=edges_limit)
            else:
                fallback = self._cache.query_contradiction_region(limit_nodes=nodes_limit, limit_edges=edges_limit)
            return self._degraded(query_kind=query_kind, fallback=fallback, error=str(exc))

    def _query_nodes(
        self,
        *,
        node_ids: list[str] | None = None,
        node_kind: str | None = None,
        min_salience: float | None = None,
        evidence_ref: str | None = None,
        limit_nodes: int,
    ) -> tuple[list[BaseSubstrateNodeV1], bool]:
        filters: list[str] = []
        values_block = ""
        if node_ids:
            values = " ".join(self._lit(node_id) for node_id in sorted(set(node_ids)))
            values_block = f"VALUES ?node_id {{ {values} }}"
        if node_kind:
            filters.append(f"?node orion:nodeKind {self._lit(node_kind)} .")
        if min_salience is not None:
            filters.append(f"FILTER(xsd:double(?salience) >= {float(min_salience):.6f})")
        if evidence_ref:
            filters.append(f"?node orion:evidenceRef {self._lit(evidence_ref)} .")

        query = f"""
PREFIX orion: <{ORION_SUBSTRATE_NS}>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
SELECT ?node_id ?payload_json ?salience
WHERE {{
  GRAPH <{self._cfg.graph_uri}> {{
    ?node a orion:SubstrateNode ;
      orion:nodeId ?node_id ;
      orion:payloadJson ?payload_json ;
      orion:salience ?salience .
    {values_block}
    {' '.join(filters)}
  }}
}}
ORDER BY DESC(xsd:double(?salience)) ?node_id
LIMIT {int(limit_nodes) + 1}
""".strip()
        rows = self._select(query)
        truncated = len(rows) > int(limit_nodes)
        bounded = rows[: int(limit_nodes)]
        nodes: list[BaseSubstrateNodeV1] = []
        for row in bounded:
            payload = self._binding_str(row, "payload_json")
            if not payload:
                continue
            nodes.append(NODE_ADAPTER.validate_json(payload))
        return nodes, truncated

    def _query_edges_for_node_ids(self, *, node_ids: list[str], limit_edges: int) -> tuple[list[SubstrateEdgeV1], bool]:
        if not node_ids:
            return [], False
        values = " ".join(self._lit(node_id) for node_id in sorted(set(node_ids)))
        query = f"""
PREFIX orion: <{ORION_SUBSTRATE_NS}>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
SELECT DISTINCT ?edge_id ?payload_json ?salience
WHERE {{
  GRAPH <{self._cfg.graph_uri}> {{
    ?edge a orion:SubstrateEdge ;
      orion:edgeId ?edge_id ;
      orion:sourceNodeId ?source_id ;
      orion:targetNodeId ?target_id ;
      orion:payloadJson ?payload_json ;
      orion:salience ?salience .
    VALUES ?focus_id {{ {values} }}
    FILTER(?source_id = ?focus_id || ?target_id = ?focus_id)
  }}
}}
ORDER BY DESC(xsd:double(?salience)) ?edge_id
LIMIT {int(limit_edges) + 1}
""".strip()
        rows = self._select(query)
        truncated = len(rows) > int(limit_edges)
        bounded = rows[: int(limit_edges)]
        edges: list[SubstrateEdgeV1] = []
        for row in bounded:
            payload = self._binding_str(row, "payload_json")
            if not payload:
                continue
            edges.append(SubstrateEdgeV1.model_validate_json(payload))
        return edges, truncated

    def _refresh_cache(self, *, nodes: list[BaseSubstrateNodeV1], edges: list[SubstrateEdgeV1]) -> None:
        for node in nodes:
            self._cache.upsert_node(identity_key=None, node=node)
        for edge in edges:
            self._cache.upsert_edge(identity_key=self._edge_identity(edge), edge=edge)

    def _degraded(self, *, query_kind: str, fallback: SubstrateQueryResultV1, error: str) -> SubstrateQueryResultV1:
        return SubstrateQueryResultV1(
            query_kind=query_kind,
            source_kind="fallback",
            degraded=True,
            error=error,
            slice=fallback.slice,
            truncated=fallback.truncated,
            limits=fallback.limits,
            details={**(fallback.details or {}), "fallback_source": fallback.source_kind},
        )

    # ---- graphdb primitives ----
    def _get_canonical_id(self, *, identity_key: str, identity_kind: str) -> str | None:
        try:
            rows = self._select(
                f"""
PREFIX orion: <{ORION_SUBSTRATE_NS}>
SELECT ?canonical_id
WHERE {{
  GRAPH <{self._cfg.graph_uri}> {{
    {self._identity_iri(identity_key)} a orion:SubstrateIdentity ;
      orion:identityKey {self._lit(identity_key)} ;
      orion:identityKind {self._lit(identity_kind)} ;
      orion:canonicalId ?canonical_id .
  }}
}}
LIMIT 1
""".strip()
            )
        except GraphDBSubstrateStoreError:
            return None
        canonical_id = self._binding_str(rows[0], "canonical_id") if rows else None
        return canonical_id or None

    def _upsert_identity(self, *, identity_key: str, canonical_id: str, identity_kind: str) -> None:
        iri = self._identity_iri(identity_key)
        update = f"""
PREFIX orion: <{ORION_SUBSTRATE_NS}>
DELETE {{ GRAPH <{self._cfg.graph_uri}> {{ {iri} ?p ?o . }} }}
INSERT {{
  GRAPH <{self._cfg.graph_uri}> {{
    {iri} a orion:SubstrateIdentity ;
      orion:identityKey {self._lit(identity_key)} ;
      orion:identityKind {self._lit(identity_kind)} ;
      orion:canonicalId {self._lit(canonical_id)} .
  }}
}}
WHERE {{ OPTIONAL {{ GRAPH <{self._cfg.graph_uri}> {{ {iri} ?p ?o . }} }} }}
""".strip()
        self._update(update)

    def _select(self, sparql: str) -> list[dict[str, dict[str, str]]]:
        try:
            response = requests.post(
                self._cfg.endpoint,
                data=sparql.encode("utf-8"),
                headers={
                    "Content-Type": "application/sparql-query; charset=utf-8",
                    "Accept": "application/sparql-results+json",
                },
                auth=self._auth(),
                timeout=self._cfg.timeout_sec,
            )
            response.raise_for_status()
            payload = response.json()
            return list(payload.get("results", {}).get("bindings", []))
        except Exception as exc:  # noqa: BLE001
            raise GraphDBSubstrateStoreError(str(exc)) from exc

    def _update(self, sparql: str) -> None:
        try:
            response = requests.post(
                self._sparql_update_endpoint(),
                data=sparql.encode("utf-8"),
                headers={"Content-Type": "application/sparql-update; charset=utf-8"},
                auth=self._auth(),
                timeout=self._cfg.timeout_sec,
            )
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            detail = ""
            try:
                if isinstance(exc, requests.HTTPError) and exc.response is not None:
                    detail = (exc.response.text or "")[:1200]
            except Exception:  # noqa: BLE001
                detail = ""
            raise GraphDBSubstrateStoreError(f"{exc}{f' | {detail}' if detail else ''}") from exc

    def _auth(self) -> tuple[str, str] | None:
        if self._cfg.user and self._cfg.password:
            return (self._cfg.user, self._cfg.password)
        return None

    @staticmethod
    def _binding_str(row: dict[str, dict[str, str]], key: str) -> Optional[str]:
        raw = row.get(key)
        if isinstance(raw, dict):
            return str(raw.get("value") or "")
        return None

    @staticmethod
    def _lit(value: Any) -> str:
        escaped = str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        return f'"{escaped}"'

    @staticmethod
    def _lit_long(value: Any) -> str:
        """SPARQL STRING_LITERAL_LONG2 — safe for JSON payloads with quotes and newlines."""
        text = str(value).replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
        return f'"""{text}"""'

    @staticmethod
    def _typed_float(value: float) -> str:
        return f'"{float(value):.6f}"^^xsd:double'

    @staticmethod
    def _typed_datetime(value: str) -> str:
        safe = str(value).replace('"', "")
        return f'"{safe}"^^xsd:dateTime'

    @staticmethod
    def _identity_suffix(raw: str) -> str:
        return sha256(str(raw).encode("utf-8", errors="ignore")).hexdigest()

    def _identity_iri(self, identity_key: str) -> str:
        return f"<urn:orion:substrate:identity:{self._identity_suffix(identity_key)}>"

    @staticmethod
    def _node_iri(node_id: str) -> str:
        return f"<urn:orion:substrate:node:{GraphDBSubstrateStore._identity_suffix(node_id)}>"

    @staticmethod
    def _edge_iri(edge_id: str) -> str:
        return f"<urn:orion:substrate:edge:{GraphDBSubstrateStore._identity_suffix(edge_id)}>"

    @staticmethod
    def _edge_identity(edge: SubstrateEdgeV1) -> str:
        return f"{edge.source.node_id}|{edge.predicate}|{edge.target.node_id}"


DEFAULT_SUBSTRATE_STORE_BACKEND = "in_memory"


def build_substrate_store_from_env() -> SubstrateGraphStore:
    backend = str(os.getenv("SUBSTRATE_STORE_BACKEND", DEFAULT_SUBSTRATE_STORE_BACKEND)).strip().lower()
    if backend in {"graphdb", "graph_db", "rdf"}:
        endpoint = str(os.getenv("SUBSTRATE_GRAPHDB_ENDPOINT", "")).strip()
        if not endpoint:
            base = str(os.getenv("GRAPHDB_URL", "")).strip()
            repo = str(os.getenv("GRAPHDB_REPO", "")).strip()
            if base and repo:
                endpoint = f"{base.rstrip('/')}/repositories/{repo}"
        if not endpoint:
            logger.warning("SUBSTRATE_STORE_BACKEND=graphdb but endpoint missing; falling back to in-memory store")
            return InMemorySubstrateGraphStore()
        cfg = GraphDBSubstrateStoreConfig(
            endpoint=endpoint,
            graph_uri=str(os.getenv("SUBSTRATE_GRAPHDB_GRAPH_URI", DEFAULT_SUBSTRATE_GRAPH_URI)).strip() or DEFAULT_SUBSTRATE_GRAPH_URI,
            timeout_sec=float(os.getenv("SUBSTRATE_GRAPHDB_TIMEOUT_SEC", "5.0")),
            user=str(os.getenv("SUBSTRATE_GRAPHDB_USER", "")).strip() or None,
            password=str(os.getenv("SUBSTRATE_GRAPHDB_PASS", "")).strip() or None,
        )
        return GraphDBSubstrateStore(cfg)
    return InMemorySubstrateGraphStore()
