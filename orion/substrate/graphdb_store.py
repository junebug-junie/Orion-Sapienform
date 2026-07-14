from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Optional
from urllib.parse import urlparse, urlunparse

import requests

from pydantic import TypeAdapter
from orion.graph.sparql_client import SparqlHttpClient, resolve_substrate_sparql_http_basic_auth
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


class SubstrateSparqlBackendUnconfiguredError(RuntimeError):
    """Raised when ``SUBSTRATE_STORE_BACKEND=sparql`` but no query/update URL could be resolved."""

    def __init__(self) -> None:
        super().__init__(
            "substrate_sparql_backend_unconfigured: set SUBSTRATE_GRAPH_QUERY_URL and "
            "SUBSTRATE_GRAPH_UPDATE_URL (or RDF_STORE_QUERY_URL / RDF_STORE_UPDATE_URL)."
        )


@dataclass(frozen=True)
class GraphDBSubstrateStoreConfig:
    endpoint: str
    graph_uri: str = DEFAULT_SUBSTRATE_GRAPH_URI
    timeout_sec: float = 5.0
    user: str | None = None
    password: str | None = None
    # Real change detection, not a blind timer (2026-07-14 incident + follow-up -- see
    # docs/superpowers/specs/2026-07-14-substrate-graph-store-snapshot-cache-spec.md).
    # An earlier version of this fix used a short blind ttl instead; removed same-day
    # once code review found it was provably dead under every real default (a 2s ttl
    # is always subsumed by this field once it's >= 2s) and its docstring contradicted
    # this one. If this store's OWN write generation hasn't moved since the last
    # successful fetch (see GraphDBSubstrateStore._write), snapshot() trusts the cache
    # for up to this many seconds without re-querying at all -- this is what actually
    # suppresses a dominant periodic caller's own solo query rate (e.g. a 5s-forever
    # tick), which a short ttl alone could never do (5s > 2s never finds itself
    # still-fresh on that caller's own cadence). Bounds worst-case staleness from
    # writes made by a DIFFERENT process this instance can't see via its own counter.
    # 0 means trust the cache forever once a write generation has been recorded (no
    # periodic forced refresh at all) -- deliberate, not an oversight; a pure-reader
    # process that never writes through this instance would then rely entirely on
    # process restart to ever see external changes, so leave this at its default
    # unless you have a specific reason not to.
    snapshot_force_refresh_ceiling_sec: float = 30.0


NODE_ADAPTER = TypeAdapter(SubstrateNodeV1)


class GraphDBSubstrateStore(SubstrateGraphStore):
    """GraphDB-backed substrate store with bounded cache fallback."""

    def __init__(self, cfg: GraphDBSubstrateStoreConfig) -> None:
        self._cfg = cfg
        self._cache = InMemorySubstrateGraphStore()
        self._result_source_kind = "graphdb"
        self._last_snapshot_at: float | None = None
        # snapshot() is called from multiple OS threads in this deployment (e.g.
        # orion-substrate-runtime's brain-frame tick runs via asyncio.to_thread) --
        # this guards the check-then-act TTL read/write and the cache refresh that
        # follows a live query, so two threads racing past an expired TTL can't both
        # fire independent live queries and race to write _last_snapshot_at/self._cache
        # (2026-07-14 review finding).
        self._snapshot_lock = threading.Lock()
        # Real change detection, not just a blind timer (2026-07-14 follow-up): every
        # write this process makes through _write() bumps this counter. snapshot()
        # trusts the cache indefinitely as long as OUR OWN write generation hasn't
        # moved -- this is the actual fix for the traced incident, since
        # orion-substrate-runtime's dominant read callers (brain-frame tick, dynamics
        # tick, attention-broadcast, endogenous-curiosity) and its writers (dynamics
        # tick, prediction-error nodes) all share ONE store instance, so a same-process
        # write is a reliable "something really changed" signal. snapshot_force_refresh_
        # ceiling_sec remains as a safety-net ceiling for changes written by a DIFFERENT
        # process this instance can't see via its own counter (e.g. cortex-exec's
        # beliefs_for_stance writing via its own separate store instance).
        self._write_generation = 0
        self._last_snapshot_generation: int | None = None

    def _write(self, sparql: str) -> None:
        """All real writes funnel through here (not directly through self._update(),
        which is per-backend and overridden in SparqlSubstrateStore) so the write
        generation counter is bumped exactly once per successful write regardless of
        backend."""
        self._update(sparql)
        self._write_generation += 1

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
        self._write(update)
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
        self._write(update)
        self._cache.upsert_edge(identity_key=identity_key, edge=edge)
        self._upsert_identity(identity_key=identity_key, canonical_id=edge.edge_id, identity_kind="edge")

    def snapshot(self) -> MaterializedSubstrateGraphState:
        with self._snapshot_lock:
            now_mono = time.monotonic()
            elapsed = (now_mono - self._last_snapshot_at) if self._last_snapshot_at is not None else None
            ceiling = float(self._cfg.snapshot_force_refresh_ceiling_sec)

            same_generation = (
                self._last_snapshot_generation is not None
                and self._last_snapshot_generation == self._write_generation
            )
            # A write since our last fetch is a KNOWN, certain change -- the cache is
            # never reused in that case, regardless of ceiling. This is the actual
            # correctness fix: without this gate, a caller that just wrote something
            # (e.g. _write_prediction_error_node) followed immediately by a caller that
            # reads it (e.g. _dynamics_tick) could be served pre-write stale data.
            # ceiling <= 0 means "trust same_generation forever" (no periodic forced
            # refresh at all); otherwise the ceiling forces a real refresh once elapsed,
            # even though same_generation is still true -- the safety net that bounds
            # staleness from writes made by a process other than this one, whose
            # write-generation counter this instance can't see.
            within_ceiling = ceiling <= 0.0 or elapsed is None or elapsed < ceiling

            if elapsed is not None and same_generation and within_ceiling:
                return self._cache.snapshot()

            # Capture the generation BEFORE issuing the live query (which releases the
            # GIL during network I/O), not after. If a write races in while this query
            # is in flight, capturing after would credit that write to data fetched
            # before it happened, silently masking the write until the next unrelated
            # write or the ceiling fires -- the exact bug this whole mechanism exists to
            # prevent. Capturing before means any such race is resolved safely: the next
            # snapshot() call sees a generation mismatch and re-fetches (one possibly-
            # redundant extra query), never a falsely-trusted stale cache. _write() is
            # deliberately not lock-protected against this method -- serializing writes
            # against reads isn't needed once the capture-before-fetch ordering holds.
            generation_at_fetch_start = self._write_generation
            try:
                nodes, _ = self._query_nodes(limit_nodes=500)
                node_ids = [n.node_id for n in nodes]
                edges, _ = self._query_edges_for_node_ids(node_ids=node_ids, limit_edges=1000)
            except GraphDBSubstrateStoreError:
                return self._cache.snapshot()
            self._refresh_cache(nodes=nodes, edges=edges)
            self._last_snapshot_at = now_mono
            self._last_snapshot_generation = generation_at_fetch_start
            return self._cache.snapshot()

    # ---- primary query layer (GraphDB-first) ----
    def query_focal_slice(self, *, node_ids: list[str], max_edges: int = 64) -> SubstrateQueryResultV1:
        node_ids = [str(node_id).strip() for node_id in node_ids if str(node_id).strip()]
        edges_limit = max(1, int(max_edges))
        if not node_ids:
            return SubstrateQueryResultV1(
                query_kind="focal_slice",
                source_kind=self._result_source_kind,
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
                source_kind=self._result_source_kind,
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
                source_kind=self._result_source_kind,
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
                source_kind=self._result_source_kind,
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
                source_kind=self._result_source_kind,
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
                source_kind=self._result_source_kind,
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
        self._write(update)

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


@dataclass(frozen=True)
class SparqlSubstrateStoreConfig:
    query_url: str
    update_url: str
    graph_uri: str = DEFAULT_SUBSTRATE_GRAPH_URI
    timeout_sec: float = 5.0
    user: str | None = None
    password: str | None = None
    auth_source: str = "none"
    snapshot_force_refresh_ceiling_sec: float = 30.0


class SparqlSubstrateStore(GraphDBSubstrateStore):
    """Substrate store over generic SPARQL query + update endpoints (Fuseki)."""

    def __init__(self, cfg: SparqlSubstrateStoreConfig) -> None:
        super().__init__(
            GraphDBSubstrateStoreConfig(
                endpoint=cfg.query_url,
                graph_uri=cfg.graph_uri,
                timeout_sec=cfg.timeout_sec,
                user=cfg.user,
                password=cfg.password,
                snapshot_force_refresh_ceiling_sec=cfg.snapshot_force_refresh_ceiling_sec,
            )
        )
        self._result_source_kind = "sparql"
        self._sparql_update_url = cfg.update_url.rstrip("/")
        self._sparql_auth_source = cfg.auth_source
        self._sparql_http = SparqlHttpClient(
            cfg.query_url,
            cfg.update_url,
            timeout_sec=cfg.timeout_sec,
            user=cfg.user,
            password=cfg.password,
        )

    def _sparql_update_endpoint(self) -> str:
        return self._sparql_update_url

    def _select(self, sparql: str) -> list[dict[str, dict[str, str]]]:
        try:
            rows = self._sparql_http.select(sparql)
            return rows
        except Exception as exc:  # noqa: BLE001
            raise GraphDBSubstrateStoreError(str(exc)) from exc

    def _update(self, sparql: str) -> None:
        try:
            self._sparql_http.update(sparql)
        except requests.HTTPError as exc:
            resp = exc.response
            if resp is not None and resp.status_code == 401:
                hint = (
                    "set_SUBSTRATE_GRAPH_USER_PASS_or_RDF_STORE_USER_PASS_or_FUSEKI_USER_PASS"
                    if self._sparql_auth_source == "none"
                    else "verify_Fuseki_update_credentials_match_server"
                )
                logger.warning(
                    "substrate_sparql_update_auth_failed status=401 credential_source=%s "
                    "update_url=%s hint=%s",
                    self._sparql_auth_source,
                    self._sparql_http.update_url_redacted,
                    hint,
                )
            detail = ""
            try:
                if resp is not None:
                    detail = (resp.text or "")[:1200]
            except Exception:  # noqa: BLE001
                detail = ""
            raise GraphDBSubstrateStoreError(f"{exc}{f' | {detail}' if detail else ''}") from exc
        except Exception as exc:  # noqa: BLE001
            detail = ""
            try:
                if isinstance(exc, requests.HTTPError) and exc.response is not None:
                    detail = (exc.response.text or "")[:1200]
            except Exception:  # noqa: BLE001
                detail = ""
            raise GraphDBSubstrateStoreError(f"{exc}{f' | {detail}' if detail else ''}") from exc


def _resolve_substrate_graphdb_endpoint() -> str:
    endpoint = str(os.getenv("SUBSTRATE_GRAPHDB_ENDPOINT", "")).strip()
    if endpoint:
        return endpoint
    base = str(os.getenv("GRAPHDB_URL", "")).strip()
    repo = str(os.getenv("GRAPHDB_REPO", "")).strip()
    if base and repo:
        return f"{base.rstrip('/')}/repositories/{repo}"
    return ""


def _resolve_substrate_sparql_http_urls() -> tuple[str, str]:
    """Resolve Fuseki/SPARQL HTTP endpoints for the substrate store.

    Prefer ``SUBSTRATE_GRAPH_*``; fall back to shared ``RDF_STORE_*`` URLs.
    """
    q = str(os.getenv("SUBSTRATE_GRAPH_QUERY_URL", "")).strip()
    if not q:
        q = str(os.getenv("RDF_STORE_QUERY_URL", "")).strip()
    u = str(os.getenv("SUBSTRATE_GRAPH_UPDATE_URL", "")).strip()
    if not u:
        u = str(os.getenv("RDF_STORE_UPDATE_URL", "")).strip()
    return q, u


def _resolve_substrate_named_graph_uri() -> str:
    raw = str(os.getenv("SUBSTRATE_GRAPH_URI", "")).strip() or str(os.getenv("SUBSTRATE_GRAPH_GRAPH_URI", "")).strip()
    return raw or DEFAULT_SUBSTRATE_GRAPH_URI


def _resolve_snapshot_force_refresh_ceiling_sec() -> float:
    """See GraphDBSubstrateStoreConfig.snapshot_force_refresh_ceiling_sec for the
    reasoning -- real change detection via a same-process write-generation counter,
    with this as the periodic safety-net ceiling for cross-process writes."""
    raw = str(os.getenv("SUBSTRATE_SNAPSHOT_FORCE_REFRESH_CEILING_SEC", "")).strip()
    if not raw:
        return 30.0
    try:
        return float(raw)
    except ValueError:
        logger.warning("substrate_snapshot_force_refresh_ceiling_invalid value=%r; using default 30.0", raw)
        return 30.0


def _redact_endpoint_for_log(endpoint: str) -> str:
    """Log-safe endpoint (host + path only; strips userinfo)."""
    p = urlparse(endpoint)
    netloc = p.hostname or ""
    if p.port:
        netloc = f"{netloc}:{p.port}"
    return urlunparse((p.scheme, netloc, p.path, p.params, p.query, p.fragment))


def build_substrate_store_from_env() -> SubstrateGraphStore:
    """Select substrate semantic store.

    **RDF Store V1 safety:** GraphDB is used only when ``SUBSTRATE_STORE_BACKEND`` is set to
    ``graphdb`` (or aliases). Global ``GRAPHDB_URL`` alone must **not** activate this store;
    that avoids accidental duplicate RDF surfaces during backend-neutral RDF writer cutover.

    Resolution:
    - ``SUBSTRATE_STORE_BACKEND`` unset / empty → in-memory (default).
    - ``in_memory`` / ``memory`` / ``mem`` / ``local`` → in-memory.
    - ``sparql`` / ``sparql_http`` → ``SparqlSubstrateStore`` when query and update URLs resolve
      (``SUBSTRATE_GRAPH_QUERY_URL`` / ``SUBSTRATE_GRAPH_UPDATE_URL``, else ``RDF_STORE_QUERY_URL`` /
      ``RDF_STORE_UPDATE_URL``). If still missing, logs ``substrate_sparql_backend_unconfigured`` and raises
      ``SubstrateSparqlBackendUnconfiguredError``.
    - ``graphdb`` / ``graph_db`` / ``rdf`` → ``GraphDBSubstrateStore`` when
      ``SUBSTRATE_GRAPHDB_ENDPOINT`` or ``GRAPHDB_URL``+``GRAPHDB_REPO`` resolves; else in-memory + warning.
    - Any other backend string → in-memory + warning.
    """
    backend = str(os.getenv("SUBSTRATE_STORE_BACKEND", "")).strip().lower()
    endpoint = _resolve_substrate_graphdb_endpoint()

    if backend in {"in_memory", "memory", "mem", "local"}:
        logger.info("substrate_store_backend_selected backend=in_memory reason=explicit_backend")
        return InMemorySubstrateGraphStore()

    if not backend:
        logger.info("substrate_store_backend_selected backend=in_memory reason=default_v1_safety")
        return InMemorySubstrateGraphStore()

    if backend in {"sparql", "sparql_http"}:
        q, u = _resolve_substrate_sparql_http_urls()
        if not q or not u:
            exc = SubstrateSparqlBackendUnconfiguredError()
            logger.error("substrate_sparql_backend_unconfigured query_resolved=%r update_resolved=%r", bool(q), bool(u))
            raise exc
        graph_uri = _resolve_substrate_named_graph_uri()
        auth_user, auth_pass, auth_src = resolve_substrate_sparql_http_basic_auth()
        auth_configured = bool(auth_user and auth_pass)
        logger.info(
            "substrate_store_backend_selected backend=sparql query_url=%s update_url=%s graph_uri=%s "
            "auth_configured=%s auth_source=%s",
            _redact_endpoint_for_log(q),
            _redact_endpoint_for_log(u),
            graph_uri,
            str(auth_configured).lower(),
            auth_src,
        )
        cfg = SparqlSubstrateStoreConfig(
            query_url=q,
            update_url=u,
            graph_uri=graph_uri,
            timeout_sec=float(os.getenv("SUBSTRATE_GRAPH_TIMEOUT_SEC", "5.0")),
            user=auth_user,
            password=auth_pass,
            auth_source=auth_src,
            snapshot_force_refresh_ceiling_sec=_resolve_snapshot_force_refresh_ceiling_sec(),
        )
        return SparqlSubstrateStore(cfg)

    if backend in {"graphdb", "graph_db", "rdf"}:
        if not endpoint:
            logger.warning("SUBSTRATE_STORE_BACKEND=graphdb but endpoint missing; falling back to in-memory store")
            logger.info("substrate_store_backend_selected backend=in_memory reason=graphdb_endpoint_missing")
            return InMemorySubstrateGraphStore()
        logger.info(
            "substrate_store_backend_selected backend=graphdb endpoint=%s",
            _redact_endpoint_for_log(endpoint),
        )
        cfg = GraphDBSubstrateStoreConfig(
            endpoint=endpoint,
            graph_uri=str(os.getenv("SUBSTRATE_GRAPHDB_GRAPH_URI", DEFAULT_SUBSTRATE_GRAPH_URI)).strip() or DEFAULT_SUBSTRATE_GRAPH_URI,
            timeout_sec=float(os.getenv("SUBSTRATE_GRAPHDB_TIMEOUT_SEC", "5.0")),
            user=str(os.getenv("SUBSTRATE_GRAPHDB_USER", "")).strip() or None,
            password=str(os.getenv("SUBSTRATE_GRAPHDB_PASS", "")).strip() or None,
            snapshot_force_refresh_ceiling_sec=_resolve_snapshot_force_refresh_ceiling_sec(),
        )
        return GraphDBSubstrateStore(cfg)

    logger.warning("Unknown SUBSTRATE_STORE_BACKEND=%r; using in-memory store", backend)
    logger.info("substrate_store_backend_selected backend=in_memory reason=unknown_backend")
    return InMemorySubstrateGraphStore()
