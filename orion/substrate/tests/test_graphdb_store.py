from __future__ import annotations

import re
from unittest.mock import MagicMock

import pytest
import requests
from datetime import datetime, timezone

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    ContradictionNodeV1,
    NodeRefV1,
    SubstrateEdgeV1,
    SubstrateGraphRecordV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
    SubstrateTemporalWindowV1,
)
from orion.graph.sparql_client import SparqlHttpClient, resolve_substrate_sparql_http_basic_auth
from orion.substrate.graphdb_store import (
    GraphDBSubstrateStore,
    GraphDBSubstrateStoreConfig,
    GraphDBSubstrateStoreError,
    SparqlSubstrateStore,
    SparqlSubstrateStoreConfig,
    SubstrateSparqlBackendUnconfiguredError,
    _resolve_snapshot_force_refresh_ceiling_sec,
    build_substrate_store_from_env,
)
from orion.substrate.materializer import SubstrateGraphMaterializer
from orion.substrate.store import InMemorySubstrateGraphStore


class _Resp:
    def __init__(self, *, payload=None, status_code: int = 200):
        self._payload = payload or {"results": {"bindings": []}}
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"status={self.status_code}")

    def json(self):
        return self._payload


def _patch_sparql_session_post(monkeypatch, fake_post):  # noqa: ANN001
    def session_post(self, url, data=None, headers=None, auth=None, timeout=None, **kwargs):  # noqa: ANN001
        return fake_post(url, data=data, headers=headers, auth=auth, timeout=timeout)

    monkeypatch.setattr("orion.graph.sparql_client.requests.Session.post", session_post)


class _FakeGraphDB:
    def __init__(self) -> None:
        self.nodes: dict[str, str] = {}
        self.edges: dict[str, str] = {}
        self.identity: dict[tuple[str, str], str] = {}

    def post(self, url, data=None, headers=None, auth=None, timeout=None):  # noqa: ANN001
        if isinstance(data, (bytes, bytearray)):
            text = data.decode("utf-8")
        else:
            text = str(data or "")
        ctype = (headers or {}).get("Content-Type") or ""
        base_ct = ctype.split(";", 1)[0].strip().lower()
        if base_ct == "application/sparql-update":
            self._handle_update(text)
            return _Resp()
        if base_ct == "application/sparql-query":
            return _Resp(payload={"results": {"bindings": self._handle_query(text)}})
        return _Resp(status_code=400)

    def _extract_literal(self, text: str, predicate: str) -> str | None:
        long_match = re.search(rf"{re.escape(predicate)}\s+\"\"\"(.*?)\"\"\"", text, flags=re.DOTALL)
        if long_match:
            return long_match.group(1)
        match = re.search(rf"{re.escape(predicate)}\s+\"((?:\\.|[^\"])*)\"", text)
        return match.group(1) if match else None

    def _extract_node_id_values(self, text: str) -> list[str]:
        match = re.search(r"VALUES \?node_id \{([^}]*)\}", text)
        if not match:
            return []
        return re.findall(r'"([^\"]+)"', match.group(1))

    def _extract_focus_values(self, text: str) -> list[str]:
        match = re.search(r"VALUES \?focus_id \{([^}]*)\}", text)
        if not match:
            return []
        return re.findall(r'"([^\"]+)"', match.group(1))

    def _extract_limit(self, text: str, default: int = 32) -> int:
        match = re.search(r"LIMIT\s+(\d+)", text)
        return int(match.group(1)) if match else default

    def _handle_update(self, text: str) -> None:
        payload_json = self._extract_literal(text, "orion:payloadJson")
        if payload_json is not None:
            payload_json = payload_json.replace('\\"', '"').replace('\\n', '\n')
        node_id = self._extract_literal(text, "orion:nodeId")
        edge_id = self._extract_literal(text, "orion:edgeId")
        identity_key = self._extract_literal(text, "orion:identityKey")
        identity_kind = self._extract_literal(text, "orion:identityKind")
        canonical_id = self._extract_literal(text, "orion:canonicalId")

        if node_id and payload_json is not None:
            self.nodes[node_id] = payload_json
            return
        if edge_id and payload_json is not None:
            self.edges[edge_id] = payload_json
            return
        if identity_key and identity_kind and canonical_id:
            self.identity[(identity_kind, identity_key)] = canonical_id

    def _handle_query(self, text: str) -> list[dict[str, dict[str, str]]]:
        if "orion:canonicalId ?canonical_id" in text:
            identity_key = self._extract_literal(text, "orion:identityKey") or ""
            identity_kind = self._extract_literal(text, "orion:identityKind") or ""
            canonical = self.identity.get((identity_kind, identity_key))
            if not canonical:
                return []
            return [{"canonical_id": {"value": canonical}}]

        if "orion:SubstrateNode" in text and "?payload_json" in text:
            node_kind = self._extract_literal(text, "orion:nodeKind")
            evidence_ref = self._extract_literal(text, "orion:evidenceRef")
            ids = set(self._extract_node_id_values(text))
            limit = self._extract_limit(text)
            out: list[dict[str, dict[str, str]]] = []
            for node_id, payload in sorted(self.nodes.items()):
                node = __import__('json').loads(payload)
                if ids and node_id not in ids:
                    continue
                if node_kind and node.get('node_kind') != node_kind:
                    continue
                evidence_refs = (((node.get('provenance') or {}).get('evidence_refs')) or [])
                if evidence_ref and evidence_ref not in evidence_refs:
                    continue
                salience = (((node.get('signals') or {}).get('salience')) or 0.0)
                out.append(
                    {
                        "node_id": {"value": node_id},
                        "payload_json": {"value": payload},
                        "salience": {"value": str(salience)},
                    }
                )
            out.sort(key=lambda row: float(row["salience"]["value"]), reverse=True)
            return out[:limit]

        if "orion:SubstrateEdge" in text and "?payload_json" in text:
            focus = set(self._extract_focus_values(text))
            limit = self._extract_limit(text)
            out: list[dict[str, dict[str, str]]] = []
            for edge_id, payload in sorted(self.edges.items()):
                edge = __import__('json').loads(payload)
                source = ((edge.get('source') or {}).get('node_id'))
                target = ((edge.get('target') or {}).get('node_id'))
                if focus and source not in focus and target not in focus:
                    continue
                salience = edge.get('salience') or 0.0
                out.append(
                    {
                        "edge_id": {"value": edge_id},
                        "payload_json": {"value": payload},
                        "salience": {"value": str(salience)},
                    }
                )
            out.sort(key=lambda row: float(row["salience"]["value"]), reverse=True)
            return out[:limit]

        return []


def _sample_record() -> SubstrateGraphRecordV1:
    now = datetime.now(timezone.utc)
    temporal = SubstrateTemporalWindowV1(observed_at=now)
    provenance = SubstrateProvenanceV1(
        authority="local_inferred",
        source_kind="test",
        source_channel="pytest",
        producer="unit",
        evidence_refs=["ev:1", "ev:2"],
    )
    node_a = ConceptNodeV1(
        node_id="node-a",
        anchor_scope="orion",
        subject_ref="orion",
        temporal=temporal,
        provenance=provenance,
        signals=SubstrateSignalBundleV1(confidence=0.8, salience=0.7),
        label="Memory Substrate",
        definition="A substrate concept",
        metadata={"concept_id": "concept-memory"},
    )
    node_b = ConceptNodeV1(
        node_id="node-b",
        anchor_scope="orion",
        subject_ref="orion",
        temporal=temporal,
        provenance=provenance,
        signals=SubstrateSignalBundleV1(confidence=0.65, salience=0.62),
        label="Graph Persistence",
        definition="Persistence concept",
        metadata={"concept_id": "concept-persistence"},
    )
    contradiction = ContradictionNodeV1(
        node_id="node-c",
        anchor_scope="orion",
        subject_ref="orion",
        temporal=temporal,
        provenance=provenance,
        signals=SubstrateSignalBundleV1(confidence=0.8, salience=0.9),
        summary="conflict",
        involved_node_ids=["node-a", "node-b"],
    )
    edge = SubstrateEdgeV1(
        edge_id="edge-a-b",
        source=NodeRefV1(node_id="node-a", node_kind="concept"),
        target=NodeRefV1(node_id="node-b", node_kind="concept"),
        predicate="supports",
        temporal=temporal,
        confidence=0.9,
        salience=0.6,
        provenance=provenance,
    )
    return SubstrateGraphRecordV1(
        graph_id="graph-phase14",
        anchor_scope="orion",
        subject_ref="orion",
        nodes=[node_a, node_b, contradiction],
        edges=[edge],
    )


def test_graphdb_queries_are_primary_and_bounded(monkeypatch):
    fake = _FakeGraphDB()
    monkeypatch.setattr("orion.substrate.graphdb_store.requests.post", fake.post)

    store = GraphDBSubstrateStore(GraphDBSubstrateStoreConfig(endpoint="http://graphdb.local/repositories/collapse"))
    materializer = SubstrateGraphMaterializer(store=store)
    materializer.apply_record(_sample_record())

    hotspot = store.query_hotspot_region(min_salience=0.65, limit_nodes=2, limit_edges=1)
    assert hotspot.source_kind == "graphdb"
    assert len(hotspot.slice.nodes) <= 2
    assert len(hotspot.slice.edges) <= 1

    contradiction = store.query_contradiction_region(limit_nodes=2, limit_edges=2)
    assert contradiction.source_kind == "graphdb"
    assert any(node.node_kind == "contradiction" for node in contradiction.slice.nodes)

    concept = store.query_concept_region(limit_nodes=2, limit_edges=2)
    assert concept.source_kind == "graphdb"
    assert all(node.node_kind == "concept" for node in concept.slice.nodes)

    provenance = store.query_provenance_neighborhood(evidence_ref="ev:1", limit_nodes=3, limit_edges=2)
    assert provenance.source_kind == "graphdb"
    assert len(provenance.slice.nodes) >= 1


def test_graphdb_cold_start_reads_do_not_require_cache(monkeypatch):
    fake = _FakeGraphDB()
    monkeypatch.setattr("orion.substrate.graphdb_store.requests.post", fake.post)

    writer_store = GraphDBSubstrateStore(GraphDBSubstrateStoreConfig(endpoint="http://graphdb.local/repositories/collapse"))
    SubstrateGraphMaterializer(store=writer_store).apply_record(_sample_record())

    # New process/store instance -> empty cache, same GraphDB backend state
    reader_store = GraphDBSubstrateStore(GraphDBSubstrateStoreConfig(endpoint="http://graphdb.local/repositories/collapse"))
    result = reader_store.query_hotspot_region(min_salience=0.5, limit_nodes=5, limit_edges=5)

    assert result.source_kind == "graphdb"
    assert result.degraded is False
    assert len(result.slice.nodes) >= 1


def test_materializer_falls_back_to_in_memory_when_graphdb_not_configured(monkeypatch):
    monkeypatch.setenv("SUBSTRATE_STORE_BACKEND", "graphdb")
    monkeypatch.delenv("SUBSTRATE_GRAPHDB_ENDPOINT", raising=False)
    monkeypatch.delenv("GRAPHDB_URL", raising=False)
    monkeypatch.delenv("GRAPHDB_REPO", raising=False)

    store = build_substrate_store_from_env()
    assert isinstance(store, InMemorySubstrateGraphStore)


def test_build_substrate_store_unset_backend_with_graphdb_env_stays_in_memory(monkeypatch):
    """V1: GRAPHDB_URL alone must not select GraphDB substrate."""
    monkeypatch.delenv("SUBSTRATE_STORE_BACKEND", raising=False)
    monkeypatch.delenv("SUBSTRATE_GRAPHDB_ENDPOINT", raising=False)
    monkeypatch.setenv("GRAPHDB_URL", "http://graphdb.test")
    monkeypatch.setenv("GRAPHDB_REPO", "collapse")

    store = build_substrate_store_from_env()
    assert isinstance(store, InMemorySubstrateGraphStore)


def test_build_substrate_store_graphdb_backend_uses_graphdb_from_url_and_repo(monkeypatch):
    monkeypatch.setenv("SUBSTRATE_STORE_BACKEND", "graphdb")
    monkeypatch.delenv("SUBSTRATE_GRAPHDB_ENDPOINT", raising=False)
    monkeypatch.setenv("GRAPHDB_URL", "http://graphdb.test")
    monkeypatch.setenv("GRAPHDB_REPO", "collapse")

    store = build_substrate_store_from_env()
    assert isinstance(store, GraphDBSubstrateStore)


def test_build_substrate_store_sparql_backend(monkeypatch):
    fake = _FakeGraphDB()
    _patch_sparql_session_post(monkeypatch, fake.post)
    monkeypatch.setenv("SUBSTRATE_STORE_BACKEND", "sparql")
    monkeypatch.setenv("SUBSTRATE_GRAPH_QUERY_URL", "http://fuseki:3030/orion/query")
    monkeypatch.setenv("SUBSTRATE_GRAPH_UPDATE_URL", "http://fuseki:3030/orion/update")
    store = build_substrate_store_from_env()
    assert isinstance(store, SparqlSubstrateStore)
    materializer = SubstrateGraphMaterializer(store=store)
    materializer.apply_record(_sample_record())
    hotspot = store.query_hotspot_region(min_salience=0.65, limit_nodes=2, limit_edges=1)
    assert hotspot.source_kind == "sparql"


def test_build_substrate_store_sparql_backend_falls_back_to_rdf_store_urls(monkeypatch):
    fake = _FakeGraphDB()
    _patch_sparql_session_post(monkeypatch, fake.post)
    monkeypatch.setenv("SUBSTRATE_STORE_BACKEND", "sparql")
    monkeypatch.delenv("SUBSTRATE_GRAPH_QUERY_URL", raising=False)
    monkeypatch.delenv("SUBSTRATE_GRAPH_UPDATE_URL", raising=False)
    monkeypatch.setenv("RDF_STORE_QUERY_URL", "http://fuseki:3030/orion/query")
    monkeypatch.setenv("RDF_STORE_UPDATE_URL", "http://fuseki:3030/orion/update")
    store = build_substrate_store_from_env()
    assert isinstance(store, SparqlSubstrateStore)


def test_build_substrate_store_sparql_backend_unconfigured_raises(monkeypatch):
    monkeypatch.setenv("SUBSTRATE_STORE_BACKEND", "sparql")
    monkeypatch.delenv("SUBSTRATE_GRAPH_QUERY_URL", raising=False)
    monkeypatch.delenv("SUBSTRATE_GRAPH_UPDATE_URL", raising=False)
    monkeypatch.delenv("RDF_STORE_QUERY_URL", raising=False)
    monkeypatch.delenv("RDF_STORE_UPDATE_URL", raising=False)
    with pytest.raises(SubstrateSparqlBackendUnconfiguredError, match="substrate_sparql_backend_unconfigured"):
        build_substrate_store_from_env()


def test_sparql_substrate_posts_select_to_query_url_and_update_to_update_url(monkeypatch):
    calls: list[tuple[str, str]] = []

    def capture_post(url, data=None, headers=None, auth=None, timeout=None):  # noqa: ANN001
        ctype = (headers or {}).get("Content-Type") or ""
        base_ct = ctype.split(";", 1)[0].strip().lower()
        calls.append((str(url), base_ct))
        if base_ct == "application/sparql-query":
            return _Resp(payload={"results": {"bindings": []}})
        if base_ct == "application/sparql-update":
            return _Resp()
        return _Resp(status_code=400)

    _patch_sparql_session_post(monkeypatch, capture_post)
    monkeypatch.setenv("SUBSTRATE_STORE_BACKEND", "sparql")
    monkeypatch.setenv("SUBSTRATE_GRAPH_QUERY_URL", "http://fuseki:3030/orion/query")
    monkeypatch.setenv("SUBSTRATE_GRAPH_UPDATE_URL", "http://fuseki:3030/orion/update")
    store = build_substrate_store_from_env()
    assert isinstance(store, SparqlSubstrateStore)
    store.get_node_by_id("missing-node")
    store.upsert_node(identity_key=None, node=_sample_record().nodes[0])
    query_urls = [u for u, ct in calls if ct == "application/sparql-query"]
    update_urls = [u for u, ct in calls if ct == "application/sparql-update"]
    assert query_urls and all(u == "http://fuseki:3030/orion/query" for u in query_urls)
    assert update_urls and all(u == "http://fuseki:3030/orion/update" for u in update_urls)


def test_build_substrate_store_explicit_in_memory_overrides_graphdb_env(monkeypatch):
    monkeypatch.setenv("SUBSTRATE_STORE_BACKEND", "in_memory")
    monkeypatch.setenv("GRAPHDB_URL", "http://graphdb.test")
    monkeypatch.setenv("GRAPHDB_REPO", "collapse")

    store = build_substrate_store_from_env()
    assert isinstance(store, InMemorySubstrateGraphStore)


def test_resolve_substrate_sparql_auth_substrate_overrides_rdf(monkeypatch):
    monkeypatch.setenv("SUBSTRATE_GRAPH_USER", "su")
    monkeypatch.setenv("SUBSTRATE_GRAPH_PASS", "sp")
    monkeypatch.setenv("RDF_STORE_USER", "ru")
    monkeypatch.setenv("RDF_STORE_PASS", "rp")
    monkeypatch.setenv("FUSEKI_USER", "fu")
    monkeypatch.setenv("FUSEKI_PASS", "fp")
    u, p, src = resolve_substrate_sparql_http_basic_auth()
    assert (u, p, src) == ("su", "sp", "SUBSTRATE_GRAPH_USER")


def test_resolve_substrate_sparql_auth_falls_back_to_rdf_then_fuseki(monkeypatch):
    monkeypatch.delenv("SUBSTRATE_GRAPH_USER", raising=False)
    monkeypatch.delenv("SUBSTRATE_GRAPH_PASS", raising=False)
    monkeypatch.setenv("RDF_STORE_USER", "ru")
    monkeypatch.setenv("RDF_STORE_PASS", "rp")
    monkeypatch.setenv("FUSEKI_USER", "fu")
    monkeypatch.setenv("FUSEKI_PASS", "fp")
    assert resolve_substrate_sparql_http_basic_auth() == ("ru", "rp", "RDF_STORE_USER")

    monkeypatch.delenv("RDF_STORE_USER", raising=False)
    monkeypatch.delenv("RDF_STORE_PASS", raising=False)
    assert resolve_substrate_sparql_http_basic_auth() == ("fu", "fp", "FUSEKI_USER")

    monkeypatch.delenv("FUSEKI_USER", raising=False)
    monkeypatch.delenv("FUSEKI_PASS", raising=False)
    assert resolve_substrate_sparql_http_basic_auth() == (None, None, "none")


def test_sparql_substrate_update_posts_basic_auth_from_substrate_env(monkeypatch):
    auths: list[tuple | None] = []

    def capture_post(url, data=None, headers=None, auth=None, timeout=None):  # noqa: ANN001
        ctype = (headers or {}).get("Content-Type") or ""
        base_ct = ctype.split(";", 1)[0].strip().lower()
        auths.append(auth)
        if base_ct == "application/sparql-query":
            return _Resp(payload={"results": {"bindings": []}})
        if base_ct == "application/sparql-update":
            return _Resp()
        return _Resp(status_code=400)

    _patch_sparql_session_post(monkeypatch, capture_post)
    monkeypatch.setenv("SUBSTRATE_STORE_BACKEND", "sparql")
    monkeypatch.setenv("SUBSTRATE_GRAPH_QUERY_URL", "http://fuseki:3030/orion/query")
    monkeypatch.setenv("SUBSTRATE_GRAPH_UPDATE_URL", "http://fuseki:3030/orion/update")
    monkeypatch.setenv("SUBSTRATE_GRAPH_USER", "u1")
    monkeypatch.setenv("SUBSTRATE_GRAPH_PASS", "p1")
    monkeypatch.delenv("RDF_STORE_USER", raising=False)
    monkeypatch.delenv("RDF_STORE_PASS", raising=False)
    store = build_substrate_store_from_env()
    assert isinstance(store, SparqlSubstrateStore)
    store.upsert_node(identity_key=None, node=_sample_record().nodes[0])
    assert ("u1", "p1") in auths


def test_sparql_substrate_update_posts_basic_auth_from_rdf_store_fallback(monkeypatch):
    auths: list[tuple | None] = []

    def capture_post(url, data=None, headers=None, auth=None, timeout=None):  # noqa: ANN001
        ctype = (headers or {}).get("Content-Type") or ""
        base_ct = ctype.split(";", 1)[0].strip().lower()
        auths.append(auth)
        if base_ct == "application/sparql-query":
            return _Resp(payload={"results": {"bindings": []}})
        if base_ct == "application/sparql-update":
            return _Resp()
        return _Resp(status_code=400)

    _patch_sparql_session_post(monkeypatch, capture_post)
    monkeypatch.setenv("SUBSTRATE_STORE_BACKEND", "sparql")
    monkeypatch.setenv("SUBSTRATE_GRAPH_QUERY_URL", "http://fuseki:3030/orion/query")
    monkeypatch.setenv("SUBSTRATE_GRAPH_UPDATE_URL", "http://fuseki:3030/orion/update")
    monkeypatch.delenv("SUBSTRATE_GRAPH_USER", raising=False)
    monkeypatch.delenv("SUBSTRATE_GRAPH_PASS", raising=False)
    monkeypatch.setenv("RDF_STORE_USER", "r1")
    monkeypatch.setenv("RDF_STORE_PASS", "r2")
    store = build_substrate_store_from_env()
    assert isinstance(store, SparqlSubstrateStore)
    store.upsert_node(identity_key=None, node=_sample_record().nodes[0])
    assert ("r1", "r2") in auths


def test_sparql_substrate_substrate_creds_override_rdf_for_auth(monkeypatch):
    auths: list[tuple | None] = []

    def capture_post(url, data=None, headers=None, auth=None, timeout=None):  # noqa: ANN001
        ctype = (headers or {}).get("Content-Type") or ""
        base_ct = ctype.split(";", 1)[0].strip().lower()
        auths.append(auth)
        if base_ct == "application/sparql-query":
            return _Resp(payload={"results": {"bindings": []}})
        if base_ct == "application/sparql-update":
            return _Resp()
        return _Resp(status_code=400)

    _patch_sparql_session_post(monkeypatch, capture_post)
    monkeypatch.setenv("SUBSTRATE_STORE_BACKEND", "sparql")
    monkeypatch.setenv("SUBSTRATE_GRAPH_QUERY_URL", "http://fuseki:3030/orion/query")
    monkeypatch.setenv("SUBSTRATE_GRAPH_UPDATE_URL", "http://fuseki:3030/orion/update")
    monkeypatch.setenv("SUBSTRATE_GRAPH_USER", "su")
    monkeypatch.setenv("SUBSTRATE_GRAPH_PASS", "sp")
    monkeypatch.setenv("RDF_STORE_USER", "ru")
    monkeypatch.setenv("RDF_STORE_PASS", "rp")
    store = build_substrate_store_from_env()
    store.upsert_node(identity_key=None, node=_sample_record().nodes[0])
    assert ("su", "sp") in auths
    assert ("ru", "rp") not in auths


def test_sparql_substrate_no_auth_when_credentials_unset(monkeypatch):
    auths: list[tuple | None] = []

    def capture_post(url, data=None, headers=None, auth=None, timeout=None):  # noqa: ANN001
        auths.append(auth)
        ctype = (headers or {}).get("Content-Type") or ""
        base_ct = ctype.split(";", 1)[0].strip().lower()
        if base_ct == "application/sparql-query":
            return _Resp(payload={"results": {"bindings": []}})
        if base_ct == "application/sparql-update":
            return _Resp()
        return _Resp(status_code=400)

    _patch_sparql_session_post(monkeypatch, capture_post)
    monkeypatch.setenv("SUBSTRATE_STORE_BACKEND", "sparql")
    monkeypatch.setenv("SUBSTRATE_GRAPH_QUERY_URL", "http://fuseki:3030/orion/query")
    monkeypatch.setenv("SUBSTRATE_GRAPH_UPDATE_URL", "http://fuseki:3030/orion/update")
    monkeypatch.delenv("SUBSTRATE_GRAPH_USER", raising=False)
    monkeypatch.delenv("SUBSTRATE_GRAPH_PASS", raising=False)
    monkeypatch.delenv("RDF_STORE_USER", raising=False)
    monkeypatch.delenv("RDF_STORE_PASS", raising=False)
    monkeypatch.delenv("FUSEKI_USER", raising=False)
    monkeypatch.delenv("FUSEKI_PASS", raising=False)
    store = build_substrate_store_from_env()
    store.get_node_by_id("x")
    store.upsert_node(identity_key=None, node=_sample_record().nodes[0])
    assert all(a is None for a in auths)


def test_sparql_substrate_update_401_logs_auth_diagnostic_without_password(monkeypatch, caplog):
    import logging

    def session_post(self, url, data=None, headers=None, auth=None, timeout=None, **kwargs):  # noqa: ANN001
        ctype = (headers or {}).get("Content-Type") or ""
        base_ct = ctype.split(";", 1)[0].strip().lower()
        if base_ct == "application/sparql-query":
            return _Resp(payload={"results": {"bindings": []}})
        r = MagicMock()
        r.status_code = 401
        r.text = "Unauthorized body with no secrets"
        r.raise_for_status.side_effect = requests.HTTPError(response=r)
        return r

    monkeypatch.setattr("orion.graph.sparql_client.requests.Session.post", session_post)
    monkeypatch.setenv("SUBSTRATE_STORE_BACKEND", "sparql")
    monkeypatch.setenv("SUBSTRATE_GRAPH_QUERY_URL", "http://user:pass@fuseki:3030/orion/query")
    monkeypatch.setenv("SUBSTRATE_GRAPH_UPDATE_URL", "http://user:wrong@fuseki:3030/orion/update")
    monkeypatch.delenv("SUBSTRATE_GRAPH_USER", raising=False)
    monkeypatch.delenv("SUBSTRATE_GRAPH_PASS", raising=False)
    monkeypatch.delenv("RDF_STORE_USER", raising=False)
    monkeypatch.delenv("RDF_STORE_PASS", raising=False)
    monkeypatch.delenv("FUSEKI_USER", raising=False)
    monkeypatch.delenv("FUSEKI_PASS", raising=False)
    caplog.set_level(logging.WARNING, logger="orion.substrate.graphdb_store")
    store = build_substrate_store_from_env()
    assert isinstance(store, SparqlSubstrateStore)
    with pytest.raises(GraphDBSubstrateStoreError):
        store.upsert_node(identity_key=None, node=_sample_record().nodes[0])
    joined = " ".join(f"{r.message}" for r in caplog.records)
    assert "substrate_sparql_update_auth_failed" in joined
    assert "credential_source=none" in joined
    assert "user:pass" not in joined


def test_sparql_http_client_strips_credentials_from_redacted_update_url():
    c = SparqlHttpClient(
        "http://fuseki:3030/orion/query",
        "http://admin:admin@fuseki:3030/orion/update",
    )
    red = c.update_url_redacted
    assert "admin" not in red
    assert "fuseki:3030/orion/update" in red


# ===========================================================================
# snapshot() TTL cache -- 2026-07-14, see docs/superpowers/specs/2026-07-14-
# substrate-graph-store-snapshot-cache-spec.md. Root cause: snapshot() issued a
# full live query on every call with no freshness check, hit by a 5s-forever
# brain-frame tick (among ~5 other periodic/per-turn callers), driving Fuseki
# to 99.99% of its memory limit while the entire chat pipeline was idle.
# ===========================================================================


def _counting_post(fake: _FakeGraphDB) -> MagicMock:
    """MagicMock wrapping _FakeGraphDB.post, so tests can assert how many live
    queries actually reached the backend via the mock's own .call_count (2 per
    live snapshot(): one node query, one edge query -- confirmed via
    _query_nodes/_query_edges_for_node_ids each issuing exactly one
    self._select() call). .call_count is a plain writable int attribute, so
    tests can reset it (e.g. after materializer setup writes) with a direct
    assignment."""
    return MagicMock(side_effect=fake.post)


def test_snapshot_same_generation_reuses_cache_no_new_query(monkeypatch):
    """The baseline case: nothing written since the first fetch -- the second call is
    served from cache with zero new live queries, regardless of how the ceiling is
    configured (well within it here)."""
    fake = _FakeGraphDB()
    counting_post = _counting_post(fake)
    monkeypatch.setattr("orion.substrate.graphdb_store.requests.post", counting_post)

    store = GraphDBSubstrateStore(
        GraphDBSubstrateStoreConfig(
            endpoint="http://graphdb.local/repositories/collapse",
            snapshot_force_refresh_ceiling_sec=60.0,
        )
    )
    SubstrateGraphMaterializer(store=store).apply_record(_sample_record())
    counting_post.call_count = 0  # ignore the materializer's own writes

    first = store.snapshot()
    calls_after_first = counting_post.call_count
    assert calls_after_first == 2  # one node query, one edge query
    assert len(first.nodes) >= 1

    second = store.snapshot()
    assert counting_post.call_count == calls_after_first  # no new live query
    assert second.nodes.keys() == first.nodes.keys()


def test_snapshot_same_generation_still_skips_live_query_over_time(monkeypatch):
    """The actual point of this fix: elapsed time alone is never sufficient to force a
    live query -- if nothing this process wrote has changed, the cache is trusted
    regardless of how much time passed (up to the ceiling, covered separately below).
    This is the fix for the traced incident: a 5s-forever tick used to always requery
    on a blind timer; now it only requeries when something real changed, or the
    ceiling forces a periodic safety-net refresh."""
    import time

    fake = _FakeGraphDB()
    counting_post = _counting_post(fake)
    monkeypatch.setattr("orion.substrate.graphdb_store.requests.post", counting_post)

    store = GraphDBSubstrateStore(
        GraphDBSubstrateStoreConfig(
            endpoint="http://graphdb.local/repositories/collapse",
            snapshot_force_refresh_ceiling_sec=60.0,
        )
    )
    SubstrateGraphMaterializer(store=store).apply_record(_sample_record())
    counting_post.call_count = 0

    store.snapshot()
    assert counting_post.call_count == 2

    time.sleep(0.02)  # real elapsed time, but nothing was written -- same generation
    store.snapshot()
    assert counting_post.call_count == 2  # no new live query


def test_snapshot_write_between_calls_forces_new_query(monkeypatch):
    """The real change-detection path: a write bumps the generation counter, so the
    very next snapshot() call issues a live query even though it's well within the
    ceiling -- a real change is detected immediately, not after waiting for a timer.
    Uses a real identity_key (not None) for upsert_node, matching its documented
    signature -- a prior draft of this test passed identity_key=None, which happens
    to be tolerated by upsert_node's own internal guard but is not representative
    real usage and would have silently masked an unrelated identity-handling bug in
    upsert_edge if reused there (upsert_edge calls _upsert_identity unconditionally,
    with no such guard)."""
    fake = _FakeGraphDB()
    counting_post = _counting_post(fake)
    monkeypatch.setattr("orion.substrate.graphdb_store.requests.post", counting_post)

    store = GraphDBSubstrateStore(
        GraphDBSubstrateStoreConfig(
            endpoint="http://graphdb.local/repositories/collapse",
            snapshot_force_refresh_ceiling_sec=60.0,
        )
    )
    SubstrateGraphMaterializer(store=store).apply_record(_sample_record())
    counting_post.call_count = 0

    store.snapshot()
    assert counting_post.call_count == 2

    extra_node = ConceptNodeV1(
        node_id="node-extra",
        anchor_scope="orion",
        subject_ref="orion",
        temporal=SubstrateTemporalWindowV1(observed_at=datetime.now(timezone.utc)),
        provenance=SubstrateProvenanceV1(
            authority="local_inferred", source_kind="test", source_channel="pytest", producer="unit"
        ),
        signals=SubstrateSignalBundleV1(confidence=0.7, salience=0.5),
        label="Extra",
        definition="Added between snapshots",
        metadata={"concept_id": "concept-extra"},
    )
    store.upsert_node(identity_key="concept-extra", node=extra_node)  # bumps the write generation
    counting_post.call_count = 0

    result = store.snapshot()
    assert counting_post.call_count == 2  # new live query, not served from stale cache
    assert "node-extra" in result.nodes


def test_snapshot_write_generation_captured_before_live_query_not_after(monkeypatch):
    """The TOCTOU fix (2026-07-14 review finding): if a write races in while a live
    query is in flight, the generation recorded for that fetch must be the value from
    BEFORE the query started, not after it completes -- otherwise a concurrent write
    gets silently credited to data that was actually fetched before it happened,
    permanently masking the write until the next unrelated write or the ceiling fires.
    Simulated here by monkeypatching _query_nodes to perform a write mid-fetch."""
    fake = _FakeGraphDB()
    monkeypatch.setattr("orion.substrate.graphdb_store.requests.post", fake.post)

    store = GraphDBSubstrateStore(
        GraphDBSubstrateStoreConfig(
            endpoint="http://graphdb.local/repositories/collapse",
            snapshot_force_refresh_ceiling_sec=60.0,
        )
    )
    SubstrateGraphMaterializer(store=store).apply_record(_sample_record())
    store.snapshot()  # establish a baseline generation

    # Force the NEXT snapshot() call past the same-generation cache-reuse gate by
    # writing first -- otherwise the call below would be served from cache without
    # ever reaching the live fetch this test needs to race against, and the test
    # would pass without exercising anything (caught by running this test and
    # finding it green for the wrong reason before this setup write was added).
    setup_node = ConceptNodeV1(
        node_id="node-setup",
        anchor_scope="orion",
        subject_ref="orion",
        temporal=SubstrateTemporalWindowV1(observed_at=datetime.now(timezone.utc)),
        provenance=SubstrateProvenanceV1(
            authority="local_inferred", source_kind="test", source_channel="pytest", producer="unit"
        ),
        signals=SubstrateSignalBundleV1(confidence=0.7, salience=0.5),
        label="Setup",
        definition="Forces the next snapshot() past cache reuse",
        metadata={"concept_id": "concept-setup"},
    )
    store.upsert_node(identity_key="concept-setup", node=setup_node)
    generation_before_race = store._write_generation

    real_query_nodes = store._query_nodes
    racing_node = ConceptNodeV1(
        node_id="node-race",
        anchor_scope="orion",
        subject_ref="orion",
        temporal=SubstrateTemporalWindowV1(observed_at=datetime.now(timezone.utc)),
        provenance=SubstrateProvenanceV1(
            authority="local_inferred", source_kind="test", source_channel="pytest", producer="unit"
        ),
        signals=SubstrateSignalBundleV1(confidence=0.7, salience=0.5),
        label="Race",
        definition="Written mid-fetch",
        metadata={"concept_id": "concept-race"},
    )

    def _query_nodes_that_races_a_write(*args, **kwargs):  # noqa: ANN001
        # Simulates a write landing in another thread while this fetch is in flight,
        # i.e. after this fetch already decided what to return but before snapshot()
        # records the generation for it.
        store.upsert_node(identity_key="concept-race", node=racing_node)
        return real_query_nodes(*args, **kwargs)

    monkeypatch.setattr(store, "_query_nodes", _query_nodes_that_races_a_write)
    store.snapshot()

    # The generation recorded for this fetch must be the pre-race value, not the
    # post-race one -- proving the next snapshot() call will see a mismatch and
    # correctly re-fetch rather than trusting data that predates the race's write.
    assert store._last_snapshot_generation == generation_before_race
    assert store._last_snapshot_generation != store._write_generation


def test_snapshot_ceiling_forces_refresh_despite_same_generation(monkeypatch):
    """The safety net: even with no writes at all (same_generation stays true
    indefinitely), the ceiling forces a periodic live re-check to catch changes made
    by a different process this instance's write counter can't see."""
    import time

    fake = _FakeGraphDB()
    counting_post = _counting_post(fake)
    monkeypatch.setattr("orion.substrate.graphdb_store.requests.post", counting_post)

    store = GraphDBSubstrateStore(
        GraphDBSubstrateStoreConfig(
            endpoint="http://graphdb.local/repositories/collapse",
            snapshot_force_refresh_ceiling_sec=0.01,
        )
    )
    SubstrateGraphMaterializer(store=store).apply_record(_sample_record())
    counting_post.call_count = 0

    store.snapshot()
    assert counting_post.call_count == 2

    time.sleep(0.02)
    store.snapshot()
    assert counting_post.call_count == 4  # ceiling forced a live query


def test_snapshot_ceiling_zero_trusts_generation_forever(monkeypatch):
    """ceiling <= 0 disables the periodic safety-net refresh entirely -- the cache is
    trusted for as long as the write generation hasn't moved, with no time bound at
    all. Documented, deliberate behavior (see the config field's docstring), not an
    oversight."""
    import time

    fake = _FakeGraphDB()
    counting_post = _counting_post(fake)
    monkeypatch.setattr("orion.substrate.graphdb_store.requests.post", counting_post)

    store = GraphDBSubstrateStore(
        GraphDBSubstrateStoreConfig(
            endpoint="http://graphdb.local/repositories/collapse",
            snapshot_force_refresh_ceiling_sec=0.0,
        )
    )
    SubstrateGraphMaterializer(store=store).apply_record(_sample_record())
    counting_post.call_count = 0

    store.snapshot()
    assert counting_post.call_count == 2

    time.sleep(0.02)
    store.snapshot()
    assert counting_post.call_count == 2  # still cached, no ceiling to force a refresh


def test_snapshot_failure_fallback_still_returns_cache(monkeypatch):
    """Pre-existing behavior must be unchanged: a live-query failure still falls
    back to the last good in-memory cache. Forces a real write between the two
    snapshot() calls so the second call can't short-circuit via same-generation
    cache reuse (default ceiling is 30s, comfortably longer than this test takes) --
    without that write, this test would never actually reach the live-query-raises
    code path at all and would falsely appear to pass."""
    fake = _FakeGraphDB()
    monkeypatch.setattr("orion.substrate.graphdb_store.requests.post", fake.post)

    store = GraphDBSubstrateStore(
        GraphDBSubstrateStoreConfig(endpoint="http://graphdb.local/repositories/collapse")
    )
    SubstrateGraphMaterializer(store=store).apply_record(_sample_record())
    warm = store.snapshot()
    assert len(warm.nodes) >= 1

    extra_node = ConceptNodeV1(
        node_id="node-extra-2",
        anchor_scope="orion",
        subject_ref="orion",
        temporal=SubstrateTemporalWindowV1(observed_at=datetime.now(timezone.utc)),
        provenance=SubstrateProvenanceV1(
            authority="local_inferred", source_kind="test", source_channel="pytest", producer="unit"
        ),
        signals=SubstrateSignalBundleV1(confidence=0.7, salience=0.5),
        label="Extra2",
        definition="Forces same_generation=False for this test",
        metadata={"concept_id": "concept-extra-2"},
    )
    store.upsert_node(identity_key="concept-extra-2", node=extra_node)

    def _raise(*args, **kwargs):  # noqa: ANN001
        raise GraphDBSubstrateStoreError("simulated backend outage")

    monkeypatch.setattr(store, "_query_nodes", _raise)
    degraded = store.snapshot()
    assert degraded.nodes.keys() == warm.nodes.keys() | {"node-extra-2"}


def test_sparql_substrate_store_ceiling_defaults_to_30_seconds_from_config():
    cfg = SparqlSubstrateStoreConfig(query_url="http://fuseki:3030/orion/query", update_url="http://fuseki:3030/orion/update")
    assert cfg.snapshot_force_refresh_ceiling_sec == 30.0
    store = SparqlSubstrateStore(cfg)
    assert store._cfg.snapshot_force_refresh_ceiling_sec == 30.0


def test_build_substrate_store_sparql_backend_reads_snapshot_ceiling_env(monkeypatch):
    monkeypatch.setenv("SUBSTRATE_STORE_BACKEND", "sparql")
    monkeypatch.setenv("SUBSTRATE_GRAPH_QUERY_URL", "http://fuseki:3030/orion/query")
    monkeypatch.setenv("SUBSTRATE_GRAPH_UPDATE_URL", "http://fuseki:3030/orion/update")
    monkeypatch.setenv("SUBSTRATE_SNAPSHOT_FORCE_REFRESH_CEILING_SEC", "45.0")
    store = build_substrate_store_from_env()
    assert isinstance(store, SparqlSubstrateStore)
    assert store._cfg.snapshot_force_refresh_ceiling_sec == 45.0


def test_resolve_snapshot_force_refresh_ceiling_invalid_value_falls_back_to_default(monkeypatch, caplog):
    import logging

    monkeypatch.setenv("SUBSTRATE_STORE_BACKEND", "sparql")
    monkeypatch.setenv("SUBSTRATE_GRAPH_QUERY_URL", "http://fuseki:3030/orion/query")
    monkeypatch.setenv("SUBSTRATE_GRAPH_UPDATE_URL", "http://fuseki:3030/orion/update")
    monkeypatch.setenv("SUBSTRATE_SNAPSHOT_FORCE_REFRESH_CEILING_SEC", "not-a-number")
    caplog.set_level(logging.WARNING, logger="orion.substrate.graphdb_store")
    store = build_substrate_store_from_env()
    assert store._cfg.snapshot_force_refresh_ceiling_sec == 30.0
    assert any("substrate_snapshot_force_refresh_ceiling_invalid" in r.message for r in caplog.records)


@pytest.mark.parametrize("raw_value", ["nan", "NaN", "-nan", "inf", "-inf"])
def test_resolve_snapshot_force_refresh_ceiling_non_finite_falls_back_to_default(monkeypatch, caplog, raw_value):
    """float("nan")/float("inf") parse without raising ValueError, so the plain
    try/except ValueError guard alone doesn't catch them. A NaN ceiling makes every
    comparison in snapshot()'s within_ceiling check False, silently disabling the
    cache entirely (every call goes live) -- the exact regression this whole
    mechanism exists to prevent. 2026-07-14 review finding."""
    import logging

    monkeypatch.setenv("SUBSTRATE_STORE_BACKEND", "sparql")
    monkeypatch.setenv("SUBSTRATE_GRAPH_QUERY_URL", "http://fuseki:3030/orion/query")
    monkeypatch.setenv("SUBSTRATE_GRAPH_UPDATE_URL", "http://fuseki:3030/orion/update")
    monkeypatch.setenv("SUBSTRATE_SNAPSHOT_FORCE_REFRESH_CEILING_SEC", raw_value)
    caplog.set_level(logging.WARNING, logger="orion.substrate.graphdb_store")
    store = build_substrate_store_from_env()
    assert store._cfg.snapshot_force_refresh_ceiling_sec == 30.0
    assert any("substrate_snapshot_force_refresh_ceiling_invalid" in r.message for r in caplog.records)


def test_snapshot_zero_ceiling_logs_warning_distinguishing_from_removed_ttl_semantics(monkeypatch, caplog):
    """The 0 sentinel's meaning inverted between the removed snapshot_cache_ttl_sec
    (0 = always live) and this field (0 = trust cache forever) -- an operator reusing
    the old 'set to 0 to force live reads' habit gets the opposite of what they
    intend. A runtime log is the only signal available short of reading source, since
    the env var was fully renamed (not aliased). 2026-07-14 review finding."""
    import logging

    monkeypatch.setenv("SUBSTRATE_SNAPSHOT_FORCE_REFRESH_CEILING_SEC", "0")
    caplog.set_level(logging.INFO, logger="orion.substrate.graphdb_store")
    value = _resolve_snapshot_force_refresh_ceiling_sec()
    assert value == 0.0
    assert any("substrate_snapshot_force_refresh_ceiling_zero_or_negative" in r.message for r in caplog.records)


def test_write_generation_bumps_on_upsert_node_and_edge(monkeypatch):
    """The counter the whole real-change-detection mechanism relies on: every real
    write through upsert_node/upsert_edge must bump it, or same_generation would
    silently paper over a real change."""
    fake = _FakeGraphDB()
    monkeypatch.setattr("orion.substrate.graphdb_store.requests.post", fake.post)

    store = GraphDBSubstrateStore(GraphDBSubstrateStoreConfig(endpoint="http://graphdb.local/repositories/collapse"))
    assert store._write_generation == 0

    record = _sample_record()
    for node in record.nodes:
        # identity_key=None is legitimate for upsert_node (it internally guards
        # _upsert_identity on a truthy check) -- unlike upsert_edge below, which has
        # no such guard and requires a real key (see the identity-collision note on
        # test_snapshot_write_between_calls_forces_new_query).
        store.upsert_node(identity_key=None, node=node)
    assert store._write_generation >= len(record.nodes)  # at least one bump per node

    gen_before_edge = store._write_generation
    for i, edge in enumerate(record.edges):
        store.upsert_edge(identity_key=f"edge-identity-{i}", edge=edge)
    assert store._write_generation > gen_before_edge
