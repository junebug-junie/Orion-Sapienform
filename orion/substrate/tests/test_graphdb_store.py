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
    SubstrateSparqlBackendUnconfiguredError,
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
