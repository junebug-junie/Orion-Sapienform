"""Read boundaries and GEXF interoperability contract for the workbench."""
from __future__ import annotations

from uuid import UUID
from xml.etree import ElementTree as ET

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch, tmp_path):
    from scripts import graph_workbench_routes as routes
    monkeypatch.setattr(routes, "ASSET_ROOT", tmp_path)
    (tmp_path / "index.html").write_text("<title>Gephi Lite</title>")
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app), routes


@pytest.mark.parametrize("path", [
    "/graph-workbench", "/gephi-lite/", "/api/graph-workbench/sources",
    "/api/graph-workbench/search/worldview", "/api/graph-workbench/export/worldview.gexf",
])
def test_workbench_uses_hubs_existing_network_boundary_without_a_second_login(client, path, monkeypatch):
    http, routes = client
    from scripts.graph_workbench import Snapshot
    monkeypatch.setattr(
        routes, "_with_graph",
        lambda _source, fn, *_args: [] if fn is routes.graph_search else Snapshot("worldview"),
    )
    response = http.get(path)
    assert response.status_code == 200
    assert "www-authenticate" not in response.headers
    assert routes.router.dependencies == []


def test_launcher_and_asset_security(client):
    http, _ = client
    page = http.get("/graph-workbench")
    assert page.status_code == 200
    assert '/static/js/graph-workbench.js' in page.text
    response = http.get("/gephi-lite/")
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    assert "connect-src 'self' blob:" in response.headers["Content-Security-Policy"]
    assert http.get("/gephi-lite/%2e%2e%2fapp/settings.py").status_code == 404


def test_gephi_fonts_stay_local_without_rewriting_the_manifest(client):
    http, routes = client
    (routes.ASSET_ROOT / "style.css").write_text('@import"https://fonts.googleapis.com/css2?family=Poppins:wght@200;300&display=swap";body{color:red}')
    (routes.ASSET_ROOT / "index.html").write_text('<link rel="manifest" href="./site.webmanifest">')
    assert http.get("/gephi-lite/style.css").text == "body{color:red}"
    assert 'rel="manifest" href="./site.webmanifest"' in http.get("/gephi-lite/").text
    assert "use-credentials" not in http.get("/gephi-lite/").text


@pytest.mark.parametrize("suffix", ["unknown.gexf", "worldview.gexf?depth=4", "worldview.gexf?limit=1001", "worldview.gexf?limit=0", "worldview.gexf?seed=1%20CREATE", "worldview.gexf?seed=99999999999999999999999"])
def test_invalid_source_and_bounds_rejected(client, suffix):
    http, _ = client
    assert http.get("/api/graph-workbench/export/" + suffix).status_code == 422


def test_arbitrary_properties_roundtrip_parallel_edges_and_limits():
    from scripts.graph_workbench import Snapshot, GEXF
    graph = Snapshot("fixture")
    graph.node("1", {"label": '<weird & "name">', "activation": .25, "enabled": True, "nested": {"source": "abc"}, "bad": "text\x00"}, 2)
    graph.node("2", {"label": "two"}, 2)
    assert not graph.node("3", {"label": "over cap"}, 2)
    graph.edge("r1", "1", "2", {"relation": "supports", "weight": 1.0})
    graph.edge("r2", "1", "2", {"relation": "refines", "weight": 2.0})
    graph.edge("r3", "1", "missing", {})
    root = ET.fromstring(graph.gexf())
    ns = {"g": GEXF}
    nodes = root.findall(".//g:node", ns)
    edges = root.findall(".//g:edge", ns)
    assert len(nodes) == len(edges) == 2
    assert nodes[0].get("label") == '<weird & "name">'
    props = {a.get("for"): a.get("value") for a in nodes[0].findall("g:attvalues/g:attvalue", ns)}
    assert props["enabled"] == "true" and props["bad"] == "text"
    assert '"source": "abc"' in props["nested"]
    assert graph.truncated


def test_substrate_expands_bounded_frontier_and_preserves_direction():
    from scripts.graph_workbench import graph_snapshot
    class Reader:
        def __init__(self): self.calls = []
        def query(self, query, params=None):
            self.calls.append((query, params))
            if "MATCH (n:Concept)" in query:
                return [{"id": 1, "labels": ["Concept"], "properties": {"label": "focus"}}]
            if "id(b) AS neighbor" in query:
                return [{"id": 7, "source": 2, "target": 1, "neighbor": 2, "labels": ["Evidence"], "properties": {}, "neighbor_properties": {"text": "proof"}, "relation": "supports"}]
            return [{"id": 7, "source": 2, "target": 1, "properties": {}, "relation": "supports"}]
    reader = Reader()
    result = graph_snapshot(reader, "substrate", "", 1, 2)
    assert result.seed == "1"
    assert result.edges["7"]["source"] == "2"
    assert result.edges["7"]["target"] == "1"
    assert reader.calls[1][1]["ids"] == [1]
    limited = graph_snapshot(Reader(), "substrate", "", 1, 1)
    assert list(limited.nodes) == ["1"] and not limited.edges and limited.truncated


def test_no_writable_fallback_or_unbounded_query():
    from scripts.graph_workbench import FalkorReader
    from redis.exceptions import ResponseError
    reader = FalkorReader.__new__(FalkorReader)
    class Redis:
        def execute_command(self, *args):
            assert args[0] == "GRAPH.RO_QUERY"
            assert args[-2:] == ("timeout", 2000)
            raise ResponseError("unknown command")
    from redis.commands.graph import Graph
    reader.redis = Redis()
    reader.graph = Graph(reader.redis, "test")
    with pytest.raises(ResponseError):
        reader.query("MATCH (n) RETURN id(n) LIMIT 1")


def test_export_errors_are_honest_and_do_not_leak_credentials(client, monkeypatch):
    http, routes = client
    monkeypatch.setattr(routes, "_with_graph", lambda *args: (_ for _ in ()).throw(RuntimeError("redis://secret:password@host")))
    response = http.get("/api/graph-workbench/export/worldview.gexf")
    assert response.status_code == 503
    assert "secret" not in response.text
    assert http.get("/api/graph-workbench/export/crystallizations.gexf").status_code == 503


def test_crystal_reads_are_bounded_readonly_and_hide_intimate():
    import asyncio
    from scripts.graph_workbench import crystal_snapshot
    cid = UUID("12345678-1234-1234-1234-123456789012")
    class Context:
        def __init__(self, value): self.value = value
        async def __aenter__(self): return self.value
        async def __aexit__(self, *args): pass
    class Conn:
        def __init__(self): self.queries = []
        def transaction(self, **kwargs):
            assert kwargs["readonly"] is True
            return Context(self)
        async def fetch(self, query, *params, **kwargs):
            self.queries.append(query)
            assert kwargs["timeout"] == 3
            if "SELECT * FROM memory_crystallizations" in query:
                assert "sensitivity" in query and "('public', 'private')" in query
                return [{"crystallization_id": cid, "subject": "A real memory", "status": "active", "projection_refs": '{"memory_card_ids": ["card-1"]}'}]
            if "SELECT DISTINCT c.*" in query:
                assert "sensitivity" in query
                return []
            if "FROM memory_crystallization_sources" in query:
                return [{"source_ref_id": cid, "crystallization_id": cid, "source_id": "turn:1", "source_kind": "chat", "strength": .8}]
            assert "LIMIT" in query
            return []
    conn = Conn()
    class Pool:
        def acquire(self): return Context(conn)
    result = asyncio.run(crystal_snapshot(Pool(), "crys_" + cid.hex, 1, 5, True))
    assert len(result.nodes) == 3 and len(result.edges) == 2
    assert result.edges["source:" + str(cid)]["properties"]["relation"] == "has_source"
    assert any(n.get("projection_kind") == "memory_card_ids" for n in result.nodes.values())
    assert not any("memory_crystallization_projection_refs" in q for q in conn.queries)
