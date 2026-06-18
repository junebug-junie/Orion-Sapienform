from __future__ import annotations

import pytest

from orion.graph.backend_config import (
    resolve_autonomy_read_query_url,
    resolve_graph_backend,
)


def test_graph_backend_fuseki_derives_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAPH_BACKEND", "fuseki")
    monkeypatch.setenv("RDF_STORE_BASE_URL", "http://orion-athena-fuseki:3030")
    monkeypatch.setenv("RDF_STORE_DATASET", "orion")
    cfg = resolve_graph_backend()
    assert cfg.backend == "fuseki"
    assert cfg.query_url == "http://orion-athena-fuseki:3030/orion/query"
    assert cfg.update_url == "http://orion-athena-fuseki:3030/orion/update"
    assert cfg.graph_store_url == "http://orion-athena-fuseki:3030/orion/data"


def test_auto_uses_rdf_store_query_url_not_graphdb(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GRAPH_BACKEND", raising=False)
    monkeypatch.setenv("RDF_STORE_QUERY_URL", "http://fuseki:3030/orion/query")
    monkeypatch.setenv("GRAPHDB_URL", "http://graphdb:7200")
    cfg = resolve_graph_backend()
    assert cfg.backend == "sparql"
    assert cfg.legacy_graphdb is False
    assert cfg.query_url == "http://fuseki:3030/orion/query"


def test_only_graphdb_url_auto_does_not_select_graphdb(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GRAPH_BACKEND", raising=False)
    monkeypatch.delenv("RDF_STORE_QUERY_URL", raising=False)
    monkeypatch.setenv("GRAPHDB_URL", "http://graphdb:7200")
    cfg = resolve_graph_backend()
    assert cfg.backend == "disabled"


def test_auto_rdf_store_fuseki_derives_urls_without_explicit_query(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GRAPH_BACKEND", raising=False)
    monkeypatch.delenv("RDF_STORE_QUERY_URL", raising=False)
    monkeypatch.setenv("RDF_STORE_BACKEND", "fuseki")
    monkeypatch.setenv("RDF_STORE_BASE_URL", "http://orion-athena-fuseki:3030")
    monkeypatch.setenv("RDF_STORE_DATASET", "orion")
    monkeypatch.delenv("GRAPHDB_URL", raising=False)
    cfg = resolve_graph_backend()
    assert cfg.backend == "fuseki"
    assert cfg.query_url == "http://orion-athena-fuseki:3030/orion/query"
    assert cfg.legacy_graphdb is False


def test_explicit_graphdb_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAPH_BACKEND", "graphdb")
    monkeypatch.setenv("GRAPHDB_URL", "http://gdb:7200")
    monkeypatch.setenv("GRAPHDB_REPO", "collapse")
    cfg = resolve_graph_backend()
    assert cfg.backend == "graphdb"
    assert cfg.legacy_graphdb is True
    assert cfg.query_url and "repositories" in cfg.query_url


def test_disabled_graph_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAPH_BACKEND", "disabled")
    cfg = resolve_graph_backend()
    assert cfg.backend == "disabled"
    assert cfg.query_url is None


def test_autonomy_read_prefers_autonomy_query_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTONOMY_GRAPH_QUERY_URL", "http://a/orion/query")
    monkeypatch.setenv("RDF_STORE_QUERY_URL", "http://b/orion/query")
    url, src = resolve_autonomy_read_query_url()
    assert url == "http://a/orion/query"
    assert src == "AUTONOMY_GRAPH_QUERY_URL"


def test_autonomy_update_prefers_autonomy_update_url(monkeypatch: pytest.MonkeyPatch) -> None:
    from orion.graph.backend_config import resolve_autonomy_graph_update_url

    monkeypatch.setenv("AUTONOMY_GRAPH_UPDATE_URL", "http://a/orion/update")
    monkeypatch.setenv("RDF_STORE_UPDATE_URL", "http://b/orion/update")
    url, src = resolve_autonomy_graph_update_url()
    assert url == "http://a/orion/update"
    assert src == "AUTONOMY_GRAPH_UPDATE_URL"
