from __future__ import annotations

import os
import sys
import asyncio
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("ORION_BUS_URL", "redis://example.test/0")
os.environ.setdefault("GRAPHDB_URL", "http://graphdb.example")
sys.path[:0] = [str(ROOT), str(SERVICE_ROOT)]

from app.rdf_store import (
    FusekiRdfStoreClient,
    GraphDbRdfStoreClient,
    GenericSparqlRdfStoreClient,
    build_rdf_store_client,
    normalize_graph_name,
    httpx_limits_for_settings,
)
from app.settings import Settings


def test_normalize_known_graphs() -> None:
    assert normalize_graph_name("orion:chat") == "http://conjourney.net/graph/orion/chat"
    assert normalize_graph_name("orion:chat:social") == "http://conjourney.net/graph/orion/chat/social"
    assert normalize_graph_name("orion:default") == "http://conjourney.net/graph/orion/default"


def test_normalize_absolute_unchanged() -> None:
    u = "http://example.com/g#x"
    assert normalize_graph_name(u) is u


def test_normalize_unknown_is_deterministic() -> None:
    assert normalize_graph_name("weird compact") == normalize_graph_name("weird compact")


def test_factory_default_graphdb_requires_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAPHDB_URL", "")
    monkeypatch.setenv("RDF_STORE_BACKEND", "graphdb")
    s = Settings()
    import httpx

    with pytest.raises(ValueError):
        build_rdf_store_client(s, httpx.AsyncClient())


def test_factory_fuseki_does_not_require_graphdb_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GRAPHDB_URL", raising=False)
    monkeypatch.setenv("RDF_STORE_BACKEND", "fuseki")
    monkeypatch.setenv("RDF_STORE_BASE_URL", "http://orion-athena-fuseki:3030")
    s = Settings()
    import httpx

    c = build_rdf_store_client(s, httpx.AsyncClient())
    assert isinstance(c, FusekiRdfStoreClient)


def test_factory_unknown_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RDF_STORE_BACKEND", "nope")
    monkeypatch.setenv("GRAPHDB_URL", "http://g.example")
    s = Settings()
    import httpx

    with pytest.raises(ValueError, match="Unknown RDF_STORE_BACKEND"):
        build_rdf_store_client(s, httpx.AsyncClient())


def test_rdf4j_requires_explicit_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RDF_STORE_BACKEND", "rdf4j")
    monkeypatch.delenv("GRAPHDB_URL", raising=False)
    monkeypatch.delenv("RDF_STORE_GRAPH_STORE_URL", raising=False)
    monkeypatch.delenv("RDF_STORE_UPDATE_URL", raising=False)
    s = Settings()
    import httpx

    with pytest.raises(ValueError, match="rdf4j"):
        build_rdf_store_client(s, httpx.AsyncClient())


def test_fuseki_endpoint_defaults() -> None:
    s = Settings(
        RDF_STORE_BACKEND="fuseki",
        RDF_STORE_BASE_URL="http://orion-athena-fuseki:3030",
        RDF_STORE_DATASET="orion",
        ORION_BUS_URL="redis://example/0",
    )
    import httpx

    c = FusekiRdfStoreClient(s, httpx.AsyncClient())
    h = asyncio.run(c.health())
    assert h["graph_store_url"].endswith("/orion/data")
    assert h["query_url"].endswith("/orion/query")


def test_graphdb_endpoint_shape() -> None:
    s = Settings(
        RDF_STORE_BACKEND="graphdb",
        GRAPHDB_URL="http://gdb:7200",
        GRAPHDB_REPO="collapse",
        ORION_BUS_URL="redis://example/0",
    )
    import httpx

    c = GraphDbRdfStoreClient(s, httpx.AsyncClient())
    assert asyncio.run(c.health())["endpoint"].endswith("/repositories/collapse/statements")


def test_limits_helper_smoke() -> None:
    s = Settings(ORION_BUS_URL="redis://example/0", GRAPHDB_URL="http://g/")
    lim = httpx_limits_for_settings(s)
    assert lim.max_connections == 64
