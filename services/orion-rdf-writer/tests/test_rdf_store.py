from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("ORION_BUS_URL", "redis://example.test/0")
os.environ.setdefault("GRAPHDB_URL", "http://graphdb.example")
sys.path[:0] = [str(ROOT), str(SERVICE_ROOT)]

from app.rdf_store import (
    FusekiRdfStoreClient,
    GraphDbRdfStoreClient,
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
    assert normalize_graph_name(u) == u


def test_normalize_unknown_is_deterministic() -> None:
    assert normalize_graph_name("weird compact") == normalize_graph_name("weird compact")


def test_factory_default_graphdb_requires_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAPHDB_URL", "")
    monkeypatch.setenv("RDF_STORE_BACKEND", "graphdb")
    s = Settings()

    async def _call() -> None:
        async with httpx.AsyncClient() as client:
            build_rdf_store_client(s, client)

    with pytest.raises(ValueError):
        asyncio.run(_call())


def test_factory_fuseki_does_not_require_graphdb_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GRAPHDB_URL", raising=False)
    monkeypatch.setenv("RDF_STORE_BACKEND", "fuseki")
    monkeypatch.setenv("RDF_STORE_BASE_URL", "http://orion-athena-fuseki:3030")
    s = Settings()

    async def _call():
        async with httpx.AsyncClient() as client:
            return build_rdf_store_client(s, client)

    c = asyncio.run(_call())
    assert isinstance(c, FusekiRdfStoreClient)


def test_factory_unknown_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RDF_STORE_BACKEND", "nope")
    monkeypatch.setenv("GRAPHDB_URL", "http://g.example")
    s = Settings()

    async def _call() -> None:
        async with httpx.AsyncClient() as client:
            build_rdf_store_client(s, client)

    with pytest.raises(ValueError, match="Unknown RDF_STORE_BACKEND"):
        asyncio.run(_call())


def test_rdf4j_requires_explicit_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RDF_STORE_BACKEND", "rdf4j")
    monkeypatch.delenv("GRAPHDB_URL", raising=False)
    monkeypatch.delenv("RDF_STORE_GRAPH_STORE_URL", raising=False)
    monkeypatch.delenv("RDF_STORE_UPDATE_URL", raising=False)
    s = Settings()

    async def _call() -> None:
        async with httpx.AsyncClient() as client:
            build_rdf_store_client(s, client)

    with pytest.raises(ValueError, match="rdf4j"):
        asyncio.run(_call())


def test_fuseki_endpoint_defaults() -> None:
    s = Settings(
        RDF_STORE_BACKEND="fuseki",
        RDF_STORE_BASE_URL="http://orion-athena-fuseki:3030",
        RDF_STORE_DATASET="orion",
        ORION_BUS_URL="redis://example/0",
    )

    async def _health():
        async with httpx.AsyncClient() as client:
            c = FusekiRdfStoreClient(s, client)
            return await c.health()

    h = asyncio.run(_health())
    assert h["graph_store_url"].endswith("/orion/data")
    assert h["query_url"].endswith("/orion/query")


def test_graphdb_endpoint_shape() -> None:
    s = Settings(
        RDF_STORE_BACKEND="graphdb",
        GRAPHDB_URL="http://gdb:7200",
        GRAPHDB_REPO="collapse",
        ORION_BUS_URL="redis://example/0",
    )

    async def _health():
        async with httpx.AsyncClient() as client:
            c = GraphDbRdfStoreClient(s, client)
            return await c.health()

    assert asyncio.run(_health())["endpoint"].endswith("/repositories/collapse/statements")


def test_rdf_store_normalize_graphdb_context_setting_default_false() -> None:
    s = Settings(
        ORION_BUS_URL="redis://example/0",
        GRAPHDB_URL="http://gdb:7200",
    )
    assert s.RDF_STORE_NORMALIZE_GRAPHDB_CONTEXT is False


def test_graphdb_write_context_uses_raw_compact_when_normalize_false() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200)

    transport = httpx.MockTransport(handler)
    s = Settings(
        ORION_BUS_URL="redis://example/0",
        GRAPHDB_URL="http://gdb:7200",
        GRAPHDB_REPO="collapse",
        RDF_STORE_NORMALIZE_GRAPHDB_CONTEXT=False,
    )

    async def _run() -> None:
        async with httpx.AsyncClient(transport=transport) as client:
            c = GraphDbRdfStoreClient(s, client)
            await c.write_graph(".\n", "orion:chat")

    asyncio.run(_run())
    assert len(captured) == 1
    assert captured[0].url.params.get("context") == "<orion:chat>"


def test_graphdb_write_context_normalizes_when_flag_true() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200)

    transport = httpx.MockTransport(handler)
    s = Settings(
        ORION_BUS_URL="redis://example/0",
        GRAPHDB_URL="http://gdb:7200",
        GRAPHDB_REPO="collapse",
        RDF_STORE_NORMALIZE_GRAPHDB_CONTEXT=True,
    )

    async def _run() -> None:
        async with httpx.AsyncClient(transport=transport) as client:
            c = GraphDbRdfStoreClient(s, client)
            await c.write_graph(".\n", "orion:chat")

    asyncio.run(_run())
    assert len(captured) == 1
    assert captured[0].url.params.get("context") == "<http://conjourney.net/graph/orion/chat>"


def test_fuseki_write_graph_query_uses_normalized_uri() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200)

    transport = httpx.MockTransport(handler)
    s = Settings(
        ORION_BUS_URL="redis://example/0",
        RDF_STORE_BACKEND="fuseki",
        RDF_STORE_BASE_URL="http://orion-athena-fuseki:3030",
        RDF_STORE_DATASET="orion",
    )

    async def _run() -> None:
        async with httpx.AsyncClient(transport=transport) as client:
            c = FusekiRdfStoreClient(s, client)
            await c.write_graph(".\n", "orion:chat")

    asyncio.run(_run())
    assert len(captured) == 1
    assert captured[0].url.params.get("graph") == "http://conjourney.net/graph/orion/chat"


def test_limits_helper_smoke() -> None:
    s = Settings(ORION_BUS_URL="redis://example/0", GRAPHDB_URL="http://g/")
    lim = httpx_limits_for_settings(s)
    assert lim.max_connections == 64
