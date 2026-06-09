from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.storage import rdf_adapter


def test_graph_iri_for_sparql_fuseki_uses_normalized_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAPH_BACKEND", "fuseki")
    assert (
        rdf_adapter.graph_iri_for_sparql("orion:chat")
        == "http://conjourney.net/graph/orion/chat"
    )
    assert (
        rdf_adapter.graph_iri_for_sparql("orion:enrichment")
        == "http://conjourney.net/graph/orion/enrichment"
    )


def test_graph_iri_for_sparql_graphdb_keeps_compact_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAPH_BACKEND", "graphdb")
    assert rdf_adapter.graph_iri_for_sparql("orion:chat") == "orion:chat"


def test_graph_iri_for_sparql_graphdb_can_normalize_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GRAPH_BACKEND", "graphdb")
    monkeypatch.setenv("RDF_STORE_NORMALIZE_GRAPHDB_CONTEXT", "true")
    assert (
        rdf_adapter.graph_iri_for_sparql("orion:chat")
        == "http://conjourney.net/graph/orion/chat"
    )


def test_fetch_rdf_chatturn_fragments_queries_normalized_chat_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GRAPH_BACKEND", "fuseki")
    monkeypatch.setattr(
        rdf_adapter.settings,
        "RECALL_RDF_ENDPOINT_URL",
        "http://orion-athena-fuseki:3030/orion/query",
    )

    captured: dict[str, str] = {}

    def _fake_post(url, data, **kwargs):
        captured["sparql"] = data
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"results": {"bindings": []}}
        return response

    monkeypatch.setattr(rdf_adapter.requests, "post", _fake_post)

    rdf_adapter.fetch_rdf_chatturn_fragments(
        query_text="Ogden Utah",
        session_id=None,
        max_items=3,
    )

    sparql = captured["sparql"]
    assert "GRAPH <http://conjourney.net/graph/orion/chat>" in sparql
    assert "GRAPH <orion:chat>" not in sparql
