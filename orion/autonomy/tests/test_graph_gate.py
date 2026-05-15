from __future__ import annotations

import pytest

from orion.autonomy.graph_gate import (
    autonomy_graph_reads_explicitly_enabled,
    resolve_autonomy_graph_read_plan,
)


def test_unset_backend_with_graphdb_url_does_not_select_graphdb(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AUTONOMY_GRAPH_BACKEND", raising=False)
    monkeypatch.setenv("GRAPHDB_URL", "http://graphdb.example:7200")
    monkeypatch.setenv("GRAPHDB_REPO", "collapse")
    for k in ("RDF_STORE_QUERY_URL", "AUTONOMY_GRAPH_QUERY_URL", "RDF_STORE_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    plan = resolve_autonomy_graph_read_plan({"verb": "chat_quick", "mode": "brain"})
    assert plan.mode == "sparql_degraded"
    assert plan.endpoint is None
    assert autonomy_graph_reads_explicitly_enabled() is False


def test_rdf_store_query_url_enables_sparql(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AUTONOMY_GRAPH_BACKEND", raising=False)
    monkeypatch.setenv("RDF_STORE_QUERY_URL", "http://orion-athena-fuseki:3030/orion/query")
    monkeypatch.delenv("GRAPHDB_URL", raising=False)
    plan = resolve_autonomy_graph_read_plan({"verb": "chat_general", "mode": "brain"})
    assert plan.mode == "sparql"
    assert plan.endpoint == "http://orion-athena-fuseki:3030/orion/query"
    assert plan.sparql_resolution_source == "RDF_STORE_QUERY_URL"


def test_autonomy_graph_query_url_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTONOMY_GRAPH_BACKEND", "auto")
    monkeypatch.setenv("AUTONOMY_GRAPH_QUERY_URL", "http://a/orion/query")
    monkeypatch.setenv("RDF_STORE_QUERY_URL", "http://b/orion/query")
    plan = resolve_autonomy_graph_read_plan({})
    assert plan.endpoint == "http://a/orion/query"
    assert plan.sparql_resolution_source == "AUTONOMY_GRAPH_QUERY_URL"


def test_explicit_graphdb_enables_with_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTONOMY_GRAPH_BACKEND", "graphdb")
    monkeypatch.setenv("GRAPHDB_URL", "http://gdb:7200")
    monkeypatch.setenv("GRAPHDB_REPO", "collapse")
    plan = resolve_autonomy_graph_read_plan({"verb": "chat_general", "mode": "brain"})
    assert plan.mode == "graphdb"
    assert plan.endpoint == "http://gdb:7200/repositories/collapse"
    assert plan.subjects == ("orion", "relationship", "juniper")
    assert autonomy_graph_reads_explicitly_enabled() is True


def test_explicit_graphdb_without_endpoint_is_degraded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTONOMY_GRAPH_BACKEND", "graphdb")
    for key in (
        "GRAPHDB_QUERY_ENDPOINT",
        "GRAPHDB_URL",
        "CONCEPT_PROFILE_GRAPHDB_ENDPOINT",
        "CONCEPT_PROFILE_GRAPHDB_URL",
    ):
        monkeypatch.delenv(key, raising=False)
    plan = resolve_autonomy_graph_read_plan({"verb": "chat_quick", "mode": "brain"})
    assert plan.mode == "graphdb_degraded"
    assert plan.endpoint is None


def test_chat_quick_bounded_defaults_sparql(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AUTONOMY_GRAPH_BACKEND", raising=False)
    monkeypatch.setenv("RDF_STORE_QUERY_URL", "http://fuseki:3030/orion/query")
    monkeypatch.delenv("AUTONOMY_QUICK_GRAPH_SUBJECTS", raising=False)
    monkeypatch.delenv("AUTONOMY_QUICK_GRAPH_SUBQUERIES", raising=False)
    monkeypatch.delenv("AUTONOMY_QUICK_GRAPH_TIMEOUT_SEC", raising=False)
    monkeypatch.setenv("AUTONOMY_GRAPH_TIMEOUT_SEC", "20.0")
    plan = resolve_autonomy_graph_read_plan({"verb": "chat_quick", "mode": "brain", "options": {}})
    assert plan.mode == "sparql"
    assert plan.subjects == ("orion",)
    assert plan.active_subqueries == ("identity",)
    assert plan.timeout_sec == 3.0


def test_chat_quick_respects_quick_timeout_not_deep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AUTONOMY_GRAPH_BACKEND", raising=False)
    monkeypatch.setenv("RDF_STORE_QUERY_URL", "http://fuseki:3030/orion/query")
    monkeypatch.setenv("AUTONOMY_GRAPH_TIMEOUT_SEC", "20.0")
    monkeypatch.setenv("AUTONOMY_QUICK_GRAPH_TIMEOUT_SEC", "2.5")
    plan = resolve_autonomy_graph_read_plan({"verb": "chat_quick", "mode": "brain", "options": {}})
    assert plan.timeout_sec == 2.5


def test_disabled_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTONOMY_GRAPH_BACKEND", "disabled")
    monkeypatch.setenv("RDF_STORE_QUERY_URL", "http://fuseki:3030/orion/query")
    plan = resolve_autonomy_graph_read_plan({"verb": "chat_quick", "mode": "brain"})
    assert plan.mode == "disabled"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("juniper,orion", ("juniper", "orion")),
        ("relationship,juniper,orion", ("relationship", "juniper", "orion")),
    ],
)
def test_quick_subjects_csv_order(monkeypatch: pytest.MonkeyPatch, raw: str, expected: tuple[str, ...]) -> None:
    monkeypatch.delenv("AUTONOMY_GRAPH_BACKEND", raising=False)
    monkeypatch.setenv("RDF_STORE_QUERY_URL", "http://fuseki:3030/orion/query")
    monkeypatch.setenv("AUTONOMY_QUICK_GRAPH_SUBJECTS", raw)
    plan = resolve_autonomy_graph_read_plan({"verb": "chat_quick", "mode": "brain", "options": {}})
    assert plan.subjects == expected
