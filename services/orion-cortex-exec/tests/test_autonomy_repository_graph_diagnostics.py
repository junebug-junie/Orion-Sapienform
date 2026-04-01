from __future__ import annotations

import logging

from orion.autonomy.repository import GraphAutonomyRepository, SUBJECT_BINDINGS
from orion.spark.concept_induction.graph_query import GraphQueryError


class _EmptyQueryClient:
    def select(self, sparql: str):
        return []


class _FailingQueryClient:
    def select(self, sparql: str):
        raise GraphQueryError("boom")


def test_subject_bindings_match_rdf_writer_entity_keys() -> None:
    assert SUBJECT_BINDINGS["orion"].entity_id == "self:orion"
    assert SUBJECT_BINDINGS["juniper"].entity_id == "user:juniper"
    assert SUBJECT_BINDINGS["relationship"].entity_id == "relationship:orion|juniper"


def test_graph_lookup_empty_logs_no_rows(caplog) -> None:
    repo = GraphAutonomyRepository(
        endpoint="http://graphdb.local/repositories/collapse",
        timeout_sec=2.0,
        query_client=_EmptyQueryClient(),
    )
    caplog.set_level(logging.INFO)

    result = repo.get_latest("orion")

    assert result.availability == "empty"
    assert "autonomy_graph_lookup subject=orion" in caplog.text
    assert "query_ok=true" in caplog.text
    assert "availability=empty" in caplog.text
    assert "empty_reason=no_rows" in caplog.text


def test_graph_lookup_unavailable_logs_query_error(caplog) -> None:
    repo = GraphAutonomyRepository(
        endpoint="http://graphdb.local/repositories/collapse",
        timeout_sec=2.0,
        query_client=_FailingQueryClient(),
    )
    caplog.set_level(logging.INFO)

    result = repo.get_latest("orion")

    assert result.availability == "unavailable"
    assert result.unavailable_reason == "query_error"
    assert "autonomy_graph_lookup subject=orion" in caplog.text
    assert "query_ok=false" in caplog.text
    assert "availability=unavailable" in caplog.text
    assert "unavailable_reason=query_error" in caplog.text
