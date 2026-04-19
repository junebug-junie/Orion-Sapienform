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


class _TimeoutOnDrivesQueryClient:
    def select(self, sparql: str):
        if "GRAPH <http://conjourney.net/graph/autonomy/drives>" in sparql:
            raise GraphQueryError("Read timed out. (read timeout=4.5)", error_type="timeout")
        if "GRAPH <http://conjourney.net/graph/autonomy/identity>" in sparql:
            return [
                {
                    "artifact_id": {"value": "id-orion-1"},
                    "summary": {"value": "holds course"},
                    "anchor_strategy": {"value": "steady"},
                    "created_at": {"value": "2026-03-31T12:00:00Z"},
                }
            ]
        return []


class _ConnectionFailureQueryClient:
    def select(self, sparql: str):
        del sparql
        raise GraphQueryError("Connection refused", error_type="connection_error")


class _DrivesOnlyFailingQueryClient:
    def select(self, sparql: str):
        if "GRAPH <http://conjourney.net/graph/autonomy/drives>" in sparql:
            raise GraphQueryError("drive_lookup_failed")
        return []


class _LegacyDriveLabelShapeClient:
    def select(self, sparql: str):
        if "?active_drive_uri <http://www.w3.org/2000/01/rdf-schema#label> ?active_drive" in sparql:
            raise GraphQueryError("bad active drive URI shape")
        if "GRAPH <http://conjourney.net/graph/autonomy/identity>" in sparql:
            return [
                {
                    "artifact_id": {"value": "id-orion-1"},
                    "summary": {"value": "holds course"},
                    "anchor_strategy": {"value": "steady"},
                    "created_at": {"value": "2026-03-31T12:00:00Z"},
                }
            ]
        if "GRAPH <http://conjourney.net/graph/autonomy/drives>" in sparql:
            return [
                {
                    "artifact_id": {"value": "drive-orion-1"},
                    "created_at": {"value": "2026-03-31T12:00:01Z"},
                    "active_drive": {"value": "relational"},
                    "drive_name": {"value": "relational"},
                    "drive_pressure": {"value": "0.91"},
                }
            ]
        return []


class _PressureOnlyDriveAuditClient:
    def select(self, sparql: str):
        if "GRAPH <http://conjourney.net/graph/autonomy/identity>" in sparql:
            return []
        if "GRAPH <http://conjourney.net/graph/autonomy/drives>" in sparql:
            return [
                {
                    "artifact_id": {"value": "drive-orion-2"},
                    "created_at": {"value": "2026-03-31T12:00:01Z"},
                    "drive_name": {"value": "predictive"},
                    "drive_pressure": {"value": "0.72"},
                },
                {
                    "artifact_id": {"value": "drive-orion-2"},
                    "created_at": {"value": "2026-03-31T12:00:01Z"},
                    "drive_name": {"value": "continuity"},
                    "drive_pressure": {"value": "0.51"},
                },
            ]
        return []


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


def test_graph_lookup_logs_failed_subquery_name_and_reason(caplog) -> None:
    repo = GraphAutonomyRepository(
        endpoint="http://graphdb.local/repositories/collapse",
        timeout_sec=2.0,
        query_client=_DrivesOnlyFailingQueryClient(),
    )
    caplog.set_level(logging.INFO)

    result = repo.get_latest("orion")

    assert result.availability == "unavailable"
    assert result.unavailable_reason == "query_error"
    assert "subquery=identity status=empty rows=0" in caplog.text
    assert "subquery=drives status=query_error" in caplog.text
    assert "reason=drive_lookup_failed" in caplog.text
    assert "failed_subquery=drives" in caplog.text


def test_graph_lookup_retains_partial_data_on_timeout(caplog) -> None:
    repo = GraphAutonomyRepository(
        endpoint="http://graphdb.local/repositories/collapse",
        timeout_sec=2.0,
        query_client=_TimeoutOnDrivesQueryClient(),
    )
    caplog.set_level(logging.INFO)

    result = repo.get_latest("orion", observer={"correlation_id": "corr-1"})

    assert result.availability == "available"
    assert result.unavailable_reason == "timeout"
    assert result.state is not None
    assert result.state.identity_summary == "holds course"
    assert "subquery=drives status=timeout" in caplog.text
    assert "correlation_id=corr-1" in caplog.text
    assert (result.subquery_diagnostics or {}).get("drives", {}).get("error_type") == "timeout"


def test_graph_lookup_connection_failure_classified() -> None:
    repo = GraphAutonomyRepository(
        endpoint="http://graphdb.local/repositories/collapse",
        timeout_sec=2.0,
        query_client=_ConnectionFailureQueryClient(),
    )
    result = repo.get_latest("orion")
    assert result.availability == "unavailable"
    assert result.unavailable_reason == "connection_error"


def test_graph_lookup_handles_legacy_active_drive_shape_without_query_error() -> None:
    repo = GraphAutonomyRepository(
        endpoint="http://graphdb.local/repositories/collapse",
        timeout_sec=2.0,
        query_client=_LegacyDriveLabelShapeClient(),
    )

    result = repo.get_latest("orion")

    assert result.availability == "available"
    assert result.state is not None
    assert result.state.dominant_drive == "relational"


def test_graph_lookup_derives_dominant_drive_from_drive_pressures() -> None:
    repo = GraphAutonomyRepository(
        endpoint="http://graphdb.local/repositories/collapse",
        timeout_sec=2.0,
        query_client=_PressureOnlyDriveAuditClient(),
    )

    result = repo.get_latest("orion")

    assert result.availability == "available"
    assert result.state is not None
    assert result.state.dominant_drive == "predictive"
