from __future__ import annotations

from datetime import datetime, timezone

from orion.core.schemas.concept_induction import ConceptProfile
from orion.spark.concept_induction.graph_mapper import (
    build_concept_profile,
    map_latest_profile_rows,
    map_profile_details_rows,
)
from orion.spark.concept_induction.graph_query import build_latest_profile_query
from orion.spark.concept_induction.profile_repository import (
    GraphConceptProfileRepository,
    LocalConceptProfileRepository,
    ShadowConceptProfileRepository,
    build_concept_profile_repository,
)


def _binding(value: str) -> dict[str, str]:
    return {"type": "literal", "value": value}


def _uri(value: str) -> dict[str, str]:
    return {"type": "uri", "value": value}


def _summary_rows() -> list[dict[str, dict[str, str]]]:
    return [
        {
            "profile": _uri("http://conjourney.net/orion/spark/concept-profile/orion/profile-7"),
            "subject": _binding("orion"),
            "profile_id": _binding("profile-7"),
            "revision": _binding("7"),
            "created_at": _binding("2026-03-27T10:00:00+00:00"),
            "window_start": _binding("2026-03-26T10:00:00+00:00"),
            "window_end": _binding("2026-03-27T10:00:00+00:00"),
            "profile_metadata_json": _binding('{"algorithm":"concept_induction.v1"}'),
        }
    ]


def _detail_rows() -> list[dict[str, dict[str, str]]]:
    base = {
        "profile": _uri("http://conjourney.net/orion/spark/concept-profile/orion/profile-7"),
        "concept": _uri("http://conjourney.net/orion/spark/concept-profile/orion/profile-7/concept/concept-1"),
        "concept_id": _binding("concept-1"),
        "concept_label": _binding("coherence"),
        "concept_type": _binding("motif"),
        "concept_salience": _binding("0.9"),
        "concept_confidence": _binding("0.8"),
        "concept_alias": _binding("consistency"),
        "concept_metadata_json": _binding('{"source":"graph"}'),
        "cluster": _uri("http://conjourney.net/orion/spark/concept-profile/orion/profile-7/cluster/cluster-1"),
        "cluster_id": _binding("cluster-1"),
        "cluster_label": _binding("core"),
        "cluster_summary": _binding("Core concepts"),
        "cluster_cohesion_score": _binding("0.75"),
        "cluster_concept": _binding("concept-1"),
        "cluster_metadata_json": _binding('{"size":"1"}'),
        "state": _uri("http://conjourney.net/orion/spark/concept-profile/orion/profile-7/state-estimate"),
        "state_confidence": _binding("0.6"),
        "state_window_start": _binding("2026-03-26T10:00:00+00:00"),
        "state_window_end": _binding("2026-03-27T10:00:00+00:00"),
        "state_dimensions_json": _binding('{"novelty":0.2}'),
        "state_trend_json": _binding('{"novelty":-0.1}'),
        "provenance": _uri("http://conjourney.net/orion/spark/concept-profile/orion/profile-7/provenance"),
        "writer_service": _binding("orion-spark-concept-induction"),
        "writer_version": _binding("0.1.0"),
        "correlation_id": _binding("corr-7"),
    }
    return [base]


class FakeGraphQueryClient:
    def __init__(self, summary_rows=None, detail_rows=None, exc: Exception | None = None):
        self.summary_rows = summary_rows if summary_rows is not None else []
        self.detail_rows = detail_rows if detail_rows is not None else []
        self.exc = exc
        self.calls = 0

    def select(self, sparql: str):
        if self.exc:
            raise self.exc
        self.calls += 1
        if "SELECT ?profile ?subject ?profile_id" in sparql:
            return self.summary_rows
        return self.detail_rows


class TestGraphReadModel:
    def test_latest_profile_query_has_deterministic_revision_semantics(self):
        sparql = build_latest_profile_query(subjects=["orion", "juniper"])
        assert "FILTER NOT EXISTS" in sparql
        assert "xsd:integer(?other_revision) > xsd:integer(?revision)" in sparql
        assert "?other_created_at > ?created_at" in sparql
        assert "STR(?other_profile_id) > STR(?profile_id)" in sparql

    def test_graph_query_mapping_maps_rows_into_concept_profile(self):
        summaries = map_latest_profile_rows(_summary_rows())
        details = map_profile_details_rows(_detail_rows())
        summary = summaries["orion"]
        profile = build_concept_profile(summary, details[summary.profile_uri])

        assert isinstance(profile, ConceptProfile)
        assert profile.subject == "orion"
        assert profile.revision == 7
        assert profile.concepts[0].concept_id == "concept-1"
        assert profile.clusters[0].cluster_id == "cluster-1"
        assert profile.state_estimate is not None
        assert profile.metadata["graph_provenance"]["writer_service"] == "orion-spark-concept-induction"

    def test_graph_repository_get_latest(self):
        repo = GraphConceptProfileRepository(
            endpoint="http://graphdb.local/repositories/collapse",
            graph_uri="http://conjourney.net/graph/spark/concept-profile",
            timeout_sec=3.0,
            query_client=FakeGraphQueryClient(summary_rows=_summary_rows(), detail_rows=_detail_rows()),
        )

        result = repo.get_latest("orion")
        assert result.availability == "available"
        assert result.profile is not None
        assert result.profile.subject == "orion"

    def test_graph_repository_list_latest(self):
        repo = GraphConceptProfileRepository(
            endpoint="http://graphdb.local/repositories/collapse",
            graph_uri="http://conjourney.net/graph/spark/concept-profile",
            timeout_sec=3.0,
            query_client=FakeGraphQueryClient(summary_rows=_summary_rows(), detail_rows=_detail_rows()),
        )

        results = repo.list_latest(["orion", "juniper"])
        by_subject = {item.subject: item for item in results}
        assert by_subject["orion"].availability == "available"
        assert by_subject["juniper"].availability == "empty"

    def test_graph_repository_distinguishes_empty_and_unavailable(self):
        unconfigured = GraphConceptProfileRepository(
            endpoint="",
            graph_uri="http://conjourney.net/graph/spark/concept-profile",
            timeout_sec=3.0,
        )
        result = unconfigured.get_latest("orion")
        assert result.availability == "unavailable"
        assert result.unavailable_reason == "graph_not_configured"

        failing = GraphConceptProfileRepository(
            endpoint="http://graphdb.local/repositories/collapse",
            graph_uri="http://conjourney.net/graph/spark/concept-profile",
            timeout_sec=3.0,
            query_client=FakeGraphQueryClient(exc=RuntimeError("boom")),
        )
        result2 = failing.get_latest("orion")
        assert result2.availability == "unavailable"
        assert result2.unavailable_reason == "query_error"

    def test_repository_factory_selection_local_graph_shadow(self, tmp_path):
        class LocalSettings:
            concept_profile_repository_backend = "local"
            store_path = str(tmp_path / "local.json")

        local_repo = build_concept_profile_repository(LocalSettings())
        assert isinstance(local_repo, LocalConceptProfileRepository)

        class GraphSettings(LocalSettings):
            concept_profile_repository_backend = "graph"
            concept_profile_graphdb_endpoint = "http://graphdb.local/repositories/collapse"
            concept_profile_graph_uri = "http://conjourney.net/graph/spark/concept-profile"
            concept_profile_graph_timeout_sec = 3.0
            concept_profile_graphdb_user = ""
            concept_profile_graphdb_pass = ""

        graph_repo = build_concept_profile_repository(GraphSettings())
        assert isinstance(graph_repo, GraphConceptProfileRepository)

        class ShadowSettings(GraphSettings):
            concept_profile_repository_backend = "shadow"

        shadow_repo = build_concept_profile_repository(ShadowSettings())
        assert isinstance(shadow_repo, ShadowConceptProfileRepository)

    def test_shadow_repository_returns_local_and_logs_parity(self, tmp_path):
        store_path = tmp_path / "profiles.json"
        store_path.write_text(
            """
            {"profiles": {"orion": {"profile_id": "profile-local", "subject": "orion", "revision": 1,
            "created_at": "2026-03-27T10:00:00+00:00", "window_start": "2026-03-27T09:00:00+00:00",
            "window_end": "2026-03-27T10:00:00+00:00", "concepts": [], "clusters": [],
            "state_estimate": null, "metadata": {}}}}
            """.strip()
        )
        local_repo = LocalConceptProfileRepository(store_path=str(store_path))
        graph_repo = GraphConceptProfileRepository(
            endpoint="http://graphdb.local/repositories/collapse",
            graph_uri="http://conjourney.net/graph/spark/concept-profile",
            timeout_sec=3.0,
            query_client=FakeGraphQueryClient(summary_rows=[], detail_rows=[]),
        )

        shadow_repo = ShadowConceptProfileRepository(local=local_repo, graph=graph_repo)
        result = shadow_repo.get_latest("orion")

        assert result.availability == "available"
        assert result.profile is not None
        assert result.profile.profile_id == "profile-local"

    def test_shadow_parity_logs_graph_unavailable_reason(self, tmp_path, caplog):
        caplog.set_level("INFO")
        store_path = tmp_path / "profiles.json"
        store_path.write_text(
            '{"profiles": {"orion": {"profile_id": "profile-local", "subject": "orion", "revision": 1,'
            '"created_at": "2026-03-27T10:00:00+00:00", "window_start": "2026-03-27T09:00:00+00:00",'
            '"window_end": "2026-03-27T10:00:00+00:00", "concepts": [], "clusters": [],'
            '"state_estimate": null, "metadata": {}}}}'
        )
        local_repo = LocalConceptProfileRepository(store_path=str(store_path))
        graph_repo = GraphConceptProfileRepository(
            endpoint="",
            graph_uri="http://conjourney.net/graph/spark/concept-profile",
            timeout_sec=3.0,
        )
        shadow_repo = ShadowConceptProfileRepository(local=local_repo, graph=graph_repo)
        shadow_repo.get_latest("orion", observer={"consumer": "concept_induction_pass", "correlation_id": "corr-a", "session_id": "sid-a"})
        assert "concept_profile_repository_parity" in caplog.text
        assert "graph_not_configured" in caplog.text

    def test_shadow_parity_logs_expected_mismatch_fields(self, tmp_path, caplog):
        caplog.set_level("INFO")
        store_path = tmp_path / "profiles.json"
        store_path.write_text(
            '{"profiles": {"orion": {"profile_id": "profile-local", "subject": "orion", "revision": 1,'
            '"created_at": "2026-03-27T10:00:00+00:00", "window_start": "2026-03-27T09:00:00+00:00",'
            '"window_end": "2026-03-27T10:00:00+00:00", "concepts": [{"concept_id":"c1","label":"alpha","aliases":[],"type":"identity","salience":1.0,"confidence":0.8,"embedding_ref":null,"evidence":[],"metadata":{}}], "clusters": [],'
            '"state_estimate": null, "metadata": {}}}}'
        )
        local_repo = LocalConceptProfileRepository(store_path=str(store_path))
        graph_repo = GraphConceptProfileRepository(
            endpoint="http://graphdb.local/repositories/collapse",
            graph_uri="http://conjourney.net/graph/spark/concept-profile",
            timeout_sec=3.0,
            query_client=FakeGraphQueryClient(summary_rows=_summary_rows(), detail_rows=_detail_rows()),
        )
        shadow_repo = ShadowConceptProfileRepository(local=local_repo, graph=graph_repo)
        shadow_repo.get_latest("orion", observer={"consumer": "chat_stance", "correlation_id": "corr-b", "session_id": "sid-b"})
        assert "concept_profile_repository_parity" in caplog.text
        assert "revision" in caplog.text
        assert "concept_ids_labels" in caplog.text
