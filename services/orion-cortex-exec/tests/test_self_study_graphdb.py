from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = SERVICE_DIR / "app"
PACKAGE_NAME = "orion_cortex_exec_graphdb"
APP_PACKAGE_NAME = f"{PACKAGE_NAME}.app"
if PACKAGE_NAME not in sys.modules:
    pkg = types.ModuleType(PACKAGE_NAME)
    pkg.__path__ = [str(SERVICE_DIR)]
    sys.modules[PACKAGE_NAME] = pkg
if APP_PACKAGE_NAME not in sys.modules:
    pkg = types.ModuleType(APP_PACKAGE_NAME)
    pkg.__path__ = [str(APP_DIR)]
    sys.modules[APP_PACKAGE_NAME] = pkg

spec = importlib.util.spec_from_file_location(f"{APP_PACKAGE_NAME}.self_study", APP_DIR / "self_study.py")
self_study = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = self_study
spec.loader.exec_module(self_study)


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http_{self.status_code}")

    def json(self) -> dict:
        return self._payload


def _bindings(rows: list[dict[str, str]]) -> dict:
    return {
        "results": {
            "bindings": [
                {key: {"type": "literal", "value": value} for key, value in row.items()}
                for row in rows
            ]
        }
    }


def _graphdb_post(url, *, data, headers, auth, timeout):
    del url, headers, auth, timeout
    query = str(data)
    if "AuthoritativeSelfFact" in query:
        return _FakeResponse(
            _bindings(
                [
                    {
                        "stable_id": "fact-1",
                        "title": "orion:rdf:enqueue",
                        "category": "channel",
                        "source_path": "orion/bus/channels.yaml",
                        "origin_kind": "channel",
                        "origin_name": "orion:rdf:enqueue",
                        "snapshot_id": "self-snapshot-1",
                    }
                ]
            )
        )
    if "InducedSelfConcept" in query and "supportedBy" not in query:
        return _FakeResponse(
            _bindings(
                [
                    {
                        "concept_uri": "http://conjourney.net/orion/self/concept/concept-1",
                        "stable_id": "concept-1",
                        "concept_kind": "recall_surface",
                        "title": "recall surface profile",
                        "description": "Persisted self-study concept about recall surfaces.",
                        "snapshot_id": "self-snapshot-1",
                        "confidence": "0.87",
                    }
                ]
            )
        )
    if "VALUES ?concept_uri" in query and "supportedBy" in query:
        return _FakeResponse(
            _bindings(
                [
                    {
                        "concept_uri": "http://conjourney.net/orion/self/concept/concept-1",
                        "snapshot_id": "self-snapshot-1",
                        "item_id": "fact-1",
                        "source_path": "orion/bus/channels.yaml",
                        "origin_kind": "channel",
                        "origin_name": "orion:rdf:enqueue",
                    }
                ]
            )
        )
    if "ReflectiveSelfFinding" in query and "derivedFromConcept" not in query and "supportedBy" not in query:
        return _FakeResponse(
            _bindings(
                [
                    {
                        "reflection_uri": "http://conjourney.net/orion/self/reflection/reflection-1",
                        "stable_id": "reflection-1",
                        "reflection_kind": "seam_risk",
                        "title": "Reflective caution",
                        "description": "Persisted reflective finding tied to a concept profile.",
                        "snapshot_id": "self-snapshot-1",
                        "confidence": "0.66",
                        "salience": "0.44",
                    }
                ]
            )
        )
    if "VALUES ?reflection_uri" in query and "supportedBy" in query:
        return _FakeResponse(
            _bindings(
                [
                    {
                        "reflection_uri": "http://conjourney.net/orion/self/reflection/reflection-1",
                        "snapshot_id": "self-snapshot-1",
                        "item_id": "fact-1",
                        "source_path": "orion/bus/channels.yaml",
                        "origin_kind": "channel",
                        "origin_name": "orion:rdf:enqueue",
                    }
                ]
            )
        )
    if "VALUES ?reflection_uri" in query and "derivedFromConcept" in query:
        return _FakeResponse(
            _bindings(
                [
                    {
                        "reflection_uri": "http://conjourney.net/orion/self/reflection/reflection-1",
                        "concept_id": "concept-1",
                        "concept_kind": "recall_surface",
                        "label": "recall surface profile",
                        "snapshot_id": "self-snapshot-1",
                    }
                ]
            )
        )
    raise AssertionError(f"unexpected query: {query}")


def test_graphdb_factual_retrieval_returns_authoritative_only(monkeypatch):
    monkeypatch.setenv("GRAPHDB_URL", "http://graphdb:7200")
    monkeypatch.setenv("GRAPHDB_REPO", "collapse")
    monkeypatch.setattr(self_study.requests, "post", _graphdb_post)

    request = self_study.SelfStudyRetrieveRequestV1.model_validate({"retrieval_mode": "factual"})
    result = self_study.retrieve_self_study(request)

    assert result.backend_used == "rdf_graph"
    assert result.counts.authoritative == 1
    assert result.counts.induced == 0
    assert result.counts.reflective == 0
    assert result.counts.facts == 1
    assert result.groups[0].items[0].source_kind == "self_repo_inspect"
    assert result.groups[0].items[0].storage_surface == "rdf_graph"


def test_graphdb_conceptual_retrieval_returns_persisted_concept_profiles(monkeypatch):
    monkeypatch.setenv("GRAPHDB_URL", "http://graphdb:7200")
    monkeypatch.setenv("GRAPHDB_REPO", "collapse")
    monkeypatch.setattr(self_study.requests, "post", _graphdb_post)

    request = self_study.SelfStudyRetrieveRequestV1.model_validate(
        {
            "retrieval_mode": "conceptual",
            "filters": {
                "stable_ids": ["fact-1", "concept-1"],
                "limit": 8,
            },
        }
    )
    result = self_study.retrieve_self_study(request)

    assert result.backend_used == "rdf_graph"
    assert result.counts.authoritative == 1
    assert result.counts.induced == 1
    assert result.counts.reflective == 0
    concept = result.groups[1].items[0]
    assert concept.stable_id == "concept-1"
    assert concept.source_kind == "self_concept_induce"
    assert concept.concept_kind == "recall_surface"
    assert concept.evidence[0].item_id == "fact-1"
    assert concept.evidence[0].source_path == "orion/bus/channels.yaml"
    assert concept.metadata["provenance"] == ["graphdb", "orion:self:induced"]

    filtered = self_study.retrieve_self_study(
        self_study.SelfStudyRetrieveRequestV1.model_validate(
            {
                "retrieval_mode": "conceptual",
                "filters": {
                    "concept_kinds": ["recall_surface"],
                    "stable_ids": ["concept-1"],
                    "limit": 8,
                },
            }
        )
    )
    assert filtered.counts.total == 1
    assert filtered.groups[0].items[0].stable_id == "concept-1"


def test_graphdb_reflective_retrieval_preserves_all_tiers_and_links(monkeypatch):
    monkeypatch.setenv("GRAPHDB_URL", "http://graphdb:7200")
    monkeypatch.setenv("GRAPHDB_REPO", "collapse")
    monkeypatch.setattr(self_study.requests, "post", _graphdb_post)

    request = self_study.SelfStudyRetrieveRequestV1.model_validate(
        {"retrieval_mode": "reflective", "filters": {"limit": 12}}
    )
    result = self_study.retrieve_self_study(request)

    tiers = {item.trust_tier for group in result.groups for item in group.items}
    assert result.backend_used == "rdf_graph"
    assert tiers == {"authoritative", "induced", "reflective"}
    reflection = result.groups[2].items[0]
    assert reflection.source_kind == "self_concept_reflect"
    assert reflection.evidence[0].item_id == "fact-1"
    assert reflection.concept_refs[0].concept_id == "concept-1"
    assert reflection.metadata["provenance"] == ["graphdb", "orion:self:reflective"]


def test_graphdb_retrieval_never_upcasts_in_factual_mode(monkeypatch):
    monkeypatch.setenv("GRAPHDB_URL", "http://graphdb:7200")
    monkeypatch.setenv("GRAPHDB_REPO", "collapse")
    monkeypatch.setattr(self_study.requests, "post", _graphdb_post)

    request = self_study.SelfStudyRetrieveRequestV1.model_validate({"retrieval_mode": "factual", "filters": {"limit": 8}})
    result = self_study.retrieve_self_study(request)

    assert all(item.trust_tier == "authoritative" for group in result.groups for item in group.items)
    assert all(item.record_type == "fact" for group in result.groups for item in group.items)
    assert all(item.source_kind != "self_concept_reflect" for group in result.groups for item in group.items)


def test_graphdb_unavailable_falls_back_explicitly_when_allowed(monkeypatch):
    monkeypatch.setenv("GRAPHDB_URL", "http://graphdb:7200")
    monkeypatch.setenv("GRAPHDB_REPO", "collapse")

    def _boom(*args, **kwargs):
        raise self_study.requests.RequestException("graphdb_down")

    monkeypatch.setattr(self_study.requests, "post", _boom)

    request = self_study.SelfStudyRetrieveRequestV1.model_validate({"retrieval_mode": "conceptual"})
    result = self_study.retrieve_self_study(request)

    assert result.backend_used == "in_process"
    statuses = {status.storage_surface: status.status for status in result.backend_status}
    assert statuses["rdf_graph"] == "unavailable"
    assert statuses["in_process"] == "used"
    assert any("fell back to in-process retrieval" in note for note in result.notes)


def test_graphdb_only_request_stays_typed_when_backend_unavailable(monkeypatch):
    monkeypatch.setenv("GRAPHDB_URL", "http://graphdb:7200")
    monkeypatch.setenv("GRAPHDB_REPO", "collapse")

    def _boom(*args, **kwargs):
        raise self_study.requests.RequestException("graphdb_down")

    monkeypatch.setattr(self_study.requests, "post", _boom)

    request = self_study.SelfStudyRetrieveRequestV1.model_validate(
        {
            "retrieval_mode": "reflective",
            "filters": {"storage_surfaces": ["rdf_graph"], "limit": 8},
        }
    )
    result = self_study.retrieve_self_study(request)

    assert result.backend_used is None
    assert result.counts.total == 0
    statuses = {status.storage_surface: status.status for status in result.backend_status}
    assert statuses["rdf_graph"] == "unavailable"
    assert statuses["in_process"] == "not_queried"


def test_graphdb_retrieval_is_stable_across_repeated_reads(monkeypatch):
    monkeypatch.setenv("GRAPHDB_URL", "http://graphdb:7200")
    monkeypatch.setenv("GRAPHDB_REPO", "collapse")
    monkeypatch.setattr(self_study.requests, "post", _graphdb_post)

    request = self_study.SelfStudyRetrieveRequestV1.model_validate({"retrieval_mode": "reflective", "filters": {"limit": 12}})
    first = self_study.retrieve_self_study(request)
    second = self_study.retrieve_self_study(request)

    first_ids = [item.stable_id for group in first.groups for item in group.items]
    second_ids = [item.stable_id for group in second.groups for item in group.items]
    assert first.backend_used == second.backend_used == "rdf_graph"
    assert first_ids == second_ids
    assert first.counts.model_dump() == second.counts.model_dump()
