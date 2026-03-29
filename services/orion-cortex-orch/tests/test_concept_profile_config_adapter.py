from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT, APP_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from app.concept_profile_config import OrchConceptProfileSettings, build_orch_concept_profile_settings, get_orch_concept_profile_settings
from orion.spark.concept_induction.profile_repository import build_concept_profile_repository


def test_orch_concept_profile_adapter_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("CONCEPT_STORE_PATH", "/tmp/orch-concepts.json")
    monkeypatch.setenv("CONCEPT_SUBJECTS", "orion,juniper")
    monkeypatch.setenv("CONCEPT_PROFILE_REPOSITORY_BACKEND", "shadow")
    monkeypatch.setenv("CONCEPT_PROFILE_BACKEND_CONCEPT_INDUCTION_PASS", "graph")
    monkeypatch.setenv("CONCEPT_PROFILE_GRAPH_CUTOVER_FALLBACK_POLICY", "fail_closed")
    monkeypatch.setenv("CONCEPT_PROFILE_GRAPHDB_URL", "http://graphdb:7200")
    monkeypatch.setenv("CONCEPT_PROFILE_GRAPHDB_REPO", "collapse")
    monkeypatch.setenv("CONCEPT_PROFILE_GRAPH_URI", "http://example.org/graph/concept-profile")
    monkeypatch.setenv("CONCEPT_PROFILE_GRAPH_TIMEOUT_SEC", "9")
    monkeypatch.setenv("CONCEPT_PROFILE_PARITY_MIN_COMPARISONS", "75")

    get_orch_concept_profile_settings.cache_clear()
    cfg = build_orch_concept_profile_settings()

    assert cfg.store_path == "/tmp/orch-concepts.json"
    assert cfg.subjects == ["orion", "juniper"]
    assert cfg.concept_profile_repository_backend == "shadow"
    assert cfg.concept_profile_backend_concept_induction_pass == "graph"
    assert cfg.concept_profile_graph_cutover_fallback_policy == "fail_closed"
    assert cfg.concept_profile_graphdb_endpoint == "http://graphdb:7200/repositories/collapse"
    assert cfg.concept_profile_graph_uri == "http://example.org/graph/concept-profile"
    assert cfg.concept_profile_graph_timeout_sec == 9
    assert cfg.concept_profile_parity_min_comparisons == 75


def test_orch_adapter_builds_local_graph_shadow_repositories_without_missing_fields(tmp_path) -> None:
    base = OrchConceptProfileSettings.model_validate(
        {
            "CONCEPT_STORE_PATH": str(tmp_path / "concepts.json"),
            "CONCEPT_PROFILE_GRAPHDB_ENDPOINT": "http://graphdb:7200/repositories/collapse",
            "CONCEPT_PROFILE_GRAPH_URI": "http://example.org/graph/concept-profile",
            "CONCEPT_PROFILE_GRAPH_TIMEOUT_SEC": 1.0,
        }
    )

    local_repo = build_concept_profile_repository(base, backend_override="local")
    graph_repo = build_concept_profile_repository(base, backend_override="graph")
    shadow_repo = build_concept_profile_repository(base, backend_override="shadow")

    assert local_repo.status().backend == "local"
    assert graph_repo.status().backend == "graph"
    assert shadow_repo.status().backend == "shadow"
