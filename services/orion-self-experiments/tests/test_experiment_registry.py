from __future__ import annotations

import os
import sys

SERVICE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SERVICE_ROOT not in sys.path:
    sys.path.insert(0, SERVICE_ROOT)


import pytest
from fastapi.testclient import TestClient

from orion.schemas.self_experiments import SelfExperimentRecordV1

from app.experiment_registry import (
    ExperimentValidationError,
    compute_dedupe_key,
    normalize_create_request,
)
from app.settings import settings
from app.store import init_db


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "experiments.sqlite3"
    monkeypatch.setattr(settings, "experiments_store_path", str(db_path))
    init_db()
    from app.main import app

    return TestClient(app)


def _now() -> str:
    return "2026-06-17T00:00:00Z"


def test_legacy_skill_id_normalizes_to_skill_probe() -> None:
    req = normalize_create_request(
        __import__("orion.schemas.self_experiments", fromlist=["SelfExperimentCreateRequestV1"]).SelfExperimentCreateRequestV1(
            skill_id="skills.system.time_now.v1",
            provenance={},
        ),
        experiment_id="exp-1",
        created_at_utc=_now(),
        allow_non_read_only=False,
    )
    spec, _ = req
    assert spec.experiment_type == "skill_probe"
    assert spec.requested_skill_id == "skills.system.time_now.v1"
    assert "time_now" in spec.question


def test_unknown_experiment_type_rejected(client) -> None:
    resp = client.post(
        "/v1/experiments",
        json={"experiment_type": "not_a_real_type", "question": "test question"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "rejected"
    assert body["message"] == "unknown_experiment_type"


def test_daily_mutation_policy_widen_rejected() -> None:
    from orion.schemas.self_experiments import SelfExperimentCreateRequestV1

    with pytest.raises(ExperimentValidationError, match="daily_mutation_forbidden"):
        normalize_create_request(
            SelfExperimentCreateRequestV1(
                experiment_type="runtime_drift_check",
                question="Check lag.",
                source="daily_metacog_v1",
                mutation_policy="proposal_only",
            ),
            experiment_id="exp-5",
            created_at_utc=_now(),
            allow_non_read_only=False,
        )


def test_daily_proposal_type_forbidden() -> None:
    from orion.schemas.self_experiments import SelfExperimentCreateRequestV1

    with pytest.raises(ExperimentValidationError, match="daily_proposal_type_forbidden"):
        normalize_create_request(
            SelfExperimentCreateRequestV1(
                experiment_type="memory_correction_candidate",
                question="Propose correction.",
                source="daily_metacog_v1",
            ),
            experiment_id="exp-5b",
            created_at_utc=_now(),
            allow_non_read_only=False,
        )


def test_dedupe_key_stable() -> None:
    a = compute_dedupe_key(
        experiment_type="runtime_drift_check",
        question="Check lag.",
        source="daily_metacog_v1",
        source_ref="2026-06-16",
    )
    b = compute_dedupe_key(
        experiment_type="runtime_drift_check",
        question="Check lag.",
        source="daily_metacog_v1",
        source_ref="2026-06-16",
    )
    assert a == b


def test_manual_review_candidate_does_not_auto_dispatch(client) -> None:
    resp = client.post(
        "/v1/experiments",
        json={
            "experiment_type": "manual_review_candidate",
            "question": "Review this hypothesis.",
        },
    )
    assert resp.json()["status"] == "validated"
    record = client.get(f"/v1/experiments/{resp.json()['experiment_id']}").json()
    assert record["status"] == "validated"
    assert record["context_exec_request"] is None


def test_unknown_skill_legacy_rejected(client) -> None:
    resp = client.post(
        "/v1/experiments",
        json={"skill_id": "definitely.not.a.real.skill.v99", "provenance": {}, "args": {}},
    )
    assert resp.json()["status"] == "rejected"
    assert resp.json()["message"] == "unknown_skill_id"


def test_non_read_only_skill_legacy_rejected(client, monkeypatch) -> None:
    from orion.cognition.skills_manifest import SkillManifestEntry

    monkeypatch.setattr(
        "app.experiment_registry.load_skill_manifest",
        lambda: [
            SkillManifestEntry(
                skill_id="mutating.skill.v1",
                label="Mutating",
                description="Not read only",
                family="test",
                read_only=False,
                idempotent=False,
                risk_class="high_impact",
            )
        ],
    )
    resp = client.post(
        "/v1/experiments",
        json={"skill_id": "mutating.skill.v1", "provenance": {}, "args": {}},
    )
    assert resp.json()["status"] == "rejected"
    assert resp.json()["message"] == "non_read_only_skill_rejected"


def test_api_dedupe_hit_returns_same_experiment_id(client) -> None:
    payload = {
        "experiment_type": "runtime_drift_check",
        "question": "Check transport reducer lag.",
        "source": "daily_metacog_v1",
        "source_ref": "2026-06-17",
    }
    first = client.post("/v1/experiments", json=payload)
    second = client.post("/v1/experiments", json=payload)
    assert first.json()["experiment_id"] == second.json()["experiment_id"]
    assert second.json()["message"] == "dedupe_hit"


def test_dedupe_allows_recreate_after_discard(client) -> None:
    payload = {
        "experiment_type": "manual_review_candidate",
        "question": "Review this hypothesis.",
    }
    created = client.post("/v1/experiments", json=payload).json()
    exp_id = created["experiment_id"]
    discard = client.post(f"/v1/experiments/{exp_id}/discard")
    assert discard.json()["status"] == "discarded"

    recreated = client.post("/v1/experiments", json=payload).json()
    assert recreated["experiment_id"] != exp_id
    assert recreated["status"] == "validated"
    assert recreated.get("message") != "dedupe_hit"


def test_insert_record_dedupe_safe_is_atomic(tmp_path, monkeypatch) -> None:
    from uuid import uuid4

    from orion.schemas.self_experiments import SelfExperimentSpecV1

    from app.store import init_db, insert_record_dedupe_safe

    db_path = tmp_path / "dedupe.sqlite3"
    monkeypatch.setattr(settings, "experiments_store_path", str(db_path))
    init_db()

    now = "2026-06-17T00:00:00Z"
    dedupe_key = compute_dedupe_key(
        experiment_type="runtime_drift_check",
        question="Check lag.",
        source="daily_metacog_v1",
        source_ref="2026-06-17",
    )
    record_a = SelfExperimentRecordV1(
        experiment_id=str(uuid4()),
        spec=SelfExperimentSpecV1(
            experiment_id=str(uuid4()),
            experiment_type="runtime_drift_check",
            question="Check lag.",
            source="daily_metacog_v1",
            source_ref="2026-06-17",
            created_at_utc=now,
        ),
        status="validated",
        dedupe_key=dedupe_key,
        created_at_utc=now,
        updated_at_utc=now,
    )
    record_b = record_a.model_copy(update={"experiment_id": str(uuid4())})

    stored_a, outcome_a = insert_record_dedupe_safe(record_a)
    stored_b, outcome_b = insert_record_dedupe_safe(record_b)

    assert outcome_a == "created"
    assert outcome_b == "dedupe_hit"
    assert stored_a.experiment_id == stored_b.experiment_id


@pytest.mark.parametrize(
    "skill_id", ["skills.docker.compose_service_bringup.v1", "skills.imagination.render_scene.v1"]
)
def test_world_changing_skills_rejected_as_read_only_probes(skill_id: str) -> None:
    """2026-10-01: both were labelled read-only; daily pulse created a
    compose_service_bringup "read-only skill probe" on 2026-08-30. Real manifest, no stub."""
    from orion.schemas.self_experiments import SelfExperimentCreateRequestV1

    with pytest.raises(ExperimentValidationError, match="non_read_only_skill_rejected"):
        normalize_create_request(
            SelfExperimentCreateRequestV1(skill_id=skill_id, provenance={}),
            experiment_id="exp-x",
            created_at_utc=_now(),
            allow_non_read_only=False,
        )


def test_dispatch_and_retry_routes_are_gone(client) -> None:
    """orion-context-exec was the only dispatch target; it was retired 2026-10-10,
    so the dispatch/retry routes were deleted rather than left pointing at nothing."""
    resp = client.post(
        "/v1/experiments",
        json={"experiment_type": "runtime_drift_check", "question": "Check transport reducer lag."},
    )
    exp_id = resp.json()["experiment_id"]
    assert client.post(f"/v1/experiments/{exp_id}/dispatch").status_code in (404, 405)
    assert client.post(f"/v1/experiments/{exp_id}/retry").status_code in (404, 405)
    assert not hasattr(settings, "self_experiments_context_exec_request_channel")
    health = client.get("/health")
    assert health.status_code == 200
    assert "dispatch_enabled" not in health.json()
