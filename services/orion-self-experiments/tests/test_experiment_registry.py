from __future__ import annotations

import os
import sys

SERVICE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SERVICE_ROOT not in sys.path:
    sys.path.insert(0, SERVICE_ROOT)

import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from orion.schemas.context_exec import ContextExecRunV1
from orion.schemas.self_experiments import SelfExperimentRecordV1, SelfExperimentSpecV1

from app.experiment_registry import (
    ExperimentValidationError,
    compile_experiment_to_context_exec_request,
    compute_dedupe_key,
    normalize_create_request,
    parse_context_exec_result,
)
from app.settings import settings
from app.store import init_db


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "experiments.sqlite3"
    monkeypatch.setattr(settings, "experiments_store_path", str(db_path))
    monkeypatch.setattr(settings, "self_experiments_dispatch_enabled", False)
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


def test_runtime_drift_check_compiles_investigation_v2() -> None:
    record = SelfExperimentRecordV1(
        experiment_id="exp-3",
        spec=SelfExperimentSpecV1(
            experiment_id="exp-3",
            experiment_type="runtime_drift_check",
            question="Check transport reducer lag.",
            source="daily_metacog_v1",
            created_at_utc=_now(),
        ),
        status="validated",
        dedupe_key="abc",
        created_at_utc=_now(),
        updated_at_utc=_now(),
    )
    req = compile_experiment_to_context_exec_request(record)
    assert req.mode == "investigation_v2"
    assert req.expected_artifact_type == "InvestigationReportV2"
    assert req.permissions.read_runtime_logs is True
    assert req.permissions.write_memory is False


def test_context_exec_mode_override_rejected() -> None:
    record = SelfExperimentRecordV1(
        experiment_id="exp-4",
        spec=SelfExperimentSpecV1(
            experiment_id="exp-4",
            experiment_type="runtime_drift_check",
            question="Check lag.",
            requested_context_exec_mode="belief_provenance",
            created_at_utc=_now(),
        ),
        status="validated",
        dedupe_key="def",
        created_at_utc=_now(),
        updated_at_utc=_now(),
    )
    with pytest.raises(ExperimentValidationError, match="context_exec_mode_override_rejected"):
        compile_experiment_to_context_exec_request(record)


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


def test_memory_correction_compiles_proposal_envelope() -> None:
    record = SelfExperimentRecordV1(
        experiment_id="exp-6",
        spec=SelfExperimentSpecV1(
            experiment_id="exp-6",
            experiment_type="memory_correction_candidate",
            question="Is this belief grounded?",
            mutation_policy="proposal_only",
            created_at_utc=_now(),
        ),
        status="validated",
        dedupe_key="ghi",
        created_at_utc=_now(),
        updated_at_utc=_now(),
    )
    req = compile_experiment_to_context_exec_request(record)
    assert req.mode == "memory_correction_proposal"
    assert req.expected_artifact_type == "ProposalEnvelopeV1"


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


def test_result_parser_extracts_proposal_id() -> None:
    parsed = parse_context_exec_result(
        {
            "run_id": "run-1",
            "status": "ok",
            "artifact_type": "ProposalEnvelopeV1",
            "runtime_debug": {"proposal_id": "prop-123", "attention_required": True},
            "artifact": {},
        }
    )
    assert parsed["proposal_id"] == "prop-123"
    assert parsed["attention_required"] is True


def test_create_and_dispatch_disabled_queues(client) -> None:
    resp = client.post(
        "/v1/experiments",
        json={
            "experiment_type": "runtime_drift_check",
            "question": "Check transport reducer lag.",
            "source": "daily_metacog_v1",
            "source_ref": "2026-06-16",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "validated"
    exp_id = body["experiment_id"]

    dispatch = client.post(f"/v1/experiments/{exp_id}/dispatch")
    assert dispatch.status_code == 200
    assert dispatch.json()["status"] == "queued"
    assert dispatch.json()["message"] == "dispatch_disabled"


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


@patch("app.main.dispatch_context_exec", new_callable=AsyncMock)
def test_dispatch_to_fake_context_exec_completes(mock_dispatch, client, monkeypatch) -> None:
    monkeypatch.setattr(settings, "self_experiments_dispatch_enabled", True)
    mock_dispatch.return_value = ContextExecRunV1(
        run_id="run-99",
        status="ok",
        mode="investigation_v2",
        text="Check lag.",
        artifact_type="InvestigationReportV2",
        final_text="done",
    )
    create = client.post(
        "/v1/experiments",
        json={
            "experiment_type": "runtime_drift_check",
            "question": "Check transport reducer lag.",
        },
    )
    exp_id = create.json()["experiment_id"]
    dispatch = client.post(f"/v1/experiments/{exp_id}/dispatch")
    assert dispatch.status_code == 200
    assert dispatch.json()["status"] == "completed"
    record = client.get(f"/v1/experiments/{exp_id}").json()
    assert record["context_exec_run_id"] == "run-99"
    assert record["artifact_type"] == "InvestigationReportV2"


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
