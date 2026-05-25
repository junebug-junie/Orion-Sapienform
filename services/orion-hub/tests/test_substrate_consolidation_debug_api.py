from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]


def _ensure_hub_scripts_import_path() -> None:
    for key in list(sys.modules):
        if key == "scripts" or key.startswith("scripts."):
            del sys.modules[key]
    for p in (str(REPO_ROOT), str(HUB_ROOT)):
        try:
            sys.path.remove(p)
        except ValueError:
            pass
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(HUB_ROOT))


_ensure_hub_scripts_import_path()

from scripts import substrate_consolidation_routes  # noqa: E402
from orion.schemas.consolidation_frame import (  # noqa: E402
    ConsolidationFrameV1,
    ExpectationV1,
    MotifObservationV1,
    SchemaCandidateV1,
    SparseTensorSliceV1,
)

MOTIF_ID = "motif:loaded_but_reliable:consolidation_policy.v1"


def _sample_consolidation_frame() -> ConsolidationFrameV1:
    return ConsolidationFrameV1(
        frame_id=(
            "consolidation.frame:2026-05-25T14:30:00+00:00:"
            "2026-05-25T15:30:00+00:00:consolidation_policy.v1"
        ),
        generated_at="2026-05-25T15:30:00+00:00",
        window_start="2026-05-25T14:30:00+00:00",
        window_end="2026-05-25T15:30:00+00:00",
        motif_observations=[
            MotifObservationV1(
                motif_id=MOTIF_ID,
                motif_kind="self_state_pattern",
                label="loaded_but_reliable",
                recurrence_count=3,
                support_score=0.6,
                confidence_score=0.7,
            )
        ],
        source_counts={"self_state": 1, "feedback": 2},
    )


def _sample_expectation() -> ExpectationV1:
    return ExpectationV1(
        expectation_id=f"expectation:{MOTIF_ID}:reliability_clear",
        trigger_motif_id=MOTIF_ID,
        expected_outcome_kind="reliability_clear",
        confidence_score=0.7,
        support_count=3,
    )


def _sample_schema_candidate() -> SchemaCandidateV1:
    return SchemaCandidateV1(
        schema_candidate_id="schema.candidate:habit:loaded_but_reliable",
        candidate_kind="habit_candidate",
        label="loaded_but_reliable",
        source_motif_ids=[MOTIF_ID],
        support_score=0.6,
        confidence_score=0.7,
    )


def _sample_tensor_slice() -> SparseTensorSliceV1:
    return SparseTensorSliceV1(
        tensor_id=(
            "tensor:field_attention_self:2026-05-25T14:30:00+00:00:"
            "2026-05-25T15:30:00+00:00"
        ),
        tensor_kind="field_attention_self",
        axes=["time_bucket", "self_condition"],
        coordinates=[{"time_bucket": "2026-05-25T15:00:00+00:00", "self_condition": "loaded"}],
        values=[0.8],
    )


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    app = FastAPI()
    app.include_router(substrate_consolidation_routes.router)
    return TestClient(app)


def test_latest_returns_frame(client) -> None:
    sample = _sample_consolidation_frame()
    with patch.object(
        substrate_consolidation_routes,
        "_load_latest_consolidation_frame",
        return_value=sample,
    ):
        resp = client.get("/api/substrate/consolidation/latest")
    assert resp.status_code == 200
    assert resp.json()["source_counts"]["feedback"] == 2


def test_motifs_returns_recent_motifs(client) -> None:
    with patch.object(
        substrate_consolidation_routes,
        "load_recent_motifs",
        return_value=[_sample_consolidation_frame().motif_observations[0]],
    ):
        resp = client.get("/api/substrate/consolidation/motifs")
    assert resp.status_code == 200
    assert resp.json()[0]["motif_id"] == MOTIF_ID


def test_expectations_returns_rows(client) -> None:
    with patch.object(
        substrate_consolidation_routes,
        "load_expectations_for_motif",
        return_value=[_sample_expectation()],
    ):
        resp = client.get(
            "/api/substrate/consolidation/expectations",
            params={"trigger_motif_id": MOTIF_ID},
        )
    assert resp.status_code == 200
    assert resp.json()[0]["expected_outcome_kind"] == "reliability_clear"


def test_schema_candidates_returns_rows(client) -> None:
    with patch.object(
        substrate_consolidation_routes,
        "load_schema_candidates",
        return_value=[_sample_schema_candidate()],
    ):
        resp = client.get("/api/substrate/consolidation/schema-candidates")
    assert resp.status_code == 200
    assert resp.json()[0]["candidate_kind"] == "habit_candidate"


def test_tensor_slices_latest_returns_rows(client) -> None:
    with patch.object(
        substrate_consolidation_routes,
        "load_latest_tensor_slices",
        return_value=[_sample_tensor_slice()],
    ):
        resp = client.get("/api/substrate/consolidation/tensor-slices/latest")
    assert resp.status_code == 200
    assert resp.json()[0]["tensor_kind"] == "field_attention_self"


def test_router_has_no_post_routes() -> None:
    post_routes = [
        route
        for route in substrate_consolidation_routes.router.routes
        if "POST" in getattr(route, "methods", set())
    ]
    assert post_routes == []
