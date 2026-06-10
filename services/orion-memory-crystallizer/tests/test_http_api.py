"""HTTP-layer tests for the governor API (no DB, no bus)."""

from __future__ import annotations

import pytest
from conftest import make_proposal
from fake_repo import FakeRepository
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routes import router
from app.settings import Settings


@pytest.fixture
def repo() -> FakeRepository:
    return FakeRepository()


@pytest.fixture
def client(repo: FakeRepository) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.state.settings = Settings()
    app.state.repo = repo
    app.state.bus = None
    app.state.cards_pool = None
    return TestClient(app)


def _propose(client: TestClient, proposal=None) -> str:
    proposal = proposal or make_proposal()
    resp = client.post(
        "/api/memory/crystallizations/propose", json=proposal.model_dump(mode="json")
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["crystallization_id"]


def _approve(client: TestClient, cid: str) -> dict:
    resp = client.post(
        f"/api/memory/crystallizations/proposals/{cid}/approve",
        json={"actor": "operator:test"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_propose_validates_and_stores(client, repo) -> None:
    cid = _propose(client)
    assert repo.rows[cid].status == "proposed"
    assert repo.rows[cid].governance.validation_status == "valid"


def test_propose_rejects_non_proposed_status(client) -> None:
    proposal = make_proposal().model_copy(update={"status": "active"})
    resp = client.post(
        "/api/memory/crystallizations/propose", json=proposal.model_dump(mode="json")
    )
    assert resp.status_code == 422


def test_repropose_governed_artifact_conflicts(client) -> None:
    cid = _propose(client)
    _approve(client, cid)
    proposal = make_proposal(crystallization_id=cid)
    resp = client.post(
        "/api/memory/crystallizations/propose", json=proposal.model_dump(mode="json")
    )
    assert resp.status_code == 409


def test_proposals_listing_route_not_shadowed(client) -> None:
    cid = _propose(client)
    listing = client.get("/api/memory/crystallizations/proposals")
    assert listing.status_code == 200
    assert [p["crystallization_id"] for p in listing.json()] == [cid]
    detail = client.get(f"/api/memory/crystallizations/proposals/{cid}")
    assert detail.status_code == 200
    assert detail.json()["crystallization_id"] == cid


def test_approve_then_active_listing(client, repo) -> None:
    cid = _propose(client)
    approved = _approve(client, cid)
    assert approved["status"] == "active"
    active = client.get("/api/memory/crystallizations").json()
    assert [c["crystallization_id"] for c in active] == [cid]
    # history recorded for both validate (on propose) and approve
    ops = [e.op for e in repo.history]
    assert "validate" in ops and "approve" in ops


def test_approve_twice_conflicts(client) -> None:
    cid = _propose(client)
    _approve(client, cid)
    resp = client.post(
        f"/api/memory/crystallizations/proposals/{cid}/approve",
        json={"actor": "operator:test"},
    )
    assert resp.status_code == 409


def test_reject_then_card_projection_blocked(client) -> None:
    cid = _propose(client)
    resp = client.post(
        f"/api/memory/crystallizations/proposals/{cid}/reject",
        json={"actor": "operator:test", "reason": "nope"},
    )
    assert resp.status_code == 200
    proj = client.post(f"/api/memory/crystallizations/{cid}/project/card")
    assert proj.status_code == 409


def test_card_projection_returns_payload_without_pool(client) -> None:
    cid = _propose(client)
    _approve(client, cid)
    proj = client.post(f"/api/memory/crystallizations/{cid}/project/card")
    assert proj.status_code == 200
    body = proj.json()
    assert body["card_created"] is False  # no cards pool wired in tests
    ref = body["card_payload"]["subschema"]["crystallization_ref"]
    assert ref["crystallization_id"] == cid


def test_supersede_via_http(client, repo) -> None:
    old_cid = _propose(client)
    _approve(client, old_cid)
    new_cid = _propose(client, make_proposal(subject="Updated stance"))
    _approve(client, new_cid)
    resp = client.post(
        f"/api/memory/crystallizations/{old_cid}/supersede",
        json={"superseded_by": new_cid, "actor": "operator:test"},
    )
    assert resp.status_code == 200
    assert repo.rows[old_cid].status == "superseded"
    assert any(
        l.relation == "supersedes" and l.target_crystallization_id == old_cid
        for l in repo.rows[new_cid].links
    )


def test_active_packet_endpoint(client) -> None:
    cid = _propose(client)
    _approve(client, cid)
    resp = client.post("/api/memory/active-packet", json={"query": "memory architecture"})
    assert resp.status_code == 200
    body = resp.json()
    packet = body["packet"]
    assert packet["crystallization_refs"] == [cid]
    assert len(packet["stance"]) == 1
    event = client.get(f"/api/memory/retrieval-events/{body['retrieval_event_id']}")
    assert event.status_code == 200


def test_repo_unavailable_returns_503(repo) -> None:
    app = FastAPI()
    app.include_router(router)
    app.state.settings = Settings()
    app.state.repo = None
    app.state.bus = None
    app.state.cards_pool = None
    client = TestClient(app)
    resp = client.get("/api/memory/crystallizations")
    assert resp.status_code == 503


def test_graphiti_disabled_returns_503(client) -> None:
    cid = _propose(client)
    resp = client.post(f"/api/memory/crystallizations/{cid}/project/graphiti")
    assert resp.status_code == 503
