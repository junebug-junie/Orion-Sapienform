"""Tests for proposal review API on orion-context-exec."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

ROOT = Path(__file__).resolve().parents[2]
SERVICE_DIR = ROOT / "services" / "orion-context-exec"
CLI = ROOT / "scripts" / "orion_proposal_cli.py"
PYTHON = ROOT / "orion_dev" / "bin" / "python"


def _seed_store(store_path: Path) -> dict:
    result = subprocess.run(
        [
            str(PYTHON if PYTHON.exists() else sys.executable),
            str(CLI),
            "seed-demo",
            "--store",
            str(store_path),
        ],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "proposals.json"


@pytest.fixture
def app_client(store_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PROPOSAL_LEDGER_STORE_PATH", str(store_path))
    monkeypatch.setenv("PROPOSAL_REVIEW_API_ENABLED", "true")

    if str(SERVICE_DIR) not in sys.path:
        sys.path.insert(0, str(SERVICE_DIR))
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    for mod in ("app.settings", "app.main", "app.api", "app.proposal_review_api"):
        sys.modules.pop(mod, None)

    from app.main import app  # noqa: WPS433

    return app


@pytest.mark.asyncio
async def test_health_includes_proposal_review_block(app_client, store_path: Path) -> None:
    _seed_store(store_path)
    transport = ASGITransport(app=app_client)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    block = data["proposal_review_api"]
    assert block["enabled"] is True
    assert block["store_configured"] is True
    assert block["store_ok"] is True


@pytest.mark.asyncio
async def test_list_proposals_requires_store_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PROPOSAL_LEDGER_STORE_PATH", raising=False)
    monkeypatch.setenv("PROPOSAL_LEDGER_STORE_PATH", "")

    if str(SERVICE_DIR) not in sys.path:
        sys.path.insert(0, str(SERVICE_DIR))
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    for mod in ("app.settings", "app.main", "app.api", "app.proposal_review_api"):
        sys.modules.pop(mod, None)

    from app.main import app  # noqa: WPS433

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/proposals")
    assert resp.status_code == 503
    assert "PROPOSAL_LEDGER_STORE_PATH" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_list_proposals_filters_pending_review(app_client, store_path: Path) -> None:
    seeded = _seed_store(store_path)
    pending_id = seeded["records"]["pending_review_memory"]

    transport = ASGITransport(app=app_client)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/proposals", params={"status": "pending_review"})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["count"] == 1
    assert payload["proposals"][0]["proposal_id"] == pending_id


@pytest.mark.asyncio
async def test_get_proposal_detail(app_client, store_path: Path) -> None:
    seeded = _seed_store(store_path)
    proposal_id = seeded["records"]["pending_review_memory"]

    transport = ASGITransport(app=app_client)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(f"/proposals/{proposal_id}")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["proposal_id"] == proposal_id
    assert payload["envelope"]["proposal_type"] == "memory_correction_proposal"
    assert "execution_eligibility" in payload


@pytest.mark.asyncio
async def test_get_proposal_not_found(app_client, store_path: Path) -> None:
    _seed_store(store_path)
    transport = ASGITransport(app=app_client)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/proposals/missing-proposal-id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_malformed_store_returns_503(app_client, store_path: Path) -> None:
    store_path.write_text("{not-json", encoding="utf-8")
    transport = ASGITransport(app=app_client)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/proposals")
    assert resp.status_code == 503
    assert "malformed" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_invalid_store_schema_returns_503(app_client, store_path: Path) -> None:
    store_path.write_text('{"records": [{"proposal_id": "x"}]}', encoding="utf-8")
    transport = ASGITransport(app=app_client)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/proposals")
    assert resp.status_code == 503
    assert "invalid" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_triage_promote_sets_pending_review(app_client, store_path: Path) -> None:
    seeded = _seed_store(store_path)
    proposal_id = seeded["records"]["stored_patch"]

    transport = ASGITransport(app=app_client)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            f"/proposals/{proposal_id}/triage",
            json={"action": "promote_to_review", "rationale": "identity memory correction"},
        )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "pending_review"
    assert payload["attention_required"] is True


@pytest.mark.asyncio
async def test_triage_store_only_does_not_create_human_chore(app_client, store_path: Path) -> None:
    seeded = _seed_store(store_path)
    proposal_id = seeded["records"]["stored_patch"]

    transport = ASGITransport(app=app_client)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            f"/proposals/{proposal_id}/triage",
            json={"action": "store_only", "rationale": "not worth human attention"},
        )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "stored"
    assert payload["attention_required"] is False


@pytest.mark.asyncio
async def test_review_reject_does_not_execute(app_client, store_path: Path) -> None:
    seeded = _seed_store(store_path)
    proposal_id = seeded["records"]["pending_review_memory"]

    transport = ASGITransport(app=app_client)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            f"/proposals/{proposal_id}/review",
            json={
                "decision": "reject",
                "rationale": "unsupported evidence",
                "reviewer_type": "human",
                "reviewer_id": "june",
            },
        )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "rejected"
    assert payload["execution_eligibility"]["eligible"] is False
    assert payload["execution_eligibility"]["execution_requested"] is False


@pytest.mark.asyncio
async def test_review_approve_creates_eligibility_not_execution(app_client, store_path: Path) -> None:
    seeded = _seed_store(store_path)
    proposal_id = seeded["records"]["pending_review_memory"]

    transport = ASGITransport(app=app_client)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            f"/proposals/{proposal_id}/review",
            json={
                "decision": "approve",
                "rationale": "bounded and reversible",
                "reviewer_type": "human",
                "reviewer_id": "june",
            },
        )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "approved"
    assert payload["execution_eligibility"]["eligible"] is True
    assert payload["execution_eligibility"]["execution_requested"] is False


@pytest.mark.asyncio
async def test_review_rejects_context_exec_reviewer(app_client, store_path: Path) -> None:
    seeded = _seed_store(store_path)
    proposal_id = seeded["records"]["pending_review_memory"]

    transport = ASGITransport(app=app_client)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            f"/proposals/{proposal_id}/review",
            json={
                "decision": "approve",
                "rationale": "bad actor",
                "reviewer_type": "human",
                "reviewer_id": "context-exec",
            },
        )
    assert resp.status_code == 403
    assert "context-exec" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_eligibility_endpoint(app_client, store_path: Path) -> None:
    seeded = _seed_store(store_path)
    proposal_id = seeded["records"]["pending_review_memory"]

    transport = ASGITransport(app=app_client)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        approve = await client.post(
            f"/proposals/{proposal_id}/review",
            json={
                "decision": "approve",
                "rationale": "ok",
                "reviewer_type": "human",
                "reviewer_id": "june",
            },
        )
        assert approve.status_code == 200

        resp = await client.get(f"/proposals/{proposal_id}/eligibility")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["eligible"] is True
    assert payload["execution_requested"] is False
