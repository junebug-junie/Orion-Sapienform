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


def _load_app(monkeypatch: pytest.MonkeyPatch, *, enabled: bool, store_path: str | None) -> object:
    if enabled:
        monkeypatch.setenv("PROPOSAL_REVIEW_API_ENABLED", "true")
    else:
        monkeypatch.delenv("PROPOSAL_REVIEW_API_ENABLED", raising=False)
        monkeypatch.setenv("PROPOSAL_REVIEW_API_ENABLED", "false")

    if store_path is None:
        monkeypatch.delenv("PROPOSAL_LEDGER_STORE_PATH", raising=False)
        monkeypatch.setenv("PROPOSAL_LEDGER_STORE_PATH", "")
    else:
        monkeypatch.setenv("PROPOSAL_LEDGER_STORE_PATH", store_path)

    if str(SERVICE_DIR) not in sys.path:
        sys.path.insert(0, str(SERVICE_DIR))
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    for mod in ("app.settings", "app.main", "app.api", "app.proposal_review_api"):
        sys.modules.pop(mod, None)

    from app.main import app  # noqa: WPS433

    return app


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "proposals.json"


@pytest.fixture
def app_client(store_path: Path, monkeypatch: pytest.MonkeyPatch):
    return _load_app(monkeypatch, enabled=True, store_path=str(store_path))


@pytest.mark.asyncio
async def test_context_exec_app_mounts_proposal_review_routes_when_enabled(
    app_client, store_path: Path
) -> None:
    seeded = _seed_store(store_path)
    proposal_id = seeded["records"]["pending_review_memory"]

    transport = ASGITransport(app=app_client)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        health = await client.get("/health")
        proposals = await client.get("/proposals")
        eligibility = await client.get(f"/proposals/{proposal_id}/eligibility")

    assert health.status_code == 200
    assert proposals.status_code == 200
    assert eligibility.status_code == 200


@pytest.mark.asyncio
async def test_proposal_review_api_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _load_app(monkeypatch, enabled=False, store_path=None)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        health = await client.get("/health")
        proposals = await client.get("/proposals")

    assert health.status_code == 200
    block = health.json()["proposal_review_api"]
    assert block["enabled"] is False
    assert block["ok"] is False
    assert block["error"]
    assert proposals.status_code == 404


@pytest.mark.asyncio
async def test_proposal_review_api_enabled_without_store_path_fails_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _load_app(monkeypatch, enabled=True, store_path=None)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        health = await client.get("/health")
        proposals = await client.get("/proposals")

    assert health.status_code == 200
    block = health.json()["proposal_review_api"]
    assert block["enabled"] is True
    assert block["store_configured"] is False
    assert block["ok"] is False
    assert "PROPOSAL_LEDGER_STORE_PATH" in (block["error"] or "")

    assert proposals.status_code == 503
    assert "PROPOSAL_LEDGER_STORE_PATH" in proposals.json()["detail"]


@pytest.mark.asyncio
async def test_health_proposal_review_block_store_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _load_app(monkeypatch, enabled=True, store_path=None)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")

    block = resp.json()["proposal_review_api"]
    assert block["enabled"] is True
    assert block["store_configured"] is False
    assert block["store_path_present"] is False
    assert block["ok"] is False
    assert block["error"]


@pytest.mark.asyncio
async def test_health_proposal_review_block_store_configured(
    app_client, store_path: Path
) -> None:
    _seed_store(store_path)
    transport = ASGITransport(app=app_client)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")

    block = resp.json()["proposal_review_api"]
    assert block["enabled"] is True
    assert block["store_configured"] is True
    assert block["store_path_present"] is True
    assert block["ok"] is True
    assert block["error"] is None


@pytest.mark.asyncio
async def test_proposal_review_api_malformed_store_returns_controlled_error(
    app_client, store_path: Path
) -> None:
    store_path.write_text("{not-json", encoding="utf-8")
    transport = ASGITransport(app=app_client)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/proposals")
    assert resp.status_code == 503
    assert "malformed" in resp.json()["detail"].lower()
    assert store_path.read_text(encoding="utf-8") == "{not-json"


@pytest.mark.asyncio
async def test_proposal_review_api_invalid_ledger_schema_returns_controlled_error(
    app_client, store_path: Path
) -> None:
    store_path.write_text('{"records": [{"proposal_id": "x"}]}', encoding="utf-8")
    transport = ASGITransport(app=app_client)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/proposals")
    assert resp.status_code == 503
    assert "invalid" in resp.json()["detail"].lower()
    assert store_path.read_text(encoding="utf-8") == '{"records": [{"proposal_id": "x"}]}'


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
async def test_proposal_review_detail_missing_returns_404(app_client, store_path: Path) -> None:
    _seed_store(store_path)
    transport = ASGITransport(app=app_client)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/proposals/missing-proposal-id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_proposal_review_triage_missing_returns_404(app_client, store_path: Path) -> None:
    _seed_store(store_path)
    transport = ASGITransport(app=app_client)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/proposals/missing-proposal-id/triage",
            json={"action": "store_only", "rationale": "missing"},
        )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_proposal_review_review_missing_returns_404(app_client, store_path: Path) -> None:
    _seed_store(store_path)
    transport = ASGITransport(app=app_client)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/proposals/missing-proposal-id/review",
            json={
                "decision": "approve",
                "rationale": "missing",
                "reviewer_type": "human",
                "reviewer_id": "june",
            },
        )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_proposal_review_eligibility_missing_returns_404(app_client, store_path: Path) -> None:
    _seed_store(store_path)
    transport = ASGITransport(app=app_client)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/proposals/missing-proposal-id/eligibility")
    assert resp.status_code == 404


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
async def test_proposal_review_approve_creates_eligibility_not_execution(
    app_client, store_path: Path
) -> None:
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
async def test_proposal_review_approve_does_not_create_receipt(
    app_client, store_path: Path
) -> None:
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
    assert "execution_receipt" not in payload
    assert "receipt" not in payload
    assert payload["execution_eligibility"].get("execution_receipt") is None


@pytest.mark.asyncio
async def test_proposal_review_approve_does_not_call_executor(
    app_client, store_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeded = _seed_store(store_path)
    proposal_id = seeded["records"]["pending_review_memory"]

    calls: list[object] = []

    def _track_run(self, *_args, **_kwargs):
        calls.append(True)
        raise AssertionError("executor must not be called from review API")

    monkeypatch.setattr("app.runner.ContextExecRunner.run", _track_run)

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
    assert calls == []
    payload = resp.json()
    assert payload["status"] == "approved"
    assert "changed_targets" not in payload
    assert payload["execution_eligibility"]["execution_requested"] is False


@pytest.mark.asyncio
async def test_proposal_review_rejects_context_exec_approval(
    app_client, store_path: Path
) -> None:
    seeded = _seed_store(store_path)
    proposal_id = seeded["records"]["pending_review_memory"]

    transport = ASGITransport(app=app_client)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            f"/proposals/{proposal_id}/review",
            json={
                "decision": "approve",
                "reviewer_type": "human",
                "reviewer_id": "context-exec",
                "rationale": "self approval",
            },
        )
        assert resp.status_code == 403
        assert "context-exec" in resp.json()["detail"].lower()

        detail = await client.get(f"/proposals/{proposal_id}")
        eligibility = await client.get(f"/proposals/{proposal_id}/eligibility")

    assert detail.json()["status"] != "approved"
    assert eligibility.json()["eligible"] is False


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



@pytest.mark.asyncio
async def test_proposal_review_api_lists_denver_pending_review(
    app_client, store_path: Path
) -> None:
    from tests.fixtures.denver_vertical_slice import run_denver_vertical_slice_async

    result = await run_denver_vertical_slice_async(store_path)
    proposal_id = result["record"].proposal_id

    transport = ASGITransport(app=app_client)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        listed = await client.get("/proposals")
        filtered = await client.get("/proposals", params={"status": "pending_review"})
        detail = await client.get(f"/proposals/{proposal_id}")
        eligibility = await client.get(f"/proposals/{proposal_id}/eligibility")

    assert listed.status_code == 200
    assert filtered.status_code == 200
    assert filtered.json()["count"] == 1
    assert filtered.json()["proposals"][0]["proposal_id"] == proposal_id
    assert any(row["proposal_id"] == proposal_id for row in listed.json()["proposals"])

    assert detail.status_code == 200
    detail_payload = detail.json()
    assert detail_payload["proposal_id"] == proposal_id
    assert detail_payload["status"] == "pending_review"
    assert detail_payload["envelope"]["proposal_type"] == "memory_correction_proposal"
    assert detail_payload["inner_artifact_summary"]["artifact_type"] == "MemoryCorrectionProposalV1"
    assert "denver" in detail_payload["inner_artifact_summary"]["current_belief"].lower()
    assert detail_payload["inner_artifact_summary"]["mutation_allowed"] is False
    assert detail_payload["inner_artifact_summary"]["requires_human_approval"] is True

    assert eligibility.status_code == 200
    assert eligibility.json()["eligible"] is False
    assert eligibility.json()["execution_requested"] is False
