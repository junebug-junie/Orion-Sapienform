from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

SERVICE_DIR = Path(__file__).resolve().parents[1]
ROOT = SERVICE_DIR.parents[1]
PYTHON = ROOT / "orion_dev" / "bin" / "python"


def _load_app(monkeypatch: pytest.MonkeyPatch, *, reload_settings: bool = False) -> object:
    if str(SERVICE_DIR) not in sys.path:
        sys.path.insert(0, str(SERVICE_DIR))
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    modules = ["app.main", "app.api", "app.proposal_review_api"]
    if reload_settings:
        modules.insert(0, "app.settings")
    for mod in modules:
        sys.modules.pop(mod, None)
    from app.main import app  # noqa: WPS433

    return app


@pytest.mark.asyncio
async def test_health(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = ASGITransport(app=_load_app(monkeypatch))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["service"] == "orion-context-exec"
    assert data["write_enabled"] is False
    assert data["max_depth"] == 1
    block = data["proposal_review_api"]
    assert "enabled" in block
    assert "store_configured" in block
    assert "store_path" in block
    assert "store_path_present" in block
    assert "store_parent_present" in block
    assert "store_parent_writable" in block
    assert "reject_store_path" in block
    assert "ok" in block
    assert "error" in block
    assert "bus_enabled" in data
    assert "dependencies" in data
    storage = data["storage"]
    assert storage["configured"] is True
    assert storage["root"]
    assert "dirs" in storage
    assert "runs" in storage["dirs"]
    assert "artifacts" in storage["dirs"]
    assert "ledger" in storage["dirs"]
    assert "workspaces" in storage["dirs"]
    assert "cache" in storage["dirs"]
    assert "tmp" in storage["dirs"]
    assert "run_ledger_enabled" in storage
    workspace = storage["workspace"]
    assert workspace["enabled"] is True
    assert workspace["root"]
    assert "present" in workspace
    assert "writable" in workspace
    assert "materialize_repo" in workspace


@pytest.mark.asyncio
async def test_health_proposal_review_block_store_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROPOSAL_REVIEW_API_ENABLED", "true")
    monkeypatch.setenv("PROPOSAL_LEDGER_STORE_PATH", "")
    app = _load_app(monkeypatch, reload_settings=True)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    block = resp.json()["proposal_review_api"]
    assert block["store_configured"] is False
    assert block["ok"] is False
    assert block["error"]


@pytest.mark.asyncio
async def test_health_proposal_review_block_store_configured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = tmp_path / "proposals.json"
    cli = ROOT / "scripts" / "orion_proposal_cli.py"
    subprocess.run(
        [
            str(PYTHON if PYTHON.exists() else sys.executable),
            str(cli),
            "seed-demo",
            "--store",
            str(store),
        ],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT)},
        check=True,
        capture_output=True,
    )
    monkeypatch.setenv("PROPOSAL_REVIEW_API_ENABLED", "true")
    monkeypatch.setenv("PROPOSAL_LEDGER_STORE_PATH", str(store))
    app = _load_app(monkeypatch, reload_settings=True)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    block = resp.json()["proposal_review_api"]
    assert block["store_configured"] is True
    assert block["store_path_present"] is True
    assert block["store_parent_present"] is True
    assert block["store_parent_writable"] is True
    assert block["ok"] is True
    assert block["error"] is None


@pytest.mark.asyncio
async def test_health_proposal_review_block_parent_writable_before_first_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = tmp_path / "ledger" / "orion-proposals.json"
    store.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("PROPOSAL_REVIEW_API_ENABLED", "true")
    monkeypatch.setenv("PROPOSAL_LEDGER_STORE_PATH", str(store))
    app = _load_app(monkeypatch, reload_settings=True)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    block = resp.json()["proposal_review_api"]
    assert block["store_configured"] is True
    assert block["store_path"] == str(store)
    assert block["store_path_present"] is False
    assert block["store_parent_present"] is True
    assert block["store_parent_writable"] is True
    assert block["reject_store_path"] == str(tmp_path / "ledger" / "orion-proposals.reject.json")
    assert block["ok"] is True
    assert block["error"] is None
