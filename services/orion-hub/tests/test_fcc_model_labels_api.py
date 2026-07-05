"""FCC model labels API for agent-claude mode."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scripts.main import app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def test_fcc_model_labels_disabled_by_default(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.settings import settings

    monkeypatch.setattr(settings, "HUB_AGENT_CLAUDE_ENABLED", False, raising=False)
    resp = client.get("/api/fcc-model-labels")
    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] is False
    assert "labels" in body


def test_fcc_model_labels_reads_fixture_env(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from scripts.settings import settings

    env_file = tmp_path / ".env"
    env_file.write_text("MODEL_HAIKU=llamacpp/quick\n", encoding="utf-8")
    monkeypatch.setattr(settings, "HUB_AGENT_CLAUDE_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "HUB_FCC_ENV_PATH", str(env_file), raising=False)
    monkeypatch.setattr(settings, "HUB_FCC_SERVER_URL", "http://127.0.0.1:59999", raising=False)

    resp = client.get("/api/fcc-model-labels")
    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] is True
    assert body["labels"] == ["MODEL_HAIKU"]
    assert body["fcc_server_ok"] is False
