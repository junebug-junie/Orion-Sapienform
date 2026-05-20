from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_ideation_local_provider_deterministic(client: TestClient) -> None:
    response = client.post(
        "/v1/ideation/run",
        json={
            "task": "Critique Knowledge Forge v1 and propose v1.1",
            "mode": "arsonist_review",
            "input_paths": ["claims/accepted"],
            "write_review": False,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "local"
    assert body["status"] == "proposed"
    assert "Arsonist critique" in body["content"]
    assert body["artifact_path"] is None


def test_ideation_write_disabled_returns_content_no_artifact(client: TestClient) -> None:
    response = client.post(
        "/v1/ideation/run",
        json={
            "task": "Write disabled ideation",
            "mode": "spec_critique",
            "write_review": True,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["content"]
    assert body["artifact_path"] is None
    assert any("write disabled" in w for w in body["warnings"])


def test_ideation_write_enabled_writes_pending_review(
    client: TestClient, corpus_root, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KNOWLEDGE_FORGE_IDEATION_WRITE_ENABLED", "true")

    from app.settings import Settings
    from app.service import KnowledgeForgeService
    from app.routers import v1

    v1.reset_service(KnowledgeForgeService(Settings()))

    response = client.post(
        "/v1/ideation/run",
        json={
            "task": "Write enabled ideation review",
            "mode": "arsonist_review",
            "write_review": True,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["artifact_path"] is not None
    assert body["artifact_path"].startswith("reviews/pending/ideation-")
    written = corpus_root / body["artifact_path"]
    assert written.is_file()
    assert "not canonical truth" in written.read_text(encoding="utf-8")


def test_ideation_invalid_mode_rejected(client: TestClient) -> None:
    response = client.post(
        "/v1/ideation/run",
        json={"task": "bad mode", "mode": "not_a_real_mode", "write_review": False},
    )
    assert response.status_code == 422


def test_ideation_anthropic_missing_api_key_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KNOWLEDGE_FORGE_IDEATION_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    from app.settings import Settings

    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        Settings()


def test_ideation_never_writes_canonical_accepted_paths(
    client: TestClient, corpus_root, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KNOWLEDGE_FORGE_IDEATION_WRITE_ENABLED", "true")

    from app.ideation.writer import assert_safe_artifact_path, build_artifact_path
    from app.settings import Settings
    from app.service import KnowledgeForgeService
    from app.routers import v1

    v1.reset_service(KnowledgeForgeService(Settings()))

    accepted_before = list((corpus_root / "claims" / "accepted").glob("*.yaml"))
    response = client.post(
        "/v1/ideation/run",
        json={
            "task": "Canonical safety check",
            "mode": "arsonist_review",
            "write_review": True,
        },
    )
    assert response.status_code == 200
    accepted_after = list((corpus_root / "claims" / "accepted").glob("*.yaml"))
    assert accepted_before == accepted_after

    with pytest.raises(ValueError, match="reviews/pending"):
        assert_safe_artifact_path(
            corpus_root,
            corpus_root / "claims" / "accepted" / "evil.md",
        )

    with pytest.raises(ValueError, match="reviews/pending"):
        assert_safe_artifact_path(
            corpus_root,
            build_artifact_path(corpus_root, task="ok").parent.parent / "claims" / "accepted" / "x.md",
        )


def test_ideation_disabled_returns_503(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KNOWLEDGE_FORGE_IDEATION_ENABLED", "false")

    from app.settings import Settings
    from app.service import KnowledgeForgeService
    from app.routers import v1

    v1.reset_service(KnowledgeForgeService(Settings()))

    response = client.post(
        "/v1/ideation/run",
        json={"task": "disabled", "mode": "arsonist_review"},
    )
    assert response.status_code == 503
