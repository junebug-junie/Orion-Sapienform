from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


SAMPLE_DOC = """# Test Design Doc

## Requirements

- Knowledge Forge should track changed design docs.
- Source ingest should propose claims, not accept them.

## Acceptance checks

- Dry run writes no files.
"""


def _write_sample(path: Path) -> None:
    path.write_text(SAMPLE_DOC, encoding="utf-8")


def test_ingest_dry_run_returns_content_no_review_path(client: TestClient, tmp_path: Path) -> None:
    src = tmp_path / "kf-test-design.md"
    _write_sample(src)
    response = client.post(
        "/v1/sources/ingest",
        json={
            "path": str(src),
            "source_id": "source:test-design-doc",
            "kind": "design_doc",
            "write_review": False,
            "dry_run": True,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["review_path"] is None
    assert body["source_path"] is None
    assert "# Source Delta Review" in body["content"]
    assert any("preview only" in w for w in body["warnings"])


def test_ingest_write_disabled_returns_null_paths(client: TestClient, tmp_path: Path) -> None:
    src = tmp_path / "kf-test-design.md"
    _write_sample(src)
    response = client.post(
        "/v1/sources/ingest",
        json={
            "path": str(src),
            "source_id": "source:test-design-doc",
            "kind": "design_doc",
            "write_review": True,
            "dry_run": False,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["review_path"] is None
    assert any("write disabled" in w for w in body["warnings"])


def test_ingest_write_requires_operator_token(
    corpus_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "kf-test-design.md"
    _write_sample(src)

    monkeypatch.setenv("KNOWLEDGE_FORGE_REPO_ROOT", str(corpus_root))
    monkeypatch.setenv("KNOWLEDGE_FORGE_WRITE_ENABLED", "true")
    monkeypatch.setenv("KNOWLEDGE_FORGE_OPERATOR_TOKEN", "secret-token")

    from app.settings import Settings
    from app.service import KnowledgeForgeService
    from app.routers import v1
    from app.main import app

    cfg = Settings()
    v1.reset_service(KnowledgeForgeService(cfg))
    write_client = TestClient(app)

    denied = write_client.post(
        "/v1/sources/ingest",
        json={
            "path": str(src),
            "source_id": "source:token-test",
            "kind": "design_doc",
            "write_review": True,
        },
    )
    assert denied.status_code == 401

    allowed = write_client.post(
        "/v1/sources/ingest",
        json={
            "path": str(src),
            "source_id": "source:token-test",
            "kind": "design_doc",
            "write_review": True,
        },
        headers={"X-Knowledge-Forge-Token": "secret-token"},
    )
    assert allowed.status_code == 200
    assert allowed.json()["review_path"] is not None


def test_ingest_write_enabled_creates_pending_review(
    client: TestClient, corpus_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "kf-test-design.md"
    _write_sample(src)

    monkeypatch.setenv("KNOWLEDGE_FORGE_WRITE_ENABLED", "true")
    from app.settings import Settings
    from app.service import KnowledgeForgeService
    from app.routers import v1

    cfg = Settings()
    v1.reset_service(KnowledgeForgeService(cfg))

    from app.main import app

    write_client = TestClient(app)
    response = write_client.post(
        "/v1/sources/ingest",
        json={
            "path": str(src),
            "source_id": "source:api-test-design",
            "kind": "design_doc",
            "write_review": True,
            "dry_run": False,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["review_path"] is not None
    assert body["review_path"].startswith("reviews/pending/source-delta-")
    assert (corpus_root / body["review_path"]).is_file()
    assert body["source_path"] == "raw/sources/api-test-design.md"
    assert (corpus_root / body["source_path"]).is_file()


def test_ingest_missing_path_returns_400(client: TestClient) -> None:
    response = client.post(
        "/v1/sources/ingest",
        json={
            "path": "/tmp/does-not-exist-kf-source.md",
            "source_id": "source:missing",
            "kind": "design_doc",
        },
    )
    assert response.status_code == 400
