from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_status_counts(client: TestClient) -> None:
    response = client.get("/v1/status")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["enabled"] is True
    assert body["write_enabled"] is False
    assert body["counts"]["claims"] >= 2
    assert body["counts"]["specs"] >= 1
    assert body["counts"]["sources"] >= 1


def test_malformed_warning_not_crash(client: TestClient, corpus_root) -> None:
    bad = corpus_root / "claims" / "accepted" / "claim-broken.yaml"
    bad.write_text("type: claim\nid: [not-a-string\n", encoding="utf-8")

    from app.settings import Settings
    from app.service import KnowledgeForgeService
    from app.routers import v1

    service = KnowledgeForgeService(Settings())
    v1.reset_service(service)

    response = client.get("/v1/status")
    assert response.status_code == 200
    warnings = response.json()["warnings"]
    assert any("claim-broken.yaml" in warning for warning in warnings)


def test_claim_search(client: TestClient) -> None:
    response = client.get("/v1/claims/search", params={"q": "Fixture claim"})
    assert response.status_code == 200
    hits = response.json()
    assert len(hits) >= 1
    assert hits[0]["id"] == "claim:test:0001"


def test_compile_excludes_disputed_stale_by_default(client: TestClient) -> None:
    response = client.post(
        "/v1/context-packs/compile",
        json={
            "task": "Compile test pack",
            "target": "cursor",
            "spec_ids": ["spec:test:compile"],
            "claim_ids": ["claim:test:bad-ref"],
            "write_file": False,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "Fixture claim for store indexing" in body["content"]
    assert "References missing claim" not in body["content"]
    assert body["path"] is None


def test_compile_include_flags(client: TestClient, corpus_root) -> None:
    stale_path = corpus_root / "claims" / "accepted" / "claim-test-stale.yaml"
    stale_path.write_text(
        "\n".join(
            [
                "type: claim",
                "id: claim:test:stale",
                "statement: Stale fixture claim.",
                "status: stale",
                "source_refs:",
                "  - source:test:fixture",
                "confidence: low",
            ]
        ),
        encoding="utf-8",
    )

    from app.settings import Settings
    from app.service import KnowledgeForgeService
    from app.routers import v1

    v1.reset_service(KnowledgeForgeService(Settings()))

    response = client.post(
        "/v1/context-packs/compile",
        json={
            "task": "Include disputed and stale",
            "target": "cursor",
            "spec_ids": [],
            "claim_ids": ["claim:test:bad-ref", "claim:test:stale"],
            "include_disputed": True,
            "include_stale": True,
            "write_file": False,
        },
    )
    assert response.status_code == 200
    content = response.json()["content"]
    assert "References missing claim" in content
    assert "Stale fixture claim" in content


def test_write_disabled_returns_content_no_path(client: TestClient) -> None:
    response = client.post(
        "/v1/context-packs/compile",
        json={
            "task": "No write",
            "target": "cursor",
            "spec_ids": ["spec:test:compile"],
            "write_file": True,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["content"]
    assert body["path"] is None
    assert any("write disabled" in warning for warning in body["warnings"])


def test_write_enabled_writes_file(client: TestClient, corpus_root, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KNOWLEDGE_FORGE_WRITE_ENABLED", "true")

    from app.settings import Settings
    from app.service import KnowledgeForgeService
    from app.routers import v1

    v1.reset_service(KnowledgeForgeService(Settings()))

    response = client.post(
        "/v1/context-packs/compile",
        json={
            "task": "Write enabled pack",
            "target": "cursor",
            "spec_ids": ["spec:test:compile"],
            "write_file": True,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["path"] is not None
    assert body["path"].startswith("context_packs/cursor/")
    written = corpus_root / body["path"]
    assert written.is_file()
    assert "Expose GET /health" in written.read_text(encoding="utf-8")
