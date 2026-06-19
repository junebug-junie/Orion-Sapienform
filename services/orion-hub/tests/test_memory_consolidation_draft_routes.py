from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (str(REPO_ROOT), str(HUB_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts.memory_consolidation_draft_routes import router

for key, value in {
    "CHANNEL_VOICE_TRANSCRIPT": "orion:voice:transcript",
    "CHANNEL_VOICE_LLM": "orion:voice:llm",
    "CHANNEL_VOICE_TTS": "orion:voice:tts",
    "CHANNEL_COLLAPSE_INTAKE": "orion:collapse:intake",
    "CHANNEL_COLLAPSE_TRIAGE": "orion:collapse:triage",
}.items():
    os.environ.setdefault(key, value)


def _sample_row(*, draft_id: str = "draft-1") -> dict[str, Any]:
    return {
        "draft_id": draft_id,
        "memory_window_id": "window-1",
        "status": "pending_review",
        "draft": {
            "ontology_version": "orionmem-2026-05",
            "utterance_ids": [],
            "entities": [{"id": "e1", "label": "Juniper", "entityKind": "person", "surfaceForms": ["hi"]}],
            "situations": [],
            "edges": [],
            "dispositions": [],
        },
        "turn_correlation_ids": ["corr-a", "corr-b"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    app.include_router(router)
    pool = MagicMock()
    app.state.memory_pg_pool = pool

    async def _need_session(_sid):
        return "sess-1"

    monkeypatch.setattr("scripts.memory_consolidation_draft_routes._need_session", _need_session)
    monkeypatch.setattr(
        "scripts.memory_consolidation_draft_routes.list_consolidation_drafts",
        AsyncMock(return_value=[_sample_row()]),
    )
    monkeypatch.setattr(
        "scripts.memory_consolidation_draft_routes.get_consolidation_draft",
        AsyncMock(return_value=_sample_row(draft_id="draft-2")),
    )
    monkeypatch.setattr(
        "scripts.memory_consolidation_draft_routes.update_consolidation_draft_status",
        AsyncMock(return_value={**_sample_row(), "status": "rejected"}),
    )
    return TestClient(app)


def test_list_consolidation_drafts(client: TestClient) -> None:
    resp = client.get(
        "/api/memory/consolidation/drafts",
        headers={"X-Orion-Session-Id": "test-session"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert body["items"][0]["draft_id"] == "draft-1"
    assert body["items"][0]["summary"]["entities"] == 1
    assert body["items"][0]["turn_count"] == 2
    assert "draft" not in body["items"][0]


def test_get_consolidation_draft(client: TestClient) -> None:
    resp = client.get(
        "/api/memory/consolidation/drafts/draft-2",
        headers={"X-Orion-Session-Id": "test-session"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["draft_id"] == "draft-2"
    assert isinstance(body.get("draft"), dict)


def test_set_consolidation_draft_status(client: TestClient) -> None:
    resp = client.post(
        "/api/memory/consolidation/drafts/draft-1/status",
        json={"status": "rejected"},
        headers={"X-Orion-Session-Id": "test-session"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "rejected"


def test_set_consolidation_draft_status_invalid(client: TestClient) -> None:
    resp = client.post(
        "/api/memory/consolidation/drafts/draft-1/status",
        json={"status": "bogus"},
        headers={"X-Orion-Session-Id": "test-session"},
    )
    assert resp.status_code == 400


def test_set_consolidation_draft_status_rejects_approved(client: TestClient) -> None:
    resp = client.post(
        "/api/memory/consolidation/drafts/draft-1/status",
        json={"status": "approved"},
        headers={"X-Orion-Session-Id": "test-session"},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "invalid_consolidation_draft_status"


def test_draft_repository_list_sql_shape() -> None:
    source = open(
        __import__("orion.memory_graph.draft_repository", fromlist=["draft_repository"]).__file__,
        encoding="utf-8",
    ).read()
    assert "FROM memory_graph_suggest_drafts" in source
    assert "async def list_consolidation_drafts" in source
    assert "async def update_consolidation_draft_status" in source
    assert "status = 'pending_review'" in source


def test_list_consolidation_drafts_memory_schema_missing(client: TestClient, monkeypatch) -> None:
    from scripts import memory_routes

    if memory_routes._AsyncpgUndefinedTableError is None:
        pytest.skip("asyncpg UndefinedTableError unavailable")

    exc = memory_routes._AsyncpgUndefinedTableError('relation "memory_graph_suggest_drafts" does not exist')

    async def _raise(*_args, **_kwargs):
        raise exc

    monkeypatch.setattr("scripts.memory_consolidation_draft_routes.list_consolidation_drafts", _raise)
    resp = client.get(
        "/api/memory/consolidation/drafts",
        headers={"X-Orion-Session-Id": "test-session"},
    )
    assert resp.status_code == 503
    assert resp.json()["detail"] == "memory_schema_missing"


@pytest.mark.parametrize(
    ("patch_target", "method", "path", "kwargs"),
    [
        (
            "get_consolidation_draft",
            "get",
            "/api/memory/consolidation/drafts/draft-x",
            {},
        ),
        (
            "update_consolidation_draft_status",
            "post",
            "/api/memory/consolidation/drafts/draft-x/status",
            {"json": {"status": "rejected"}},
        ),
    ],
)
def test_consolidation_draft_routes_memory_schema_missing(
    client: TestClient,
    monkeypatch,
    patch_target: str,
    method: str,
    path: str,
    kwargs: dict,
) -> None:
    from scripts import memory_routes

    if memory_routes._AsyncpgUndefinedTableError is None:
        pytest.skip("asyncpg UndefinedTableError unavailable")

    exc = memory_routes._AsyncpgUndefinedTableError('relation "memory_graph_suggest_drafts" does not exist')

    async def _raise(*_args, **_kwargs):
        raise exc

    monkeypatch.setattr(f"scripts.memory_consolidation_draft_routes.{patch_target}", _raise)
    resp = getattr(client, method)(
        path,
        headers={"X-Orion-Session-Id": "test-session"},
        **kwargs,
    )
    assert resp.status_code == 503
    assert resp.json()["detail"] == "memory_schema_missing"
