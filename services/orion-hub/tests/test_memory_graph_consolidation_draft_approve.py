from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (str(REPO_ROOT), str(HUB_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)



def _router():
    # Imported per test: conftest purges scripts.* before every test, so a
    # module-level `router` belongs to a copy the string monkeypatches no
    # longer reach.
    from scripts.memory_graph_routes import router

    return router

for key, value in {
    "CHANNEL_VOICE_TRANSCRIPT": "orion:voice:transcript",
    "CHANNEL_VOICE_LLM": "orion:voice:llm",
    "CHANNEL_VOICE_TTS": "orion:voice:tts",
    "CHANNEL_COLLAPSE_INTAKE": "orion:collapse:intake",
    "CHANNEL_COLLAPSE_TRIAGE": "orion:collapse:triage",
    "MEMORY_GRAPH_DEFAULT_NAMED_GRAPH": "http://example.org/graph/test",
}.items():
    os.environ.setdefault(key, value)

_MIN_DRAFT = {
    "ontology_version": "orionmem-2026-05",
    "utterance_ids": [],
    "entities": [],
    "situations": [],
    "edges": [],
    "dispositions": [],
}


@pytest.fixture
def approve_client(monkeypatch):
    app = FastAPI()
    app.include_router(_router())
    pool = MagicMock()
    app.state.memory_pg_pool = pool

    async def _ensure_session(_sid, _bus=None):
        return "sess-1"

    fake_main = MagicMock()
    fake_main.bus = MagicMock()
    monkeypatch.setitem(sys.modules, "scripts.main", fake_main)
    monkeypatch.setattr("scripts.memory_graph_routes.ensure_session", _ensure_session)

    approve_result = SimpleNamespace(ok=True, violations=[], card_ids=["card-1"])
    monkeypatch.setattr(
        "scripts.memory_graph_routes.approve_memory_graph_draft",
        AsyncMock(return_value=approve_result),
    )
    monkeypatch.setattr(
        "scripts.memory_graph_routes.ensure_draft_utterance_text",
        lambda draft, supplemental=None: draft,
    )

    update_mock = AsyncMock(return_value={"draft_id": "draft-abc", "status": "approved"})
    monkeypatch.setattr(
        "orion.memory_graph.draft_repository.update_consolidation_draft_status",
        update_mock,
    )
    monkeypatch.setattr(
        "scripts.memory_graph_routes._consolidation_supplemental_utterance_text",
        AsyncMock(return_value={}),
    )
    # Approve also re-reads each new card and announces it to the crystallizer
    # (29be26815). That needs a real card store and bus; stub it and check the
    # card ids reach it.
    emit_mock = AsyncMock(return_value=None)
    monkeypatch.setattr("scripts.memory_graph_routes._emit_approved_cards_for_crystallizer", emit_mock)
    return TestClient(app), update_mock


def test_graph_approve_marks_consolidation_draft(approve_client) -> None:
    client, update_mock = approve_client
    resp = client.post(
        "/api/memory/graph/approve",
        json={"draft": _MIN_DRAFT, "consolidation_draft_id": "draft-abc"},
        headers={"X-Orion-Session-Id": "test-session"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["consolidation_draft_marked"] is True
    update_mock.assert_awaited_once()
    assert update_mock.await_args.kwargs["status"] == "approved"
    import scripts.memory_graph_routes as routes

    routes._emit_approved_cards_for_crystallizer.assert_awaited_once()
    assert routes._emit_approved_cards_for_crystallizer.await_args.args[1] == ["card-1"]


def test_graph_approve_reports_failed_consolidation_mark(approve_client, monkeypatch) -> None:
    client, update_mock = approve_client
    update_mock.return_value = None
    resp = client.post(
        "/api/memory/graph/approve",
        json={"draft": _MIN_DRAFT, "consolidation_draft_id": "draft-missing"},
        headers={"X-Orion-Session-Id": "test-session"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["consolidation_draft_marked"] is False
    assert body["consolidation_draft_status"] == "update_failed"


def test_graph_approve_without_consolidation_draft_omits_inbox_fields(approve_client) -> None:
    client, update_mock = approve_client
    resp = client.post(
        "/api/memory/graph/approve",
        json={"draft": _MIN_DRAFT},
        headers={"X-Orion-Session-Id": "test-session"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert "consolidation_draft_id" not in body
    assert "consolidation_draft_marked" not in body
    update_mock.assert_not_awaited()


def _validate_app(monkeypatch, *, with_pool: bool):
    app = FastAPI()
    app.include_router(_router())
    if with_pool:
        app.state.memory_pg_pool = MagicMock()

    async def _ensure_session(_sid, _bus=None):
        return "sess-1"

    fake_main = MagicMock()
    fake_main.bus = MagicMock()
    monkeypatch.setitem(sys.modules, "scripts.main", fake_main)
    monkeypatch.setattr("scripts.memory_graph_routes.ensure_session", _ensure_session)
    return TestClient(app)


def test_validate_plain_draft_does_not_need_the_memory_store(monkeypatch) -> None:
    """Regression: validate is a read-only preview; it 503'd for every draft whenever
    the memory Postgres pool was absent, though only consolidation drafts read it."""
    client = _validate_app(monkeypatch, with_pool=False)
    resp = client.post(
        "/api/memory/graph/validate",
        json={"draft": _MIN_DRAFT},
        headers={"X-Orion-Session-Id": "test-session"},
    )
    assert resp.status_code == 200, resp.text
    assert {"ok", "violations", "preview"} <= set(resp.json())


def test_validate_consolidation_draft_still_needs_the_memory_store(monkeypatch) -> None:
    client = _validate_app(monkeypatch, with_pool=False)
    resp = client.post(
        "/api/memory/graph/validate",
        json={"draft": _MIN_DRAFT, "consolidation_draft_id": "draft-abc"},
        headers={"X-Orion-Session-Id": "test-session"},
    )
    assert resp.status_code == 503
    assert resp.json()["detail"] == "memory_store_unavailable"
