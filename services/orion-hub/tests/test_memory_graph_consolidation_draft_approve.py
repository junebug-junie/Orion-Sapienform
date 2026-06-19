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

from scripts.memory_graph_routes import router

for key, value in {
    "CHANNEL_VOICE_TRANSCRIPT": "orion:voice:transcript",
    "CHANNEL_VOICE_LLM": "orion:voice:llm",
    "CHANNEL_VOICE_TTS": "orion:voice:tts",
    "CHANNEL_COLLAPSE_INTAKE": "orion:collapse:intake",
    "CHANNEL_COLLAPSE_TRIAGE": "orion:collapse:triage",
    "MEMORY_GRAPH_DEFAULT_NAMED_GRAPH": "http://example.org/graph/test",
    "GRAPHDB_URL": "http://graphdb.example",
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
    app.include_router(router)
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
    monkeypatch.setattr(
        "orion.memory_graph.rdf_target.resolve_memory_graph_rdf_target",
        lambda: SimpleNamespace(kind="fuseki"),
    )

    update_mock = AsyncMock(return_value={"draft_id": "draft-abc", "status": "approved"})
    monkeypatch.setattr(
        "orion.memory_graph.draft_repository.update_consolidation_draft_status",
        update_mock,
    )
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
