from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]
for candidate in (str(REPO_ROOT), str(HUB_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

for key, value in {
    "CHANNEL_VOICE_TRANSCRIPT": "orion:voice:transcript",
    "CHANNEL_VOICE_LLM": "orion:voice:llm",
    "CHANNEL_VOICE_TTS": "orion:voice:tts",
    "CHANNEL_COLLAPSE_INTAKE": "orion:collapse:intake",
    "CHANNEL_COLLAPSE_TRIAGE": "orion:collapse:triage",
}.items():
    os.environ.setdefault(key, value)


def test_memory_cards_returns_503_when_pool_unconfigured() -> None:
    import scripts.main as hub_main

    with TestClient(hub_main.app) as client:
        resp = client.post(
            "/api/memory/cards",
            json={
                "types": ["fact"],
                "title": "Test",
                "summary": "Body",
                "provenance": "operator_highlight",
            },
            headers={"X-Orion-Session-Id": "test-session"},
        )
    assert resp.status_code == 503
    assert resp.json().get("detail") == "memory_store_unavailable"


def test_memory_history_requires_filter() -> None:
    import scripts.main as hub_main

    with TestClient(hub_main.app) as client:
        resp = client.get("/api/memory/history", headers={"X-Orion-Session-Id": "test-session"})
    assert resp.status_code == 400


def test_memory_history_rejects_invalid_edge_id_uuid() -> None:
    import scripts.main as hub_main

    with TestClient(hub_main.app) as client:
        resp = client.get(
            "/api/memory/history?edge_id=not-a-uuid",
            headers={"X-Orion-Session-Id": "test-session"},
        )
    assert resp.status_code == 400
    assert resp.json().get("detail") == "invalid_edge_id"


def test_memory_reverse_rejects_invalid_history_id_uuid() -> None:
    import scripts.main as hub_main

    with TestClient(hub_main.app) as client:
        resp = client.post(
            "/api/memory/history/not-a-uuid/reverse",
            headers={"X-Orion-Session-Id": "test-session"},
        )
    assert resp.status_code == 400
    assert resp.json().get("detail") == "invalid_history_id"


def test_memory_distill_returns_501() -> None:
    import scripts.main as hub_main

    with TestClient(hub_main.app) as client:
        resp = client.post(
            "/api/memory/sessions/sid-1/distill",
            headers={"X-Orion-Session-Id": "test-session"},
        )
    assert resp.status_code == 501
