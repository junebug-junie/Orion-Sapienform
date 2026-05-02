from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]


def _ensure_hub_scripts_import_path() -> None:
    """Repo-root ``scripts/`` shadows Hub when pytest mixes repo tests with Hub tests (import order)."""
    for key in list(sys.modules):
        if key == "scripts" or key.startswith("scripts."):
            del sys.modules[key]
    for p in (str(REPO_ROOT), str(HUB_ROOT)):
        try:
            sys.path.remove(p)
        except ValueError:
            pass
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(HUB_ROOT))


_ensure_hub_scripts_import_path()

for key, value in {
    "CHANNEL_VOICE_TRANSCRIPT": "orion:voice:transcript",
    "CHANNEL_VOICE_LLM": "orion:voice:llm",
    "CHANNEL_VOICE_TTS": "orion:voice:tts",
    "CHANNEL_COLLAPSE_INTAKE": "orion:collapse:intake",
    "CHANNEL_COLLAPSE_TRIAGE": "orion:collapse:triage",
}.items():
    os.environ.setdefault(key, value)


@pytest.mark.skipif(not Path(REPO_ROOT / "tests/fixtures/memory_graph/joey_cats_draft.json").is_file(), reason="fixture missing")
def test_memory_graph_validate_fixture_roundtrip() -> None:
    _ensure_hub_scripts_import_path()
    import scripts.main as hub_main

    raw = json.loads((REPO_ROOT / "tests/fixtures/memory_graph/joey_cats_draft.json").read_text(encoding="utf-8"))
    with TestClient(hub_main.app) as client:
        resp = client.post(
            "/api/memory/graph/validate",
            json=raw,
            headers={"X-Orion-Session-Id": "test-session"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert "ok" in body and "violations" in body and "preview" in body


def test_memory_graph_approve_requires_graphdb_or_named_graph() -> None:
    _ensure_hub_scripts_import_path()
    import scripts.main as hub_main

    with TestClient(hub_main.app) as client:
        resp = client.post(
            "/api/memory/graph/approve",
            json={"ontology_version": "orionmem-2026-05", "utterance_ids": [], "entities": [], "situations": [], "edges": [], "dispositions": []},
            headers={"X-Orion-Session-Id": "test-session"},
        )
    assert resp.status_code in (400, 503)
