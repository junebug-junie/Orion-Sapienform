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


def test_memory_graph_approve_requires_named_graph(monkeypatch) -> None:
    """memory-graph approve no longer gates on an RDF backend (removed
    2026-07-22, see orion/memory_graph/approve.py) -- named_graph_iri is now
    the only remaining precondition before touching Postgres.

    Bare TestClient(hub_main.app) doesn't mock the DB pool, so whether this
    test can even reach the named_graph_iri_required check depends on
    whether a real Postgres pool attached during this run's lifespan
    (environment-dependent, not something this test controls) -- if it
    didn't, _pool() raises its own 503 "memory_store_unavailable" first.
    That's why the original version of this test (pre-2026-07-22, when it
    was gated on the now-removed RDF backend instead) already tolerated
    both outcomes rather than asserting a single status code.
    """
    _ensure_hub_scripts_import_path()
    monkeypatch.setenv("MEMORY_GRAPH_DEFAULT_NAMED_GRAPH", "")
    import importlib

    import app.settings as hub_app_settings

    hub_app_settings.get_settings.cache_clear()
    import scripts.settings as scripts_settings

    importlib.reload(scripts_settings)
    import scripts.main as hub_main

    importlib.reload(hub_main)

    with TestClient(hub_main.app) as client:
        resp = client.post(
            "/api/memory/graph/approve",
            json={"ontology_version": "orionmem-2026-05", "utterance_ids": [], "entities": [], "situations": [], "edges": [], "dispositions": []},
            headers={"X-Orion-Session-Id": "test-session"},
        )
    assert resp.status_code in (400, 503)
    if resp.status_code == 400:
        assert resp.json()["detail"] == "named_graph_iri_required"
    else:
        assert resp.json()["detail"] == "memory_store_unavailable"
