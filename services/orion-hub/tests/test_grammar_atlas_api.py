"""Hub Grammar Atlas read API (substrate trace/graph introspection)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]


def _ensure_hub_scripts_import_path() -> None:
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
    "DATABASE_URL": "postgresql://postgres:postgres@127.0.0.1:55432/conjourney",
    "GRAMMAR_ATLAS_ENABLED": "true",
}.items():
    os.environ.setdefault(key, value)


def _atlas_test_app() -> FastAPI:
    from scripts.grammar_atlas_routes import router

    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client() -> TestClient:
    import app.settings as hub_app_settings

    hub_app_settings.get_settings.cache_clear()
    _ensure_hub_scripts_import_path()
    return TestClient(_atlas_test_app())


async def _mock_with_session(fn):
    return fn(MagicMock())


def test_list_traces_empty(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import grammar_atlas_routes

    monkeypatch.setattr(grammar_atlas_routes, "_with_session", _mock_with_session)
    fake_query = type("Q", (), {"list_traces": staticmethod(lambda sess, session_id=None, limit=50: [])})
    monkeypatch.setattr(grammar_atlas_routes, "_grammar_query", lambda: fake_query)

    r = client.get("/api/substrate/atlas/traces?limit=10")
    assert r.status_code == 200
    assert r.json()["items"] == []


def test_get_trace_not_found(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import grammar_atlas_routes

    monkeypatch.setattr(grammar_atlas_routes, "_with_session", _mock_with_session)
    fake_query = type("Q", (), {"get_trace": staticmethod(lambda sess, trace_id: None)})
    monkeypatch.setattr(grammar_atlas_routes, "_grammar_query", lambda: fake_query)

    r = client.get("/api/substrate/atlas/traces/missing-trace")
    assert r.status_code == 404
    assert r.json()["detail"] == "trace_not_found"


def test_get_atom_not_found(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import grammar_atlas_routes

    monkeypatch.setattr(grammar_atlas_routes, "_with_session", _mock_with_session)
    fake_query = type("Q", (), {"get_atom_provenance": staticmethod(lambda sess, atom_id: None)})
    monkeypatch.setattr(grammar_atlas_routes, "_grammar_query", lambda: fake_query)

    r = client.get("/api/substrate/atlas/atoms/missing-atom/provenance")
    assert r.status_code == 404
    assert r.json()["detail"] == "atom_not_found"


def test_grammar_atlas_disabled(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAMMAR_ATLAS_ENABLED", "false")
    import app.settings as hub_app_settings

    hub_app_settings.get_settings.cache_clear()
    _ensure_hub_scripts_import_path()
    disabled_client = TestClient(_atlas_test_app())

    r = disabled_client.get("/api/substrate/atlas/traces?limit=10")
    assert r.status_code == 503
    assert r.json()["detail"] == "grammar_atlas_disabled"


def test_resolve_sql_writer_root_prefers_orion_repo_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import grammar_atlas_routes

    monkeypatch.setenv("ORION_REPO_ROOT", str(REPO_ROOT))
    resolved = grammar_atlas_routes._resolve_sql_writer_root()
    assert resolved == REPO_ROOT / "services" / "orion-sql-writer"
    assert resolved.is_dir()


def test_grammar_atlas_no_database_url(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("GRAMMAR_ATLAS_POSTGRES_URI", raising=False)
    import app.settings as hub_app_settings

    hub_app_settings.get_settings.cache_clear()
    _ensure_hub_scripts_import_path()
    no_db_client = TestClient(_atlas_test_app())

    r = no_db_client.get("/api/substrate/atlas/traces?limit=10")
    assert r.status_code == 503
    assert r.json()["detail"] == "grammar_atlas_database_unconfigured"
