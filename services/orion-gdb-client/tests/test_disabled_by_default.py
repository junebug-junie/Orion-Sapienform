from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]


def _reload_app(monkeypatch: pytest.MonkeyPatch) -> object:
    monkeypatch.setenv("ORION_BUS_URL", os.environ.get("ORION_BUS_URL", "redis://bus.test/0"))
    import app.settings as settings_mod
    import app.main as main_mod

    importlib.reload(settings_mod)
    importlib.reload(main_mod)
    return main_mod


def test_default_settings_gdb_client_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GDB_CLIENT_ENABLED", raising=False)
    main_mod = _reload_app(monkeypatch)
    assert main_mod.settings.GDB_CLIENT_ENABLED is False


def test_disabled_startup_skips_graphdb_and_listener(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GDB_CLIENT_ENABLED", "false")
    monkeypatch.setenv("ORION_BUS_ENABLED", "true")
    main_mod = _reload_app(monkeypatch)
    with (
        patch.object(main_mod, "wait_for_graphdb") as w,
        patch.object(main_mod, "ensure_repo_exists") as e,
        patch.object(main_mod, "listener_worker") as lw,
        patch.object(main_mod.threading, "Thread") as th,
    ):
        with TestClient(main_mod.app):
            pass
    w.assert_not_called()
    e.assert_not_called()
    lw.assert_not_called()
    th.assert_not_called()


def test_health_reports_enabled_false_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GDB_CLIENT_ENABLED", "false")
    main_mod = _reload_app(monkeypatch)
    with TestClient(main_mod.app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body.get("enabled") is False
