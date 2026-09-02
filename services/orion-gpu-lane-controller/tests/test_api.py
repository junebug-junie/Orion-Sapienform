"""Tests for app/main.py's HTTP surface: auth gate on POST
/v1/gpu-lane/flip, pass-through of GET /v1/gpu-lane/status, and the
fail-closed behavior when GPU_LANE_CONTROLLER_TOKEN is unset.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

SERVICE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = SERVICE_DIR / "app"
PACKAGE_NAME = "orion_gpu_lane_controller"
APP_PACKAGE_NAME = f"{PACKAGE_NAME}.app"
if PACKAGE_NAME not in sys.modules:
    pkg = types.ModuleType(PACKAGE_NAME)
    pkg.__path__ = [str(SERVICE_DIR)]
    sys.modules[PACKAGE_NAME] = pkg
if APP_PACKAGE_NAME not in sys.modules:
    pkg = types.ModuleType(APP_PACKAGE_NAME)
    pkg.__path__ = [str(APP_DIR)]
    sys.modules[APP_PACKAGE_NAME] = pkg

REPO_ROOT = SERVICE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{APP_PACKAGE_NAME}.{name}", APP_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


settings_module = _load("settings")
sys.modules[f"{APP_PACKAGE_NAME}.settings"] = settings_module
lane_control_module = _load("lane_control")
sys.modules[f"{APP_PACKAGE_NAME}.lane_control"] = lane_control_module
main_module = _load("main")


@pytest.fixture
def client(monkeypatch):
    # Heartbeat chassis talks to a real bus -- irrelevant to these HTTP
    # contract tests and not something to stand a real Redis up for.
    monkeypatch.setattr(main_module.settings, "ORION_BUS_ENABLED", False)
    monkeypatch.setattr(
        main_module, "build_heartbeat_chassis", lambda: (_ for _ in ()).throw(RuntimeError("no bus in tests"))
    )
    with TestClient(main_module.app) as c:
        yield c


def test_health_no_auth_needed(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_status_no_auth_needed(client, monkeypatch):
    monkeypatch.setattr(
        main_module.lane_control, "get_status", lambda: {"active": "agent", "affect": {}, "agent": {}}
    )
    resp = client.get("/v1/gpu-lane/status")
    assert resp.status_code == 200
    assert resp.json()["active"] == "agent"


def test_flip_fails_closed_when_token_unset(client, monkeypatch):
    monkeypatch.setattr(main_module.settings, "GPU_LANE_CONTROLLER_TOKEN", "")
    resp = client.post("/v1/gpu-lane/flip", json={"target": "agent"})
    assert resp.status_code == 503


def test_flip_rejects_missing_bearer_token(client, monkeypatch):
    monkeypatch.setattr(main_module.settings, "GPU_LANE_CONTROLLER_TOKEN", "secret-token")
    resp = client.post("/v1/gpu-lane/flip", json={"target": "agent"})
    assert resp.status_code == 401


def test_flip_rejects_wrong_bearer_token(client, monkeypatch):
    monkeypatch.setattr(main_module.settings, "GPU_LANE_CONTROLLER_TOKEN", "secret-token")
    resp = client.post(
        "/v1/gpu-lane/flip",
        json={"target": "agent"},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert resp.status_code == 401


def test_flip_accepts_correct_bearer_token(client, monkeypatch):
    monkeypatch.setattr(main_module.settings, "GPU_LANE_CONTROLLER_TOKEN", "secret-token")
    monkeypatch.setattr(
        main_module.lane_control,
        "flip",
        AsyncMock(return_value={"status": "noop", "target": "agent", "user_facing_summary": "already there"}),
    )
    resp = client.post(
        "/v1/gpu-lane/flip",
        json={"target": "agent"},
        headers={"Authorization": "Bearer secret-token"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "noop"


def test_flip_rejects_invalid_target_body(client, monkeypatch):
    monkeypatch.setattr(main_module.settings, "GPU_LANE_CONTROLLER_TOKEN", "secret-token")
    resp = client.post(
        "/v1/gpu-lane/flip",
        json={"target": "not-a-real-lane"},
        headers={"Authorization": "Bearer secret-token"},
    )
    assert resp.status_code == 422


def test_flip_returns_409_when_busy(client, monkeypatch):
    monkeypatch.setattr(main_module.settings, "GPU_LANE_CONTROLLER_TOKEN", "secret-token")
    monkeypatch.setattr(
        main_module.lane_control,
        "flip",
        AsyncMock(return_value={"status": "busy", "target": "agent", "user_facing_summary": "in progress"}),
    )
    resp = client.post(
        "/v1/gpu-lane/flip",
        json={"target": "agent"},
        headers={"Authorization": "Bearer secret-token"},
    )
    assert resp.status_code == 409


def test_flip_surfaces_failure_status_as_502(client, monkeypatch):
    monkeypatch.setattr(main_module.settings, "GPU_LANE_CONTROLLER_TOKEN", "secret-token")
    monkeypatch.setattr(
        main_module.lane_control,
        "flip",
        AsyncMock(return_value={"status": "stop_failed", "target": "agent", "user_facing_summary": "nope"}),
    )
    resp = client.post(
        "/v1/gpu-lane/flip",
        json={"target": "agent"},
        headers={"Authorization": "Bearer secret-token"},
    )
    assert resp.status_code == 502
