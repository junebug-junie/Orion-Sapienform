"""Tests for Hub cabinet cooling latest and history APIs."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]

for _key, _val in (
    ("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript"),
    ("CHANNEL_VOICE_LLM", "orion:voice:llm"),
    ("CHANNEL_VOICE_TTS", "orion:voice:tts"),
    ("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake"),
    ("CHANNEL_COLLAPSE_TRIAGE", "orion:collapse:triage"),
):
    os.environ.setdefault(_key, _val)


def _ensure_hub_scripts_import_path() -> None:
    for key in list(sys.modules):
        if key == "scripts" or key.startswith("scripts."):
            del sys.modules[key]
        if key == "app" or key.startswith("app."):
            del sys.modules[key]
    for path in (str(REPO_ROOT), str(HUB_ROOT)):
        try:
            sys.path.remove(path)
        except ValueError:
            pass
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(HUB_ROOT))


_ensure_hub_scripts_import_path()

from scripts import cabinet_cooling_routes  # noqa: E402


NOW = datetime(2026, 9, 25, 15, 0, 5, tzinfo=timezone.utc)
SAMPLE_TS = datetime(2026, 9, 25, 15, 0, 0, tzinfo=timezone.utc)


def _row(**overrides) -> dict:
    payload = {
        "ts": SAMPLE_TS,
        "cooling_watts": 412.5,
        "cooling_volts": 120.1,
        "switch_on": True,
        "controller_ready": True,
        "device_online": True,
        "payload_json": {
            "device": {"product": "Shelly Wave Plug", "online": True},
            "state": {"switch_on": True},
        },
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        cabinet_cooling_routes.settings, "CABINET_AMBIENT_HISTORY_NODE", "athena"
    )
    monkeypatch.setattr(
        cabinet_cooling_routes.settings, "CABINET_AMBIENT_HISTORY_MAX_POINTS", 800
    )
    monkeypatch.setattr(
        cabinet_cooling_routes.settings, "CABINET_SENSORS_STALE_AFTER_SEC", 30.0
    )
    monkeypatch.setattr(cabinet_cooling_routes, "_now_utc", lambda: NOW)

    async def no_rows(*, node: str, hours: int):
        assert node == "athena"
        assert hours == 24
        return []

    async def no_latest(*, node: str):
        assert node == "athena"
        return None

    monkeypatch.setattr(cabinet_cooling_routes, "_history_query", no_rows)
    monkeypatch.setattr(cabinet_cooling_routes, "_latest_query", no_latest)
    app = FastAPI()
    app.include_router(cabinet_cooling_routes.router)
    return TestClient(app)


def test_router_prefix():
    assert cabinet_cooling_routes.router.prefix == "/api/cabinet/cooling"


def test_parse_window_accepts_only_supported_windows():
    assert cabinet_cooling_routes.parse_window("24h") == 24
    assert cabinet_cooling_routes.parse_window("3d") == 72
    assert cabinet_cooling_routes.parse_window("7d") == 168
    with pytest.raises(ValueError):
        cabinet_cooling_routes.parse_window("1h")


def test_row_to_sample_omits_absent_optional_values():
    sample = cabinet_cooling_routes.row_to_sample(_row(cooling_watts=None, cooling_volts=None))
    assert sample["ts"] == "2026-09-25T15:00:00Z"
    assert "cooling_watts" not in sample
    assert "cooling_volts" not in sample
    assert sample["switch_on"] is True
    assert sample["product"] == "Shelly Wave Plug"


def test_rows_to_points_never_zero_fills_watts():
    points = cabinet_cooling_routes.rows_to_points(
        [{"t": SAMPLE_TS, "cooling_watts": None, "cooling_volts": 120.0}]
    )
    assert points == [{"t": "2026-09-25T15:00:00Z", "cooling_volts": 120.0}]


def test_latest_no_rows_returns_ok_false_and_null_sample(client):
    body = client.get("/api/cabinet/cooling/latest").json()
    assert body == {"ok": False, "age_sec": None, "sample": None}


def test_latest_fresh_row_returns_sample(client, monkeypatch):
    async def latest(*, node: str):
        return _row()

    monkeypatch.setattr(cabinet_cooling_routes, "_latest_query", latest)
    body = client.get("/api/cabinet/cooling/latest").json()
    assert body["ok"] is True
    assert body["age_sec"] == pytest.approx(5.0)
    assert body["sample"]["cooling_watts"] == 412.5
    assert body["sample"]["switch_on"] is True
    assert body["sample"]["product"] == "Shelly Wave Plug"


def test_latest_stale_keeps_last_sample(client, monkeypatch):
    async def latest(*, node: str):
        return _row(ts=datetime(2026, 9, 25, 14, 59, 0, tzinfo=timezone.utc))

    monkeypatch.setattr(cabinet_cooling_routes, "_latest_query", latest)
    body = client.get("/api/cabinet/cooling/latest").json()
    assert body["ok"] is False
    assert body["age_sec"] == pytest.approx(65.0)
    assert body["sample"]["cooling_watts"] == 412.5


def test_history_defaults_to_24h_and_returns_empty_points(client):
    body = client.get("/api/cabinet/cooling/history").json()
    assert body == {
        "ok": True,
        "node": "athena",
        "window": "24h",
        "points": [],
        "stats": {"n_raw": 0, "n": 0, "watts_min": None, "watts_max": None},
    }


def test_history_accepts_window_24h(client, monkeypatch):
    seen = {}

    async def rows(*, node: str, hours: int):
        seen["hours"] = hours
        return [
            {"t": SAMPLE_TS, "cooling_watts": 100.0, "cooling_volts": 119.0},
            {
                "t": datetime(2026, 9, 25, 15, 1, 0, tzinfo=timezone.utc),
                "cooling_watts": 400.0,
                "cooling_volts": 120.0,
            },
        ]

    monkeypatch.setattr(cabinet_cooling_routes, "_history_query", rows)
    body = client.get("/api/cabinet/cooling/history?window=24h").json()
    assert seen["hours"] == 24
    assert body["ok"] is True
    assert body["window"] == "24h"
    assert len(body["points"]) == 2
    assert body["stats"] == {
        "n_raw": 2,
        "n": 2,
        "watts_min": 100.0,
        "watts_max": 400.0,
    }


def test_history_invalid_window_returns_400(client):
    response = client.get("/api/cabinet/cooling/history?window=1h")
    assert response.status_code == 400


def test_history_db_failure_returns_ok_false(client, monkeypatch):
    async def failed(*, node: str, hours: int):
        raise OSError("db unavailable")

    monkeypatch.setattr(cabinet_cooling_routes, "_history_query", failed)
    body = client.get("/api/cabinet/cooling/history").json()
    assert body["ok"] is False
    assert body["points"] == []
    assert body["error"] == "cooling_history_unavailable"
