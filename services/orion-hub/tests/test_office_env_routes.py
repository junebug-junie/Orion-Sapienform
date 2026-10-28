"""Tests for Hub office environment latest and history APIs."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
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

from scripts import office_env_routes  # noqa: E402


NOW = datetime(2026, 10, 27, 20, 0, 30, tzinfo=timezone.utc)
SAMPLE_TS = datetime(2026, 10, 27, 20, 0, 0, tzinfo=timezone.utc)


def _row(**overrides) -> dict:
    row = {
        "ts": SAMPLE_TS,
        "office_temp_c": 21.4,
        "office_humidity_pct": 33.0,
        "sensor_online": True,
        "stale": False,
        "sample_age_sec": 2.0,
        "payload_json": None,
    }
    row.update(overrides)
    return row


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(office_env_routes.settings, "OFFICE_ENV_NODE", "pan-1")
    monkeypatch.setattr(office_env_routes.settings, "OFFICE_ENV_STALE_AFTER_SEC", 90.0)
    monkeypatch.setattr(office_env_routes.settings, "CABINET_AMBIENT_HISTORY_MAX_POINTS", 800)
    monkeypatch.setattr(office_env_routes, "_now_utc", lambda: NOW)

    async def no_rows(*, node: str, hours: int):
        assert node == "pan-1"
        assert hours == 24
        return []

    async def no_latest(*, node: str):
        assert node == "pan-1"
        return None

    monkeypatch.setattr(office_env_routes, "_history_query", no_rows)
    monkeypatch.setattr(office_env_routes, "_latest_query", no_latest)
    app = FastAPI()
    app.include_router(office_env_routes.router)
    return TestClient(app)


def test_router_prefix():
    assert office_env_routes.router.prefix == "/api/office/env"


def test_row_to_sample_omits_absent_readings():
    sample = office_env_routes.row_to_sample(
        _row(office_temp_c=None, office_humidity_pct=None, sensor_online=False, stale=True)
    )
    assert sample == {"ts": "2026-10-27T20:00:00Z", "sensor_online": False}


def test_rows_to_points_never_zero_fills():
    points = office_env_routes.rows_to_points(
        [
            {"t": SAMPLE_TS, "office_temp_c": 21.0, "office_humidity_pct": None},
            {"t": SAMPLE_TS, "office_temp_c": None, "office_humidity_pct": None},
        ]
    )
    assert points == [{"t": "2026-10-27T20:00:00Z", "office_temp_c": 21.0}]


def test_downsample_averages_buckets_and_skips_absent_fields():
    pts = [
        {"t": (SAMPLE_TS + timedelta(seconds=30 * i)).isoformat().replace("+00:00", "Z"), "office_temp_c": float(i)}
        for i in range(10)
    ]
    out = office_env_routes.downsample_points(pts, 2)
    assert [p["office_temp_c"] for p in out] == [2.0, 7.0]
    assert all("office_humidity_pct" not in p for p in out)


def test_latest_no_rows(client):
    assert client.get("/api/office/env/latest").json() == {"ok": False, "age_sec": None, "sample": None}


def test_latest_fresh_row_is_ok(client, monkeypatch):
    async def latest(*, node: str):
        return _row()

    monkeypatch.setattr(office_env_routes, "_latest_query", latest)
    body = client.get("/api/office/env/latest").json()
    assert body["ok"] is True
    assert body["age_sec"] == pytest.approx(30.0)
    assert body["sample"]["office_temp_c"] == 21.4
    assert body["sensor_stale"] is False


def test_latest_sensor_stale_row_is_not_ok_and_has_no_readings(client, monkeypatch):
    async def latest(*, node: str):
        return _row(office_temp_c=None, office_humidity_pct=None, sensor_online=False, stale=True)

    monkeypatch.setattr(office_env_routes, "_latest_query", latest)
    body = client.get("/api/office/env/latest").json()
    assert body["ok"] is False
    assert body["sensor_stale"] is True
    assert "office_temp_c" not in body["sample"]


def test_latest_old_row_is_not_ok_even_if_sensor_was_fine(client, monkeypatch):
    async def latest(*, node: str):
        return _row(ts=datetime(2026, 10, 27, 19, 0, 0, tzinfo=timezone.utc))

    monkeypatch.setattr(office_env_routes, "_latest_query", latest)
    body = client.get("/api/office/env/latest").json()
    assert body["ok"] is False
    assert body["sensor_stale"] is False


def test_history_rejects_unknown_window(client):
    assert client.get("/api/office/env/history?window=1h").status_code == 400


def test_history_returns_points_and_stats(client, monkeypatch):
    async def rows(*, node: str, hours: int):
        return [
            {"t": SAMPLE_TS, "office_temp_c": 20.0, "office_humidity_pct": 30.0},
            {"t": SAMPLE_TS + timedelta(seconds=30), "office_temp_c": 22.0, "office_humidity_pct": None},
        ]

    monkeypatch.setattr(office_env_routes, "_history_query", rows)
    body = client.get("/api/office/env/history?window=24h").json()
    assert body["ok"] is True
    assert len(body["points"]) == 2
    assert body["stats"]["temp_min"] == 20.0 and body["stats"]["temp_max"] == 22.0
    assert body["stats"]["humidity_min"] == 30.0 and body["stats"]["humidity_max"] == 30.0


def test_history_db_failure_is_ok_false_not_empty_success(client, monkeypatch):
    async def boom(*, node: str, hours: int):
        raise RuntimeError("db down")

    monkeypatch.setattr(office_env_routes, "_history_query", boom)
    body = client.get("/api/office/env/history?window=24h").json()
    assert body["ok"] is False
    assert body["error"] == "office_env_history_unavailable"
    assert body["points"] == []


def test_latest_publisher_dead_hides_the_old_reading(client, monkeypatch):
    """Publisher down 2h: the API must not hand the UI a temperature it could show as live."""
    async def latest(*, node: str):
        return _row(ts=datetime(2026, 10, 27, 18, 0, 0, tzinfo=timezone.utc))

    monkeypatch.setattr(office_env_routes, "_latest_query", latest)
    body = client.get("/api/office/env/latest").json()
    assert body["ok"] is False
    assert body["publisher_stale"] is True
    assert "office_temp_c" not in body["sample"]
    assert "office_humidity_pct" not in body["sample"]


def test_latest_db_failure_is_flagged_as_an_error(client, monkeypatch):
    async def boom(*, node: str):
        raise RuntimeError("db down")

    monkeypatch.setattr(office_env_routes, "_latest_query", boom)
    body = client.get("/api/office/env/latest").json()
    assert body["ok"] is False
    assert body["error"] == "office_env_latest_unavailable"


def test_downsample_never_averages_across_an_outage():
    from datetime import timedelta as td

    start = datetime(2026, 10, 27, 0, 0, 0, tzinfo=timezone.utc)
    iso = lambda s: (start + td(seconds=s)).isoformat().replace("+00:00", "Z")
    before = [{"t": iso(30 * i), "office_temp_c": 20.0} for i in range(100)]
    after = [{"t": iso(30 * i + 3 * 3600), "office_temp_c": 30.0} for i in range(100)]
    out = office_env_routes.downsample_points(before + after, 20)
    assert not any(p["office_temp_c"] not in (20.0, 30.0) for p in out)  # no 25.0 blend
    assert [p.get("gap") for p in out].count(True) == 1
    first_after = next(p for p in out if p.get("gap"))
    assert first_after["office_temp_c"] == 30.0
